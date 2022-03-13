import numpy as np
import tensorflow as tf
from tensorflow import keras
from tqdm import tqdm
from datetime import datetime
from tensorflow_probability.python.distributions.mvn_diag import MultivariateNormalDiag
import scipy.signal
from self_balancer_env_v2 import SelfBalancerEnv_v2
import json

layers = keras.layers


# Buffer for storing trajectories
class Buffer:
    def __init__(self, observation_dimensions, action_dimensions, size, gamma, lam):
        self.observation_buffer = np.zeros(
            (size, observation_dimensions), dtype=np.float32
        )
        self.action_buffer = np.zeros((size, action_dimensions), dtype=np.float32)
        self.advantage_buffer = np.zeros(size, dtype=np.float32)
        self.reward_buffer = np.zeros(size, dtype=np.float32)
        self.return_buffer = np.zeros(size, dtype=np.float32)
        self.value_buffer = np.zeros(size, dtype=np.float32)
        self.log_probability_buffer = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.pointer, self.trajectory_start_index = 0, 0

    # Append one step of agent-environment interaction
    def store(self, observation, action, reward, value, log_probability):
        self.observation_buffer[self.pointer] = observation
        self.action_buffer[self.pointer] = action
        self.reward_buffer[self.pointer] = reward
        self.value_buffer[self.pointer] = value
        self.log_probability_buffer[self.pointer] = log_probability
        self.pointer += 1

    # Discounted cumulative sums of vectors for computing rewards-to-go and advantage estimates
    @staticmethod
    def _discounted_cumulative_sums(x, discount):
        return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

    # Finish the trajectory by computing advantage estimates and rewards-to-go
    def finish_trajectory(self, last_value=0):
        path_slice = slice(self.trajectory_start_index, self.pointer)
        rewards = np.append(self.reward_buffer[path_slice], last_value)
        values = np.append(self.value_buffer[path_slice], last_value)

        deltas = rewards[:-1] + self.gamma * values[1:] - values[:-1]

        self.advantage_buffer[path_slice] = self._discounted_cumulative_sums(
            deltas, self.gamma * self.lam
        )
        self.return_buffer[path_slice] = self._discounted_cumulative_sums(
            rewards, self.gamma
        )[:-1]

        self.trajectory_start_index = self.pointer

    # Get all data of the buffer and normalize the advantages
    def get(self):
        self.pointer, self.trajectory_start_index = 0, 0
        advantage_mean, advantage_std = (
            np.mean(self.advantage_buffer),
            np.std(self.advantage_buffer),
        )
        self.advantage_buffer = (self.advantage_buffer - advantage_mean) / advantage_std
        return (
            self.observation_buffer,
            self.action_buffer,
            self.advantage_buffer,
            self.return_buffer,
            self.log_probability_buffer,
        )


# Initialize the actor as keras models
class Actor(keras.Model):
    def __init__(self, observation_dimension, action_dimensions, policy_learning_rate, clip_ratio):
        super(Actor, self).__init__()
        input_layer = keras.Input(shape=(observation_dimension,), dtype=tf.float32)
        dense_layer_1 = layers.Dense(units=64, activation=tf.tanh)(input_layer)
        dense_layer_2 = layers.Dense(units=64, activation=tf.tanh)(dense_layer_1)
        mu_layer = layers.Dense(units=action_dimensions, activation=tf.tanh)(dense_layer_2)
        sigma_layer = layers.Dense(units=action_dimensions, activation=tf.nn.softplus)(dense_layer_2)
        self._actor = keras.Model(inputs=input_layer,
                                  outputs=keras.layers.concatenate([mu_layer, sigma_layer]))
        # self._actor = keras.models.load_model('saved_models/actor-continuous-2022-03-12 02:31:48.210542')

        self._policy_optimizer = keras.optimizers.Adam(learning_rate=policy_learning_rate)

        self.action_dimension = action_dimensions

        self.clip_ratio = clip_ratio

    # Train the policy by maximizing the PPO-Clip objective
    @tf.function
    def train_policy(self, observation_buffer, action_buffer, log_probability_buffer, advantage_buffer):
        with tf.GradientTape() as tape:  # Record operations for automatic differentiation.
            ratio = tf.exp(
                PPO.log_probability(self._actor(observation_buffer), action_buffer, self.action_dimension)
                - log_probability_buffer
            )
            min_advantage = tf.where(
                advantage_buffer > 0,
                (1 + self.clip_ratio) * advantage_buffer,
                (1 - self.clip_ratio) * advantage_buffer,
            )
            policy_loss = -tf.reduce_mean(tf.minimum(ratio * advantage_buffer, min_advantage))

        policy_grads = tape.gradient(policy_loss, self._actor.trainable_variables)
        self._policy_optimizer.apply_gradients(zip(policy_grads, self._actor.trainable_variables))

        kl = tf.reduce_mean(
            log_probability_buffer
            - PPO.log_probability(self._actor(observation_buffer), action_buffer, self.action_dimension)
        )
        kl = tf.reduce_sum(kl)
        return kl

    def call(self, observation):
        return self._actor(observation)

    def summary(self):
        self._actor.summary()

    def save(self, path):
        self._actor.save(path)


class Critic(keras.Model):
    def __init__(self, observation_dimension, value_function_learning_rate):
        super(Critic, self).__init__()
        input_layer = keras.Input(shape=(observation_dimension,), dtype=tf.float32)
        dense_layer_1 = layers.Dense(units=64, activation=tf.tanh)(input_layer)
        dense_layer_2 = layers.Dense(units=64, activation=tf.tanh)(dense_layer_1)
        value_layer = layers.Dense(units=1, activation=None)(dense_layer_2)

        self._critic = keras.Model(inputs=input_layer, outputs=value_layer)
        # self._critic = keras.models.load_model('saved_models/critic-continuous-2022-03-12 02:31:48.210542')

        self.value_optimizer = keras.optimizers.Adam(learning_rate=value_function_learning_rate)

    # Train the value function by regression on mean-squared error
    @tf.function
    def train_value_function(self, observation_buffer, return_buffer):
        with tf.GradientTape() as tape:  # Record operations for automatic differentiation.
            value_loss = tf.reduce_mean((return_buffer - self._critic(observation_buffer)) ** 2)
        value_grads = tape.gradient(value_loss, self._critic.trainable_variables)
        self.value_optimizer.apply_gradients(zip(value_grads, self._critic.trainable_variables))

    def call(self, observation):
        return self._critic(observation)

    def summary(self):
        self._critic.summary()

    def save(self, path):
        self._critic.save(path)


# PPO algorithm
class PPO:
    def __init__(self,
                 observation_dimension,
                 action_dimension,
                 steps_per_epoch,
                 clip_ratio,
                 policy_learning_rate,
                 value_function_learning_rate,
                 train_policy_iterations,
                 train_value_iterations,
                 target_kl,
                 gamma,
                 lam):
        self.actor = Actor(observation_dimension, action_dimension, policy_learning_rate, clip_ratio)
        self.critic = Critic(observation_dimension, value_function_learning_rate)
        self.buffer = Buffer(observation_dimension, action_dimension, steps_per_epoch, gamma, lam)

        self.action_dimension = action_dimension
        self._steps_per_epoch = steps_per_epoch
        self.train_policy_iterations = train_policy_iterations
        self.train_value_iterations = train_value_iterations
        self.target_kl = target_kl

    # Sample action from actor, returns probability distribution for each actions and sampled action from it
    @tf.function
    def sample_action(self, observation):
        logits = self.actor(observation)

        mu, sigma = logits[0, :self.action_dimension], logits[0, self.action_dimension:]

        action = tf.random.normal((1, self.action_dimension), mu, sigma)

        return logits, action

    # Compute the log-probability of taking actions a by using joint-distribution of logits
    @staticmethod
    def log_probability(logits, action, action_dimension):
        mu, sigma = logits[0, :action_dimension], logits[0, action_dimension:]
        dist = MultivariateNormalDiag(mu, sigma)
        return dist.log_prob(action)

    # Get the value and log-probability of the action
    def store_transition(self, observation, logits, action, reward):
        value_t = self.critic(observation)
        log_probability_t = PPO.log_probability(logits, action, self.action_dimension)
        self.buffer.store(observation, action, reward, value_t, log_probability_t)

    def finish_trajectory(self, observation):
        last_value = 0 if done else self.critic(observation)
        self.buffer.finish_trajectory(last_value)

    def train_network(self):
        (
            observation_buffer,
            action_buffer,
            advantage_buffer,
            return_buffer,
            log_probability_buffer,
        ) = self.buffer.get()

        # Update the policy and implement early stopping using KL divergence
        for _ in range(self.train_policy_iterations):
            kl = self.actor.train_policy(observation_buffer,
                                         action_buffer,
                                         log_probability_buffer,
                                         advantage_buffer)
            if kl > 1.5 * self.target_kl:
                # Early Stopping
                break

        # Update the value function
        for _ in range(self.train_value_iterations):
            self.critic.train_value_function(observation_buffer, return_buffer)
        pass

    # Save both the models
    def save_models(self):
        now = datetime.now()
        self.actor.save(f'models/actor-continuous-{now}')
        self.critic.save(f'models/critic-continuous-{now}')
        with open(f'reports/hand-balance-continuous-rewards-{now}.txt', 'w') as f:
            f.write(json.dumps(epoch_rewards))


# Initialize the environment
env = SelfBalancerEnv_v2()
observation, episode_return, episode_length = env.reset(), 0, 0

steps_per_epoch = 4000

# Initialize PPO algorithm with hyperparameter
ppo = PPO(observation_dimension=env.observation_space.shape[0],
          action_dimension=env.action_space.shape[0],
          steps_per_epoch=steps_per_epoch,
          clip_ratio=0.2,
          policy_learning_rate=3e-4,
          value_function_learning_rate=1e-3,
          train_policy_iterations=80,
          train_value_iterations=80,
          target_kl=0.01,
          gamma=0.99,
          lam=0.95)

render = True
noise = False
render_freq = 1
noise_freq = 1
epochs = 200

# Iterate over the number of epochs
pb = tqdm(tqdm(range(epochs)))
epoch_rewards = []
for epoch in pb:
    # Initialize the sum of the returns, lengths and number of episodes for each epoch
    sum_return = 0
    sum_length = 0
    num_episodes = 0

    rewards = []

    # Iterate over the steps of each epoch
    for t in range(steps_per_epoch):
        if noise and t % noise_freq == 0:
            env.noise()

        if render and t % render_freq == 0:
            env.render()

        observation = observation.reshape(1, -1)
        logits, action = ppo.sample_action(observation)

        observation_new, reward, done, _ = env.step(tf.reshape(action, -1))
        episode_return += reward
        episode_length += 1

        ppo.store_transition(observation, logits, action, reward)

        # Update the observation
        observation = observation_new

        rewards.append(reward)

        # print(reward)

        # Finish trajectory if reached to a terminal state
        terminal = done
        if terminal or t == steps_per_epoch - 1:
            ppo.finish_trajectory(observation.reshape(1, -1))

            sum_return += episode_return
            sum_length += episode_length
            num_episodes += 1
            observation, episode_return, episode_length = env.reset(), 0, 0

    # Train the network (update actor and critic using loss)
    ppo.train_network()

    # Print mean return and length for each epoch
    mean_return = sum_return / num_episodes
    mean_length = sum_length / num_episodes
    pb.set_description(f" Epoch:{epoch + 1}, Mean Reward:{mean_return / mean_length}, "
                       f"Mean Return:{mean_return}, Mean Length:{mean_length}")

    # Save reward of each time step for every epochs
    epoch_rewards.append(rewards)

ppo.save_models()
