import numpy as np
import tensorflow as tf
from tensorflow import keras
from tqdm import tqdm
from datetime import datetime
from tensorflow_probability.python.distributions.mvn_diag import MultivariateNormalDiag

layers = keras.layers

import scipy.signal
from self_balancer_env import SelfBalancerEnv
import json


def discounted_cumulative_sums(x, discount):
    # Discounted cumulative sums of vectors for computing rewards-to-go and advantage estimates
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


class Buffer:
    # Buffer for storing trajectories
    def __init__(self, observation_dimensions, action_dimensions, size, gamma=0.99, lam=0.95):
        # Buffer initialization
        self.observation_buffer = np.zeros(
            (size, observation_dimensions), dtype=np.float32
        )
        self.action_buffer = np.zeros((size, action_dimensions), dtype=np.float32)
        self.advantage_buffer = np.zeros(size, dtype=np.float32)
        self.reward_buffer = np.zeros(size, dtype=np.float32)
        self.return_buffer = np.zeros(size, dtype=np.float32)
        self.value_buffer = np.zeros(size, dtype=np.float32)
        self.logprobability_buffer = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.pointer, self.trajectory_start_index = 0, 0

    def store(self, observation, action, reward, value, logprobability):
        # Append one step of agent-environment interaction
        self.observation_buffer[self.pointer] = observation
        self.action_buffer[self.pointer] = action
        self.reward_buffer[self.pointer] = reward
        self.value_buffer[self.pointer] = value
        self.logprobability_buffer[self.pointer] = logprobability
        self.pointer += 1

    def finish_trajectory(self, last_value=0):
        # Finish the trajectory by computing advantage estimates and rewards-to-go
        path_slice = slice(self.trajectory_start_index, self.pointer)
        rewards = np.append(self.reward_buffer[path_slice], last_value)
        values = np.append(self.value_buffer[path_slice], last_value)

        deltas = rewards[:-1] + self.gamma * values[1:] - values[:-1]

        self.advantage_buffer[path_slice] = discounted_cumulative_sums(
            deltas, self.gamma * self.lam
        )
        self.return_buffer[path_slice] = discounted_cumulative_sums(
            rewards, self.gamma
        )[:-1]

        self.trajectory_start_index = self.pointer

    def get(self):
        # Get all data of the buffer and normalize the advantages
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
            self.logprobability_buffer,
        )


def mlp(x, sizes, activation=tf.tanh, output_activation=None):
    # Build a feedforward neural network
    for size in sizes[:-1]:
        x = layers.Dense(units=size, activation=activation)(x)
    return layers.Dense(units=sizes[-1], activation=output_activation)(x)


def logprobabilities(logits, action):
    # Compute the log-probabilities of taking actions a by using the logits (i.e. the output of the actor)

    mu, sigma = logits[0, :num_actions], logits[0, num_actions:]

    dist = MultivariateNormalDiag(mu, sigma)

    return dist.log_prob(action)


# Sample action from actor
@tf.function
def sample_action(observation):
    logits = actor(observation)

    mu, sigma = logits[0, :num_actions], logits[0, num_actions:]

    action = tf.random.normal((1, num_actions), mu, sigma)

    return logits, action


# Train the policy by maxizing the PPO-Clip objective
@tf.function
def train_policy(
        observation_buffer, action_buffer, logprobability_buffer, advantage_buffer
):
    with tf.GradientTape() as tape:  # Record operations for automatic differentiation.
        ratios = tf.exp(
            logprobabilities(actor(observation_buffer), action_buffer)
            - logprobability_buffer)

        surrogate1 = ratios * advantage_buffer
        cr = tf.keras.backend.clip(ratios, min_value=1 - clip_ratio,
                                   max_value=1 + clip_ratio)
        surrogate2 = np.transpose(cr) * advantage_buffer
        # loss is the mean of the minimum of either of the surrogates
        loss_actor = - tf.keras.backend.mean(tf.keras.backend.minimum(surrogate1, surrogate2))
        # entropy bonus in accordance with move37 explanation https://youtu.be/kWHSH2HgbNQ
        sigma = action_buffer[:, num_actions:]
        variance = tf.keras.backend.square(sigma)
        loss_entropy = entropy_loss_ratio * tf.keras.backend.mean(
            -(tf.keras.backend.log(2 * np.pi * variance) + 1) / 2)  # see move37 chap 9.5
        # total bonus is all losses combined. Add MSE-value-loss here as well?
        policy_loss = loss_actor + loss_entropy

    policy_grads = tape.gradient(policy_loss, actor.trainable_variables)
    policy_optimizer.apply_gradients(zip(policy_grads, actor.trainable_variables))

    kl = tf.reduce_mean(
        logprobability_buffer
        - logprobabilities(actor(observation_buffer), action_buffer)
    )
    kl = tf.reduce_sum(kl)
    return kl


# Train the value function by regression on mean-squared error
@tf.function
def train_value_function(observation_buffer, return_buffer):
    with tf.GradientTape() as tape:  # Record operations for automatic differentiation.
        value_loss = tf.reduce_mean((return_buffer - critic(observation_buffer)) ** 2)
    value_grads = tape.gradient(value_loss, critic.trainable_variables)
    value_optimizer.apply_gradients(zip(value_grads, critic.trainable_variables))


# Hyperparameters of the PPO algorithm
steps_per_epoch = 5000
epochs = 100
clip_ratio = 0.2
policy_learning_rate = 3e-4
value_function_learning_rate = 1e-3
train_policy_iterations = 80
train_value_iterations = 80
target_kl = 0.01
hidden_sizes = (64, 64)
entropy_loss_ratio = 0.001

# Initialize the environment and get the dimensionality of the
# observation space and the number of possible actions
env = SelfBalancerEnv()
observation_dimensions = env.observation_space.shape[0]
num_actions = env.action_space.shape[0]

# Initialize the buffer
buffer = Buffer(observation_dimensions, num_actions, steps_per_epoch)

# Initialize the actor as keras models
observation_input = keras.Input(shape=(observation_dimensions,), dtype=tf.float32)
dense = mlp(observation_input, list(hidden_sizes), tf.tanh, tf.tanh)
mu_layer = layers.Dense(units=num_actions, activation=tf.tanh)(dense)
sigma_layer = layers.Dense(units=num_actions, activation=tf.nn.softplus)(dense)
actor = keras.Model(inputs=observation_input, outputs=keras.layers.concatenate([mu_layer, sigma_layer]))
# actor = keras.models.load_model('saved_models/actor-continuous-2022-03-07 18:44:11.742029')
actor.summary()

# Initialize the critic as keras models
value = tf.squeeze(
    mlp(observation_input, list(hidden_sizes) + [1], tf.tanh, None), axis=1
)
critic = keras.Model(inputs=observation_input, outputs=value)
# critic = keras.models.load_model('saved_models/critic-continuous-2022-03-07 18:46:02.947460')
critic.summary()

# Initialize the policy and the value function optimizers
policy_optimizer = keras.optimizers.Adam(learning_rate=policy_learning_rate)
value_optimizer = keras.optimizers.Adam(learning_rate=value_function_learning_rate)

# Initialize the observation, episode return and episode length
observation, episode_return, episode_length = env.reset(), 0, 0

# True if you want to render the environment
render = True
render_freq = 1
noise_freq = 10
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
        if t % noise_freq == 0:
            env.noise()

        if render and t % render_freq == 0:
            env.render()

        # Get the logits, action, and take one step in the environment
        observation = observation.reshape(1, -1)
        logits, action = sample_action(observation)

        _action = action[0:, ]
        target_rotation = np.where((_action >= -np.pi / 2) & (_action <= np.pi / 2), _action, 0)

        observation_new, reward, done, _ = env.step(target_rotation)
        episode_return += reward
        episode_length += 1

        # Get the value and log-probability of the action
        value_t = critic(observation)
        logprobability_t = logprobabilities(logits, action)

        # Store obs, act, rew, v_t, logp_pi_t
        buffer.store(observation, action, reward, value_t, logprobability_t)

        # Update the observation
        observation = observation_new

        pb.set_description(f'Reward:{reward}')
        rewards.append(reward)

        # Finish trajectory if reached to a terminal state
        terminal = done
        if terminal or (t == steps_per_epoch - 1):
            last_value = 0 if done else critic(observation.reshape(1, -1))
            buffer.finish_trajectory(last_value)
            sum_return += episode_return
            sum_length += episode_length
            num_episodes += 1
            observation, episode_return, episode_length = env.reset(), 0, 0

    # Get values from the buffer
    (
        observation_buffer,
        action_buffer,
        advantage_buffer,
        return_buffer,
        logprobability_buffer,
    ) = buffer.get()

    # Update the policy and implement early stopping using KL divergence
    for _ in range(train_policy_iterations):
        kl = train_policy(
            observation_buffer, action_buffer, logprobability_buffer, advantage_buffer
        )
        if kl > 1.5 * target_kl:
            # Early Stopping
            break

    # Update the value function
    for _ in range(train_value_iterations):
        train_value_function(observation_buffer, return_buffer)

    # Print mean return and length for each epoch
    print(
        f" Epoch: {epoch + 1}. Mean Return: {sum_return / num_episodes}. Mean Length: {sum_length / num_episodes}"
    )

    epoch_rewards.append(rewards)

# save trained model
now = datetime.now()
actor.save(f'saved_models/actor-continuous-{now}')
critic.save(f'saved_models/critic-continuous-{now}')
with open(f'reports/hand-balance-continuous-rewards-{now}.txt', 'w') as f:
    f.write(json.dumps(epoch_rewards))
