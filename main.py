from itertools import count

from self_balancer_env import SelfBalancerEnv

env = SelfBalancerEnv()

motor_x = 0
motor_y = 0
for i in count():
    env.render()

    observation, reward, done, info = env.step([motor_x, motor_y])

    # PROBLEM
    # A simple PID controller didn't work due to inertial force in mujoco
    # BUT hardware has enough damping/stopper in servo motor that let state to hold at one position
    #
    # SOLUTION
    # 1. Model a servo motor in mujoco OR
    # 2. Change servo motor to freely rotate and use torque instead of angle inputs

    if observation[0] < 0:
        motor_x += 1
    else:
        motor_x += -1

    if observation[1] < 0:
        motor_y += 1
    else:
        motor_y += -1

    print(reward)
