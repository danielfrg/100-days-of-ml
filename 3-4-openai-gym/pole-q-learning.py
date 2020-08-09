import math

import gym
import numpy as np

env = gym.make("CartPole-v0")
env.reset()

buckets = (1, 1, 6, 12)
print(buckets + (env.action_space.n,))
Q = np.zeros(buckets + (env.action_space.n,))

n_episodes = 1000
alpha = 0.7
min_alpha = 0.1
ada_divisor = 25
min_epsilon = 0.1
gamma = 1.0


def get_alpha(t):
    return max(min_alpha, min(1.0, 1.0 - math.log10((t + 1) / ada_divisor)))


def get_epsilon(t):
    return max(min_epsilon, min(1, 1.0 - math.log10((t + 1) / ada_divisor)))


def choose_action(env, Q, state, epsilon):
    return (
        env.action_space.sample()
        if (np.random.random() <= epsilon)
        else np.argmax(Q[state])
    )


def discretize(env, obs):
    upper_bounds = [
        env.observation_space.high[0],
        0.5,
        env.observation_space.high[2],
        math.radians(50),
    ]
    lower_bounds = [
        env.observation_space.low[0],
        -0.5,
        env.observation_space.low[2],
        -math.radians(50),
    ]
    ratios = [
        (obs[i] + abs(lower_bounds[i])) / (upper_bounds[i] - lower_bounds[i])
        for i in range(len(obs))
    ]
    new_obs = [int(round((buckets[i] - 1) * ratios[i])) for i in range(len(obs))]
    new_obs = [min(buckets[i] - 1, max(0, new_obs[i])) for i in range(len(obs))]
    return tuple(new_obs)


for episode in range(n_episodes):
    state_ = env.reset()
    state = discretize(env, state_)
    alpha = get_alpha(episode)
    epsilon = get_epsilon(episode)

    done = False
    total_reward, reward = 0, 0

    while done != True:
        env.render()

        action = choose_action(env, Q, state, epsilon)
        state2_, reward, done, info = env.step(action)
        state2 = discretize(env, state2_)

        Q[state][action] += alpha * (
            reward + gamma * np.max(Q[state2]) - Q[state][action]
        )

        state = state2
        total_reward += reward

    if episode % 50 == 0:
        print("Episode {} Total Reward: {}".format(episode, total_reward))
# print(Q)
