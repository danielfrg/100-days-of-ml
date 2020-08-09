import gym
import numpy as np

env = gym.make("Taxi-v3")
env.reset()

Q = np.zeros([env.observation_space.n, env.action_space.n])
print(Q)

n_episodes = 2000
alpha = 0.7

for episode in range(n_episodes):
    done = False
    total_reward, reward = 0, 0
    state = env.reset()

    while done != True:
        action = np.argmax(Q[state])
        state2, reward, done, info = env.step(action)
        Q[state, action] = (1 - alpha) * Q[state, action] + alpha * (
            reward + np.max(Q[state2])
        )
        total_reward += reward
        state = state2

    if episode % 50 == 0:
        print("Episode {} Total Reward: {}".format(episode, total_reward))
print(Q)
