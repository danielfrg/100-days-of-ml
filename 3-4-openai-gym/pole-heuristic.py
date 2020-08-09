import gym

env = gym.make("CartPole-v0")
env.reset()

n_iters = 100
n_episodes = 10
total_reward = 0

for i_episode in range(n_episodes):
    observation = env.reset()
    episode_reward = 0

    for t in range(n_iters):
        env.render()

        # action = env.action_space.sample()
        action = 1 if observation[2] > 0 else 0
        observation, reward, done, info = env.step(action)
        episode_reward += reward

        if done:
            print("Episode finished after {} timesteps".format(t + 1))
            total_reward += episode_reward
            break
env.close()

print("Avg reward:", total_reward / n_episodes)
