#!/usr/bin/env python

# https://towardsdatascience.com/this-is-how-reinforcement-learning-works-5080b3a335d6

import gym

env = gym.make('FrozenLake-v0')
#env = gym.make('CartPole-v0')
print('action_space', env.action_space)
print('observation_space', env.observation_space)
env.reset()
for _ in range(1):
#    env.render()
    observation, reward, done, info = env.step(env.action_space.sample()) # take a random action
    print('observation', observation)
    print('reward', reward)
    print('done', done)
    print('info', info)
env.close()
