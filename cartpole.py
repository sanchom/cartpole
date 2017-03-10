from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import xrange

import gym
import math
import numpy as np
import tensorflow as tf

class QLearnerAgent(object):
  def __init__(self, env):
    self.q = {}
    self.action_space = [0, 1]
    self.state_bounds = list(zip(env.observation_space.low, env.observation_space.high))
    self.state_bounds[1] = [-0.5, 0.5]
    self.state_bounds[3] = [-math.radians(50), math.radians(50)]
    self.state_bins = [1, 1, 6, 3]

  def _discretize_observation(self, observation):
    discrete = [0, 0, 0, 0]
    for i, val in enumerate(observation):
      low, high = self.state_bounds[i]
      if val < low:
        discrete[i] = 0
      elif val > high:
        discrete[i] = self.state_bins[i] + 1
      else:
        relative_val = (val - low) / (high - low)
        bin_size = (high - low) / self.state_bins[i]
        b = int(relative_val / bin_size) + 1
        discrete[i] = b

    return tuple(discrete)

  def get_action(self, observation, explore_rate):
    discrete_observation = self._discretize_observation(observation)
    if discrete_observation in self.q:
      if np.random.rand() < explore_rate:
        # Occasionally, just move randomly.
        return np.random.choice(self.action_space)
      else:
        # Most of the time, choose the till-now best guess about the
        # optimal action.
        return np.argmax(self.q[discrete_observation])
    else:
      # We haven't seen this state before. Just pick randomly.
      return np.random.choice(self.action_space)

  def learn(self, observation, action, reward, new_observation):
    discrete_observation = self._discretize_observation(observation)
    discrete_new_observation = self._discretize_observation(new_observation)
    if discrete_observation in self.q:
      old_q = self.q[discrete_observation][action]
    else:
      old_q = 0
    if discrete_new_observation in self.q:
      now_q = self.q[discrete_new_observation]
    else:
      now_q = np.zeros_like(self.action_space, np.float32)
    
    learning_rate = 1
    discount_factor = 0.99
    new_q = old_q + learning_rate * (reward + discount_factor * np.max(now_q) - old_q)
    if discrete_observation in self.q:
      self.q[discrete_observation][action] = new_q
    else:
      self.q[discrete_observation] = np.zeros_like(self.action_space, np.float32)
      self.q[discrete_observation][action] = new_q

def main(_):
  env = gym.make('CartPole-v0')
  agent = QLearnerAgent(env)
  explore_rate = 0.2
  for episode in xrange(20000):
    explore_rate *= 0.99
    observation = env.reset()
    #env.render()
    for i in xrange(1000):
      action = agent.get_action(observation, explore_rate)
      new_observation, reward, done, _ = env.step(action)
      agent.learn(observation, action, reward, new_observation)
      observation = new_observation
      #env.render()
      if done:
        print('Episode {} finished after {} steps.'.format(episode + 1, i + 1))
        break

if __name__ == '__main__':
  tf.app.run()
