from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import xrange

import gym
from gym import wrappers
import math
import numpy as np

# State-space taken from Matthew Chan
# (https://medium.com/@tuzzer/cart-pole-balancing-with-q-learning-b54c6068d947)
class QLearnerAgent(object):
  def __init__(self, env):
    self.q = {}
    self.action_space = range(env.action_space.n)
    self.state_bounds = list(zip(env.observation_space.low, env.observation_space.high))
    # The bounds for this dimension (x_dot) are too broad (basically
    # the entire range of float32). Let's only consider a small
    # portion of the range.
    self.state_bounds[1] = [-0.5, 0.5]
    # Similarly with the theta_dot range.
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
        relative_val = (val - low)
        bin_size = (high - low) / self.state_bins[i]
        b = int(relative_val / bin_size)
        discrete[i] = b + 1
    return tuple(discrete)

  def get_action(self, state, explore_rate):
    if state in self.q:
      if np.random.rand() < explore_rate:
        # Occasionally, just move randomly.
        return np.random.choice(self.action_space)
      else:
        # Most of the time, choose the till-now best guess about the
        # optimal action.
        return np.argmax(self.q[state])
    else:
      # We haven't seen this state before. Just pick randomly.
      return np.random.choice(self.action_space)

  def learn(self, starting_state, action, reward, state, learning_rate):
    try:
      best_q_forward = np.max(self.q[state])
    except KeyError:
      best_q_forward = 0.0

    discount_factor = 0.99
    try:
      original_q = self.q[starting_state][action]
    except KeyError:
      original_q = 0.0

    new_q = original_q + learning_rate * (reward + discount_factor * best_q_forward - original_q)
    try:
      self.q[starting_state][action] = new_q
    except KeyError:
      self.q[starting_state] = np.zeros_like(self.action_space, np.float32)
      self.q[starting_state][action] = new_q

def main():
  env = gym.make('CartPole-v0')
  env = wrappers.Monitor(env, '/tmp/cartpole_experiment')
  agent = QLearnerAgent(env)
  for episode in xrange(1000):
    observation = env.reset()
    state = agent._discretize_observation(observation)
    # Learning rate decay and exploration rate decay taken from
    # Matthew Chan
    # (https://medium.com/@tuzzer/cart-pole-balancing-with-q-learning-b54c6068d947)
    learning_rate = max(0.1, min(0.5, 1.0 - math.log10((episode + 1)/25)))
    explore_rate = max(0.01, min(1, 1.0 - math.log10((episode + 1)/25)))
    env.render()
    for i in xrange(1000):
      action = agent.get_action(state, explore_rate)
      observation, reward, done, _ = env.step(action)
      new_state = agent._discretize_observation(observation)
      agent.learn(state, action, reward, new_state, learning_rate)
      state = new_state
      env.render()
      if done:
        print('Episode {} finished after {} steps.'.format(episode + 1, i + 1))
        break

if __name__ == '__main__':
  main()
