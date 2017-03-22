from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

class QLearner(object):
  def __init__(self, env, discount_factor):
    self.q = {}
    self.action_space = range(env.action_space.n)
    self.discount_factor = discount_factor

  def get_action(self, state, explore_rate):
    if state in self.q:
      if np.random.rand() < explore_rate:
        # Occasionally, move randomly.
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

    try:
      original_q = self.q[starting_state][action]
    except KeyError:
      original_q = 0.0

    new_q = original_q + learning_rate * (reward + self.discount_factor * best_q_forward - original_q)
    try:
      self.q[starting_state][action] = new_q
    except KeyError:
      self.q[starting_state] = np.zeros_like(self.action_space, np.float32)
      self.q[starting_state][action] = new_q
