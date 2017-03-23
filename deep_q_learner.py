from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

class DeepQLearner(object):
  def __init__(self, env, discount_factor):
    # TODO(sanchom): Define network
    self.action_space = range(env.action_space.n)
    self.discount_factor = discount_factor

  def get_action(self, state, explore_rate):
    # Feed state through network and choose max action.
    # Or, choose a random action.
    return np.random.choice(self.action_space)
    pass

  def learn(self, starting_state, action, reward, state, learning_rate):
    # Add state/action/reward/state into the history.
    # Sometimes, playback a batch of sequences and do some learning.
    pass
