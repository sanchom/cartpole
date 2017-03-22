from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

class StateBinner:
  @classmethod
  def get_binner(classname, environment_name, env):
    if environment_name == 'CartPole-v0':
      return CartpoleStateBinner(env)
    if environment_name == 'Acrobot-v1':
      return AcrobotStateBinner(env)

# State-space taken from Matthew Chan
# (https://medium.com/@tuzzer/cart-pole-balancing-with-q-learning-b54c6068d947)
class CartpoleStateBinner(object):
  def __init__(self, env):
    self.state_bounds = list(zip(env.observation_space.low, env.observation_space.high))
    # The bounds for this dimension (x_dot) are too broad (basically
    # the entire range of float32). Let's only consider a small
    # portion of the range to be relevant.
    self.state_bounds[1] = [-0.5, 0.5]
    # Similarly with the theta_dot range.
    self.state_bounds[3] = [-math.radians(15), math.radians(15)]
    self.state_bins = [1, 1, 6, 2]

  def discretize(self, observation):
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

class AcrobotStateBinner(object):
  def __init__(self, env):
    self.state_bounds = list(zip(env.observation_space.low, env.observation_space.high))
    self.state_bins = [5, 5, 5, 5, 10, 10]

  def discretize(self, observation):
    discrete = [0, 0, 0, 0, 0, 0]
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
