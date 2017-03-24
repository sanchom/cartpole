from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import numpy as np
import tensorflow as tf

class DeepQLearner(object):
  def __init__(self, env, discount_factor):
    self.action_space = range(env.action_space.n)
    self.observation_size = np.prod(env.observation_space.shape)
    self.discount_factor = discount_factor

    self.x = tf.placeholder(tf.float32, shape=(1, self.observation_size))
    self.action = tf.placeholder(tf.int32, shape=(1))
    self.target = tf.placeholder(tf.float32, shape=(1))
    weights = tf.Variable(tf.random_normal([self.observation_size, len(self.action_space)],
                                           stddev= 1.0 / math.sqrt(float(self.observation_size))),
                          name='weights')
    bias = tf.Variable(tf.constant(0.01, shape=[len(self.action_space)]), name='biases')
    self.y = tf.nn.relu(tf.matmul(self.x, weights) + bias)

    action_q = self.y[0, self.action[0]]
    # TODO: This isn't learning. The loss is all over the
    # place. Probably because of the well-known issues with
    # non-stationary and correlated examples when sampling on-policy
    # from the environment. I need to use a big batch of examples
    # samples from a memory of historic observations in each training
    # iteration.
    self.loss = tf.nn.l2_loss(action_q - self.target)

    self.global_step = tf.Variable(0, name='global_step', trainable=False)
    self.train_op = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(
      self.loss, global_step=self.global_step)

  def get_action(self, state, explore_rate, session):
    x = np.reshape(np.array(state), [1, len(state)])
    if np.random.rand() < explore_rate:
      return np.random.choice(self.action_space)
    else:
      y = session.run(self.y, feed_dict={self.x: x})
      return np.argmax(y)

  def learn(self, starting_state, action, reward, state, learning_rate, session):
    # TODO: Add state/action/reward/state into the history.
    # Sometimes, playback a batch of sequences and do some learning.
    x = np.reshape(np.array(starting_state), [1, len(state)])
    action  = np.reshape(np.array(action), [1])

    next_x = np.reshape(np.array(state), [1, len(state)])
    next_y = session.run(self.y, feed_dict={self.x: next_x})

    target_y = np.reshape(np.array(reward + self.discount_factor * np.max(next_y)), [1])

    print(session.run(self.loss, feed_dict={self.x: x, self.action: action, self.target: target_y}))
    session.run(self.train_op, feed_dict={self.x: x, self.action: action, self.target: target_y})
