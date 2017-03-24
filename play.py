from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import xrange

import math
import numpy as np

import tensorflow as tf
flags = tf.app.flags
FLAGS = flags.FLAGS

import gym
from gym import wrappers

from q_learner import QLearner
from deep_q_learner import DeepQLearner
from states import BinnerFactory

flags.DEFINE_string('environment', '', 'The OpenAI environment to load.')
flags.DEFINE_integer('render_every', 0, 'Render every x iterations.')
flags.DEFINE_float('discount_factor', 0.99, 'The discount factor to use for the QLearner.')
flags.DEFINE_integer('max_episodes', 100000, 'The maximum number of episodes to run.')
flags.DEFINE_integer('max_steps', 2000, 'The maximum number of steps to run per episode.')
flags.DEFINE_float('min_learning_rate', 0.1, 'The smallest that learning rate will decay to.')
flags.DEFINE_float('starting_learning_rate', 0.5, 'The initial learning rate.')
flags.DEFINE_float('min_exploration_rate', 0.01, 'The smallest that exploration rate will decay to.')
flags.DEFINE_float('starting_exploration_rate', 0.2, 'The initial exploration rate.')
flags.DEFINE_string('agent', 'q-learner', 'One of [\'q-learner\', \'deep-q-learner\'].')

def render(env, iteration):
  if FLAGS.render_every and iteration % FLAGS.render_every == 0:
    env.render()

# Form of rate decay taken from Matthew Chan.
# (https://medium.com/@tuzzer/cart-pole-balancing-with-q-learning-b54c6068d947)
def get_learning_rate(episode):
  return max(FLAGS.min_learning_rate,
             min(FLAGS.starting_learning_rate, 1.0 - math.log10((episode + 1) / 25)))

def get_explore_rate(episode):
  return max(FLAGS.min_exploration_rate,
             min(FLAGS.starting_exploration_rate, 1.0 - math.log10((episode + 1) / 25)))

def get_agent(environment):
  if FLAGS.agent == 'q-learner':
    return QLearner(environment, FLAGS.discount_factor)
  elif FLAGS.agent == 'deep-q-learner':
    return DeepQLearner(environment, FLAGS.discount_factor)
  else:
    raise ValueError('Unknown agent: {}'.format(FLAGS.agent))

def main(_):
  # Environment-specific setup.
  env = gym.make(FLAGS.environment)
  binner = BinnerFactory.get_binner(FLAGS.environment, env)
  agent = get_agent(env)


  with tf.train.MonitoredTrainingSession(
      checkpoint_dir="/tmp/deep_q_learner"
  ) as session:
    for episode in xrange(FLAGS.max_episodes):
      observation = env.reset()
      # state = binner.discretize(observation)
      state = observation
      learning_rate = get_learning_rate(episode)
      explore_rate = get_explore_rate(episode)
      render(env, episode)
      cumulative_reward = 0
      for i in xrange(FLAGS.max_steps):
        action = agent.get_action(state, explore_rate, session)
        observation, reward, done, _ = env.step(action)
        cumulative_reward += reward
        # new_state = binner.discretize(observation)
        new_state = observation
        agent.learn(state, action, reward, new_state, learning_rate, session)
        state = new_state
        render(env, episode)
        if done:
          print('Episode {} finished after {} steps. Reward = {}'.format(episode + 1, i + 1, cumulative_reward))
          break

if __name__ == '__main__':
  tf.app.run()
