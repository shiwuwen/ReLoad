import tensorflow as tf 
import numpy as np
import time
import gc

from reload_environment import Environment
from reload_rl_DDPG import Actor, Critic, Memory


#####################  hyper parameters  ####################

MAX_EPISODES = 100
MAX_EP_STEPS = 20

LR_A = 0.001    # learning rate for actor
LR_C = 0.001    # learning rate for critic
GAMMA = 0.9     # reward discount

REPLACEMENT = [
    dict(name='soft', tau=0.01),
    dict(name='hard', rep_iter_a=600, rep_iter_c=500)
][0]            # you can try different target replacement strategies

MEMORY_CAPACITY = 500
BATCH_SIZE = 16

OUTPUT_GRAPH = False


env = Environment()
state_dim = env.state_dim
action_dim = env.action_dim

print(env.a, env.N, env.c)
print(env.g)

with tf.name_scope('S'):
	S = tf.placeholder(tf.float32, shape=[None, state_dim], name='s')
with tf.name_scope('R'):
	R = tf.placeholder(tf.float32, [None, 1], name='r')
with tf.name_scope('S_'):
	S_ = tf.placeholder(tf.float32, shape=[None, state_dim], name='s_')

sess = tf.Session()

actor = Actor(sess, action_dim, LR_A, REPLACEMENT, S, R, S_)
critic = Critic(sess, state_dim, action_dim, LR_C, GAMMA, REPLACEMENT, actor.a, actor.a_, S, R, S_)
actor.add_grad_to_graph(critic.a_grads)

sess.run(tf.global_variables_initializer())

M = Memory(MEMORY_CAPACITY, dims=2 * state_dim + action_dim + 1)

if OUTPUT_GRAPH:
	tf.summary.FileWriter("logs/", sess.graph)

t1 = time.time()
for i in range(MAX_EPISODES):
	s = env.reset()
	ep_reward = 0

	for j in range(MAX_EP_STEPS):

		a = actor.choose_action(s)
		s_, r = env.step(a)

		M.store_transition(s, a, r, s_)

		if M.pointer > MEMORY_CAPACITY:
			b_M = M.sample(BATCH_SIZE)
			b_s = b_M[:, :state_dim]
			b_a = b_M[:, state_dim:state_dim+action_dim]
			b_r = b_M[:, -state_dim-1: -state_dim]
			b_s_ = b_M[:, -state_dim:]

			critic.learn(b_s, b_a, b_r, b_s_)
			actor.learn(b_s)

		s = s_
		ep_reward += r

		if j == MAX_EP_STEPS-1:
			print('action probability: ', a)
			# print('state : ', s)
			print('reward : ', r)
			print('Episode:', i, ' Reward: %i' % int(ep_reward))

print('Running time: ', time.time()-t1)

del env
gc.collect()