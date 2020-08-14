import tensorflow as tf 
import numpy as np
import time
import gc
import matplotlib.pyplot as plt

from reload_environment import Environment
from reload_rl_DDPG import Actor_ddpg, Critic_ddpg, Memory_ddpg
from reload_rl_AC import Actor_ac, Critic_ac


def rl_ac(env):

	reward = []

	sess = tf.Session()

	actor_ac = Actor_ac(sess, state_dim, action_dim, LR_A)
	critic_ac = Critic_ac(sess, state_dim, action_dim, LR_C)

	sess.run(tf.global_variables_initializer())

	if OUTPUT_GRAPH:
		tf.summary.FileWriter('./logs/', sess.graph)

	t1 = time.time()
	for i in range(MAX_EPISODES):
		s = env.reset()
		ep_reward = 0

		for step in range(MAX_EP_STEPS):

			a = actor_ac.choose_action(s)
			# a = np.zeros_like(a_temp)
			# a[np.argmax(a_temp)] = 1

			s_, r = env.step(a)
			ep_reward += r

			td_error = critic_ac.learn(s, r, s_)
			actor_ac.learn(s, a, td_error)

			s = s_

			# if step == MAX_EP_STEPS-1:
			# 	print('action probability: ', a)
			# 	print('state : ', s)
			# 	print('reward : ', r)
			# 	print('Episode:', i, ' Reward: %i' % int(ep_reward))
		reward.append(ep_reward/MAX_EP_STEPS)

	print('Running time of rl_ac: ', time.time()-t1)
	return reward

def local_only(env):
	reward = []

	a = np.zeros([action_dim], dtype=np.int)
	a[0] = 1

	t1 = time.time()
	for i in range(MAX_EPISODES):
		s = env.reset()
		ep_reward = 0 

		for step in range(MAX_EP_STEPS):
			s_ , r = env.step(a)

			ep_reward += r

			s = s_

			# if step == MAX_EP_STEPS-1:
			# 	print('action probability: ', a)
			# 	print('state : ', s)
			# 	print('reward : ', r)
			# 	print('Episode:', i, ' Reward: %i' % int(ep_reward))
		reward.append(ep_reward/MAX_EP_STEPS)

	print('Running time of rl_ac: ', time.time()-t1)
	return reward 


def rl_ddpg(env):

	reward = []

	with tf.name_scope('S'):
		S = tf.placeholder(tf.float32, shape=[None, state_dim], name='s')
	with tf.name_scope('R'):
		R = tf.placeholder(tf.float32, [None, 1], name='r')
	with tf.name_scope('S_'):
		S_ = tf.placeholder(tf.float32, shape=[None, state_dim], name='s_')

	sess = tf.Session()

	actor_ddpg = Actor_ddpg(sess, action_dim, LR_A, REPLACEMENT, S, R, S_)
	critic_ddpg = Critic_ddpg(sess, state_dim, action_dim, LR_C, GAMMA, REPLACEMENT, actor_ddpg.a, actor_ddpg.a_, S, R, S_)
	actor_ddpg.add_grad_to_graph(critic_ddpg.a_grads)

	sess.run(tf.global_variables_initializer())

	M = Memory_ddpg(MEMORY_CAPACITY, dims=2 * state_dim + action_dim + 1)

	# print('OUTPUT_GRAPH: ' , OUTPUT_GRAPH)

	if OUTPUT_GRAPH:
		tf.summary.FileWriter("logs/", sess.graph)

	t1 = time.time()
	for i in range(MAX_EPISODES):
		s = env.reset()
		ep_reward = 0

		for step in range(MAX_EP_STEPS):

			a = actor_ddpg.choose_action(s)
			# if i == 0 and step == 0:
			# 	a = np.zeros(env.action_dim)
			# 	a[0] = 1
			# a = sess.run(tf.nn.softmax(a_))

			s_, r = env.step(a)

			M.store_transition(s, a, r, s_)

			if M.pointer > MEMORY_CAPACITY:
				b_M = M.sample(BATCH_SIZE)
				b_s = b_M[:, :state_dim]
				b_a = b_M[:, state_dim:state_dim+action_dim]
				b_r = b_M[:, -state_dim-1: -state_dim]
				b_s_ = b_M[:, -state_dim:]

				critic_ddpg.learn(b_s, b_a, b_r, b_s_)
				actor_ddpg.learn(b_s)

			s = s_
			ep_reward += r

			# reward.append(r)

			if (i == 0 and step == 0) or (i == MAX_EPISODES-1 and step == MAX_EP_STEPS-1): #if step == MAX_EP_STEPS-1:
				# print('action_ probability: ', a_)
				print('action probability : ', a)
				# print('state : ', s)
				# print('reward : ', r)
				# print('Episode:', i, ' Reward: %i' % int(ep_reward))
		reward.append(ep_reward/MAX_EP_STEPS)

	print('Running time of rl_ddpg: ', time.time()-t1)
	return reward

#####################  hyper parameters  ####################

MAX_EPISODES = 100
MAX_EP_STEPS = 100

LR_A = 0.001    # learning rate for actor  	0.001
LR_C = 0.001    # learning rate for critic 	0.001
GAMMA = 0.9     # reward discount

REPLACEMENT = [
    dict(name='soft', tau=0.01),
    dict(name='hard', rep_iter_a=600, rep_iter_c=500)
][0]            # you can try different target replacement strategies

MEMORY_CAPACITY = 1000
BATCH_SIZE = 32

OUTPUT_GRAPH = False

np.random.seed(1)


if __name__ == '__main__':
	env = Environment()
	state_dim = env.state_dim
	action_dim = env.action_dim

	print(env.a, env.N, env.c)
	# print(env.g)

	

	# rac = rl_ac(env)
	# print('episode reward of ac : ')
	# print(rac)

	rddpg = rl_ddpg(env)
	# print('episode reward of ddpg : ')
	# print(rddpg)

	# rlo = local_only(env)
	# print('episode reward of local only : ')
	# print(rlo)
	
	# y = (rddpg - np.mean(rddpg)) / np.std(rddpg)
	# print(y)

	x = [i for i in range(MAX_EPISODES)]
	plt.figure()
	plt.plot(x, rddpg, color='red')
	# plt.plot(x, rac, color='blue')
	# plt.plot(x, rlo, color='black')

	plt.show()

	print('ok')