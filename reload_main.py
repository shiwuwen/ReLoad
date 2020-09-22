import tensorflow as tf 
import numpy as np
import time
import gc
import matplotlib.pyplot as plt

from reload_environment import Environment
from reload_rl_DDPG import Actor_ddpg, Critic_ddpg, Memory_ddpg
from reload_rl_AC import Actor_ac, Critic_ac

import os
os.environ['CUDA_VISIBLE_DEVICES'] = "1"

def rl_ac(env):
	'''
	使用actor-critic算法实现actions的决策生成
	'''

	reward = []

	sess = tf.Session()

	#actor网络
	actor_ac = Actor_ac(sess, state_dim, action_dim, LR_A)
	#critic网络
	critic_ac = Critic_ac(sess, state_dim, action_dim, LR_C)

	sess.run(tf.global_variables_initializer())

	#if true，输出graph
	if OUTPUT_GRAPH:
		tf.summary.FileWriter('./logs/', sess.graph)

	t1 = time.time()
	for i in range(MAX_EPISODES):
		s = env.reset()
		ep_reward = 0

		for step in range(MAX_EP_STEPS):

			#返回actions，并保留两位小数
			a = actor_ac.choose_action(s).round(2)
			# a = np.zeros_like(a_temp)
			# a[np.argmax(a_temp)] = 1

			#获取state_和reward
			s_, r = env.step(a)
			#一个训练周期的reward之和
			ep_reward += r

			#训练
			td_error = critic_ac.learn(s, r, s_)
			actor_ac.learn(s, a, td_error)

			s = s_

			if step%50==0:#if (i == 0 and step == 0) or step == MAX_EP_STEPS-1: # if step == MAX_EP_STEPS-1:
				print('EPISODE: ', i, ' action probability of rl_ac: ', a)
			# 	print('state : ', s)
			# 	print('reward : ', r)
			# 	print('Episode:', i, ' Reward: %i' % int(ep_reward))
		reward.append(round(ep_reward/MAX_EP_STEPS,2))

	print('Running time of rl_ac: ', time.time()-t1)
	return reward

def rl_choose_by_uplink_b(env):
	rewrad = []

	a = np.zeros_like([action_dim], dtype=np.int)

	t1 = time.time()
	for i in range(MAX_EPISODES):
		s = env.reset()
		ep_reward = 0

		for step in range(MAX_EP_STEPS):
			b = s[3:env.N+3]
			a[np.argmax(b)] = 1

			s_, r = env.step(a)

			ep_reward += r

			s = s_

			# if (i == 0 and step == 0) or step == MAX_EP_STEPS-1: # if step == MAX_EP_STEPS-1:
			# 	print('action probability of rl_ac: ', a)
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
		reward.append(round(ep_reward/MAX_EP_STEPS,2))

	print('Running time of rl_ac: ', time.time()-t1)
	return reward 


def rl_ddpg(env):
	'''
	使用ddpg算法生成二进制决策算法
	'''

	reward = []

	with tf.name_scope('S'):
		S = tf.placeholder(tf.float32, shape=[None, state_dim], name='s')
	with tf.name_scope('R'):
		R = tf.placeholder(tf.float32, [None, 1], name='r')
	with tf.name_scope('S_'):
		S_ = tf.placeholder(tf.float32, shape=[None, state_dim], name='s_')

	sess = tf.Session()

	#生成网络
	actor_ddpg = Actor_ddpg(sess, action_dim, LR_A, REPLACEMENT, S, R, S_)
	critic_ddpg = Critic_ddpg(sess, state_dim, action_dim, LR_C, GAMMA, REPLACEMENT, actor_ddpg.a, actor_ddpg.a_, S, R, S_)
	actor_ddpg.add_grad_to_graph(critic_ddpg.a_grads)

	sess.run(tf.global_variables_initializer())

	#更新记忆库
	M = Memory_ddpg(MEMORY_CAPACITY, dims=2 * state_dim + action_dim + 1)

	# print('OUTPUT_GRAPH: ' , OUTPUT_GRAPH)

	if OUTPUT_GRAPH:
		tf.summary.FileWriter("logs/", sess.graph)

	t1 = time.time()
	for i in range(MAX_EPISODES):
		#初始化
		s = env.reset()
		ep_reward = 0

		for step in range(MAX_EP_STEPS):

			#获取actions，并保留两位小数
			a_temp = actor_ddpg.choose_action(s).round(2)
			#将数组转换为one_hot数组
			a = np.zeros_like(a_temp)
			a[np.argmax(a_temp)] = 1
			# if i == 0 and step == 0:
			# 	a = np.zeros(env.action_dim)
			# 	a[0] = 1
			# a = sess.run(tf.nn.softmax(a_))

			#获取state_, reward
			s_, r = env.step(a)

			#存储训练数据
			M.store_transition(s, a, r, s_)

			#当存储数据大于阈值时开始训练
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

			if (i == 0 and step == 0) or step == MAX_EP_STEPS-1: #(i == MAX_EPISODES-1 and step == MAX_EP_STEPS-1): ##if step == MAX_EP_STEPS-1:
				print('EPISODE: ', i,' action probability of rddpg: ', a)
				# print('action_ probability: ', a_)
				# print('state : ', s)
				# print('reward : ', r)
				# print('Episode:', i, ' Reward: %i' % int(ep_reward))
		reward.append(round(ep_reward/MAX_EP_STEPS,2))

	print('Running time of rl_ddpg: ', time.time()-t1)
	return reward

#####################  hyper parameters  ####################

#当EPISODES较大，STEPS较小时能获得较好效果
MAX_EPISODES = 200
MAX_EP_STEPS = 200

LR_A = 0.001    # learning rate for actor  	0.001
LR_C = 0.001    # learning rate for critic 	0.001
GAMMA = 0.9     # reward discount

#ddpg的网络参数更新方式
REPLACEMENT = [
    dict(name='soft', tau=0.01),
    dict(name='hard', rep_iter_a=600, rep_iter_c=500)
][0]            # you can try different target replacement strategies
#ddpg的记忆库大小
MEMORY_CAPACITY = 1000
BATCH_SIZE = 32

#是否输出graph
OUTPUT_GRAPH = False

#使用该代码将使随机数可以预测，使用1时将使N固定为9
# np.random.seed(1)


if __name__ == '__main__':

	#初始化环境
	env = Environment()
	#获取状态空间和动作空间的维度
	state_dim = env.state_dim
	action_dim = env.action_dim

	#输出环境的基本信息，包括 用户设备的计算能力 边缘服务器的数量 N个边缘服务器的计算能力
	print(env.a, env.N, env.c)
	# print(env.g)

	
	#action-critic算法
	rac = rl_ac(env)
	print('episode reward of ac : ')
	print(rac)

	#ddpg算法
	rddpg = rl_ddpg(env)
	print('episode reward of ddpg : ')
	print(rddpg)

	#local-only算法
	# rlo = local_only(env)
	# print('episode reward of local only : ')
	# print(rlo)
	
	# y = (rddpg - np.mean(rddpg)) / np.std(rddpg)
	# print(y)

	#绘制reward图表
	# x = [i for i in range(MAX_EPISODES)]
	# plt.figure()
	# plt.plot(x, rac, color='blue')
	# plt.plot(x, rddpg, color='red')
	# # plt.plot(x, rlo, color='black')

	# plt.show()

	print('ok')