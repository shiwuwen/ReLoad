#coding=utf-8

import tensorflow as tf 
import numpy as np
import time
import gc
import matplotlib.pyplot as plt

from reload_environment import Environment
from reload_rl_DDPG import Actor_ddpg, Critic_ddpg, Memory_ddpg
from reload_rl_AC import Actor_ac, Critic_ac

#是否使用GPU，0使用，-1不使用
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "-1"


def rl_ac(env, bound=0):
	'''
	使用actor-critic算法实现actions的决策生成
	'''
	ES1 = []
	ES2 = []
	ES3 = []
	ES4 = []
	ES5 = []

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
	for episode in range(MAX_EPISODES):
		s = env.reset()
		ep_reward = 0

		for step in range(MAX_EP_STEPS):

			#返回actions
			a = actor_ac.choose_action(s)

			#使用超参数bound对action进行裁减
			for j in range(action_dim):
				if a[j]<=bound:
					a[np.argmax(a)] += a[j]
					a[j] = 0

			#保留两位小数
			a = a.round(2)
			# a = np.zeros_like(a_temp)
			# a[np.argmax(a_temp)] = 1

			if step == MAX_EP_STEPS-1:
				ES1.append(a[1])
				ES2.append(a[2])
				ES3.append(a[3])
				ES4.append(a[4])
				ES5.append(a[5])

			#获取state_和reward
			s_, r = env.step(a)
			#一个训练周期的reward之和
			ep_reward += r

			#训练
			td_error = critic_ac.learn(s, r, s_)
			actor_ac.learn(s, a, td_error)

			s = s_

			if step%50==0:#if (i == 0 and step == 0) or step == MAX_EP_STEPS-1: # if step == MAX_EP_STEPS-1:
				print('EPISODE: ', episode, ' action probability of rl_ac: ', a)
			# 	print('state : ', s)
			# 	print('reward : ', r)
			# 	print('Episode:', i, ' Reward: %i' % int(ep_reward))
		reward.append(round(ep_reward/MAX_EP_STEPS,2))

	print('Running time of rl_ac: ', time.time()-t1)
	return ES1, ES2, ES3, ES4, ES5

#####################  hyper parameters  ####################

#当EPISODES较大，STEPS较小时能获得较好效果
MAX_EPISODES = 200
MAX_EP_STEPS = 100

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

clip_bound = 0.09

#使用该代码将使随机数可以预测，使用1时将使N固定为9
#n-0:badresult  n-1:goodresult
#5: 2-1(quxiaohao), 5-1(quxianhao**), 9-0, 11-1(quxianhao**), 18-0, 21-1
#6: 3-1(quxianhao**), 4-0, 7-0, 8-1, 12-0, 16-0, 17-0
#7: 13-0, 19-1, 
#8: 6-1, 
#9: 1-0, 10-1, 14-1, 15-1, 20-1, 22-0
np.random.seed(5)


if __name__ == '__main__':

	#初始化环境
	env = Environment()
	#获取状态空间和动作空间的维度
	state_dim = env.state_dim
	action_dim = env.action_dim

	#输出环境的基本信息，包括 用户设备的计算能力 边缘服务器的数量 N个边缘服务器的计算能力
	print(env.a, env.N, env.c)
	# print(env.g)

	#local-only算法
	# rlo = local_only(env)
	# print('episode reward of local only : ')
	# print(rlo)


	#action-critic算法
	ES1, ES2, ES3, ES4, ES5 = rl_ac(env, clip_bound)

	
	x = [i for i in range(MAX_EPISODES)]
	plt.figure()
	# plt.plot(x[130:], ES1[130:], color='blue', label='ES1')
	# plt.plot(x[130:], ES2[130:], color='orange', label='ES2')
	# plt.plot(x[130:], ES3[130:], color='red', label='ES3')
	# plt.plot(x[130:], ES4[130:], color='green', label='ES4')
	# plt.plot(x[130:], ES5[130:], color='cyan', label='ES5')

	plt.plot(x, ES1, color='blue', label='EdgeServer1')
	# plt.plot(x, ES2, color='orange', label='ES2')
	# plt.plot(x, ES3, color='red', label='ES3')
	plt.plot(x, ES4, color='green', label='EdgeServer2')
	plt.plot(x, ES5, color='cyan', label='EdgeServer3')

	plt.legend()

	plt.xlabel('time')
	plt.ylabel('workload')

	plt.show()

	print('ok')