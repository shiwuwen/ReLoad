#coding=utf-8

import tensorflow as tf 
import numpy as np
import time
import gc
import matplotlib.pyplot as plt

from reload_environment import Environment
from reload_rl_DDPG import Actor_ddpg, Critic_ddpg, Memory_ddpg
from reload_rl_AC import Actor_ac, Critic_ac
import toolbar

#是否使用GPU，0使用，-1不使用
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "-1"


def rl_ac(env, bound=0):
	'''
	使用actor-critic算法实现actions的决策生成
	'''
	actionList = []
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

			aCopy = a.copy()

			#使用超参数bound对action进行裁减
			for j in range(action_dim):
				if a[j]<=bound:
					# a[np.argmax(a)] += a[j]
					a[np.argmax(a[1:])] += a[j]
					a[j] = 0

			#保留两位小数
			a = a.round(2)
			# a = np.zeros_like(a_temp)
			# a[np.argmax(a_temp)] = 1
			
			#for MS start
			for j in range(1, action_dim):
				if aCopy[j]<=bound:
					# a[np.argmax(a)] += a[j]
					aCopy[np.argmax(aCopy[1:])] += aCopy[j]
					aCopy[j] = 0
			aCopy = aCopy.round(2)
			actionList.append(aCopy)
			#for MS end

			# print('a: ',a)
			# print('acopy: ', aCopy)

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
				# print(actionList[step])
				# print('服务器延迟： ', env.shortest_g[env.n])
			# 	print('state : ', s)
			# 	print('reward : ', r)
			# 	print('Episode:', i, ' Reward: %i' % int(ep_reward))
		reward.append(round(ep_reward/MAX_EP_STEPS,2))

	print('Running time of rl_ac: ', time.time()-t1)
	return reward, actionList


def rl_choose_by_uplink_b(env):
	'''
	选择用户当前所在的位置的服务器进行卸载
	SS-B
	'''
	reward = []

	t1 = time.time()
	for episode in range(MAX_EPISODES):
		s = env.reset()
		ep_reward = 0

		for step in range(MAX_EP_STEPS):
			a = np.zeros([action_dim], dtype=np.int)
			b = s[3:env.N+3]
			a[np.argmax(b)+1] = 1

			s_, r = env.step(a)

			ep_reward += r

			s = s_

			if step%50==0: # if (i == 0 and step == 0) or step == MAX_EP_STEPS-1: # if step == MAX_EP_STEPS-1:
				print('EPISODE: ', episode, 'action probability of rlb: ', a)
			# 	print('state : ', s)
			# 	print('reward : ', r)
			# 	print('Episode:', i, ' Reward: %i' % int(ep_reward))
		reward.append(round(ep_reward/MAX_EP_STEPS,2))

	print('Running time of rl_choose_by_uplink_b: ', time.time()-t1)
	return reward


def rl_choose_by_pending_queue(env):
	'''
	基于等待队列来选择服务器，选择等待队列最小的服务器
	大多数时间等待队列都为空，相当于随机选择服务器
	SS-W
	'''
	reward = []

	t1 = time.time()
	for episode in range(MAX_EPISODES):
		s = env.reset()
		ep_reward = 0

		for step in range(MAX_EP_STEPS):
			a = np.zeros([action_dim], dtype=np.int32)
			pendingQ = s[env.N+3:]
			#当多个服务器等待队列相同时，从这些服务器中随机选择
			temp = [i for i in range(len(pendingQ)) if pendingQ[i] == np.min(pendingQ)]
			index = np.random.choice(temp)
			# print(pendingQ[pendingQ == np.min(pendingQ)].tolist().index())
			a[index] = 1

			s_, r = env.step(a)

			ep_reward += r

			s = s_

			if step%50==0: # if (i == 0 and step == 0) or step == MAX_EP_STEPS-1: # if step == MAX_EP_STEPS-1:
				print('EPISODE: ', episode, 'action probability of rlq: ', a)
				# print('pendingQ: ' ,pendingQ)
			# 	print('state : ', s)
			# 	print('reward : ', r)
			# 	print('Episode:', i, ' Reward: %i' % int(ep_reward))
		reward.append(round(ep_reward/MAX_EP_STEPS,2))

	print('Running time of rl_choose_by_pending_queue: ', time.time()-t1)
	return reward


def rl_choose_by_uplinkb_and_pendingqueue(env):
	'''
<<<<<<< HEAD
	基于上传带宽和任务队列选择actions
=======
	基于等待队列大小以及用户所在位置选择两个服务器进行卸载
	DS-BW
>>>>>>> 66adf975b24cd9de9a717de1c88cb28c86ebf230
	'''
	reward = []

	t1 = time.time()
	for episode in range(MAX_EPISODES):
		s = env.reset()
		ep_reward = 0

		for step in range(MAX_EP_STEPS):
			a = np.zeros([action_dim], dtype=np.float32)
			b = s[3:env.N+3]
			a[np.argmax(b)+1] += 0.5

			pendingQ = s[env.N+3:]
			#当多个服务器等待队列相同时，从这些服务器中随机选择
			temp = [i for i in range(len(pendingQ)) if pendingQ[i] == np.min(pendingQ)]
			index = np.random.choice(temp)
			# print(pendingQ[pendingQ == np.min(pendingQ)].tolist().index())
			a[index] += 0.5

			s_, r = env.step(a)

			ep_reward += r

			s = s_

			if step%50==0: # if (i == 0 and step == 0) or step == MAX_EP_STEPS-1: # if step == MAX_EP_STEPS-1:
				print('EPISODE: ', episode, 'action probability of rlbq: ', a)
				
			# 	print('state : ', s)
			# 	print('reward : ', r)
			# 	print('Episode:', i, ' Reward: %i' % int(ep_reward))
		reward.append(round(ep_reward/MAX_EP_STEPS,2))

	print('Running time of rl_choose_by_uplinkb_and_pendingqueue: ', time.time()-t1)
	return reward


def MS(env, actionList):
	'''
<<<<<<< HEAD
	首先使用Reload生成卸载策略，之后在不改变分配比列的情况下，按照服务器间延迟重新选择服务器
=======
	基于Reload算法的卸载策略，选择相同数量的延迟最小的服务器进行卸载
	MS
>>>>>>> 66adf975b24cd9de9a717de1c88cb28c86ebf230
	'''
	reward = []

	t1 = time.time()
	for episode in range(MAX_EPISODES):
		s = env.reset()
		ep_reward = 0

		for step in range(MAX_EP_STEPS):
			# print(episode*MAX_EP_STEPS+step)

			a = np.zeros([action_dim], dtype=np.float32)
			currentAction = actionList[episode*MAX_EP_STEPS+step]
			#对action进行排序，从小到大
			actionListIndexSorted = np.argsort(currentAction[1:])
			#对当前位置的服务器与其他服务器的传输延迟进行排序，从小到大
			#shortest_gIndexSorted.reverse()
			shortest_gIndexSorted = np.argsort(env.shortest_g[env.n])
			#对action进行重排，使得延迟最小的获得最大的卸载量
			for i in range(action_dim-1):
				a[shortest_gIndexSorted[i]] = currentAction[actionListIndexSorted[i]+1]
			a[0] = currentAction[0]

			s_, r = env.step(a)

			ep_reward += r

			s = s_

			if step%50==0: # if (i == 0 and step == 0) or step == MAX_EP_STEPS-1: # if step == MAX_EP_STEPS-1:
				print('EPISODE: ', episode, 'action probability of MS: ', a)
				# print('currentAction: ', currentAction)
			# 	print('state : ', s)
			# 	print('reward : ', r)
			# 	print('Episode:', i, ' Reward: %i' % int(ep_reward))
		reward.append(round(ep_reward/MAX_EP_STEPS,2))

	print('Running time of MS: ', time.time()-t1)
	return reward


#####################  hyper parameters  ####################
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

#对rl_ac的action空间进行裁减，确保分配的任务不少于bound
#该参数能显著改善性能，且N越大，bound应当越小
#当N=(5,10)时，0.07可能出现负优化，0.09有较好效果；过大会退化为二进制策略
#n=20:clip_bound=0.09
#n=10:clip_bound=0.09
clip_bound = 0.09

#使用该代码将使随机数可以预测，使用1时将使N固定为9
#n-0:badresult  n-1:goodresult
#5: 2-1(quxiaohao), 5-1(quxianhao**), 9-0, 11-1(quxianhao**), 18-0, 21-1
#6: 3-1(quxianhao**), 4-0, 7-0, 8-1, 12-0, 16-0, 17-0
#7: 13-0, 19-1, 
#8: 6-1, 
#9: 1-0, 10-1, 14-1, 15-1, 20-1, 22-0
#n=10,lamda=[20,50]: 1-0, 2-1, 3-0, 4-0, 5-1*, 6-0, 7-0
#n=10,lamda=[40,70]: 1-0, 2-1, 3-0, 4-0, 5-0, 6-1, 7-1*
#n=20,lamda=[20,50]: 1-0, 2-0, 3-0, 4-0, 5-1, 6-0, 7-1*(epi100), 8-0, 9-0, 10-0, 11-0, 12-0, 13-0
#n=20,lamda=[40,70]: 1-0, 2-0, 3-0, 4-0, 5-0, 6-0, 7-0, 8-1(epi90), 9-0, 10-0, 11-0, 12-0, 13-1*(100)
np.random.seed(5)


#当EPISODES较大，STEPS较小时能获得较好效果
#n=10,lamda=[20,50]:MAX_EPISODES=260
#n=10,lamda=[40,70]:MAX_EPISODES=300
#n=20,lamda=[20,50]:MAX_EPISODES=100
#n=20,lamda=[40,70]:MAX_EPISODES=100
MAX_EPISODES = 260
MAX_EP_STEPS = 100


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


	#action-critic算法 Reload
	rac, actionListForMS = rl_ac(env, clip_bound)
	# print(actionListForMS[0:5])

	#Reload with no clip bound
	#重置计算图
	# tf.reset_default_graph()
	# rac_bound_0, _ = rl_ac(env)
	# print('episode reward of ac : ')
	# print(rac)

	# SS-B
	# rl_choose_by_uplink_b
	# rlb = rl_choose_by_uplink_b(env)
	# print('episode reward of rlb : ')
	# print(rlb)

	# SS-W
	# rl_choose_by_pending_queue
	# rlq = rl_choose_by_pending_queue(env)
	# print('episode reward of rlq : ')
	# print(rlq)
	
	# DS-BW
	# rl_choose_by_uplinkb_and_pendingqueue
	# rlbq = rl_choose_by_uplinkb_and_pendingqueue(env)
	# print('episode reward of rlbq : ')
	# print(rlbq)


	# MS
	# rMS = MS(env, actionListForMS)
	# print('episode reward of MS : ')
	# print(rMS)


	# 查看deadline之前完成的任务数
	# racnum = toolbar.countNumber(rac)
	# rmsnum = toolbar.countNumber(rMS)
	# rlbnum = toolbar.countNumber(rlb)
	# print(round(racnum/MAX_EPISODES*MAX_EP_STEPS, 2))
	# print(round(rmsnum/MAX_EPISODES*MAX_EP_STEPS, 2))
	# print(round(rlbnum/MAX_EPISODES*MAX_EP_STEPS, 2))


	# 将算法运算结果写入文本中
	# list2txtList = []
	# list2txtList.append(rac)
	# list2txtList.append(rac_bound_0)
	# # list2txtList.append(rlb)
	# # list2txtList.append(rlq)
	# # list2txtList.append(rlbq)
	# # list2txtList.append(rMS)
	# filename = 'n10lamda25_compare.txt'
	# toolbar.list2txt(list2txtList, filename)
	

	#绘制reward图表
	x = [i for i in range(MAX_EPISODES)]
	plt.figure()
	plt.plot(x, rac, color='blue', label='Reload')
	# plt.plot(x, rac_bound_0, color='orange', label='Reload_nobound')
	# plt.plot(x, rlb, color='green', label='SS-B')
	# plt.plot(x, rlq, color='cyan', label='SS-W')
	# plt.plot(x, rlbq, color='grey', label='DS-BW')
	# plt.plot(x, rMS, color='yellow', label='MS')
	# plt.plot(x, rddpg, color='red', label='rddpg')

	plt.legend()

	plt.xlabel('Episode')
	plt.ylabel('Reward')

	plt.show()

	print('ok')



	#ddpg算法
	# rddpg = rl_ddpg(env)
	# print('episode reward of ddpg : ')
	# print(rddpg)


	# y = (rddpg - np.mean(rddpg)) / np.std(rddpg)
	# print(y)

	# #绘制reward图表，去除开始时的不稳定的迭代期
	# x = [i-20 for i in range(20, MAX_EPISODES)]
	# plt.figure()
	# plt.plot(x, rac[20:], color='blue')
	# plt.plot(x, rddpg[20:], color='red')
	# plt.plot(x, rlb[20:], color='green')
	# plt.plot(x, rlq[20:], color='yellow')



'''
#该算法未使用
def local_only(env):
	reward = []

	a = np.zeros([action_dim], dtype=np.int)
	a[0] = 1

	t1 = time.time()
	for episode in range(MAX_EPISODES):
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
	
	# 使用ddpg算法生成二进制决策算法
	

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
	for episode in range(MAX_EPISODES):
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

			if (episode == 0 and step == 0) or step == MAX_EP_STEPS-1: #(i == MAX_EPISODES-1 and step == MAX_EP_STEPS-1): ##if step == MAX_EP_STEPS-1:
				print('EPISODE: ', episode,' action probability of rddpg: ', a)
				# print('action_ probability: ', a_)
				# print('state : ', s)
				# print('reward : ', r)
				# print('Episode:', i, ' Reward: %i' % int(ep_reward))
		reward.append(round(ep_reward/MAX_EP_STEPS,2))

	print('Running time of rl_ddpg: ', time.time()-t1)
	return reward
'''
