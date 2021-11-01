import numpy as np
from toolbar import dijkstra

#当此行存在时，np.random生成的随机数始终相同，用于比较两个不同算法对于同一输入的不同性能
# np.random.seed(15)

class Environment:
	'''
	reload environment
	这是项目的1.0版本，由于缺少一些真实的数据，
	本版中通过随机生成的数据代替

	对a，c，w等进行标准化，在原数据的基础上除以1000
	'''

### init start ###
	def __init__(self):
		#用户设备的计算能力 GHz
		self.a = np.random.uniform(0.010,0.040)
		#边缘服务器的数量 10、20 [5,10), [10,20)
		self.N = 10 #np.random.randint(5,10)
		#N个边缘服务器的计算能力 GHz
		self.c = np.random.uniform(0.100,0.200, size=self.N)
		#任意两个边缘服务器之间的传播延迟，这里通过随机数生成 s
		self.g = self._get_es_propagation_delay(self.N)
		#获取任意两个服务器之间的最短延迟
		self.shortest_g = self._get_shortest_es_delay(self.g)
		#任务到达时间所服从的指数分布的均值 [20,50] [40,70]
		self.lamda = np.random.uniform(0.2, 0.5)
		#用户设备和N个服务器之间的上传带宽 MHz
		self.uplink = self._get_user_server_uplink()
		#动作空间的纬度
		self.action_dim = self.N + 1
		#状态空间的纬度
		self.state_dim = 4+2*self.N

	def _get_es_propagation_delay(self, num):
		'''
		通过均匀分布获取任意两个服务器之间的传播延迟
		结果保存在N*N的矩阵中
		'''
		g = np.random.uniform(0.010, 0.100,size=(num,num))

		#使得g[i,j]==g[j,i]
		for i in range(num):
			g[i,i] = 0
			for j in range(i):
				g[i,j] = g[j,i]

		return g

	def _get_shortest_es_delay(self, cost):
		'''
		获取任意两个服务器之间的最短传播延迟
		'''
		delay = []
		for v0 in range(len(cost)):
			result = dijkstra(v0, cost)
			delay.append(list(result))
		return np.asarray(delay)

	def _get_user_server_uplink(self):
		'''
		获取用户设备与边缘服务器之间的上传带宽
		本版本中通过均匀分布随机采样
		'''
		return np.random.uniform(1.000, 2.000, size=self.N)

### init end ###

### del start ###
	# def __del__(self):
	# 	'''
	# 	注销class
	# 	'''
	# 	class_name = self.__class__.__name__
	# 	print('注销')
### del end ###

### reset start ###
	def reset(self):
		'''
		初始化environment
		'''

		#两个相邻任务的到达时间差，服从指数分布
		self.delta_t = np.random.exponential(self.lamda)
		
		#0时刻任务f对应的工作量
		self.w = np.random.uniform(0.100,0.400)
		
		#当前用户所处的边缘服务器位置
		self.n = np.random.randint(self.N)
		# print('n : ', self.n)

		#0时刻任务f的deadline
		d_max = self.w/self.a
		d_min = self.w/(self.a + self.c[self.n])
		self.d = np.random.uniform(d_min, d_max)

		#0时刻任务f的inputsize  Mbit
		self.s = np.random.uniform(2.000, 4.000)

		#0时刻用户的上传带宽  MHz
		self.b = np.zeros(self.N, dtype=np.float32)
		self.b[self.n] = self.uplink[self.n]

		#0时刻用户设备与边缘服务器的任务队列大小，此时为0
		self.P = np.zeros(self.N+1, dtype=np.float32)

		state = list()
		state.append(self.w)
		state.append(self.d)
		state.append(self.s)
		for item in list(self.b):
			state.append(item)
		for item in list(self.P):
			state.append(item)

		#返回初始状态s0
		return np.array(state)

### reset end ###

### step start ###
	def step(self, action):
		'''
		通过actiuon获得t时刻reward以及t+1时刻的state
		'''
		reward = self._get_reward(action)
		state_ = self._get_state(action)
		
		return state_, reward

### get reward start ###
	def _get_reward(self, action):
		'''
		获取reward
		其中t由三部分组成，分别为上传时间，传播延迟和工作时间
		'''
		t1 = self._get_uplink_time()
		t2 = self._get_total_propagation_delay(action)
		t3 = self._get_work_time(action, t2)
		total_time = t1+t2+t3

		reward = (self.d - total_time) / self.d
		return reward


	def _get_uplink_time(self):
		'''
		计算用户上传任务到所在服务器的时间
		'''
		return self.s/np.max(self.b)

	def _get_total_propagation_delay(self, action):
		'''
		计算任务在服务器间传播的延迟时间
		'''
		cost = 0

		for i in range(1, self.action_dim):
			if action[i] > 0:
				cost += self.shortest_g[self.n, i-1]

		# print('propagation delay: ', cost)
		return cost

	def _get_work_time(self, action, t2):
		'''
		计算用户设备或边缘服务器完成各自分配任务所需的时间
		'''
		#本地任务时间等于本地完成任务的时间减去任务在服务器间传播的时间
		user_time = (self.P[0] + action[0]*self.w) / self.a - t2

		server_time = []
		for i in range(1, self.action_dim):
			#任务量除以边缘服务器计算能力
			result = (self.P[i]+action[i]*self.w) / self.c[i-1]
			server_time.append(result)
		server_time_max = np.max(server_time)

		if user_time > server_time_max:
			return user_time
		else:
			return server_time_max
### get reward end ###


### get state start ###
	def _get_state(self, action):
		'''
		获取t+1时刻的state
		参数含义未特别说明的则与reset()中相同
		'''
		
		#计算t+1时刻任务队列的大小
		self.P = self._get_pending_queue(action, self.delta_t)

		self.delta_t = np.random.exponential(self.lamda)

		self.w = np.random.uniform(0.100,0.400)
		
		self.n = np.random.randint(self.N)
		# print('n : ', self.n)

		d_max = self.w/self.a
		d_min = self.w/(self.a + self.c[self.n])
		self.d = np.random.uniform(d_min, d_max)

		self.s = np.random.uniform(2.000, 4.000)

		self.b = np.zeros(self.N, dtype=np.float32)
		self.b[self.n] = self.uplink[self.n]

		state_ = list()
		state_.append(self.w)
		state_.append(self.d)
		state_.append(self.s)
		for item in list(self.b):
			state_.append(item)
		for item in list(self.P):
			state_.append(item)

		return np.array(state_)

	def _get_pending_queue(self, action, delta_t):
		'''
		t+1时刻任务队列的大小由t时刻任务队列大小与t时刻分配的任务量之和与delta_t时间内处理的任务量的差值决定
		并且该值必须大于等于0
		'''

		#用户设备的任务队列
		p0 = self.P[0] + action[0]*self.w - delta_t*self.a
		if p0<0:
			self.P[0] = 0 
		else:
			self.P[0] = p0

		#边缘服务器的任务队列
		for i in range(1, self.N+1):
			p = self.P[i] + action[i]*self.w - delta_t*self.c[i-1]
			# print('p : ', p )
			if p > 0:
				self.P[i] = p
			else:
				self.P[i] = 0
		return self.P

### get state end ###
### step end ###

if __name__ == '__main__':

	env = Environment()
	# print(env._get_es_propagation_delay(4))
	print(env.reset())
	print('es delay: ' )
	print(env.shortest_g)

	a1 = [0.2, 0.3, 0.1, 0.4, 0, 0]
	a2 = [0.1, 0, 0.1, 0.4, 0.2, 0.2]
	a3 = [0, 0.3, 0, 0, 0.6, 0.1]
	a = []
	a.append(a1)
	a.append(a2)
	a.append(a3)
	result = []
	for i in range(3):
		print('------------------------------')
		s_,r = env.step(a[i])
		print('time %d:' %(i+1) ,env.delta_t)
		print('state %d :' %(i+1))
		print(s_)
		print('reward {} :'.format(i+1), r)
		print('------------------------------')
		# print(r)
	