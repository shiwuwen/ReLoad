import numpy as np



class Environment:

### init start ###
	def __init__(self):
		self.a = np.random.uniform(10,40)
		self.N = np.random.randint(10,20)
		self.c = np.random.uniform(100,200, size=self.N)
		self.g = self._get_es_propagation_delay(self.N)
		self.lamda = np.random.uniform(20,50)
		self.uplink = self._get_user_server_uplink()
		self.action_dim = self.N + 1
		self.state_dim = 4+2*self.N

	def _get_es_propagation_delay(self, num):
		g = np.random.uniform(10,100,size=(num,num))

		for i in range(num):
			g[i,i] = 0
			for j in range(i):
				g[i,j] = g[j,i]

		return g

	def _get_user_server_uplink(self):
		return np.random.uniform(200, 500, size=self.N)

### init end ###

### reset start ###
	def reset(self):
		self.delta = np.random.exponential(self.lamda)
		
		self.w = np.random.uniform(100,400)
		
		n = np.random.randint(self.N)

		d_min = self.w/self.a
		d_max = self.w/(self.a + self.c[n])
		self.d = np.random.uniform(d_min, d_max)

		self.s = np.random.uniform(500,2000)

		self.b = np.zeros(self.N, dtype=np.int)
		self.b[n] = self.uplink[n]

		self.P = np.zeros(self.N+1, dtype=np.int)

		state = list()
		state.append(self.w)
		state.append(self.d)
		state.append(self.s)
		for item in list(self.b):
			state.append(item)
		for item in list(self.P):
			state.append(item)


		return np.array(state)

### reset end ###

### step start ###
	def step(self, action):

		reward = self._get_reward(action)
		state_ = self._get_state(action)
		
		return state_, reward

### get reward start ###
	def _get_reward(self, action):
		
		t1 = self._get_uplink_time()
		t2 = self._get_total_propagation_delay(action)
		t3 = self._get_work_time(action, t2)
		total_time = t1+t2+t3

		reward = (self.d - total_time) / self.d
		return reward


	def _get_uplink_time(self):
		return self.s/np.max(self.b)

	def _get_total_propagation_delay(self):
		return 0

	def _get_work_time(self, action, t2):
		user_time = (self.P[0] + action[0]*self.w) / self.a - t2
		server_time = np.max((self.P[1:]+action[1:]*self.w) / self.c)

		if user_time > server_time:
			return user_time
		else:
			return server_time
### get reward end ###


### get state start ###
	def _get_state(self, action):
		
		self.P = self._get_pending_queue(action, self.delta_t)

		self.delta_t = np.random.exponential(self.lamda)

		self.w = np.random.uniform(100,400)
		
		n = np.random.randint(self.N)

		d_min = self.w_t/self.a
		d_max = self.w_t/(self.a + self.c[n])
		self.d = np.random.uniform(d_min, d_max)

		self.s = np.random.uniform(500,2000)

		self.b = np.zeros(self.N, dtype=np.int)
		self.b[n] = self.uplink[n]

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

		p0 = self.P[0] + action[0]*self.w - delta_t*self.a
		if p0<0:
			self.P[0] = 0 
		else:
			self.P[0] = p0

		for i in range(1, self.N+1):
			p = self.P[i] + action[i]*self.w - delta_t*self.c[i]
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
	print(env.state_dim)