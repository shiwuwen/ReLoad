import numpy as np 

# the1 = np.random.uniform(20,50,[1])
# print(the1)

# result = np.random.exponential(the1, [20])
# print(np.mean(result))
# print(result)

b_t = np.ones(5, dtype=np.int)
b_t[2] = np.random.randint(100,300)
print(200/np.max(b_t))
c = [3,4,5,6,7,8]
d = (b_t*2+c[1:]) / c[1:]
print(d)
print(np.max(d))
# state = list()
# state.append(list(result))
# state.append(list(b_t))
# print(state)

# list1 = [[1,2],[3,4]]

# list2 = [[5,6],[7,8]]

# print(type(list1))
# print(np.array(list1)*np.array(list2))
# print(np.matmul(list1,list2))

# class test:
# 	def __init__(self):
# 		self.t = 1

# 	def get_F(self, n):
# 		self.F = n+1
# 		return self.F

# 	def get_p(self):
# 		p = self.F + 1
# 		return p

# env = test()
# print(env.get_F(1))
# print(env.get_p())