import numpy as np
import copy

def dijkstra(v0, cost):
	'''
	用于获取图中一点到其余各点的距离
	'''
	passed = [v0]
	nopassed = [x for x in range(len(cost)) if x != v0]

	distance = copy.deepcopy(cost[v0])
	# distance = cost[v0]

	while len(nopassed):
		idx = nopassed[0]

		for i in nopassed:
			if distance[i] < distance[idx]:
				idx = i

		nopassed.remove(idx)
		passed.append(idx)

		for i in nopassed:
			if distance[idx] + cost[idx][i] < distance[i]: 
				distance[i] = distance[idx] + cost[idx][i] 

	return distance

# np.random.seed(1)
def get_es_propagation_delay(num):
	'''
	生成服务器间的传播延迟
	'''
	np.random.seed(1)
	g = np.random.uniform(0.010, 0.100,size=(num,num))

	for i in range(num):
		g[i,i] = 0
		for j in range(i):
			g[i,j] = g[j,i]

	return g



def get_action_space(nums:int, sums:int, actions:list, temp:list):
	'''
	获取不同个数服务器时可选的动作空间数
	'''
	if nums<1:
		print('nums must greater than 0')
	elif nums == 1:
		temp.append(sums/10)
		node = copy.deepcopy(temp)
		actions.append(node)
		temp.pop()
	else:
		for i in range(sums+1):
			temp.append(i/10)
			get_action_space(nums-1, sums-i, actions, temp)
			temp.pop()



if __name__ == "__main__":
	# cost = get_es_propagation_delay(5)
	# print("cost : ")
	# print(cost)

	# distance = []
	# for v0 in range(len(cost)):
	# 	# print(v0)
	# 	result = dijkstra(v0, cost)
	# 	distance.append(result)
	# 	# distance.append(list(dijkstra(v0, cost)))
	# # ditance = dijkstra(0, cost)
	# print(cost)
	# print(np.asarray(distance))

	actions = []
	temp = []

	nums = 3
	sums = 10 
	get_action_space(nums, sums, actions, temp)

	print('{}个服务器时共有{}种可能'.format(nums, len(actions)))
	print(actions)