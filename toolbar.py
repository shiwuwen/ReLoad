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
	用于测试
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
	当对所有动作空间进行穷举时使用
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


def countNumber(inList, minBound=0):
	'''
	计算inList中值大于minBound的个数
	'''
	listLenth = len(inList)
	returnNum = 0.0
	for i in range(listLenth):
		if inList[i] > minBound:
			returnNum += 1
	return returnNum


def list2txt(inList, filename):
	'''
	将算法训练结果inList保存到文件filename中
	'''
	t = ''
	with open (filename,'w') as f:
		for content in inList:
			for index in range(len(content)):
				t = t + str(content[index]) + ' '
			f.write(t.strip(' '))
			f.write('\n')
			t = ''


def txt2list(filename):
	'''
	从文件filename中读取数据并返回list
	'''
	returnList = []
	with open(filename, 'r') as f:
		for line in f.readlines():
			strResult = line.strip('\n').split(' ')
			temp = map(eval, strResult)
			returnList.append(list(temp))

	return returnList


def sumListValue(inList, moveStep=10):
	'''
	在inList中以moveStep为步长，对步长内的数据进行求均值，从而减少inList长度
	'''
	listLen = len(inList)
	renList = []

	for i in range(0, listLen, moveStep):
		temp = 0.0
		for j in range(moveStep):
			temp += inList[i+j]
		renList.append(round(temp/moveStep, 2))

	return renList





if __name__ == "__main__":

	# a = [0.49,0.5,0.57,0.49,0.56,0.53,0.57,0.49,0.46,-2.85] 
	# res = sumListValue(a, 2)
	# print(res)



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

	nums = 10
	sums = 10 
	get_action_space(nums, sums, actions, temp)

	print('{}个服务器时共有{}种可能'.format(nums, len(actions)))
	# print(actions)


	# a = [[1.2,2,3,4],
	# 	 [3,4,5.4,6.4],
	# 	 [7,8,9,2]]
	# list2txt(a, 'n10lamda25.txt')

	# b = txt2list('n10lamda25.txt')
	# rac = b[0]
	# rMS = b[1]
	# rlb = b[2]
	# rlq = b[3]
	# rlbq = b[4]

	# MAX_EPISODES = 260
	# MAX_EP_STEPS = 100

	# for i in range(5):
	# 	res = countNumber(b[i])
	# 	print(round(res/MAX_EPISODES*MAX_EP_STEPS, 2))