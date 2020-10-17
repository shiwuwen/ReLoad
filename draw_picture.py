#绘制论文所需的算法使用的数据的分析图
#绘制 边缘服务器与工作负载不平衡 直方图
import numpy as np 
import matplotlib.pyplot as plt

np.random.seed(5)


def getData():
	'''
	获取相关实验数据
	'''
	workloadList = []
	deadlineList = []
	inputsizeList = []

	for i in range(51):
		n = np.random.randint(5)
		a = np.random.uniform(0.010,0.040)
		c = np.random.uniform(0.100,0.200, size=5)
		
		workload = np.random.uniform(0.100, 0.400) #GHZ

		deadline = np.random.uniform(workload/(a+c[n]), workload/a) #s

		inputsize = np.random.uniform(2000, 4000)  #Kbit

		workloadList.append(workload*1000)
		deadlineList.append(deadline)
		inputsizeList.append(inputsize)

	return workloadList, deadlineList, inputsizeList

def autolabel(rects):
	for rect in rects:
		height = rect.get_height()
		plt.text(rect.get_x()+rect.get_width()/2.- 0.2, 1.03*height, '%s' % int(height))

def drawHistogram(inX, inY, inColor, inXlabel, inYlabel):
	'''
	绘制直方图
	'''
	plt.bar(inX, inY, color=inColor)
	plt.xlabel(inXlabel)
	plt.ylabel(inYlabel)
	plt.show()



workload, deadline, inputsize = getData()

# print(deadline)

taskIndex = [i for i in range(0,51)]
xlabel = 'index of task'

drawHistogram(taskIndex, workload, 'blue', xlabel, 'workload (GHZ)')
drawHistogram(taskIndex, deadline, 'blue', xlabel, 'deadline (s)')
drawHistogram(taskIndex, inputsize, 'blue', xlabel, 'inputsize (KBit)')