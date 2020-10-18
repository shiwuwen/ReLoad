import matplotlib.pyplot as plt

import toolbar


def autolabel(rects):
	for rect in rects:
		height = rect.get_height()
		plt.text(rect.get_x()+rect.get_width()/8, 1.01*height, '%s' % float(height))

def countNumberOfTasksFinishedBeforeDeadLines(filename, MAX_EPISODES, inXlabel, inYlabel):
	countList = toolbar.txt2list(filename)
	countListLen = len(countList)

	MAX_EP_STEPS = 100

	yLabel = []

	for i in range(countListLen):
		res = toolbar.countNumber(countList[i])
		temp = round(float(res/MAX_EPISODES*MAX_EP_STEPS), 2)
		yLabel.append(temp)
		# print(temp)
	print(yLabel)
	xLabel = ('Reload', 'SS-B', 'SS-W', 'DS-BW', 'MS')

	#显示网格
	plt.grid(True, linestyle='--', axis='both', zorder=1)


	rects = plt.bar(xLabel, yLabel, width=0.5, zorder=10)
	autolabel(rects)
	plt.xlabel(inXlabel)
	plt.ylabel(inYlabel)

	#设置横纵坐标轴范围
	# plt.xlim(-0.4, 1.6)
	plt.ylim(0, 100)

	#显示图示
	# plt.legend()

	plt.show()



if __name__ == '__main__':
	#n=10,lamda=[20,50]:MAX_EPISODES=260
	#n=10,lamda=[40,70]:MAX_EPISODES=300
	#n=20,lamda=[20,50]:MAX_EPISODES=101
	#n=20,lamda=[40,70]:MAX_EPISODES=101

	inHisXLabel = 'Name of different algorithms'
	inHisYLabel = 'Number of tasks finished before deadlines(%)'
	inHisFilename = 'n20lamda47.txt'
	inHisEpisode = 101
	countNumberOfTasksFinishedBeforeDeadLines(inHisFilename, inHisEpisode, inHisXLabel, inHisYLabel)

