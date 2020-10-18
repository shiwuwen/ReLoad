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


def rewardAndEpisode(filename, MAX_EPISODES, inXlabel, inYlabel):
	originList = toolbar.txt2list(filename)
	originListLen = len(originList)
	modifiedList = []
	moveStep = 10
	#n20lamda25: 20
	#n20lamda47: 10
	#n10lamda47: 100
	#n10lamda25: 120
	startStep = 120
	#n20lamda25: MAX_EPISODES
	#n20lamda47: MAX_EPISODES
	#n10lamda47: 210
	#n10lamda25: 260
	endStep = 260

	for i in range(originListLen):
		modifiedList.append(toolbar.sumListValue(originList[i][startStep:endStep], moveStep))
	

	#绘制折线图
	x = [i for i in range(startStep, endStep, moveStep)]
	plt.figure()
	plt.plot(x, modifiedList[0], color='blue', label='Reload', marker='o', zorder=1)
	# plt.plot(x, rac_bound_0, color='orange', label='reload_nobound')
	plt.plot(x, modifiedList[1], color='green', label='SS-B', marker='v', zorder=1)
	plt.plot(x, modifiedList[2], color='cyan', label='SS-W', marker='D', zorder=1)
	plt.plot(x, modifiedList[3], color='grey', label='DS-BW', marker='^', zorder=1)
	plt.plot(x, modifiedList[4], color='magenta', label='MS', marker='x', zorder=1)

	#n20lamda25: plt.legend(loc='best', bbox_to_anchor=(0.96, 0.55))
	#n20lamda47: plt.legend(loc='lower right')
	#n10lamda47: plt.legend(loc='lower right', bbox_to_anchor=(1, 0.2))
	#n10lamda25: plt.legend(loc='lower right', bbox_to_anchor=(1, 0.2))
	plt.legend(loc='lower right', bbox_to_anchor=(1, 0.2))

	plt.xlabel('Episode')
	plt.ylabel('Reward')

	#显示网格
	plt.grid(True, linestyle='--', axis='both', zorder=0)

	plt.show()
	

def reloadCompare(filename, MAX_EPISODES, inXlabel, inYlabel):
	originList = toolbar.txt2list(filename)
	originListLen = len(originList)
	modifiedList = []

	moveStep = 20
	startStep = 40
	endStep = 200

	for i in range(originListLen):
		modifiedList.append(toolbar.sumListValue(originList[i][startStep:endStep], moveStep))
	

	#绘制折线图
	x = [i for i in range(startStep, endStep, moveStep)]
	plt.figure()
	plt.plot(x, modifiedList[0], color='blue', label='Reload', marker='o', zorder=1)
	plt.plot(x, modifiedList[1], color='orange', label='Reload_nobound', marker='v', zorder=1)
	 
	plt.legend(loc='lower right')

	plt.xlabel('Episode')
	plt.ylabel('Reward')

	#显示网格
	plt.grid(True, linestyle='--', axis='both', zorder=0)

	plt.show()




if __name__ == '__main__':
	#n=10,lamda=[20,50]:MAX_EPISODES=260
	#n=10,lamda=[40,70]:MAX_EPISODES=300
	#n=20,lamda=[20,50]:MAX_EPISODES=100
	#n=20,lamda=[40,70]:MAX_EPISODES=100

	inLineCortXLabel = 'Reward'
	inLineCortYLabel = 'Episode'
	inLineCortFilename = 'n10lamda25_compare.txt'
	inLineCortEpisode = 200
	reloadCompare(inLineCortFilename, inLineCortEpisode, inLineCortXLabel, inLineCortYLabel)

	# 绘制直方图
	# inHisXLabel = 'Name of different algorithms'
	# inHisYLabel = 'Number of tasks finished before deadlines(%)'
	# inHisFilename = 'n20lamda47.txt'
	# inHisEpisode = 101
	# countNumberOfTasksFinishedBeforeDeadLines(inHisFilename, inHisEpisode, inHisXLabel, inHisYLabel)

	# 绘制折线图
	# inLineCortXLabel = 'Reward'
	# inLineCortYLabel = 'Episode'
	# inLineCortFilename = 'n10lamda25.txt'
	# inLineCortEpisode = 260
	# rewardAndEpisode(inLineCortFilename, inLineCortEpisode, inLineCortXLabel, inLineCortYLabel)


