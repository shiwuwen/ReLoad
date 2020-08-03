import numpy as np

def dijkstra(v0, cost):
	passed = [v0]
	nopassed = [x for x in range(len(cost)) if x != v0]

	distance = cost[v0]

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

def get_es_propagation_delay(num):

	g = np.random.uniform(0.010, 0.100,size=(num,num))

	for i in range(num):
		g[i,i] = 0
		for j in range(i):
			g[i,j] = g[j,i]

	return g


if __name__ == "__main__":
	cost = get_es_propagation_delay(5)
	print("cost : ")
	print(cost)

	distance = []
	for v0 in range(len(cost)):
		# print(v0)
		result = dijkstra(v0, cost)
		distance.append(result)
		# distance.append(list(dijkstra(v0, cost)))
	# ditance = dijkstra(0, cost)
	print(np.asarray(distance)[0])