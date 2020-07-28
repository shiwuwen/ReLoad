import numpy as np 

# the1 = np.random.uniform(20,50,[1])
# print(the1)

# result = np.random.exponential(the1, [20])
# print(np.mean(result))
# print(result)

list1 = [[1,2],[3,4]]

list2 = [[5,6],[7,8]]

print(type(list1))
print(np.array(list1)*np.array(list2))
print(np.matmul(list1,list2))