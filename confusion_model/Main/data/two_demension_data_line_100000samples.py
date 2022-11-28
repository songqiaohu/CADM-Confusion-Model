import numpy as np
import matplotlib.pyplot as plt
import csv
max_sample = 100000



X = np.random.uniform(-10,10,max_sample)
Y = np.random.uniform(-10,10,max_sample)
data = np.column_stack((X, Y))

l, w = data.shape

label = np.zeros((l, 1))
flag = 1
for i in range(l):
    if flag == 1:
        if data[i, 0] + data[i, 1] >= 0:
            label[i] = 0
        else:
            label[i] = 1
    else:
        if data[i, 0] + data[i, 1] <= 0:
            label[i] = 0
        else:
            label[i] = 1
    if i % 5000 == 0:
        flag *= -1

plt.figure(0)
X1 = X[20000 : 25000]
Y1 = Y[20000 : 25000]
Z1 = label[20000 : 25000]
plt.scatter(X1, Y1, marker='h', c=Z1)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Decision Boundary---Line: y = -x, before drift')


plt.figure(1)
X1 = X[15000 : 20000]
Y1 = Y[15000 : 20000]
Z1 = label[15000 : 20000]
plt.scatter(X1, Y1, marker='h', c=Z1)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Decision Boundary---Line: y = -x, after drift')
data = np.column_stack((data, label))
with open('line_100000.csv', 'w', encoding='utf-8', newline='') as fp:
    writer = csv.writer(fp)
    writer.writerows(data)

plt.show()
