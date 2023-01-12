import numpy as np
import matplotlib.pyplot as plt
import csv
max_sample = 100000

rr = np.random.uniform(0, 100, max_sample)
theta = np.random.uniform(0, 2 * np.pi, max_sample)
X = np.sqrt(rr) * np.cos(theta)
Y = np.sqrt(rr) * np.sin(theta)#to be uniform
data = np.column_stack((X, Y))

l, w = data.shape
label = np.zeros((l, 1))

#a flag to control drifts
flag = 1
for i in range(l):
    if flag == 1:
        if data[i, 0] ** 2 + data[i, 1] ** 2 >= 50:
            label[i] = 0
        else:
            label[i] = 1
    else:
        if data[i, 0] ** 2 + data[i, 1] ** 2 <= 50:
            label[i] = 0
        else:
            label[i] = 1
    if i % 5000 == 0:
        flag *= -1

#######before drift#################
plt.figure(0)
X1 = X[0:5000]
Y1 = Y[0:5000]
Z1 = label[0:5000]
plt.scatter(X1, Y1, marker='h', c=Z1)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Decision Boundary---Circle: x$^2$ + y$^2$ = 50, before drift')


####### after drift ##############
plt.figure(1)
X1 = X[5000:10000]
Y1 = Y[5000:10000]
Z1 = label[5000:10000]
plt.scatter(X1, Y1, marker='h', c=Z1)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Decision Boundary---Circle: x$^2$ + y$^2$ = 50, after drift')

#######  writer  #####################
data = np.column_stack((data, label))
with open('circle_100000.csv', 'w', encoding='utf-8', newline='') as fp:
    writer = csv.writer(fp)
    writer.writerows(data)

plt.show()
