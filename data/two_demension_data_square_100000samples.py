import numpy as np
import matplotlib.pyplot as plt
import csv
max_sample = 100000



X = np.random.uniform(-10, 10, max_sample)
Y = np.random.uniform(-10, 10, max_sample)
data = np.column_stack((X, Y))

l, w = data.shape

label = np.zeros((l, 1))
flag = 1
for i in range(l):
    if flag == 1:
        if data[i, 0] >= -5 * np.sqrt(2) and data[i, 0] <= 5 * np.sqrt(2) \
                and data[i, 1] >= -5 * np.sqrt(2) and data[i, 1] <= 5 * np.sqrt(2):
            label[i] = 0
        else:
            label[i] = 1
    else:
        if data[i, 0] >= -5 * np.sqrt(2) and data[i, 0] <= 5 * np.sqrt(2) \
                and data[i, 1] >= -5 * np.sqrt(2) and data[i, 1] <= 5 * np.sqrt(2):
            label[i] = 1
        else:
            label[i] = 0
    if i % 5000 == 0:
        flag *= -1

plt.figure(0)
X1 = X[0:5000]
Y1 = Y[0:5000]
Z1 = label[0:5000]
plt.scatter(X1, Y1, marker='h', c=Z1)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Decision Boundary---Square: x, y = $\pm$5$\sqrt{2}$, before drift')

plt.figure(1)
X1 = X[5000:10000]
Y1 = Y[5000:10000]
Z1 = label[5000:10000]
plt.scatter(X1, Y1, marker='h', c=Z1)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Decision Boundary---Square: x, y = $\pm$5$\sqrt{2}$, after drift')
data = np.column_stack((data, label))
with open('square_100000.csv', 'w', encoding='utf-8', newline='') as fp:
    writer = csv.writer(fp)
    writer.writerows(data)

plt.show()
