import matplotlib.pyplot as plt

from random import random

inside = 0
n = 10**3

x_inside = []
y_inside = []
x_outside = []
y_outside = []

for _ in range(n):
    x = random()
    y = random()
    if x**2+y**2 <= 1:
        inside += 1
        x_inside.append(x)
        y_inside.append(y)
    else:
        x_outside.append(x)
        y_outside.append(y)

pi = 4*inside/n
print(pi)

fig, ax = plt.subplots(1)
ax.set_aspect('equal')
plt.scatter(x_inside, y_inside, color='b', marker='s')
plt.scatter(x_outside, y_outside, color='r', marker='s')
plt.show()
