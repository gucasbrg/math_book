import numpy as np
import matplotlib.pyplot as plt

transfer_matrix = np.array([[0.5,0.3,0.2],[0.2,0.4,0.4],[0.1,0.3,0.6]],dtype='float32')
start_matrix = np.array([[0.1,0.3,0.6]],dtype='float32')

value1 = []
value2 = []
value3 = []
for i in range(30):
    start_matrix = np.dot(start_matrix,transfer_matrix)
    value1.append(start_matrix[0][0])
    value2.append(start_matrix[0][1])
    value3.append(start_matrix[0][2])
print(start_matrix)

x = np.arange(30)
plt.plot(x,value1,label='positive')
plt.plot(x,value2,label='other')
plt.plot(x,value3,label='negative')
plt.legend()
plt.show()