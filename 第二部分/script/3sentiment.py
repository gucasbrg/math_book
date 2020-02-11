import numpy as np
matrix = np.matrix([[0.5,0.3,0.2],[0.2,0.4,0.4],[0.1,0.3,0.6]], dtype=float)
vector1 = np.matrix([[0.1,0.3,0.6]], dtype=float)

for i in range(20):
    vector1 = vector1*matrix

	
    print("%s" % (vector1))