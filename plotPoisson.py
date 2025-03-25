import numpy as np
import matplotlib.pyplot as plt

# Load the CSV file
data = np.loadtxt("solution.csv", delimiter=',')
N = data.shape[0]

X = np.linspace(0.0, 1.0, N)
Y = np.linspace(0.0, 1.0, N)
X, Y = np.meshgrid(X, Y)
plt.pcolor(X, Y, data, cmap='jet')
plt.colorbar()
plt.xlabel(r'$x$')
plt.ylabel(r'$y$')
plt.show()