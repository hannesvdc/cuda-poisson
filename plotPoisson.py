import numpy as np
import numpy.linalg as lg
import matplotlib.pyplot as plt

# Load the CSV file
data = np.loadtxt("solution.csv", delimiter=',')
data_cuda = np.loadtxt("solution_cuda.csv", delimiter=',')
print(lg.norm(data - data_cuda, ord=np.inf))
N = data.shape[0]

X = np.linspace(0.0, 1.0, N)
Y = np.linspace(0.0, 1.0, N)
X, Y = np.meshgrid(X, Y)
plt.pcolor(X, Y, data, cmap='jet')
plt.colorbar()
plt.xlabel(r'$x$')
plt.ylabel(r'$y$')
plt.title('CPU Reference Solver')

plt.figure()
plt.pcolor(X, Y, data_cuda, cmap='jet')
plt.colorbar()
plt.xlabel(r'$x$')
plt.ylabel(r'$y$')
plt.title('GPU Accelerated Solver')
plt.show()