import random, math
import matplotlib.pyplot as plt
import numpy as np
def gibbs():
    X = []
    Y = []
    def sampling(N=2000):
        x = 0
        y = 0
        for i in range(N):
            x = random.gauss(0 - (1 / 1.33) * (-0.667) * (y - 0), 1)
            X.append(x)
            Y.append(y)
            y = random.gauss(0 - (1 / 1.33) * (-0.667) * (x - 0), 1)
            X.append(x)
            Y.append(y)
    sampling()
    plt.plot(X, Y, 'r-')
    plt.xlabel('x_1')
    plt.ylabel('x_2')
    plt.show()

def show_multi_normal():
    mean = [0, 0]
    cov = [[1, 0.5], [0.5, 1]]
    x, y = np.random.multivariate_normal(mean, cov, 5000).T
    plt.plot(x, y, 'r-')
    plt.show()

#show_multi_normal()
gibbs()
