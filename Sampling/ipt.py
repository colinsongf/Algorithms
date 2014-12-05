import matplotlib.pyplot as plt
import math
import random
# Based on MLAPP, Page 816. Chapter 23, Monte Carlo Inference
# exponential: y = ax* exp(-ax)
Lambda = 1.0

def exponential(Lambda, x):
    return Lambda *  math.exp(-1 * Lambda * x)

def ipt(N = 100000):
    # Inverse Probability Transform
    # F ^-1 (x) = -1 * ln(x) / lambda
    lst_sample = []
    for i in range(N):
        u = random.uniform(0, 1)
        sample = - 1 * math.log(u) / Lambda
        lst_sample.append(sample)
    plt.hist(lst_sample, 50, normed=0.8, facecolor='g', alpha=1)

def plot_exponential():
    X = []
    Y = []
    for x in range(1, 10):
        X.append(x)
        y = exponential(Lambda, x)
        Y.append(y)
    plt.plot(X, Y, 'r-')
    plt.show()

ipt()
plot_exponential()
