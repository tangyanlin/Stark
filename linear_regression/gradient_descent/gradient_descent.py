#encoding:utf-8

import sys
import os
import numpy as np

dir = os.path.dirname(__file__)

loop_max = 100
epsolon = 0.0001
alpha = 0.005

def get_training_data(path):
    x_array = []
    y_array = []
    with open(path, 'r') as f:
        for line in f:
            y = line.split(',')[-1]
            x = line.split(',')[:-1]
            x.insert(0, 1)
            x_array.append(map(int, x))
            y_array.append(int(y))
    return x_array, y_array

def batch_gradient_descent(m, theta, x, y):
    count = 0
    print m, theta, x, y
    while count < loop_max:
        count += 1
        hypothesis = np.dot(x, theta)
        loss = hypothesis - y
        gradient = np.dot(x.transpose(), loss)
        theta = theta - alpha * gradient
    return theta

def inc_gradient_descent(m, theta, x, y):
    count = 0
    while count < loop_max:
        count += 1
        for i in range(m):
            diff = np.dot(theta, x[i]) -y[i]
            theta = theta - alpha * diff * x[i]
    return theta

if __name__ == '__main__':
    x, y = get_training_data(os.path.join(dir, 'training_data'))
    if not x:
        sys.exit(0)
    m = len(x)
    theta = np.array([1 for i in x[0]])
    x = np.array(x)
    y = np.array(y)
    print batch_gradient_descent(m, theta, x, y)
    print inc_gradient_descent(m, theta, x, y)
