





############   @copy by sobhan siamak

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
# from LogisticRegression import *
from sklearn import datasets



# Importing the dataset from scratch
# dataset = pd.read_csv('iris.csv')
#looking at the first 5 values of the dataset
# print(dataset.head())

# Importing the dataset from sklearn
iris = load_iris()

xsetosa = iris.data[0:50,:]
ysetosa = iris.target[0:50]
xversicolor = iris.data[50:100,:]
yversicolor = iris.target[50:100]
xvirginica = iris.data[100:150,:]
yvirginica = iris.target[100:150]



# Splitting the dataset into the Training set and Test set
x_train1, x_test1, y_train1, y_test1 = train_test_split(xsetosa, ysetosa, test_size = 0.20, random_state = 82)#, random_state = 82
x_train2, x_test2, y_train2, y_test2 = train_test_split(xversicolor, yversicolor, test_size = 0.20, random_state = 82)#, random_state = 82
x_train3, x_test3, y_train3, y_test3 = train_test_split(xvirginica, yvirginica, test_size = 0.20, random_state = 82)#, random_state = 82

x_train = np.concatenate([x_train1, x_train2, x_train3])
y_train = np.concatenate([y_train1, y_train2, y_train3])

y0binary=[]
y1binary=[]
y2binary=[]
ind = y_train.shape[0]

for i in range(ind):
   y0binary.append(1) if y_train[i]==0 else y0binary.append(0)
   y1binary.append(1) if y_train[i]==1 else y1binary.append(0)
   y2binary.append(1) if y_train[i]==2 else y2binary.append(0)

# print(y2binary)

x_test = np.concatenate([x_test1, x_test2, x_test3])
y_test = np.concatenate([y_test1, y_test2, y_test3])


# m, n= x_train.shape
# for j in range(n):
#         mx = np.max(x_train[:, j])
#         mi = np.min(x_train[:, j])
#         x_train[:,j] = (x_train[:,j]-mi)/(mx-mi)

# add bias column in training data
# x_train = pd.DataFrame(x_train)
# x_train.insert(0,"bias",1)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def cost(theta, x, y):
    h = sigmoid(x @ theta)#@ means dot
    m = len(y)
    cost = 1 / m * np.sum(
        -y * np.log(h) - (1 - y) * np.log(1 - h))
    grad = 1 / m * ((y - h) @ x)
    return cost, grad


def fit(x, y, max_iter=6500, alpha=0.1):
    x = np.insert(x, 0, 1, axis=1)
    thetas = []
    classes = np.unique(y)
    costs = np.zeros(max_iter)

    for c in classes:
        # one vs. all  binary classification
        binary_y = np.where(y == c, 1, 0)

        theta = np.zeros(x.shape[1])
        for epoch in range(max_iter):
            costs[epoch], grad = cost(theta, x, binary_y)
            theta += alpha * grad

        thetas.append(theta)
    return thetas, classes, costs


def predict(classes, thetas, x):
    x = np.insert(x, 0, 1, axis=1)# add verticaly columns of one(axis=1) to x
    preds = [np.argmax(
        [sigmoid(xi @ theta) for theta in thetas]) for xi in x]
    return [classes[p] for p in preds]




thetas, classes, costs = fit(x_train, y_train)
plt.plot(costs)
plt.xlabel('Number Epochs'); plt.ylabel('Cost')
plt.title("One versus all")
plt.show()


def score(classes, theta, x, y):
    return (predict(classes, theta, x) == y).mean()


AccTrain = score(classes, thetas, x_train, y_train)
AccTest =  score(classes, thetas, x_test, y_test)
print("Accuracy on Train Data is:", AccTrain*100)
print("Accuracy on Test Data is:", AccTest*100)


