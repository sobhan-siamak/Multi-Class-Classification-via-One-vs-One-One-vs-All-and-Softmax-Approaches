



############   @copy by sobhan siamak

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
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
x_train1, x_test1, y_train1, y_test1 = train_test_split(xsetosa, ysetosa, test_size = 0.20, random_state = 21)#, random_state = 82
x_train2, x_test2, y_train2, y_test2 = train_test_split(xversicolor, yversicolor, test_size = 0.20, random_state = 21)#, random_state = 82
x_train3, x_test3, y_train3, y_test3 = train_test_split(xvirginica, yvirginica, test_size = 0.20, random_state = 21)#, random_state = 82


x_train = np.concatenate([x_train1, x_train2, x_train3])
y_train = np.concatenate([y_train1, y_train2, y_train3])


# m, n= x_train.shape
# # print("xtrain before normalization", x_train)
# for j in range(n):
#         mx = np.max(x_train[:, j])
#         mi = np.min(x_train[:, j])
#         x_train[:,j] = (x_train[:,j]-mi)/(mx-mi)

# x_train = pd.DataFrame(x_train)
# x_train.insert(0, "bias", 1)

xtrain01 = np.concatenate([x_train1, x_train2])
ytrain01 = np.concatenate([y_train1, y_train2])

xtrain02 = np.concatenate([x_train1, x_train3])
ytrain02 = np.concatenate([y_train1, y_train3])

ytrain02 = np.where(ytrain02 == 2, 1, 0)



xtrain12 = np.concatenate([x_train2, x_train3])
ytrain12 = np.concatenate([y_train2, y_train3])

ytrain12 = np.where(ytrain12 == 1, 0, 1)





xtest = np.concatenate([x_test1, x_test2, x_test3])
ytest = np.concatenate([y_test1, y_test2, y_test3])




def sigmoid(scores):
    return 1 / (1 + np.exp(-scores))


# def log_likelihood(features, target, weights):
#     scores = np.dot(features, weights)
#     ll = np.sum( target*scores - np.log(1 + np.exp(scores)) )
#     return ll


def logistic_regression(features, target, num_steps, learning_rate, add_intercept=False):
    if add_intercept:
        intercept = np.ones((features.shape[0], 1))
        features = np.hstack((intercept, features))

    weights = np.zeros(features.shape[1])

    for step in range(num_steps):
        scores = np.dot(features, weights)
        predictions = sigmoid(scores)

        # Update weights with log likelihood gradient
        output_error_signal = target - predictions

        gradient = np.dot(features.T, output_error_signal)
        weights += learning_rate * gradient

        # Print log-likelihood every so often
        # if step % 10000 == 0:
        #     print(log_likelihood(features, target, weights))
    return weights


Thetas01 = logistic_regression(xtrain01, ytrain01,
                     num_steps = 50000, learning_rate = 1/(10^4), add_intercept=True)# learning_rate = 5e-1
Thetas02 = logistic_regression(xtrain02, ytrain02,
                     num_steps = 50000, learning_rate = 1/(10^4), add_intercept=True)# learning_rate = 5e-1

Thetas12 = logistic_regression(xtrain12, ytrain12,
                     num_steps = 50000, learning_rate = 1/(10^4), add_intercept=True)# learning_rate = 5e-1


print("Thetas for classifier01 is:", Thetas01)
print("=======================================")
print("Thetas for classifier02 is:", Thetas02)
print("=======================================")
print("Thetas for classifier12 is:", Thetas12)




####classifier01
final_scores01 = np.dot(np.hstack((np.ones((xtrain01.shape[0], 1)),
                                 xtrain01)), Thetas01)
preds01 = np.round(sigmoid(final_scores01))
print("===========================================")
print('Accuracy from classifier01 is: {0}'.format((preds01 == ytrain01).sum().astype(float) / len(preds01)))


####classifier02
final_scores02 = np.dot(np.hstack((np.ones((xtrain02.shape[0], 1)),
                                 xtrain02)), Thetas02)
preds02 = np.round(sigmoid(final_scores02))
# preds02 = np.where(preds02 == 2, 1, 0)

print('Accuracy from classifier02 is: {0}'.format((preds02 == ytrain02).sum().astype(float) / len(preds02)))


####classifier12
final_scores12 = np.dot(np.hstack((np.ones((xtrain12.shape[0], 1)),
                                 xtrain12)), Thetas12)
preds12 = np.round(sigmoid(final_scores12))
# preds12 = np.where(preds12 == 2, 0, 1)
# print(np.hstack((np.ones((xtrain12.shape[0], 1)), xtrain12)))
print('Accuracy from classifier12 is: {0}'.format((preds12 == ytrain12).sum().astype(float) / len(preds12)))

# add bias again
x_train = pd.DataFrame(x_train)
x_train.insert(0,"bias",1)
xtest = pd.DataFrame(xtest)
xtest.insert(0,"bias",1)



#for training data
m, n = x_train.shape
classes = np.zeros([m,3])
#for classifier01
for i in range(m):
    cls = np.round(sigmoid(np.dot(Thetas01, x_train.iloc[i,:])))
    classes[i,0] = cls
#for classifier02
for i in range(m):
    cls1 = np.round(sigmoid(np.dot(Thetas02, x_train.iloc[i,:])))
    if cls1 == 0 :
        classes[i, 1] = cls1
    else:classes[i, 1] = 2
#for classifier12
for i in range(m):
    cls1 = np.round(sigmoid(np.dot(Thetas12, x_train.iloc[i,:])))
    if cls1 == 0 :
        classes[i, 2] = 1
    else:classes[i, 2] = 2

# print(classes)


#for test data
m1, n1 = xtest.shape
classes1 = np.zeros([m1, 3])
#for classifier01
for i in range(m1):
    cls = np.round(sigmoid(np.dot(Thetas01, xtest.iloc[i,:])))
    classes1[i,0] = cls
#for classifier02
for i in range(m1):
    cls1 = np.round(sigmoid(np.dot(Thetas02, xtest.iloc[i,:])))
    if cls1 == 0 :
        classes1[i, 1] = cls1
    else:classes1[i, 1] = 2
#for classifier12
for i in range(m1):
    cls1 = np.round(sigmoid(np.dot(Thetas12, xtest.iloc[i,:])))
    if cls1 == 0 :
        classes1[i, 2] = 1
    else:classes1[i, 2] = 2
# print(classes1)


#####counting for training data

predict = np.zeros([m,1])
predict1 = np.zeros([m1,1])
classnum = 3
# count = np.zeros([1,classnum])
# counter = np.zeros([m,3])
for i in range(m):
    count = np.zeros([1, classnum])
    for j in range(classnum):
        if (classes[i][j] == 0):
            count[0][0] += 1
        if (classes[i][j] == 1):
            count[0][1] += 1
        if (classes[i][j] == 2):
            count[0][2] += 1
    # counter[i,:] = count
    predict[i] = np.argmax(count)
# print(predict)
# print(y_train)


for i in range(m1):
    count1 = np.zeros([1, classnum])
    for j in range(classnum):
        if (classes1[i][j] == 0):
            count1[0][0] += 1
        if (classes1[i][j] == 1):
            count1[0][1] += 1
        if (classes1[i][j] == 2):
            count1[0][2] += 1
    # counter[i,:] = count
    predict1[i] = np.argmax(count1)



print("============================================")
print('Accuracy from Total train data is: {0}'.format((predict.transpose() == y_train).sum().astype(float) / len(predict)))
print('Accuracy from test data is: {0}'.format((predict1.transpose() == ytest).sum().astype(float) / len(predict1)))



