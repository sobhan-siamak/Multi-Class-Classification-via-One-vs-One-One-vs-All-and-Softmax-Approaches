




############   @copy by sobhan siamak

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import math
import warnings
warnings.filterwarnings('ignore')



def phi(i,theta,x):  #i goes from 1 to k
    mat_theta = np.matrix(theta[i])
    mat_x = np.matrix(x)
    num = math.exp(np.dot(mat_theta,mat_x.T))
    den = 0
    for j in range(0,k):
        mat_theta_j = np.matrix(theta[j])
        den = den + math.exp(np.dot(mat_theta_j,mat_x.T))
    phi_i = num/den
    return phi_i




def indicator(a,b):
    if a == b: return 1
    else: return 0


def computegrad(j,theta):
    sum = np.array([0 for i in range(0,n+1)])
    for i in range(0,m):
        p = indicator(y[i],j) - phi(j,theta,x.loc[i])
        sum = sum + (x.loc[i] *p)
    grad = -sum/m
    return grad



def gradient_descent(theta,alpha= 1/(10^4),iters=100):#iters=1000
    for j in range(0,k):
        for iter in range(iters):
            theta[j] = theta[j] - alpha * computegrad(j,theta)
    print('running iterations')
    return theta


#give us the probability of obtaining each outcome given a vector of features x
def h_theta(x):
    x = np.matrix(x)
    h_matrix = np.empty((k,1))
    den = 0
    for j in range(0,k):
        den = den + math.exp(np.dot(theta_dash[j], x.T))
    for i in range(0,k):
        h_matrix[i] = math.exp(np.dot(theta_dash[i],x.T))
    h_matrix = h_matrix/den
    return h_matrix



iris = pd.read_csv('Iris.csv')
iris = iris.drop(['Id'],axis=1)
print(iris.head())

####

ir = load_iris()

xsetosa = ir.data[0:50,:]
ysetosa = ir.target[0:50]
xversicolor = ir.data[50:100,:]
yversicolor = ir.target[50:100]
xvirginica = ir.data[100:150,:]
yvirginica = ir.target[100:150]

# Splitting the dataset into the Training set and Test set
x_train1, x_test1, y_train1, y_test1 = train_test_split(xsetosa, ysetosa, test_size = 0.20, random_state = 21)#, random_state = 82
x_train2, x_test2, y_train2, y_test2 = train_test_split(xversicolor, yversicolor, test_size = 0.20, random_state = 21)#, random_state = 82
x_train3, x_test3, y_train3, y_test3 = train_test_split(xvirginica, yvirginica, test_size = 0.20, random_state = 21)#, random_state = 82

x_train = np.concatenate([x_train1, x_train2, x_train3])
y_train = np.concatenate([y_train1, y_train2, y_train3])

xtest = np.concatenate([x_test1, x_test2, x_test3])
ytest = np.concatenate([y_test1, y_test2, y_test3])



####


train, test = train_test_split(iris, test_size = 0.2, random_state=82)# in this our main data is split into train and test
train = train.reset_index()
test = test.reset_index()


x = train[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
n = x.shape[1]
m = x.shape[0]



y = train['Species']
k = len(y.unique())
y =y.map({'Iris-setosa':0,'Iris-versicolor':1,'Iris-virginica':2})
y.value_counts()

#bias in train data
x[5] = np.ones([x.shape[0]])
print(x.shape)

theta = np.empty((k,n+1))

theta_dash = gradient_descent(theta)

print(theta_dash)


x_u = test[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
n = x_u.shape[1]
m = x_u.shape[0]


y_true = test['Species']
k = len(y_true.unique())
y_true =y_true.map({'Iris-setosa':0,'Iris-versicolor':1,'Iris-virginica':2})
y_true.value_counts()

#bias in test data
x_u[5] = np.ones([x_u.shape[0]])
# x_u.shape


# Train Data Accuracy
for index1,row1 in x.iterrows():
    h_matrix = h_theta(row1)
    prediction = int(np.where(h_matrix == h_matrix.max())[0])
    x.loc[index1,'prediction'] = prediction

results1 = x
results1['actual'] = y

compare1 = results1['prediction'] == results1['actual']
correct1 = compare1.value_counts()[1]
accuracy1 = correct1/len(results1)

#Test Data accuracy
for index,row in x_u.iterrows():
    h_matrix = h_theta(row)
    prediction = int(np.where(h_matrix == h_matrix.max())[0])
    x_u.loc[index,'prediction'] = prediction




results = x_u
results['actual'] = y_true
print(results.head(10))


compare = results['prediction'] == results['actual']
correct = compare.value_counts()[1]
accuracy = correct/len(results)




print("The accuracy in Train Data is:")
print(accuracy1*100)


print("The accuracy in Test Data is:")
print(accuracy*100)  #best = 97.777777 or best =100


