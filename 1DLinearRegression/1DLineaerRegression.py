import numpy as np
import matplotlib.pyplot as plt

X=[]
Y=[]

#load data
for line in open('data_1d.csv'):
    x,y=line.split(',')
    X.append(float(x))
    Y.append(float(y))

#convert lists to numpy array
X=np.array(X)
Y=np.array(Y)

#Visualize the data
plt.figure('Data')
plt.scatter(X,Y)
plt.show()

#Find slope(a) and y-intercept(b) values
#such that MSE is minimized

denominator=np.dot(X,X) - X.mean()*X.sum()
a=(X.dot(Y)-Y.mean()*X.sum())/denominator
b=(Y.mean()*X.dot(X)-X.mean()*X.dot(Y))/denominator

#Calculate prediction line(line of best fit)
Yhat=a*X+b

#plot data and the line of best fit
plt.figure('Line of best fit')
plt.scatter(X,Y)
plt.plot(X,Yhat)
plt.title('1D Linear Regression')
plt.show()
