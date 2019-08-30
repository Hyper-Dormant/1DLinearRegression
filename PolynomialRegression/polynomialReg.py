import numpy as np
import matplotlib.pyplot as plt


#loading data
X=[]
Y=[]
for line in open('data_poly.csv'):
    x,y=line.split(',')
    X.append([float(x),float(x)**2,1])
    Y.append(float(y))

X=np.array(X)
Y=np.array(Y)

#plotting the data
plt.figure('Data')
plt.scatter(X[:,0],Y)
plt.show()


#calculate weight to find Yhat
#Same as Multiple Linear Regression
W=np.linalg.solve(np.dot(X.T,X),np.dot(X.T,Y))
Yhat=np.dot(X,W)

#plotting prediction line
plt.figure('line of best fit')
plt.scatter(X[:,0],Y)
#sorting is to avoid a strange case
#guarantees point are in the right order(quadratic function is monotinically increasing)
plt.plot(sorted(X[:,0]),sorted(Yhat))
plt.show()

#Assesment of the model
SSres=np.dot(Y-Yhat,Y-Yhat)
SStot=np.dot(Y-Y.mean(),Y-Y.mean())

print('R2 value: {0}'.format(1-SSres/SStot))
