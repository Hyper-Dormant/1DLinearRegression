import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

#load data
X=[]
Y=[]

for line in open('data_2d.csv'):
    x1,x2,y=line.split(',')
    X.append([float(x1),float(x2),1])
    Y.append(float(y))

X=np.array(X)
Y=np.array(Y)

#plotting the data
fig=plt.figure()
ax=fig.add_subplot(111,projection='3d')
ax.scatter(X[:,0],X[:,1],Y)
plt.show()

#calculate weights since Yhat=f(W)

W=np.linalg.solve(np.dot(X.T,X),np.dot(X.T,Y))

#prediction line
Yhat=np.dot(X,W)

#plotting prediction line on Data
#plotting the data
plt.figure('Sureface of best fit')
Xmesh, Ymesh = np.meshgrid(X[:,0], X[:,1])
Zmesh = W[0]*Xmesh + W[1]*Ymesh + W[2]
ax.scatter(X[:,0],X[:,1],Y)
surf=ax.plot_surface(Xmesh, Ymesh, Zmesh)
plt.show()

#Model's Accuracy using R2

SSres= np.dot(Y-Yhat,Y-Yhat)
SStot=np.dot(Y-Y.mean(),Y-Y.mean())
R2=1-SSres/SStot

print("The value of R2: {0}".format(R2))
