import numpy as np
import matplotlib.pyplot as plt

N=50

X=np.linspace(0,10,N)
#The function has some additional noise
Y=0.5*X+np.random.randn(N)

#add outliers, to check the effect of l1 regularization
Y[-1]+=25
Y[-2]+=25

plt.figure('Data')
plt.scatter(X,Y)
plt.show()

#add bias
X=np.vstack([X,np.ones(N)]).T
#computing the weights withput penalizing outliers (Maximum Liklihood solution)
W_ml=np.linalg.solve(np.dot(X.T,X),np.dot(X.T,Y))
Yhat_ml=np.dot(X,W_ml)
#plotting the maximum liklihood solution
plt.figure('Maximum liklihood solution')
plt.scatter(X[:,0],Y)
plt.plot(X[:,0],Yhat_ml)
plt.show()

#computing weights using l2 regularization

l2=1000.0
W_map=np.linalg.solve(l2*np.eye(2)+np.dot(X.T,X),np.dot(X.T,Y))
Yhat_map=np.dot(X,W_map)

#plot the Maximum A posteriori solution
plt.figure('MAP solution')
plt.scatter(X[:,0],Y)
plt.plot(X[:,0],Yhat_map)
plt.show()


