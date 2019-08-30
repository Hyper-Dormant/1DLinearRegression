import numpy as np
import matplotlib.pyplot as plt


#create a fat matrix to use L1 regularization
N=50
D=50

#create data
X=10*np.random.random((N,D))-5
#review weights creation !
true_w=np.array([1,0.5,-0.5]+[0]*(D-3))

Y=np.dot(X,true_w)+0.5*np.random.randn(N)

#apply gradient descent to the weights:

costs=[]

w=np.random.randn(D)/np.sqrt(D)
learning_rate=0.001
l1=10.0

for t in range(500):
    Yhat=np.dot(X,w)
    delta=Yhat-Y
    w=w-learning_rate*(X.T.dot(delta)+l1*np.sign(w))

    mse=delta.dot(delta)/N
    costs.append(mse)

plt.plot(costs)
plt.show()

print("Final W:{0}".format(w))

plt.plot(true_w,label='true w')
plt.plot(w,label='w_map')
plt.legend()
plt.show()
