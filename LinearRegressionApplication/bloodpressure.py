#Data file:
#X1:blood pressure
#X2:age in years
#X3: weight in pounds
#We want to predict blood pressure
#given age and weight

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df=pd.read_excel('mlr02.xls')
X=df.values

#plotting relation between age and blood pressure
plt.figure('f(age)')
plt.scatter(X[:,1],X[:,0])
plt.show()

#plotting relation between weight and blood pressure
plt.figure('f(weight)')
plt.scatter(X[:,2],X[:,0])
plt.show()

#Add bias

df['ones']=1
#Uncomment the next line to test the effect of adding noise to data
#df['noise']=np.random.randn(11)
#labels
Y=df['X1']
#independent variables(preparing for 3 seperate linear regressions) ////Add 'noise' to X to include it in data assesment
X=df[['X2','X3','ones']]

age=df[['X2','ones']]
weight=df[['X3','ones']]

#Assesment of model's accuracy

def get_R2(Y,Yhat):
    SSres=np.dot(Y-Yhat,Y-Yhat)
    SStot=np.dot(Y-Y.mean(),Y-Y.mean())
    R2=1-SSres/SStot
    return R2

def get_Yhat(X,Y):
    W=np.linalg.solve(np.dot(X.T,X),
        np.dot(X.T,Y))
    Yhat=np.dot(X,W)
    return Yhat

#computer accuracy for bloodpressure=f(age)
R2_age=get_R2(Y,get_Yhat(age,Y))
R2_weight=get_R2(Y,get_Yhat(weight,Y))
R2_age_weight=get_R2(Y,get_Yhat(X,Y))

print("R2_age: {0}".format(R2_age))
print("R2_weight: {0}".format(R2_weight))
print("R2_age/weight: {0}".format(R2_age_weight))


