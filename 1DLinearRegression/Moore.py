import re
import numpy as np
import matplotlib.pyplot as plt

X = []
Y = []

# some numbers show up as 1,170,000,000 (commas)
# some numbers have references in square brackets after them
non_decimal = re.compile(r'[^\d]+')

for line in open('moore.csv'):
    r = line.split('\t')

    x = int(non_decimal.sub('', r[2].split('[')[0]))
    y = int(non_decimal.sub('', r[1].split('[')[0]))
    X.append(x)
    Y.append(y)


X = np.array(X)
Y = np.array(Y)

plt.scatter(X, Y)
plt.show()


#compress data to linear by taking log
plt.figure('Data')
Y=np.log(Y)
plt.scatter(X, Y)
plt.show()

denominator=np.dot(X,X) - X.mean()*X.sum()
a=(X.dot(Y)-Y.mean()*X.sum())/denominator
b=(Y.mean()*X.dot(X)-X.mean()*X.dot(Y))/denominator

Yhat=a*X+b

plt.figure('Line of best fit')
plt.scatter(X,Y)
plt.plot(X,Yhat)
plt.show()

#Assesment
SSres=np.dot(Y-Yhat,Y-Yhat)
SStot=np.dot(Y-Y.mean(),Y-Y.mean())

R2=1-SSres/SStot
print(R2)
