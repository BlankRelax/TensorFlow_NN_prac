import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
#-------------------------------------------------

Ntrain = 100
Ntest = 100
x_train = []
y_train = []
x_test = []
y_test = []
gaus_Noise = tf.random.normal([Ntrain,1], 0,0.05, seed=2)
xmax = np.pi*2
xmin = 0
#----------------------------------------------------

def sim_sin(xmin, xmax):
    x =np.random.random()*(xmax-xmin)+xmin
    y = np.sin(x)
    return x, y
#------------------------------------------------------
for i in range(Ntrain):
    x, y = sim_sin(xmin, xmax)
    y = y*(1 + np.random.random()*gaus_Noise[i,0]) #add noise
    x_train.append(x)
    y_train.append(y)

for i in range(Ntest):
    x, y = sim_sin(xmin, xmax)
    y = y*(1 + np.random.random()*gaus_Noise[i,0]) #add noise
    x_test.append(x)
    y_test.append(y)

x_test = np.array(x_test)
y_test = np.array(y_test)
x_train = np.array(x_train)
y_train = np.array(y_train)

plt.scatter(x_train, y_train)
plt.show()
