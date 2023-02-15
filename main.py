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


model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(128, input_dim=1, activation=tf.keras.layers.LeakyReLU(alpha=0.1)),
    tf.keras.layers.Dense(128,activation=tf.keras.layers.LeakyReLU(alpha=0.1)),
    tf.keras.layers.Dropout(0.01),
    tf.keras.layers.Dense(1)
])

loss_fn = tf.keras.losses.MeanSquaredError()
learning_rate = 0.001

model.compile(optimizer=tf.optimizers.Adam(learning_rate), loss=loss_fn)
history = model.fit(x_train, y_train, batch_size=5, epochs=100, validation_split=0.2,)

y_pred = model.predict(x_test)
plt.scatter(x_test, y_pred)


delta = []
deltaf = []
for i in range(Ntest):
    delta_n = y_pred[i] - y_test[i]
    delta.append(delta_n)
    if x_test[i]:
        deltaf.append(delta_n/x_test[i])
    else:
        deltaf.append(0)
plt.scatter(x_test, delta)
plt.title('Model accuracy')
plt.xlabel('x')
plt.ylabel('$\widehat{y}-y$')
plt.scatter(x_test, deltaf)
plt.legend(['Model', 'Prediction','$\delta y$','$\delta y frac$'])
plt.show()