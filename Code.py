import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,Activation,Flatten,Conv2D,MaxPooling2D

# the data, split between train and test sets

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
#print(x_train.shape)

#plt.imshow(x_train[0])
#plt.show()

#gives b/w image
#plt.imshow(x_train[0],cmap=plt.cm.binary)
#plt.show()

# stores array of 28*28
#print(x_train[0])
# stores the actual digit value
#print(y_train[0])

# normalizes the image 0-255 --> 0-1
x_train = tf.keras.utils.normalize(x_train,axis = 1)
x_test = tf.keras.utils.normalize(x_test,axis = 1)
#plt.imshow(x_train[0],cmap=plt.cm.binary)
#plt.show()

#print(x_train[0])

#increasing by one dimension of the image to apply the convolution operation
IMG_SIZE = 28

x_trainer = np.array(x_train).reshape(-1,IMG_SIZE,IMG_SIZE,1) #increasing dimension for operation
x_tester = np.array(x_test).reshape(-1,IMG_SIZE,IMG_SIZE,1)
#print("New dimensions :",x_trainer.shape)
#print("New dimensions :",x_tester.shape)

model = Sequential()

model.add(Conv2D(64,(3,3),input_shape = x_trainer.shape[1:]))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64,(3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64,(3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation("relu"))

model.add(Dense(32))
model.add(Activation("relu"))

model.add(Dense(10))
model.add(Activation("softmax"))


#model.summary()

print("Total training samples = ",len(x_trainer))
model.compile(loss="sparse_categorical_crossentropy",optimizer="adam",metrics=['accuracy'])

model.fit(x_trainer,y_train,epochs=5,validation_split=0.3)
