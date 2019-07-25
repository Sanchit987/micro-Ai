from keras.models import Sequential
from keras.layers import Dense,Conv2D,Dropout,Flatten,MaxPooling2D
import tensorflow as tf
import matplotlib.pyplot as plt
'''
I AM GETTING THE DATA TO BE USED FROM MNIST DATA AVAILABLE FREELY ON ITS SITE. THIS DATA IS BEING IMPORTED VIA KERAS.
'''
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

'''
THESE FOUR STATEMENTS ARE USED TO CHECK THE TRAINNING IMAGES AND ITS LABEL WHICH TELLS THE NUMBER
'''
image_index = 7777 # You may select anything up to 60,000
print(y_train[image_index]) # The label is 8
plt.imshow(x_train[image_index])
plt.show()
'''
As i need to know the shape of the dataset to channel it to the convolutional neural network. Therefore “shape” attribute of numpy array is being used.
It returns a 3D array.
'''
x_train.shape

'''
For using keras with numpy we need 4D array hence we need to reshape it.There are four arguments in reshape(this is the number of training images,length,breath,height(for images its one))
'''
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
input_shape = (28, 28, 1)
# Making sure that the values are float so that we can get decimal points after division
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
# Normalizing the RGB codes by dividing it to the max RGB value.
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print('Number of images in x_train', x_train.shape[0])
print('Number of images in x_test', x_test.shape[0])

# Creating a Sequential Model and adding the layers
model = Sequential()
model.add(Conv2D(28, kernel_size=(3,3), input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten()) # Flattening the 2D arrays for fully connected layers
model.add(Dense(128, activation=tf.nn.relu))
model.add(Dropout(0.2))
model.add(Dense(10,activation=tf.nn.softmax))

'''
Model compiling
'''
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(x=x_train,y=y_train, epochs=10)


model.evaluate(x_test, y_test)


'''
model use
'''
image_index = 4444
plt.imshow(x_test[image_index].reshape(28, 28))
plt.show()
pred = model.predict(x_test[image_index].reshape(1, 28, 28, 1))
print(pred.argmax())
