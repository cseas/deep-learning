# 2.1 Loading the MNIST dataset in Keras
from keras.datasets import mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()


# 2.2 The network architecture
from keras import models
from keras import layers

network = models.Sequential()
network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
network.add(layers.Dense(10, activation='softmax'))


# 2.3 The compilation step
network.compile(
    optimizer='rmsprop',
    loss='categorical_crossentropy',
    metrics=['accuracy'])


# 2.4 Preparing the image data
# We transform uint8 values in the [0, 255] interval
# to float32 values between 0 and 1.
train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype('float32') / 255


# 2.5 Preparing the labels
from keras.utils import to_categorical
# categorically encode the labels (refer chapter 3)
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# train the network
network.fit(train_images, train_labels, epochs=5, batch_size=128)

# check accuracy on test data
test_loss, test_acc = network.evaluate(test_images, test_labels)
print('test_acc', test_acc)


# 2.6 Displaying the fourth digit
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
digit = train_images[4]
import matplotlib.pyplot as plt
plt.imshow(digit, cmap=plt.cm.binary)
plt.show()