# 3.1 Loading the MNIST dataset
# num_words=10000 means you'll only keep the top 10,000 most 
# frequently occuring words in the training data.
from keras.datasets import imdb
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(
    num_words=10000)

# print(train_data[0])
# print(train_labels[0])

# Decode review back to English
word_index = imdb.get_word_index()
reverse_word_index = dict(
    [(value, key) for (key, value) in word_index.items()])

# indices are offset by 3 because 0,1,2 are reserved for
# "padding", "start of sequence", "unknown"
decoded_review = ' '.join(
    [reverse_word_index.get(i-3, '?') for i in train_data[0]])

# print(decoded_review)

# Preparing the data
# 3.2 Encoding the integer sequences into a binary matrix
import numpy as np
def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1
    return results

x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

# print(x_train[0])

# also vectorize the labels
y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')

# now the data is ready to be fed into a neural network

# 3.3 the model definition
from keras import models
from keras import layers

model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

# 3.4 Compiling the model
# model.compile(optimizer='rmsprop',
#     loss = 'binary_crossentropy',
#     metrics = ['accuracy'])

# 3.5 Configuring the optimizer
# from keras import optimizers
# model.compile(optimizer=optimizers.RMSprop(lr=0.001),
#     loss='binary_crossentropy',
#     metrics = ['accuracy'])

# 3.6 Using custom losses and metrics
# from keras import losses
# from keras import metrics
# model.compile(optimizer=optimizers.RMSprop(lr=0.001),
#     loss=losses.binary_crossentropy,
#     metrics=[metrics.binary_accuracy])

# 3.7 Setting aside a validation set
x_val = x_train[:10000]
partial_x_train = x_train[10000:]

y_val = y_train[:10000]
partial_y_train = y_train[10000:]

# 3.8 Training your model
model.compile(optimizer='rmsprop',
    loss = 'binary_crossentropy',
    metrics = ['acc'])

history = model.fit(partial_x_train,
    partial_y_train,
    epochs=20,
    batch_size=512,
    validation_data=(x_val, y_val))

history_dict = history.history
# print(history_dict.keys())
# print(history_dict['acc'])

# 3.9 Plotting the training and validation loss
import matplotlib.pyplot as plt
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']

epochs = range(1, len(history_dict['acc']) + 1)
# print(epochs)

plt.plot(epochs, loss_values, 'bo', label='Training loss')
plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

# 3.10 plotting the training and validation accuracy
plt.clf()