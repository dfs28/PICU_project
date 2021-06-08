### Sample bit of code to practice making autoencoder

# Setup
import keras
from keras import layers
from keras.datasets import mnist
import numpy as np
import pandas as pd
import math

training_data = pd.read_csv('Project/PICU_project/practice_data/datatraining.txt')
training_array = np.array(training_data.loc[:, training_data.columns[1:-1]])
testing_data = pd.read_csv('Project/PICU_project/practice_data/datatest.txt')
testing_array = np.array(testing_data.loc[:, testing_data.columns[1:-1]])

#Normalise the data
def normalise_array(array):
    """
    Function to normalise the data so that it is between 0 and 1
    Plan to add in functionality to consider capping outliers for example?
    Or for certain lab values could maybe log scale them and then normalise
    """

    for i in range(array.shape[1]):

        max_value = np.max(array[:, i])
        array[:, i] = array[:, i]/max_value

    return array


#Make 3d training array
def make_3d_array(array, length):
    """
    Function to make an array into slices and stack them
    Takes a 2d array, slices it by the length specified
    Expects the time series to be longer than the number of vars
    """

    #Get the shape, work out what shape the new array should be
    shape = array.shape
    z_dim = math.floor(max(shape)/length)
    x_dim = length
    y_dim = min(shape)
    array_3d = np.empty((z_dim, y_dim, x_dim))

    for i in range(z_dim):
        start_position = i*length
        end_position = (i + 1)*length
        temp_array = array[start_position:end_position, :]
        array_3d[i, :, :] = temp_array.transpose()

    return array_3d

length = 100
normalised_array = normalise_array(training_array)
training_array3d = make_3d_array(normalised_array, length)
input_shape = training_array3d.shape
input_timeseries = keras.Input(shape = input_shape[1:])

#Now make 1d conv net
# This is the encoder - set the shape to be the shape of the timeseries data
x = layers.Conv1D(50, 3, activation='relu', padding = 'same')(input_timeseries)
x = layers.Conv1D(25, 3, activation='relu', padding = 'same')(x)
encoded = layers.Conv1D(10, 3, activation='relu', padding = 'same')(x)

#Now make the docoder
x = layers.Conv1D(25, 3, activation = 'relu', padding = 'same')(encoded)
x = layers.Conv1D(50, 3, activation= 'relu', padding = 'same')(x)
decoded = layers.Conv1D(100, 3, activation = 'sigmoid', padding = 'same')(x)

autoencoder = keras.Model(input_timeseries, decoded)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')


"""
# This is the size of our encoded representations
encoding_dim = 32  # 32 floats -> compression of factor 24.5, assuming the input is 784 floats

input_img = keras.Input(shape=(28, 28, 1))

x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
x = layers.MaxPooling2D((2, 2), padding='same')(x)
x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = layers.MaxPooling2D((2, 2), padding='same')(x)
x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
encoded = layers.MaxPooling2D((2, 2), padding='same')(x)

# at this point the representation is (4, 4, 8) i.e. 128-dimensional

x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
x = layers.UpSampling2D((2, 2))(x)
x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = layers.UpSampling2D((2, 2))(x)
x = layers.Conv2D(16, (3, 3), activation='relu')(x)
x = layers.UpSampling2D((2, 2))(x)
decoded = layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

autoencoder = keras.Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')


(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))
x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))
print(x_train.shape)
print(x_test.shape)
"""

#Format the test data
normalised_test = normalise_array(testing_array)
testing_3d = make_3d_array(normalised_test, length)

autoencoder.fit(training_array3d, training_array3d,
                epochs=1000,
                batch_size=81,
                shuffle=True,
                validation_data=(testing_3d, testing_3d))


# Encode and decode some digits
# Note that we take them from the *test* set
encoded_imgs = encoder.predict(x_test)
decoded_imgs = decoder.predict(encoded_imgs)


# Use Matplotlib (don't ask)
import matplotlib.pyplot as plt

n = 10  # How many digits we will display
plt.figure(figsize=(20, 4))
for i in range(n):
    # Display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # Display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

#No GUI on Subliminal so need to save figures to view them
plt.savefig('Project/PICU_project/scripts/MNIST_autencoded.png')


