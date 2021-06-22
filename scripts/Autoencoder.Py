### Sample bit of code to practice making autoencoder

# Setup
import keras
from keras import layers
from keras.datasets import mnist
import numpy as np
import pandas as pd
import math


#Read in the data
flowsheet = pd.read_csv('~/Project/Project_data/files/flowsheet_output.csv', parse_dates = ['taken_datetime'])

#


"""
training_data = pd.read_csv('Project/PICU_project/practice_data/datatraining.txt')
training_array = np.array(training_data.loc[:, training_data.columns[1:-1]])
testing_data = pd.read_csv('Project/PICU_project/practice_data/datatest.txt')
testing_array = np.array(testing_data.loc[:, testing_data.columns[1:-1]])
"""

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

cols = ['Ventilation', 'IPAP', 'EPAP', 'ETCO2', 'FiO2', 'O2Flow', #Need to add back HFO, Tracheostomy
        'MeanAirwayPressure', 'SysBP', 'DiaBP', 'HR', 'Comfort', 'AVPU', 'GCS', #need to add back MAP at some point
        'CRT', 'SpO2', 'Inotropes']

#Make 3d training array
def make_3d_array(array, length, cols):
    """
    Function to make an pandas into a 3d np.array of slices and stack them \n
    Takes a dataframe, slices it by the length specified 
    Expects the time series to be longer than the number of vars
    Specify the length
    """

    #Get the shape, work out what shape the new array should be
    array_use = np.array(array.loc[:, cols])
    
    #Make this an np array with no nans
    for i in range(array_use.shape[1]):
        nas = np.isnan(array_use[:, i])
        if sum(nas == False) < 1:
            print(i)
            continue
        min = np.min(array_use[nas == False, i])
        max = np.max(array_use[nas == False, i])
        array_use[nas, i] = min

        #Now normalise
        array_use[:, i] -= min
        array_use[:, i] /= (max - min) 

        #Could also consider then normalising by centile/ alternatively normalising by centile
        
    shape = array_use.shape
    z_dim = math.floor(np.max(shape)/length)
    
    #Get the slices to use
    to_use = list()
    unique_patients = array['project_id'].unique()
    slicesPerPatient = np.zeros(len(unique_patients))

    for i in range(z_dim):
        start_position = i*length
        end_position = (i + 1)*length
        
        #Skip if more than one patient
        patients = array.loc[range(start_position, end_position), 'project_id'].unique()
        if patients.shape[0] == 1:
            to_use.append(i)
        
        #Record number of slices per patient (in order)
        patient_loc = unique_patients == patients
        slicesPerPatient[patient_loc] += 1

    x_dim = length
    y_dim = np.min(shape)
    array_3d = np.empty((len(to_use), y_dim, x_dim))

    #Outcomes
    outcomes = np.zeros((len(to_use), 3))
    pt_slices = list()

    for position, i in enumerate(to_use):
        start_position = i*length
        end_position = (i + 1)*length
        temp_array = array_use[start_position:end_position, :]
        array_3d[position, :, :] = temp_array.transpose()

        #Build the outcomes
        if array.loc[end_position, 'died'] == 'Y':
            outcomes[position, 0] = 1
        if array.loc[end_position, 'time_to_death'] <= 2/365:
            outcomes[position, 1] = 1
        if np.isnan(array.loc[end_position, 'time_to_death']):
            outcomes[position, 2] = 70
        else:
            outcomes[position, 2] = array.loc[end_position, 'time_to_death']
    
    na_loc = np.isnan(array_3d)
    array_3d[na_loc] = 0

    return array_3d, outcomes, slicesPerPatient

def test_trainsplit(array, split):
    """
    Function to split up 3d slices into test, train, validate
    split is an np.ndarray
    """

    #Flexibly choose dimension to split along
    shape = array3d.shape
    z_dim = np.max(shape)

    #Ensure splits add up to 1, get indicies based on splits
    split = split/sum(split)
    indices = np.floor(z_dim*split)

    #Get cumulative indices, ensure start from beginning and end at end
    cumulative_indices = np.cumsum(indices).astype(int)
    cumulative_indices = np.insert(cumulative_indices, 0, 0)
    cumulative_indices[-1] = z_dim
    split_array = list()

    for i in range(len(split)):
        start = cumulative_indices[i]
        finish = cumulative_indices[i + 1]
        temp_array = array[start:finish, ]
        split_array.append(temp_array)
    
    return split_array


length = 60
array3d, outcomes, slicesPerPatient = make_3d_array(flowsheet, length, cols)
split_array = test_trainsplit(array, np.array([70, 15, 15]))
train_array = split_array[0]
testing_array = split_array[1]
input_shape = train_array.shape
input_timeseries = keras.Input(shape = input_shape[1:])

#Now make 1d conv net
# This is the encoder - set the shape to be the shape of the timeseries data
x = layers.Conv1D(30, 16, activation='relu', padding = 'same')(input_timeseries)
x = layers.Conv1D(15, 16, activation='relu', padding = 'same')(x)
encoded = layers.Conv1D(18, 16, activation='relu', padding = 'same')(x)

#Now make the docoder
x = layers.Conv1D(15, 16, activation = 'relu', padding = 'same')(encoded)
x = layers.Conv1D(30, 16, activation= 'relu', padding = 'same')(x)
decoded = layers.Conv1D(60, 16, activation = 'sigmoid', padding = 'same')(x)

autoencoder = keras.Model(input_timeseries, decoded)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

autoencoder.fit(train_array, train_array,
                epochs=1000,
                batch_size=81,
                shuffle=True,
                validation_data=(testing_array, testing_array))
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


