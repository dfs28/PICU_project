### Sample bit of code to practice making autoencoder

# Setup
import keras
from keras import layers
from keras.datasets import mnist
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import seaborn as sns

#Read in the data
flowsheet = pd.read_csv('~/Project/Project_data/files/flowsheet_output.csv', parse_dates = ['taken_datetime'])


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
    outcomes = np.zeros((len(to_use), 5))
    pt_slices = list()

    for position, i in enumerate(to_use):
        start_position = i*length
        end_position = (i + 1)*length
        temp_array = array_use[start_position:end_position, :]
        array_3d[position, :, :] = temp_array.transpose()

        #Build the outcomes
        #Death
        if array.loc[end_position, 'died'] == 'Y':
            outcomes[position, 0] = 1
        
        #Time to death less than 48h
        if array.loc[end_position, 'time_to_death'] <= 2/365:
            outcomes[position, 1] = 1

        #Time to death as outcome (if doesn't die then 70yrs to death)
        if np.isnan(array.loc[end_position, 'time_to_death']):
            outcomes[position, 2] = 70
        else:
            outcomes[position, 2] = array.loc[end_position, 'time_to_death']
        
        #Make sure you can see which patients these are coming from
        patient_id = np.where(array.loc[end_position, 'project_id'] == unique_patients)[0]
        outcomes[position, 3] = patient_id[0]

        #Age of the patient
        outcomes[position, 4] = array.loc[end_position, 'Age_yrs']
    
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
split_array = test_trainsplit(array3d, np.array([70, 15, 15]))
split_outcomes = test_trainsplit(outcomes, np.array([70, 15, 15]))
train_array = split_array[0]
train_outcomes = split_outcomes[0]
testing_array = split_array[1]
testing_outcomes = split_outcomes[1]
input_shape = train_array.shape
input_timeseries = keras.Input(shape = input_shape[1:])

#Now make 1d conv net
# This is the encoder - set the shape to be the shape of the timeseries data
x = layers.Conv1D(30, 16, activation='relu', padding = 'same')(input_timeseries)
x = layers.Conv1D(15, 16, activation='relu', padding = 'same')(x)
encoded = layers.Conv1D(8, 16, activation='relu', padding = 'same')(x)

#Now make the docoder
x = layers.Conv1D(15, 16, activation = 'relu', padding = 'same')(encoded)
x = layers.Conv1D(30, 16, activation= 'relu', padding = 'same')(x)
decoded = layers.Conv1D(60, 16, activation = 'sigmoid', padding = 'same')(x)

# This model maps an input to its encoded representation
encoder = keras.Model(input_timeseries, encoded)

autoencoder = keras.Model(input_timeseries, decoded)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy',  metrics=['accuracy'])

history = autoencoder.fit(train_array, train_array,
                epochs=100,
                batch_size=81,
                shuffle=True,
                validation_data=(testing_array, testing_array))


#Plot the history
# list all data in history
print(history.history.keys())

fig, (ax1, ax2) = plt.subplots(1, 2)
fig.suptitle('1D Convolutional Autoencoder')

# summarize history for accuracy
ax1.plot(history.history['accuracy'])
ax1.plot(history.history['val_accuracy'])
ax1.set_title('model accuracy')
ax1.set(xlabel = 'epoch', ylabel = 'accuracy')
ax1.legend(['train', 'test'], loc='upper left')

# summarize history for loss
ax2.plot(history.history['loss'])
ax2.plot(history.history['val_loss'])
ax2.set_title('model loss')
ax2.set(xlabel = 'epoch', ylabel = 'loss')
ax2.legend(['train', 'test'], loc='upper left')

for ax in fig.get_axes():
    ax.label_outer()

#Save it
plt.savefig('Project/PICU_project/figs/PICU/Network_training/Practice_plots/3layer1Dautoencoder.png')



## Now train the output from the autoencoder onto the outcomes
x = layers.Conv1D(15, 16, activation = 'relu', padding = 'same')(encoded)
x = layers.Conv1D(30, 16, activation= 'relu', padding = 'same')(x)
decoded = layers.Conv1D(60, 16, activation = 'sigmoid', padding = 'same')(x)

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

#### Now apply the encoding to some digits
# Note that we take them from the *test* set
encoded_imgs = encoder.predict(testing_array)
flattened_encoded = np.reshape(encoded_imgs, (len(testing_array), 8*16))



#### Now do PCA on encodings
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(flattened_encoded)
principalDf = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2'])

#Attach the outcomes
testingDf = pd.DataFrame(testing_outcomes, columns = ['Death', 'DeathIn48', 'TimetoDeath', 'project_id', 'Age'])
finalDf = pd.concat([principalDf, testingDf], axis = 1)

#Plot death
fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('Autoencoder PCA', fontsize = 20)
targets = [1, 0]
colors = ['r', 'g']
for target, color in zip(targets,colors):
    indicesToKeep = finalDf['Death'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
               , finalDf.loc[indicesToKeep, 'principal component 2']
               , c = color
               , s = 50)
ax.legend(['Died', 'Survived'])
ax.grid()
plt.savefig('Project/PICU_project/figs/PICU/Network_training/Practice_plots/AutoencodedPCA.png')

#PCA by age 
fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('Autoencoder PCA coloured by Age', fontsize = 20)
cmap = sns.cubehelix_palette(start=2.8, rot=.1. as_cmap = True)
points = ax.scatter(finalDf['principal component 1'], finalDf['principal component 2'], 
                    c=finalDf['Age'], s=20, cmap=cmap)
fig.colorbar(points, ax=ax)
ax.grid()
plt.savefig('Project/PICU_project/figs/PICU/Network_training/Practice_plots/AutoencodedPCA_age.png')

#PCA by patient
fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('Autoencoder PCA coloured by Patient', fontsize = 20)
cmap = sns.cubehelix_palette(start=2.8, rot=.1. as_cmap = True)
points = ax.scatter(finalDf['principal component 1'], finalDf['principal component 2'], 
                    c=finalDf['project_id'], s=20, cmap=cmap)
fig.colorbar(points, ax=ax)
ax.grid()
plt.savefig('Project/PICU_project/figs/PICU/Network_training/Practice_plots/AutoencodedPCA_patient.png')


#Repeat the PCA and plot but without encodings
flattenedInputs = np.reshape(array3d, (len(array3d), 60*16))
PCAInputs = pca.fit_transform(flattenedInputs)
PCAInputsDf = pd.DataFrame(data = PCAInputs, columns = ['principal component 1', 'principal component 2'])

#Attach the outcomes
outcomesDf = pd.DataFrame(outcomes, columns = ['Death', 'DeathIn48', 'TimetoDeath', 'project_id', 'Age'])
finalDf = pd.concat([PCAInputsDf, outcomesDf], axis = 1)

#Plot it
fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('PCA of Inputs', fontsize = 20)
targets = [1, 0]
colors = ['r', 'g']
for target, color in zip(targets,colors):
    indicesToKeep = finalDf['Death'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
               , finalDf.loc[indicesToKeep, 'principal component 2']
               , c = color
               , s = 50)
ax.legend(['Died', 'Survived'])
ax.grid()
plt.savefig('Project/PICU_project/figs/PICU/Network_training/Practice_plots/PCAInputs.png')


#Plot age
fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('PCA of Inputs by Age', fontsize = 20)
cmap = sns.cubehelix_palette(as_cmap=True)
points = ax.scatter(finalDf['principal component 1'], finalDf['principal component 2'], 
                    c=finalDf['Age'], s=20, cmap=cmap)
fig.colorbar(points, ax=ax)
ax.grid()
plt.savefig('Project/PICU_project/figs/PICU/Network_training/Practice_plots/PCAInputs_age.png')