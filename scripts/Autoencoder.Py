### Sample bit of code to practice making autoencoder

# Setup
import tensorflow as tf
import keras 
from keras import layers
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import seaborn as sns
import shap

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
    array_2d = np.empty((len(to_use), y_dim))

    #Outcomes
    outcomes = np.zeros((len(to_use), 10))
    pt_slices = list()

    for position, i in enumerate(to_use):
        start_position = i*length
        end_position = (i + 1)*length
        temp_array = array_use[start_position:end_position, :]
        array_3d[position, :, :] = temp_array.transpose()
        
        ##Build the outcomes
        #Make sure you can see which patients these are coming from
        patient_id = np.where(array.loc[end_position, 'project_id'] == unique_patients)[0]
        project_id = array.loc[end_position, 'project_id']
        outcomes[position, 0] = patient_id[0]

        #Age of the patient
        outcomes[position, 1] = array.loc[end_position, 'Age_yrs']
        
        #Time to death as triplicate value
        if array.loc[end_position, 'time_to_death'] <= 2/365:
            outcomes[position, 2] = 1
        elif array.loc[end_position, 'died'] == 'Y':
            outcomes[position, 3] = 1
        else: 
            outcomes[position, 4] = 1

        #Length of stay as triplicate
        end_of_section = array.loc[end_position, 'taken_datetime']
        all_dates = array.loc[array['project_id'] == project_id, 'taken_datetime']
        discharge_date = all_dates[all_dates.index[-1]]
        time_to_discharge = discharge_date - end_of_section

        #Now assign depending on how soon:
        if time_to_discharge < np.timedelta64(2, 'D'):
            outcomes[position, 5] = 1
        elif time_to_discharge < np.timedelta64(7, 'D'):
            outcomes[position, 6] = 1
        else:
            outcomes[position, 7] = 1

        #Time to death as outcome (if doesn't die then 70yrs to death)
        if np.isnan(array.loc[end_position, 'time_to_death']):
            outcomes[position, 8] = 70
        else:
            outcomes[position, 9] = array.loc[end_position, 'time_to_death']

        ##Now get pointwise variables
        medians = [np.median(temp_array[:, i]) for i in range(np.shape(temp_array)[1])]
        array_2d[position, :] = np.array(medians)
        

    
    na_loc = np.isnan(array_3d)
    array_3d[na_loc] = 0

    return array_3d, array_2d, outcomes, slicesPerPatient

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
array3d, array2d, outcomes, slicesPerPatient = make_3d_array(flowsheet, length, cols)

"""
Plan:
Leave one out cross validation if possible
Code it up with grid search - not sure if it is possible to change architecture or just hyperparameters
If doing LOOCV could then calculate a prediction interval (I suppose could repeat this a few times to get prediction interval)
    If doing prediction interval otherwise can just repeat the final model a lot of times
Want LSTM, 1d convnet, 2d convnet with and without multihead
Should probably use more than one loss function/ report more than one loss function
Make use of callbacks - both the early stopping one and the reduce learning rate one
For logistic regression should include median value and variability for each of the time series things
Unfortunately can't do skfda due to implementation issue - is there another way I could do this - fit splines and use coefficients?
Could do xgboost instead?
Shapley values - looks like these aren't possible with functional API at present - could build a simplified one head sequential model to test importance?
Maybe change outcomes to binary for simplicity of understanding? Then can do the p/1-p logit link thing
"""


#Split up testing and outcomes
split_array3d = test_trainsplit(array3d, np.array([70, 15, 15]))
split_array2d = test_trainsplit(array2d, np.array([70, 15, 15]))
split_outcomes = test_trainsplit(outcomes, np.array([70, 15, 15]))
train_array3d = split_array3d[0]
train_array2d = split_array2d[0]
train_outcomes = split_outcomes[0]
testing_array3d = split_array3d[1]
testing_array2d = split_array2d[1]
testing_outcomes = split_outcomes[1]

#Set the input shape
input_shape3d = train_array3d.shape
input_timeseries = keras.Input(shape = input_shape3d[1:])
input_flat = keras.Input(shape = train_array2d.shape[1:])

####Now make 1d conv net
# This is the encoder - set the shape to be the shape of the timeseries data
x = layers.Conv1D(30, 16, activation='relu', padding = 'same')(input_timeseries)
x = layers.Conv1D(15, 16, activation='relu', padding = 'same')(x)
encoded = layers.Conv1D(8, 16, activation='relu', padding = 'same')(x)

##Now make the other head with input
y = layers.Dense(8, activation = 'relu')(input_flat)

#Now make the other head
flattened = layers.Flatten()(encoded)
concatted = layers.Concatenate()([y, flattened])
dense1 = layers.Dense(32, activation = 'relu', use_bias = True)(concatted)
dense2 = layers.Dense(8, activation = 'relu', use_bias = True)(dense1)

#Make this a multihead output
death_head = layers.Dense(3, activation = 'softmax', use_bias = True)(dense2)
time_head = layers.Dense(3, activation = 'softmax', use_bias = True)(dense2)

#This is the full model with death and LOS as the outcome
full_model = keras.Model([input_timeseries, input_flat], [death_head, time_head])
full_model.compile(optimizer='adam', loss='categorical_crossentropy',  metrics=['accuracy'])

#Now fit the outcome head having trained the autoencoder
full_model_history = full_model.fit([train_array3d, train_array2d], [train_outcomes[:, 5:8], train_outcomes[:, 2:5]], 
                                    epochs = 10,
                                    batch_size = 100,
                                    shuffle = True, 
                                    validation_data = ([testing_array3d, testing_array2d], [testing_outcomes[:, 5:8], testing_outcomes[:, 2:5]]))

#Plot the history
# list all data in history
print(full_model_history.history.keys())

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

### Consider some of the convolutional filters - I think there is something here to talk about but might have to leave this to the end
filters, biases = full_model.layers[1].get_weights()
fig, (ax1, ax2) = plt.subplots(2, 1)
fig.suptitle('Convolutional Filters')
ax1.plot(filters[:, 1, 1])
ax1.set_title('Filter 1')
ax1.set(xlabel = 'unit', ylabel = 'Value')

ax1.plot(filters[:, 2, 1])
ax1.set_title('Filter 2')
ax1.set(xlabel = 'unit', ylabel = 'Value')

plt.savefig('Project/PICU_project/figs/PICU/Network_training/Practice_plots/2filters.png')



encoded_outcomes = full_model.predict([testing_array3d, testing_array2d])




## Now build LSTM
####Now make 1d conv net
input_shape3d = train_array3d.shape
input_timeseries = keras.Input(shape = input_shape3d[1:])
input_flat = keras.Input(shape = train_array2d.shape[1:])

# This is the encoder - set the shape to be the shape of the timeseries data
x = layers.LSTM(30)(input_timeseries)

##Now make the other head with input
y = layers.Dense(8, activation = 'relu')(input_flat)

#Now make the other head
flattened = layers.Flatten()(x)
concatted = layers.Concatenate()([y, flattened])
dense1 = layers.Dense(32, activation = 'relu', use_bias = True)(concatted)
dense2 = layers.Dense(8, activation = 'relu', use_bias = True)(dense1)

#Make this a multihead output
death_head = layers.Dense(3, activation = 'softmax', use_bias = True)(dense2)
time_head = layers.Dense(3, activation = 'softmax', use_bias = True)(dense2)

#This is the full model with death and LOS as the outcome
full_model_LSTM = keras.Model([input_timeseries, input_flat], [death_head, time_head])
full_model_LSTM.compile(optimizer='adam', loss='categorical_crossentropy',  metrics=['accuracy'])

#Now fit the outcome head having trained the autoencoder
full_model_LSTM = full_model_LSTM.fit([train_array3d, train_array2d], [train_outcomes[:, 5:8], train_outcomes[:, 2:5]], 
                                    epochs = 10,
                                    batch_size = 100,
                                    shuffle = True, 
                                    validation_data = ([testing_array3d, testing_array2d], [testing_outcomes[:, 5:8], testing_outcomes[:, 2:5]]))








## Now build a 2d convnet
array_image = np.reshape(array3d, (array3d.shape[0], array3d.shape[1], array3d.shape[2], 1))
input_time_image = keras.Input(shape = array_image.shape[1:])
input_flat = keras.Input(shape = train_array2d.shape[1:])
split_image = test_trainsplit(array_image, np.array([70, 15, 15]))
split_array2d = test_trainsplit(array2d, np.array([70, 15, 15]))
split_outcomes = test_trainsplit(outcomes, np.array([70, 15, 15]))
train_image = split_image[0]
train_array2d = split_array2d[0]
train_outcomes = split_outcomes[0]
testing_image = split_image[1]
testing_array2d = split_array2d[1]
testing_outcomes = split_outcomes[1]

# Convolutional input
x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(input_time_image)
x = layers.MaxPooling2D((2, 2), padding='same')(x)
x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = layers.MaxPooling2D((2, 2), padding='same')(x)
x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
encoded = layers.MaxPooling2D((2, 2), padding='same')(x)

##Now make the other input
y = layers.Dense(8, activation = 'relu')(input_flat)
y = layers.Dense(4, activation = 'relu')(y)

#Now combine and make 2 head
flattened = layers.Flatten()(encoded)
concatted = layers.Concatenate()([y, flattened])
dense1 = layers.Dense(32, activation = 'relu', use_bias = True)(concatted)
dense2 = layers.Dense(8, activation = 'relu', use_bias = True)(dense1)

#Make this a multihead output
death_head = layers.Dense(3, activation = 'softmax', use_bias = True)(dense2)
time_head = layers.Dense(3, activation = 'softmax', use_bias = True)(dense2)

#This is the full model with death and LOS as the outcome
full_model_2d = keras.Model([input_time_image, input_flat], [death_head, time_head])
full_model_2d.compile(optimizer='adam', loss='categorical_crossentropy',  metrics=['accuracy'])

#Now fit the outcome head having trained the autoencoder
full_model_2d_history = full_model_2d.fit([train_image, train_array2d], [train_outcomes[:, 5:8], train_outcomes[:, 2:5]], 
                                    epochs = 10,
                                    batch_size = 100,
                                    shuffle = True, 
                                    validation_data = ([testing_image, testing_array2d], [testing_outcomes[:, 5:8], testing_outcomes[:, 2:5]]))



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



### Calculate shapley values - have to build a smaller sequential model for this
#Although have averaged across all here should probably split by outcome - the contributions of different things to different points by outcome
#Then if I do the 2d one I can get 'images' to look at which might make more sense
#Could actually consider representing all of the lines as a heatmap of some description anyway? Then I effectively have 2x2 or 3x2 heatmaps
# this should also normalise for input so ones that are more important appear more important
# When talk about this will have to talk about limitations of shapley values

model_m = keras.Sequential()
model_m.add(layers.Conv1D(30, 16, activation='relu', padding = 'same', input_shape=train_array3d.shape[1:]))
model_m.add(layers.Conv1D(15, 16, activation='relu', padding = 'same',))
model_m.add(layers.MaxPooling1D(2))
model_m.add(layers.Conv1D(15, 4, activation='relu', padding = 'same',))
model_m.add(layers.Flatten())
model_m.add(layers.Dense(3, activation='softmax'))
print(model_m.summary())

model_m.compile(optimizer='adam', loss='categorical_crossentropy',  metrics=['accuracy'])

#Now fit the simplified 1d convnet
model_m_history = model_m.fit(train_array3d, train_outcomes[:, 5:8], 
                                    epochs = 10,
                                    batch_size = 100,
                                    shuffle = True, 
                                    validation_data = (testing_array3d, testing_outcomes[:, 5:8]))

explainer = shap.DeepExplainer(model_m, train_array3d)

#Now work through all of the samples to take an average shapley value for each point:
shap_values = explainer.shap_values(testing_array3d[:, :, :])
#Shap values gives an value for contribution to each of the outputs, so the shap_values output gives (n_outputs, z_dim, y_dim, x_dim) shape 
ave_shap_values = np.average(shap_values, axis = 1)

#Now plot some shapley values
fig, axs = plt.subplots(8, 2, figsize = (20, 13))
grid = plt.GridSpec(2, 2, wspace=0.2, hspace=0.5)

for i in range(16):
    axs[i % 8, i // 8].plot(ave_shap_values[0, i, :])
    axs[i % 8, i // 8].plot(ave_shap_values[1, i, :])
    axs[i % 8, i // 8].plot(ave_shap_values[2, i, :])
    axs[i % 8, i // 8].title.set_text(cols[i])
    axs[i % 8, i // 8].legend(['Discharge <2d', 'Discharge 2-7d', 'Discharge >7d'])

for ax in axs.flat:
    ax.set(xlabel='Filter position', ylabel='importance')

plt.savefig('Project/PICU_project/figs/PICU/Network_training/Practice_plots/Plotting_shap_values.png')


#Now get shapley values for 2d cnn
model_2dCNN = tf.keras.Sequential()
model_2dCNN.add(tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same',  input_shape=train_image.shape[1:]))
model_2dCNN.add(tf.keras.layers.MaxPooling2D((2, 2), padding='same'))
model_2dCNN.add(tf.keras.layers.Conv2D(8, (3, 3), activation='relu', padding='same'))
model_2dCNN.add(tf.keras.layers.MaxPooling2D((2, 2), padding='same'))
model_2dCNN.add(tf.keras.layers.Conv2D(8, (3, 3), activation='relu', padding='same'))
model_2dCNN.add(tf.keras.layers.MaxPooling2D((2, 2), padding='same'))
model_2dCNN.add(tf.keras.layers.Flatten())
model_2dCNN.add(tf.keras.layers.Dense(32, activation = 'relu', use_bias = True))
model_2dCNN.add(tf.keras.layers.Dense(8, activation = 'relu', use_bias = True))
model_2dCNN.add(tf.keras.layers.Dense(3, activation = 'softmax', use_bias = True))
model_2dCNN.compile(optimizer='adam', loss='categorical_crossentropy',  metrics=['accuracy'])

#Now fit the simplified 1d convnet
model_2dCNN_history = model_2dCNN.fit(train_image, train_outcomes[:, 5:8], 
                                    epochs =20,
                                    batch_size = 100,
                                    shuffle = True, 
                                    validation_data = (testing_image, testing_outcomes[:, 5:8]))



#Now get some shapley values
explainer = shap.DeepExplainer(model_2dCNN, train_image)

#Now work through all of the samples to take an average shapley value for each point:
shap_values = explainer.shap_values(testing_image[0:100, :, :])
#Shap values gives an value for contribution to each of the outputs, so the shap_values output gives (n_outputs, z_dim, y_dim, x_dim) shape 
ave_shap_values = np.average(shap_values, axis = 1)
shap_shape = np.shape(ave_shap_values)
ave_shap_values = ave_shap_values.reshape(shap_shape[:-1])

#Now plot them as 3 images - should change this to 9 but would need to change inputs
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize = (20, 13))
ax1.imshow(ave_shap_values[0, :, :])
for i, j in enumerate(cols):
    ax1.text(-5, i, , fontdict=None, **kwargs)
ax2.imshow(ave_shap_values[1, :, :])
ax3.imshow(ave_shap_values[2, :, :])

plt.savefig('Project/PICU_project/figs/PICU/Network_training/Practice_plots/Plotting_2d_shap_values.png')

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