### Shapley value production
# Setup
import shap
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import random

#Should then consider removing values with low shapley values and repeating network to see how it performs
#Read in the data
data = np.load('/store/DAMTP/dfs28/PICU_data/np_arrays.npz')
array3d = data['d3']
array2d = data['d2']
outcomes = data['outcomes']
series_cols = data['series_cols']


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


#Split up testing and outcomes
array3d = np.transpose(array3d, (0, 2, 1))
split_array3d = test_trainsplit(array3d, np.array([70, 15, 15]))
split_array2d = test_trainsplit(array2d, np.array([70, 15, 15]))
split_outcomes = test_trainsplit(outcomes, np.array([70, 15, 15]))
train_array3d = split_array3d[0]
train_array2d = split_array2d[0]
train_outcomes = split_outcomes[0]
test_array3d = split_array3d[1]
test_array2d = split_array2d[1]
test_outcomes = split_outcomes[1]
validate_array3d = split_array3d[2]
validate_array2d = split_array2d[2]
validate_outcomes = split_outcomes[2]


#Make the final net
kernal_regulariser = bias_regulariser = tf.keras.regularizers.l2(1e-5)
#tf.keras.regularizers.l1(l1_value)
#tf.keras.regularizers.l1_l2(l1 = l1_value, l2 = l2_value)

#Set the input shape
input_shape3d = train_array3d.shape
input_timeseries = keras.Input(shape = input_shape3d[1:])
input_flat = keras.Input(shape = train_array2d.shape[1:])

#Init
init = tf.keras.initializers.GlorotUniform()

model_1dCNN = tf.keras.Sequential()

#Convolutional block1
model_1dCNN.add(tf.keras.layers.Conv1D(140, 10, activation='relu', padding = 'same',  kernel_initializer = init,
                  kernel_regularizer= kernal_regulariser,
                  bias_regularizer= bias_regulariser, input_shape = train_array3d.shape[1:]))
model_1dCNN.add(tf.keras.layers.Conv1D(40, 35, activation='relu', padding = 'same',  kernel_initializer = init,
                  kernel_regularizer= kernal_regulariser,
                  bias_regularizer= bias_regulariser))
model_1dCNN.add(tf.keras.layers.MaxPooling1D(4, padding = 'same'))

#Now flatten
model_1dCNN.add(tf.keras.layers.Flatten())
#With dropout
model_1dCNN.add(tf.keras.layers.Dropout(0.5))

#Now a couple of dense layers for output
model_1dCNN.add(tf.keras.layers.Dense(40, activation = 'relu', use_bias = True, kernel_initializer = init,
                      kernel_regularizer= kernal_regulariser,
                      bias_regularizer= bias_regulariser)

model_1dCNN.add(tf.keras.layers.Dense(20, activation = 'relu', use_bias = True, kernel_initializer = init,
                      kernel_regularizer= kernal_regulariser,
                      bias_regularizer= bias_regulariser))
model_1dCNN.add(tf.keras.layers.Dense(3, activation = 'softmax', use_bias = True, kernel_initializer = init,
                      kernel_regularizer= kernal_regulariser,
                      bias_regularizer= bias_regulariser))

#This is the full model with death and LOS as the outcome
model_1dCNN.compile(optimizer = 'adam', loss='categorical_crossentropy',  metrics=['accuracy', 
                        'mse', tf.keras.metrics.MeanAbsoluteError(), 
                        tf.keras.metrics.AUC()])               

#Dont forget to add batch size back in 160
#Now fit the model
model_1dCNN_history = model_1dCNN.fit(train_array3d, train_outcomes[:, 2:5],
                                    epochs = 20,
                                    batch_size = 160,
                                    shuffle = True, 
                                    validation_data = (test_array3d, test_outcomes[:, 2:5]),
                                    callbacks = [tf.keras.callbacks.EarlyStopping(patience=2)])


#Now get some shapley values
explainer_death = shap.DeepExplainer(model_1dCNN, test_array3d)

#Need to recompile the model every time
model_1dCNN.compile(optimizer = 'adam', loss='categorical_crossentropy',  metrics=['accuracy', 
                        'mse', tf.keras.metrics.MeanAbsoluteError(), 
                        tf.keras.metrics.AUC()])      

model_1dCNN_history = model_1dCNN.fit(train_array3d, train_outcomes[:, 5:8],
                                    epochs = 20,
                                    batch_size = 160,
                                    shuffle = True, 
                                    validation_data = (test_array3d, test_outcomes[:, 5:8]),
                                    callbacks = [tf.keras.callbacks.EarlyStopping(patience=2)])
explainer_LOS = shap.DeepExplainer(model_1dCNN, test_array3d)

#Need to recompile the model every time
model_1dCNN.compile(optimizer = 'adam', loss='categorical_crossentropy',  metrics=['accuracy', 
                        'mse', tf.keras.metrics.MeanAbsoluteError(), 
                        tf.keras.metrics.AUC()])      


model_1dCNN_history = model_1dCNN.fit(train_array3d, train_outcomes[:, 8:11],
                                    epochs = 20,
                                    batch_size = 160,
                                    shuffle = True, 
                                    validation_data = (test_array3d, test_outcomes[:, 5:8]),
                                    callbacks = [tf.keras.callbacks.EarlyStopping(patience=2)])
explainer_PEWS = shap.DeepExplainer(model_1dCNN, test_array3d)

#Now work through all of the samples to take an average shapley value for each point:
values = random.sample(range(test_array3d.shape[0]), 100)
shap_values = explainer_death.shap_values(test_array3d[values, :, :], check_additivity=False)
np.savez('/store/DAMTP/dfs28/PICU_data/SHAP_values.npz', shap = shap_values)
shap_values_LOS = explainer_LOS.shap_values(test_array3d[values, :, :], check_additivity=False)
shap_values_PEWS = explainer_PEWS.shap_values(test_array3d[values, :, :], check_additivity=False)
#Shap values gives an value for contribution to each of the outputs, so the shap_values output gives (n_outputs, z_dim, y_dim, x_dim) shape 
ave_shap_values = np.average(shap_values, axis = 1)
shap_shape = np.shape(ave_shap_values)
ave_shap_values = np.transpose(ave_shap_values, (0, 2, 1))

#Now plot them as 3 images - should change this to 9 but would need to change inputs
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize = (50, 33))
ax1.imshow(ave_shap_values[0, :, :])
ax1.axes.get_yaxis().set_ticks([])
ax1.set_title('Death <48h')
ax2.imshow(ave_shap_values[1, :, :])
ax2.axes.get_yaxis().set_ticks([])
ax2.set_title('Death >48h')
ax3.imshow(ave_shap_values[2, :, :])
ax3.axes.get_yaxis().set_ticks([])
ax3.set_title('Survived')
fig.suptitle('Average SHAP values for NN predicting mortality')

#I think the naming is upside down but not sure how to fix thisx
for i, j in enumerate(series_cols):
    ax1.text(-1, i + 0.5, j, horizontalalignment='right')
    ax2.text(-1, i + 0.5, j, horizontalalignment='right')
    ax3.text(-1, i + 0.5, j, horizontalalignment='right')

plt.savefig('/mhome/damtp/q/dfs28/Project/PICU_project/figs/PICU/SHAP_plots/1D_shap_death2.png')

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize = (50, 33))
my_cmap = sns.color_palette("icefire", as_cmap=True)
sns.heatmap(ave_shap_values[0, :, :],  yticklabels = series_cols, ax = ax1, cmap = my_cmap)
ax1.set_title('Death <48h')
sns.heatmap(ave_shap_values[1, :, :], yticklabels = series_cols, ax = ax2, cmap = my_cmap)
ax2.set_title('Death >48h')
sns.heatmap(ave_shap_values[2, :, :], yticklabels = series_cols, ax = ax3, cmap = my_cmap)
ax3.set_title('Survived')
#plt.suptitle('Average SHAP values for 1D Convnet predicting mortality')
fig.tight_layout()
plt.savefig('/mhome/damtp/q/dfs28/Project/PICU_project/figs/PICU/SHAP_plots/1D_shap_death_sns2.png')

#Now do the same for the flat values
kernal_regulariser = bias_regulariser = tf.keras.regularizers.l2(1e-5)
#tf.keras.regularizers.l1(l1_value)
#tf.keras.regularizers.l1_l2(l1 = l1_value, l2 = l2_value)

#Set the input shape
input_shape3d = train_array3d.shape
input_timeseries = keras.Input(shape = input_shape3d[1:])
input_flat = keras.Input(shape = train_array2d.shape[1:])

#Init
init = tf.keras.initializers.GlorotUniform()

#Now make the flat model
model_flat = tf.keras.Sequential()

model_flat.add(tf.keras.layers.Dense(20, activation = 'relu', use_bias = True, kernel_initializer = init,
                      kernel_regularizer= kernal_regulariser,
                      bias_regularizer= bias_regulariser, input_shape = train_array2d.shape[1:]))

#With dropout
model_flat.add(tf.keras.layers.Dropout(0.5))
model_flat.add(tf.keras.layers.Dense(3, activation = 'softmax', use_bias = True, kernel_initializer = init,
                      kernel_regularizer= kernal_regulariser,
                      bias_regularizer= bias_regulariser))

#This is the full model with death and LOS as the outcome
model_flat.compile(optimizer = 'adam', loss='categorical_crossentropy',  metrics=['accuracy', 
                        'mse', tf.keras.metrics.MeanAbsoluteError(), 
                        tf.keras.metrics.AUC()])               

#Dont forget to add batch size back in 160
#Now fit the model
model_flat_history = model_flat.fit(train_array2d, [train_outcomes[:, 2:5],
                                    epochs = 20,
                                    batch_size = 160,
                                    shuffle = True, 
                                    validation_data = (test_array2d, test_outcomes[:, 2:5]),
                                    callbacks = [tf.keras.callbacks.EarlyStopping(patience=2)])