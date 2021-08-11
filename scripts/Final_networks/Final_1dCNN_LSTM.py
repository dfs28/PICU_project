### Final 1dCNN + LSTM

#Setup
import tensorflow as tf
from tensorflow import keras 
from tensorflow.keras import layers
import numpy as np
import keras_tuner as kt
import json


#Read in the data
data = np.load('/store/DAMTP/dfs28/PICU_data/np_arrays.npz')
array3d = data['d3']
array2d = data['d2']
outcomes = data['outcomes']



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



#### Define the 1d convnet as a function
kernal_regulariser = bias_regulariser = tf.keras.regularizers.l2(1e-5)

#Set the input shape
input_shape3d = train_array3d.shape
input_timeseries = keras.Input(shape = input_shape3d[1:])
input_flat = keras.Input(shape = train_array2d.shape[1:])

#Init
init = tf.keras.initializers.GlorotUniform()

####Now make 1d conv net
# This is the encoder (drawn from autoencoder thing) - set the shape to be the shape of the timeseries data




x = layers.Conv1D(120, 10, activation='relu', padding = 'same',  kernel_initializer = init,
                kernel_regularizer= kernal_regulariser,
                bias_regularizer= bias_regulariser)(input_timeseries)
x = layers.Conv1D(120, 10, activation='relu', padding = 'same', kernel_initializer = init,
                kernel_regularizer= kernal_regulariser,
                bias_regularizer= bias_regulariser)(x)
            
x = layers.MaxPooling1D(3, padding = 'same')(x)

#Add LSTM layer

lstm_out = tf.keras.layers.Bidirectional(layers.LSTM(110, return_sequences = True))(x)

##Now make the other head with input
y = layers.Dense(20, activation = 'relu', kernel_initializer = init,
                kernel_regularizer= kernal_regulariser,
                bias_regularizer= bias_regulariser)(input_flat)

#Now make the other head
flattened = layers.Flatten()(lstm_out)
concatted = layers.Concatenate()([y, flattened])
        
#With dropount
concatted = layers.Dropout(0.5)(concatted)

dense2 = layers.Dense(40, activation = 'relu', use_bias = True, kernel_initializer = init,
                    kernel_regularizer= kernal_regulariser,
                    bias_regularizer= bias_regulariser)(concatted)

    #Make this a multihead output
death_head = layers.Dense(20, activation = 'relu', use_bias = True, kernel_initializer = init,
                        kernel_regularizer= kernal_regulariser,
                        bias_regularizer= bias_regulariser)(dense2)
death_head = layers.Dense(3, activation = 'softmax', use_bias = True, kernel_initializer = init,
                        kernel_regularizer= kernal_regulariser,
                        bias_regularizer= bias_regulariser)(death_head)
time_head = layers.Dense(20, activation = 'relu', use_bias = True, kernel_initializer = init,
                        kernel_regularizer= kernal_regulariser,
                        bias_regularizer= bias_regulariser)(dense2)
time_head = layers.Dense(3, activation = 'softmax', use_bias = True, kernel_initializer = init,
                        kernel_regularizer= kernal_regulariser,
                        bias_regularizer= bias_regulariser)(time_head)
PEWS_head = layers.Dense(20, activation = 'relu', use_bias = True, kernel_initializer = init,
                        kernel_regularizer= kernal_regulariser,
                        bias_regularizer= bias_regulariser)(dense2)
PEWS_head = layers.Dense(3, activation = 'softmax', use_bias = True, kernel_initializer = init,
                        kernel_regularizer= kernal_regulariser,
                        bias_regularizer= bias_regulariser)(PEWS_head)

#This is the full model with death and LOS as the outcome
full_model = keras.Model([input_timeseries, input_flat], [death_head, time_head, PEWS_head])

full_model.compile(optimizer = 'adam', loss='categorical_crossentropy',  metrics=['accuracy', 
                        'mse', tf.keras.metrics.MeanAbsoluteError(), 
                        tf.keras.metrics.AUC()])               

#Should possibly change this back to patience = 2?
full_model_history = full_model.fit([train_array3d, train_array2d], [train_outcomes[:, 2:5], train_outcomes[:, 5:8], train_outcomes[:, 8:11]],
                                    epochs = 20,
                                    batch_size = 220,
                                    shuffle = True, 
                                    validation_data = ([test_array3d, test_array2d], [test_outcomes[:, 2:5], test_outcomes[:, 5:8], test_outcomes[:, 8:11]]),
                                    callbacks = [tf.keras.callbacks.EarlyStopping(patience=2)])

tf.keras.utils.plot_model(full_model, to_file='/mhome/damtp/q/dfs28/Project/PICU_project/models/1dCNN_LSTM.png', show_shapes=True, expand_nested = True)



