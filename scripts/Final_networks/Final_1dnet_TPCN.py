### Script for testing of 1d convnet

#Setup
import tensorflow as tf
from tensorflow import keras 
from tensorflow.keras import layers
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error, mean_absolute_error, auc, confusion_matrix, roc_curve, precision_score, recall_score, f1_score
import json
import matplotlib.pyplot as plt

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
    shape = array.shape
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
split_array3d2 = test_trainsplit(array3d, np.array([85, 15]))
split_array2d2 = test_trainsplit(array2d, np.array([85, 15]))
split_outcomes2 = test_trainsplit(outcomes, np.array([85, 15]))
train_array3d = split_array3d[0]
train_array2d = split_array2d[0]
train_outcomes = split_outcomes[0]
test_array3d = split_array3d[1]
test_array2d = split_array2d[1]
test_outcomes = split_outcomes[1]
validate_array3d = split_array3d[2]
validate_array2d = split_array2d[2]
validate_outcomes = split_outcomes[2]

all_train_array3d = split_array3d2[0]
all_train_array2d = split_array2d2[0]
all_train_outcomes = split_outcomes2[0]

all_test_array3d = split_array3d2[1]
all_test_array2d = split_array2d2[1]
all_test_outcomes = split_outcomes2[1]


def make_1DNET(model_type):
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

    ####Now make 1d conv net
    # This is the encoder (drawn from autoencoder thing) - set the shape to be the shape of the timeseries data

    x = layers.Conv1D(20, 15, activation='relu', padding = 'causal',  kernel_initializer = init,
                    kernel_regularizer= kernal_regulariser,
                    bias_regularizer= bias_regulariser)(input_timeseries)
    y = layers.Conv1D(20, 1, activation='relu', padding = 'causal',  kernel_initializer = init,
                    kernel_regularizer= kernal_regulariser,
                    bias_regularizer= bias_regulariser)(input_timeseries)
    x = tf.keras.layers.Concatenate(axis=2)([x, y])

    z = layers.Conv1D(40, 40, activation='relu', padding = 'causal', kernel_initializer = init,
                    kernel_regularizer= kernal_regulariser,
                    bias_regularizer= bias_regulariser, 
                    dilation_rate = 2)(x)
    y = layers.Conv1D(40, 1, activation='relu', padding = 'causal',  kernel_initializer = init,
                    kernel_regularizer= kernal_regulariser,
                    bias_regularizer= bias_regulariser, 
                    dilation_rate = 2)(x)
    x = tf.keras.layers.Concatenate(axis=2)([z, y])
        
    x = layers.MaxPooling1D(2, padding = 'same')(x)

    z = layers.Conv1D(140, 15, activation='relu', padding = 'causal',  kernel_initializer = init,
                    kernel_regularizer= kernal_regulariser,
                    bias_regularizer= bias_regulariser, 
                    dilation_rate = 3)(x)
    y = layers.Conv1D(140, 1, activation='relu', padding = 'causal',  kernel_initializer = init,
                    kernel_regularizer= kernal_regulariser,
                    bias_regularizer= bias_regulariser, 
                    dilation_rate = 3)(x)
    x = tf.keras.layers.Concatenate(axis=2)([z, y])

    z = layers.Conv1D(120, 5, activation='relu', padding = 'causal', kernel_initializer = init,
                    kernel_regularizer= kernal_regulariser,
                    bias_regularizer= bias_regulariser, 
                    dilation_rate = 4)(x)
    y = layers.Conv1D(130, 1, activation='relu', padding = 'causal',  kernel_initializer = init,
                    kernel_regularizer= kernal_regulariser,
                    bias_regularizer= bias_regulariser, 
                    dilation_rate = 4)(x)
    x = tf.keras.layers.Concatenate(axis=2)([z, y])
        
    x = layers.MaxPooling1D(4, padding = 'same')(x)

    ##Now make the other head with input
    y = layers.Dense(20, activation = 'relu', kernel_initializer = init,
                    kernel_regularizer= kernal_regulariser,
                    bias_regularizer= bias_regulariser)(input_flat)

    #Now make the other head
    flattened = layers.Flatten()(x)
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
    death_model = keras.Model([input_timeseries, input_flat], death_head)
    discharge_model = keras.Model([input_timeseries, input_flat], [time_head])
    PEWS_model = keras.Model([input_timeseries, input_flat], [PEWS_head])
    
    #Allow this to return one of 3 different model structures
    if model_type == 'full':
        return full_model
    elif model_type == 'death':
        return death_model
    elif model_type == 'discharge':
        return discharge_model
    elif model_type == 'PEWS':
        return PEWS_model

#Set up some storage for the different metrics
AUC_death_full = list()
AUC_LOS_full = list()
AUC_PEWS_full = list()
acc_death_full = list()
acc_LOS_full = list()
acc_PEWS_full= list()
MSE_death_full = list()
MSE_LOS_full = list()
MSE_PEWS_full = list()
MAE_death_full = list()
MAE_LOS_full = list()
MAE_PEWS_full = list()
recall_death_full = list()
recall_LOS_full = list()
recall_PEWS_full = list()
precision_death_full = list()
precision_LOS_full = list()
precision_PEWS_full = list()
F1_death_full = list()
F1_LOS_full = list()
F1_PEWS_full = list()


#Run this 10 times
for i in range(10):
    full_model = make_1DNET('full')

    full_model.compile(optimizer = 'adam', loss='categorical_crossentropy',  metrics=['accuracy', 
                            'mse', tf.keras.metrics.MeanAbsoluteError(), 
                            tf.keras.metrics.AUC()])               

    #Dont forget to add batch size back in 160
    #Now fit the model
    full_model_history = full_model.fit([all_train_array3d, all_train_array2d], [all_train_outcomes[:, 2:5], all_train_outcomes[:, 5:8], all_train_outcomes[:, 8:11]],
                                        epochs = 20,
                                        batch_size = 160,
                                        shuffle = True, 
                                        validation_data = ([all_test_array3d, all_test_array2d], [all_test_outcomes[:, 2:5], all_test_outcomes[:, 5:8], all_test_outcomes[:, 8:11]]),
                                        callbacks = [tf.keras.callbacks.EarlyStopping(patience=1)])
    y_pred1, y_pred2, y_pred3 = full_model.predict([all_test_array3d, all_test_array2d])
    recall_death_full.append(recall_score(np.argmax(y_pred1, axis = 1), np.argmax(all_test_outcomes[:, 2:5], axis = 1), average = 'macro'))
    recall_LOS_full.append(recall_score(np.argmax(y_pred2, axis = 1), np.argmax(all_test_outcomes[:, 5:8], axis = 1), average = 'macro'))
    recall_PEWS_full.append(recall_score(np.argmax(y_pred3, axis = 1), np.argmax(all_test_outcomes[:, 8:11], axis = 1), average = 'macro'))
    precision_death_full.append(precision_score(np.argmax(y_pred1, axis = 1), np.argmax(all_test_outcomes[:, 2:5], axis = 1), average = 'macro'))
    precision_LOS_full.append(precision_score(np.argmax(y_pred2, axis = 1), np.argmax(all_test_outcomes[:, 5:8], axis = 1), average = 'macro'))
    precision_PEWS_full.append(precision_score(np.argmax(y_pred3, axis = 1), np.argmax(all_test_outcomes[:, 8:11], axis = 1), average = 'macro'))
    F1_death_full.append(f1_score(np.argmax(y_pred1, axis = 1), np.argmax(all_test_outcomes[:, 2:5], axis = 1), average = 'macro'))
    F1_LOS_full.append(f1_score(np.argmax(y_pred2, axis = 1), np.argmax(all_test_outcomes[:, 5:8], axis = 1), average = 'macro'))
    F1_PEWS_full.append(f1_score(np.argmax(y_pred3, axis = 1), np.argmax(all_test_outcomes[:, 8:11], axis = 1), average = 'macro'))

    keys = [i for i in full_model_history.history.keys()]
    AUC_death_full.append(full_model_history.history[keys[23]][-1])
    AUC_LOS_full.append(full_model_history.history[keys[27]][-1])
    AUC_PEWS_full.append(full_model_history.history[keys[31]][-1])
    acc_death_full.append(full_model_history.history[keys[20]][-1]) 
    acc_LOS_full.append(full_model_history.history[keys[24]][-1]) 
    acc_PEWS_full.append(full_model_history.history[keys[28]][-1]) 
    MSE_death_full.append(full_model_history.history[keys[21]][-1]) 
    MSE_LOS_full.append(full_model_history.history[keys[25]][-1]) 
    MSE_PEWS_full.append(full_model_history.history[keys[29]][-1]) 
    MAE_death_full.append(full_model_history.history[keys[22]][-1]) 
    MAE_LOS_full.append(full_model_history.history[keys[26]][-1]) 
    MAE_PEWS_full.append(full_model_history.history[keys[30]][-1]) 
    
conf_mat1 = confusion_matrix(np.argmax(y_pred1, axis = 1), np.argmax(all_test_outcomes[:, 2:5], axis = 1))
conf_mat2 = confusion_matrix(np.argmax(y_pred2, axis = 1), np.argmax(all_test_outcomes[:, 5:8], axis = 1))
conf_mat3_2 = confusion_matrix(np.argmax(y_pred3, axis = 1), np.argmax(all_test_outcomes[:, 8:11], axis = 1))
#Aim to tune outputs of 1 and 3
tf.keras.utils.plot_model(full_model, to_file='/mhome/damtp/q/dfs28/Project/PICU_project/models/1d_CNN_TPCN.png', show_shapes=True, expand_nested = True)


#Storage for individual models
AUC_death_individual = list()
AUC_LOS_individual = list()
AUC_PEWS_individual = list()
acc_death_individual = list()
acc_LOS_individual = list()
acc_PEWS_individual= list()
MSE_death_individual = list()
MSE_LOS_individual = list()
MSE_PEWS_individual = list()
MAE_death_individual = list()
MAE_LOS_individual = list()
MAE_PEWS_individual = list()
recall_death_individual = list()
recall_LOS_individual = list()
recall_PEWS_individual = list()
precision_death_individual = list()
precision_LOS_individual = list()
precision_PEWS_individual = list()
F1_death_individual = list()
F1_LOS_individual = list()
F1_PEWS_individual = list()

#Mortality prediction
for i in range(10):
    full_model = make_1DNET('death')

    full_model.compile(optimizer = 'adam', loss='categorical_crossentropy', metrics = ['mse', tf.keras.metrics.MeanAbsoluteError(), 
                        tf.keras.metrics.AUC()])               

    #Dont forget to add batch size back in 160
    #Now fit the model
    full_model_history = full_model.fit([all_train_array3d, all_train_array2d], [all_train_outcomes[:, 2:5]],
                                        epochs = 20,
                                        batch_size = 160,
                                        shuffle = True, 
                                        validation_data = ([all_test_array3d, all_test_array2d], [all_test_outcomes[:, 2:5]]),
                                        callbacks = [tf.keras.callbacks.EarlyStopping(patience=1)])
    y_pred1 = full_model.predict([all_test_array3d, all_test_array2d])  
    acc_death_individual.append(accuracy_score(np.argmax(y_pred1, axis = 1), np.argmax(all_test_outcomes[:, 2:5], axis = 1)))
    recall_death_individual.append(recall_score(np.argmax(y_pred1, axis = 1), np.argmax(all_test_outcomes[:, 2:5], axis = 1), average = 'macro'))
    precision_death_individual.append(precision_score(np.argmax(y_pred1, axis = 1), np.argmax(all_test_outcomes[:, 2:5], axis = 1), average = 'macro'))
    F1_death_individual.append(f1_score(np.argmax(y_pred1, axis = 1), np.argmax(all_test_outcomes[:, 2:5], axis = 1), average = 'macro'))

    keys = [i for i in full_model_history.history.keys()]
    AUC_death_individual.append(full_model_history.history[keys[7]][-1])
    MSE_death_individual.append(full_model_history.history[keys[5]][-1]) 
    MAE_death_individual.append(full_model_history.history[keys[6]][-1]) 




#LOS prediction
for i in range(10):
    full_model = make_1DNET('discharge')

    full_model.compile(optimizer = 'adam', loss='categorical_crossentropy', metrics = ['mse', tf.keras.metrics.MeanAbsoluteError(), 
                        tf.keras.metrics.AUC()])               

    #Dont forget to add batch size back in 160
    #Now fit the model
    full_model_history = full_model.fit([all_train_array3d, all_train_array2d], [all_train_outcomes[:, 5:8]],
                                        epochs = 20,
                                        batch_size = 160,
                                        shuffle = True, 
                                        validation_data = ([all_test_array3d, all_test_array2d], [all_test_outcomes[:, 5:8]]),
                                        callbacks = [tf.keras.callbacks.EarlyStopping(patience=1)])
    y_pred1 = full_model.predict([all_test_array3d, all_test_array2d])  
    acc_LOS_individual.append(accuracy_score(np.argmax(y_pred1, axis = 1), np.argmax(all_test_outcomes[:, 5:8], axis = 1)))
    recall_LOS_individual.append(recall_score(np.argmax(y_pred1, axis = 1), np.argmax(all_test_outcomes[:, 5:8], axis = 1), average = 'macro'))
    precision_LOS_individual.append(precision_score(np.argmax(y_pred1, axis = 1), np.argmax(all_test_outcomes[:, 5:8], axis = 1), average = 'macro'))
    F1_LOS_individual.append(f1_score(np.argmax(y_pred1, axis = 1), np.argmax(all_test_outcomes[:, 5:8], axis = 1), average = 'macro'))

    keys = [i for i in full_model_history.history.keys()]
    AUC_LOS_individual.append(full_model_history.history[keys[7]][-1])
    MSE_LOS_individual.append(full_model_history.history[keys[5]][-1]) 
    MAE_LOS_individual.append(full_model_history.history[keys[6]][-1]) 

#Deterioration prediction
for i in range(10):
    full_model = make_1DNET('PEWS')

    full_model.compile(optimizer = 'adam', loss='categorical_crossentropy', metrics = ['mse', tf.keras.metrics.MeanAbsoluteError(), 
                        tf.keras.metrics.AUC()])               

    #Dont forget to add batch size back in 160
    #Now fit the model
    full_model_history = full_model.fit([all_train_array3d, all_train_array2d], [all_train_outcomes[:, 8:11]],
                                        epochs = 20,
                                        batch_size = 160,
                                        shuffle = True, 
                                        validation_data = ([all_test_array3d, all_test_array2d], [all_test_outcomes[:, 8:11]]),
                                        callbacks = [tf.keras.callbacks.EarlyStopping(patience=1)])
    y_pred1 = full_model.predict([all_test_array3d, all_test_array2d])  
    acc_PEWS_individual.append(accuracy_score(np.argmax(y_pred1, axis = 1), np.argmax(all_test_outcomes[:, 8:11], axis = 1)))
    recall_PEWS_individual.append(recall_score(np.argmax(y_pred1, axis = 1), np.argmax(all_test_outcomes[:, 8:11], axis = 1), average = 'macro'))
    precision_PEWS_individual.append(precision_score(np.argmax(y_pred1, axis = 1), np.argmax(all_test_outcomes[:,8:11], axis = 1), average = 'macro'))
    F1_PEWS_individual.append(f1_score(np.argmax(y_pred1, axis = 1), np.argmax(all_test_outcomes[:, 8:11], axis = 1), average = 'macro'))

    keys = [i for i in full_model_history.history.keys()]
    AUC_PEWS_individual.append(full_model_history.history[keys[7]][-1])
    MSE_PEWS_individual.append(full_model_history.history[keys[5]][-1]) 
    MAE_PEWS_individual.append(full_model_history.history[keys[6]][-1]) 

conf_mat3 = confusion_matrix(np.argmax(y_pred1, axis = 1), np.argmax(all_test_outcomes[:, 8:11], axis = 1))

results = {'acc_death_individual_mean' : np.mean(acc_death_individual), 
            'acc_death_individual_std' : np.std(acc_death_individual),
            'acc_death_full_mean' : np.mean(acc_death_full), 
            'acc_death_full_std' : np.std(acc_death_full), 
            'acc_LOS_individual_mean' : np.mean(acc_LOS_individual), 
            'acc_LOS_individual_std' : np.std(acc_LOS_individual),
            'acc_LOS_full_mean' : np.mean(acc_LOS_full), 
            'acc_LOS_full_std' : np.std(acc_LOS_full),
            'acc_PEWS_individual_mean' : np.mean(acc_PEWS_individual), 
            'acc_PEWS_individual_std' : np.std(acc_PEWS_individual),
            'acc_PEWS_full_mean' : np.mean(acc_PEWS_full), 
            'acc_PEWS_full_std' : np.std(acc_PEWS_full),
            'AUC_death_individual_mean' : np.mean(AUC_death_individual), 
            'AUC_death_individual_std' : np.std(AUC_death_individual),
            'AUC_death_full_mean' : np.mean(AUC_death_full), 
            'AUC_death_full_std' : np.std(AUC_death_full), 
            'AUC_LOS_individual_mean' : np.mean(AUC_LOS_individual), 
            'AUC_LOS_individual_std' : np.std(AUC_LOS_individual),
            'AUC_LOS_full_mean' : np.mean(AUC_LOS_full), 
            'AUC_LOS_full_std' : np.std(AUC_LOS_full),
            'AUC_PEWS_individual_mean' : np.mean(AUC_PEWS_individual), 
            'AUC_PEWS_individual_std' : np.std(AUC_PEWS_individual),
            'AUC_PEWS_full_mean' : np.mean(AUC_PEWS_full), 
            'AUC_PEWS_full_std' : np.std(AUC_PEWS_full),
            'MSE_death_individual_mean' : np.mean(MSE_death_individual), 
            'MSE_death_individual_std' : np.std(MSE_death_individual),
            'MSE_death_full_mean' : np.mean(MSE_death_full), 
            'MSE_death_full_std' : np.std(MSE_death_full), 
            'MSE_LOS_individual_mean' : np.mean(MSE_LOS_individual), 
            'MSE_LOS_individual_std' : np.std(MSE_LOS_individual),
            'MSE_LOS_full_mean' : np.mean(MSE_LOS_full), 
            'MSE_LOS_full_std' : np.std(MSE_LOS_full),
            'MSE_PEWS_individual_mean' : np.mean(MSE_PEWS_individual), 
            'MSE_PEWS_individual_std' : np.std(MSE_PEWS_individual),
            'MSE_PEWS_full_mean' : np.mean(MSE_PEWS_full), 
            'MSE_PEWS_full_std' : np.std(MSE_PEWS_full),
            'MAE_death_individual_mean' : np.mean(MAE_death_individual), 
            'MAE_death_individual_std' : np.std(MAE_death_individual),
            'MAE_death_full_mean' : np.mean(MAE_death_full), 
            'MAE_death_full_std' : np.std(MAE_death_full), 
            'MAE_LOS_individual_mean' : np.mean(MAE_LOS_individual), 
            'MAE_LOS_individual_std' : np.std(MAE_LOS_individual),
            'MAE_LOS_full_mean' : np.mean(MAE_LOS_full), 
            'MAE_LOS_full_std' : np.std(MAE_LOS_full),
            'MAE_PEWS_individual_mean' : np.mean(MAE_PEWS_individual), 
            'MAE_PEWS_individual_std' : np.std(MAE_PEWS_individual),
            'MAE_PEWS_full_mean' : np.mean(MAE_PEWS_full), 
            'MAE_PEWS_full_std' : np.std(MAE_PEWS_full), 
            'precision_death_individual_mean' : np.mean(precision_death_individual), 
            'precision_death_individual_std' : np.std(precision_death_individual),
            'precision_death_full_mean' : np.mean(precision_death_full), 
            'precision_death_full_std' : np.std(precision_death_full), 
            'precision_LOS_individual_mean' : np.mean(precision_LOS_individual), 
            'precision_LOS_individual_std' : np.std(precision_LOS_individual),
            'precision_LOS_full_mean' : np.mean(precision_LOS_full), 
            'precision_LOS_full_std' : np.std(precision_LOS_full),
            'precision_PEWS_individual_mean' : np.mean(precision_PEWS_individual), 
            'precision_PEWS_individual_std' : np.std(precision_PEWS_individual),
            'precision_PEWS_full_mean' : np.mean(precision_PEWS_full), 
            'precision_PEWS_full_std' : np.std(precision_PEWS_full), 
            'recall_death_individual_mean' : np.mean(recall_death_individual), 
            'recall_death_individual_std' : np.std(recall_death_individual),
            'recall_death_full_mean' : np.mean(recall_death_full), 
            'recall_death_full_std' : np.std(recall_death_full), 
            'recall_LOS_individual_mean' : np.mean(recall_LOS_individual), 
            'recall_LOS_individual_std' : np.std(recall_LOS_individual),
            'recall_LOS_full_mean' : np.mean(recall_LOS_full), 
            'recall_LOS_full_std' : np.std(recall_LOS_full),
            'recall_PEWS_individual_mean' : np.mean(recall_PEWS_individual), 
            'recall_PEWS_individual_std' : np.std(recall_PEWS_individual),
            'recall_PEWS_full_mean' : np.mean(recall_PEWS_full), 
            'recall_PEWS_full_std' : np.std(recall_PEWS_full), 
            'F1_death_individual_mean' : np.mean(F1_death_individual), 
            'F1_death_individual_std' : np.std(F1_death_individual),
            'F1_death_full_mean' : np.mean(F1_death_full), 
            'F1_death_full_std' : np.std(F1_death_full), 
            'F1_LOS_individual_mean' : np.mean(F1_LOS_individual), 
            'F1_LOS_individual_std' : np.std(F1_LOS_individual),
            'F1_LOS_full_mean' : np.mean(F1_LOS_full), 
            'F1_LOS_full_std' : np.std(F1_LOS_full),
            'F1_PEWS_individual_mean' : np.mean(F1_PEWS_individual), 
            'F1_PEWS_individual_std' : np.std(F1_PEWS_individual),
            'F1_PEWS_full_mean' : np.mean(F1_PEWS_full), 
            'F1_PEWS_full_std' : np.std(F1_PEWS_full)}

a_file = open("/mhome/damtp/q/dfs28/Project/PICU_project/files/1Dnet_TPCN_results", "w")
json.dump(results, a_file)
a_file.close()

print(conf_mat1)
print(conf_mat2)
print(conf_mat3)
print(conf_mat3_2)

"""
#### Now ablate bits
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

####Now make 1d conv net
# This is the encoder (drawn from autoencoder thing) - set the shape to be the shape of the timeseries data
x = layers.Conv1D(160, 32, activation='relu', padding = 'same',  kernel_initializer = init,
                  kernel_regularizer= kernal_regulariser,
                  bias_regularizer= bias_regulariser)(input_timeseries)
x = layers.Conv1D(160, 32, activation='relu', padding = 'same', kernel_initializer = init,
                  kernel_regularizer= kernal_regulariser,
                  bias_regularizer= bias_regulariser)(x)
    
x = layers.MaxPooling1D(2, padding = 'same')(x)

x = layers.Conv1D(140, 25, activation='relu', padding = 'same',  kernel_initializer = init,
                  kernel_regularizer= kernal_regulariser,
                  bias_regularizer= bias_regulariser)(input_timeseries)
x = layers.Conv1D(140, 25, activation='relu', padding = 'same', kernel_initializer = init,
                  kernel_regularizer= kernal_regulariser,
                  bias_regularizer= bias_regulariser)(x)
    
x = layers.MaxPooling1D(2, padding = 'same')(x)


##Now make the other head with input
y = layers.Dense(20, activation = 'relu', kernel_initializer = init,
                 kernel_regularizer= kernal_regulariser,
                 bias_regularizer= bias_regulariser)(input_flat)

#Now make the other head
flattened = layers.Flatten()(x)
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
ablated_full_model1 = keras.Model([input_timeseries, input_flat], [death_head, time_head, PEWS_head])
ablated_full_model1.compile(optimizer = 'adam', loss='categorical_crossentropy',  metrics=['accuracy', 
                        'mse', tf.keras.metrics.MeanAbsoluteError(), 
                        tf.keras.metrics.AUC()])               

#Dont forget to add batch size back in 160
#Now fit the model
ablated_full_model1_history = ablated_full_model1.fit([train_array3d, train_array2d], [train_outcomes[:, 2:5], train_outcomes[:, 5:8], train_outcomes[:, 8:11]],
                                    epochs = 20,
                                    batch_size = 160,
                                    shuffle = True, 
                                    validation_data = ([test_array3d, test_array2d], [test_outcomes[:, 2:5], test_outcomes[:, 5:8], test_outcomes[:, 8:11]]),
                                    callbacks = [tf.keras.callbacks.EarlyStopping(patience=2)])
"""