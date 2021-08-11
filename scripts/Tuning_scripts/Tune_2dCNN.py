### Script for hyperparameter training of 2d convnet

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


### Split up test and train
#Note that this has to be a 4d image with a single colour channel
array_image = np.reshape(array3d, (array3d.shape[0], array3d.shape[1], array3d.shape[2], 1))
input_time_image = keras.Input(shape = array_image.shape[1:])
input_flat = keras.Input(shape = array2d.shape[1:])
split_image = test_trainsplit(array_image, np.array([70, 15, 15]))
split_array2d = test_trainsplit(array2d, np.array([70, 15, 15]))
split_outcomes = test_trainsplit(outcomes, np.array([70, 15, 15]))
train_image = split_image[0]
train_array2d = split_array2d[0]
train_outcomes = split_outcomes[0]
test_image = split_image[1]
test_array2d = split_array2d[1]
test_outcomes = split_outcomes[1]
validate_image = split_image[2]
validate_array2d = split_array2d[2]
validate_outcomes = split_outcomes[2]



### Now make the tunable model

def convnet_2d(hp):
    """ This is the keras hyperparameter search object
    """

    #Set regularisers
    l1_value = hp.Choice("reg_weight1", [0., 1e-5, 1e-4, 1e-3])
    l2_value = hp.Choice("reg_weight2", [0., 1e-5, 1e-4, 1e-3])
    tf.keras.regularizers.l2(l2_value)
    tf.keras.regularizers.l1(l1_value)
    tf.keras.regularizers.l1_l2(l1 = l1_value, l2 = l2_value)

    kernal_regulariser = hp.Choice('kernal_reg', ['l1', 'l2', 'l1_l2'])
    bias_regulariser = hp.Choice('bias_reg', ['l1', 'l2', 'l1_l2'])
    activity_regulariser = hp.Choice('activity_reg', ['l1', 'l2', 'l1_l2'])

    #Set initialiser
    if hp.Choice("init" , ['glorot_uniform', 'uniform', 'normal']) == 'glorot_uniform':
        init = initializer = tf.keras.initializers.GlorotUniform()
    elif  hp.Choice("init", ['glorot_uniform', 'uniform', 'normal']) == 'normal': 
        init = tf.keras.initializers.RandomNormal(mean=0., stddev=1.)
    elif  hp.Choice("init", ['glorot_uniform', 'uniform', 'normal']) == 'uniform':
        init = tf.keras.initializers.RandomUniform(minval=-0.05, maxval=0.05, seed=None)


    # Convolutional input
    filter_size1 = hp.Int('filter_size1', min_value = 2, max_value = 5, step = 1, default = 3)
    x = layers.Conv2D(hp.Int('filter_num1', min_value = 8, max_value = 64, step = 8, default = 32), 
                      (filter_size1, filter_size1), activation='relu', padding='same', kernel_initializer = init,
                      kernel_regularizer= kernal_regulariser,
                      bias_regularizer= bias_regulariser,
                      activity_regularizer= activity_regulariser)(input_time_image)

    pool_size1 = hp.Int('pool_size1', min_value = 1, max_value = 4, step = 1, default = 2)
    x = layers.MaxPooling2D((pool_size1, pool_size1), padding='same')(x)
    
    #Choose how many convolutional layers to have
    filter_num2 = hp.Int('filter_num2', min_value = 8, max_value = 64, step = 8, default = 32)
    filter_size2 = hp.Int('filter_size2', min_value = 2, max_value = 5, step = 1, default = 3)
    pool_size2 = hp.Int('pool_size2', min_value = 1, max_value = 4, step = 1, default = 2)

    for i in range(hp.Int("conv_step", min_value=1, max_value=4, step=1, default = 3)):
        x = layers.Conv2D(filter_num2, (filter_size2, filter_size2), 
                          activation='relu', padding='same', 
                          kernel_initializer = init,
                          kernel_regularizer= kernal_regulariser,
                          bias_regularizer= bias_regulariser,
                          activity_regularizer= activity_regulariser)(x)
        x = layers.MaxPooling2D((pool_size2, pool_size2), padding='same')(x)


    ##Now make the other head with input
    y = layers.Dense(hp.Int("layer_size2", min_value=10, max_value=50, step=5, default = 30), activation = 'relu', kernel_initializer = init,
                      kernel_regularizer= kernal_regulariser,
                      bias_regularizer= bias_regulariser,
                      activity_regularizer= activity_regulariser)(input_flat)

    #Now make the other head
    flattened = layers.Flatten()(x)
    concatted = layers.Concatenate()([y, flattened])
    
    if hp.Choice('dropout', ['True', 'False']):
        concatted = layers.Dropout(0.5)(concatted)

    dense1 = layers.Dense(hp.Int("layer_size4", min_value=10, max_value=50, step=5, default = 30), activation = 'relu', use_bias = True, kernel_initializer = init,
                      kernel_regularizer= kernal_regulariser,
                      bias_regularizer= bias_regulariser,
                      activity_regularizer= activity_regulariser)(concatted)
    dense2 = layers.Dense(hp.Int("layer_size5", min_value=10, max_value=50, step=5, default = 10), activation = 'relu', use_bias = True, kernel_initializer = init,
                      kernel_regularizer= kernal_regulariser,
                      bias_regularizer= bias_regulariser,
                      activity_regularizer= activity_regulariser)(dense1)

    #Make this a multihead output
    death_head = layers.Dense(3, activation = 'softmax', use_bias = True, kernel_initializer = init,
                      kernel_regularizer= kernal_regulariser,
                      bias_regularizer= bias_regulariser,
                      activity_regularizer= activity_regulariser)(dense2)
    time_head = layers.Dense(3, activation = 'softmax', use_bias = True, kernel_initializer = init,
                      kernel_regularizer= kernal_regulariser,
                      bias_regularizer= bias_regulariser,
                      activity_regularizer= activity_regulariser)(dense2)
    PEWS_head = layers.Dense(3, activation = 'softmax', use_bias = True, kernel_initializer = init,
                      kernel_regularizer= kernal_regulariser,
                      bias_regularizer= bias_regulariser,
                      activity_regularizer= activity_regulariser)(dense2)

    #This is the full model with death and LOS as the outcome
    full_model_2d = keras.Model([input_time_image, input_flat], [death_head, time_head, PEWS_head])
    full_model_2d.compile(optimizer='adam', loss='categorical_crossentropy',  metrics=['accuracy'])

    return full_model_2d


class MyTuner(kt.tuners.Hyperband):
  def run_trial(self, trial, *args, **kwargs):
    # You can add additional HyperParameters for preprocessing and custom training loops
    # via overriding `run_trial`
    kwargs['batch_size'] = trial.hyperparameters.Int('batch_size', 32, 256, step=32)
    kwargs['epochs'] = trial.hyperparameters.Int('epochs', 10, 50)
    super(MyTuner, self).run_trial(trial, *args, **kwargs)# Uses same arguments as the BayesianOptimization Tuner.



#Instantiate the search object
tuner = MyTuner(
    convnet_2d,
    objective="val_loss",
    max_epochs=30,
    executions_per_trial=1,
    overwrite=False,
    directory="/store/DAMTP/dfs28/PICU_data",
    project_name="convnet2d_log",
)

#Do the search
tuner.search(
    [train_image, train_array2d], [train_outcomes[:, 2:5], train_outcomes[:, 5:8], train_outcomes[:, 8:11]],
    validation_data=([test_image, test_array2d], [test_outcomes[:, 2:5], test_outcomes[:, 5:8], test_outcomes[:, 8:11]]),
    callbacks=[tf.keras.callbacks.EarlyStopping(patience=2)],
)

#Save the best model
best_model = tuner.get_best_models(1)[0]
best_model.save('/mhome/damtp/q/dfs28/Project/PICU_project/models/Best_tuned_2dCNN')

#Save the best hyperparameters
best_hyperparameters = tuner.get_best_hyperparameters(1)[0]

a_file = open("/mhome/damtp/q/dfs28/Project/PICU_project/models/Best_tuned_2dCBB.json", "w")
json.dump(best_hyperparameters.values, a_file)
a_file.close()
