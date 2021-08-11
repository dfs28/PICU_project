### Script for hyperparameter training of non-causal keras tcn

#Setup
import tensorflow as tf
from tensorflow import keras 
from tensorflow.keras import layers
import numpy as np
import keras_tuner as kt
import json
from tcn import TCN, tcn_full_summary
from tcn import compiled_tcn

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
array3d = np.transpose(array3d, (0, 2, 1))
split_array3d = test_trainsplit(array3d, np.array([70, 15, 15]))
split_array2d = test_trainsplit(array2d, np.array([70, 15, 15]))
split_outcomes = test_trainsplit(outcomes, np.array([70, 15, 15]))
split_transpose = test_trainsplit(np.transpose(array3d, (0, 2, 1)), np.array([70, 15, 15])) #Produce transpose version so can do pointwise convs
train_array3d = split_array3d[0]
train_array2d = split_array2d[0]
train_outcomes = split_outcomes[0]
train_transpose = split_transpose[0]
test_array3d = split_array3d[1]
test_array2d = split_array2d[1]
test_outcomes = split_outcomes[1]
test_transpose = split_transpose[1]
validate_array3d = split_array3d[2]
validate_array2d = split_array2d[2]
validate_outcomes = split_outcomes[2]
validate_transpose = split_transpose[2]


#### Define the 1d convnet as a function
def TCN_keras(hp):

    """This is the function for the 1d convnet that we will pass to the gridsearch
    It takes lots of different values that can be used for hyperparameter optimisation
    Params needs to be a dict
    layer_size0, layer_size1, layer_size2, layer_size3, layer_size4, 
                               layer_size5, filter_size0, filter_size1, filter_size2, 
                               pool_size, num_layers, dropout, optimizer, batch_size, reg_type_kernel, 
                               reg_type_bias, reg_type_activity, reg_weight1, reg_weight2
    """

    #Other things to think about - momentum, dropout, sparsity, more layers
    #Set regularisers
    l1_value = hp.Choice("reg_weight1", [0., 1e-5, 1e-4, 1e-3])
    l2_value = hp.Choice("reg_weight2", [0., 1e-5, 1e-4, 1e-3])
    tf.keras.regularizers.l2(l2_value)
    tf.keras.regularizers.l1(l1_value)
    tf.keras.regularizers.l1_l2(l1 = l1_value, l2 = l2_value)

    depthwise_regulariser = hp.Choice('depthwise_reg', ['l1', 'l2', 'l1_l2'])
    bias_regulariser = hp.Choice('bias_reg', ['l1', 'l2', 'l1_l2'])
    pointwise_regulariser = hp.Choice('pointwise_reg', ['l1', 'l2', 'l1_l2'])
    kernal_regulariser = hp.Choice('kernel_reg', ['l1', 'l2', 'l1_l2'])
    init = hp.Choice('init', ['glorot_uniform', 'he_normal'])

    #Set the input shape
    input_shape3d = train_array3d.shape
    input_timeseries = keras.Input(shape = input_shape3d[1:])
    input_flat = keras.Input(shape = train_array2d.shape[1:])


    ####Now make 1d conv net
    # Loop over how many times want to have depthwise separable convolutions
    x = TCN(nb_filters = hp.Int("num_filters", min_value=10, max_value=100, step=5, default = 50), 
            kernel_size = hp.Int('kernel_size', min_value=1, max_value=20, step=1, default = 3),
            nb_stacks=hp.Int('num_stacks', min_value=1, max_value=4, step=1, default = 1), 
            dilations=(1, 2, 4, 8, 16, 32), padding='same',
            use_skip_connections= hp.Boolean('use_skip', default = True), 
            dropout_rate=hp.Float('dropout', min_value = 0, max_value = 0.5, step = 0.1, default = 0.2),
            return_sequences=hp.Boolean('return_seqs', default = True),
            activation='relu',
            kernel_initializer=init,
            use_layer_norm=hp.Boolean('layer_norm'))(input_timeseries)

    ##Now make the other head with input
    y = layers.Dense(20, activation = 'relu', kernel_initializer = init,
                    kernel_regularizer= kernal_regulariser,
                    bias_regularizer= bias_regulariser)(input_flat)

    #Now make the other head
    flattened = layers.Flatten()(x)
    concatted = layers.Concatenate()([y, flattened])
        
    #With dropount
    concatted = layers.Dropout(0.5)(concatted)

    dense2 = layers.Dense(hp.Int('dense2', min_value = 30, max_value = 70, step = 5, default = 40), activation = 'relu', use_bias = True, kernel_initializer = init,
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

    return full_model




class MyTuner(kt.tuners.Hyperband):
  def run_trial(self, trial, *args, **kwargs):
    # You can add additional HyperParameters for preprocessing and custom training loops
    # via overriding `run_trial`
    kwargs['batch_size'] = trial.hyperparameters.Int('batch_size', 64, 256, step=32)
    kwargs['epochs'] = trial.hyperparameters.Int('epochs', 10, 50)
    super(MyTuner, self).run_trial(trial, *args, **kwargs)# Uses same arguments as the BayesianOptimization Tuner.


#Instantiate the search object
tuner = MyTuner(
    TCN_keras,
    objective="val_loss",
    max_epochs=30,
    executions_per_trial=1,
    overwrite=True,
    directory="/store/DAMTP/dfs28/PICU_data",
    project_name="TCN_noncausal_keras_log",
)

#Do the search
tuner.search(
    [train_array3d, train_array2d], [train_outcomes[:, 2:5], train_outcomes[:, 5:8], train_outcomes[:, 8:11]],
    validation_data=([test_array3d, test_array2d], [test_outcomes[:, 2:5], test_outcomes[:, 5:8], test_outcomes[:, 8:11]]),
    callbacks=[tf.keras.callbacks.EarlyStopping(patience=2)],
)

#Save the best model
best_model = tuner.get_best_models(1)[0]
best_model.save('/mhome/damtp/q/dfs28/Project/PICU_project/models/Best_tuned_keras_TCN_noncausal')

#Save the best hyperparameters
best_hyperparameters = tuner.get_best_hyperparameters(1)[0]

a_file = open("/mhome/damtp/q/dfs28/Project/PICU_project/models/Best_tuned_keras_TCN_noncausal.json", "w")
json.dump(best_hyperparameters.values, a_file)
a_file.close()