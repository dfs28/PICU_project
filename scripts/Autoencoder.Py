### Sample bit of code to practice making autoencoder

# Setup
import tensorflow as tf
from tensorflow import keras 
from tensorflow.keras import layers
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import seaborn as sns
import shap
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
import talos
import keras_tuner as kt


#Read in the data
data = np.load('/store/DAMTP/dfs28/PICU_data/practice_arrays.npz')
array3d = data['d3']
array2d = data['d2']
characteristics = data['chars']
splines = data['splines']
outcomes = data['outcomes']
slices_per_pt = data['per_pt']

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

"""
Plan:
Leave one out cross validation if possible - done but k-fold due to time constraints
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
test_array3d = split_array3d[1]
test_array2d = split_array2d[1]
test_outcomes = split_outcomes[1]
validate_array3d = split_array3d[2]
validate_array2d = split_array2d[2]
validate_outcomes = split_outcomes[2]





#### Define the 1d convnet as a function
def temporal_convolutional_net(hp):

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

    #Set the input shape
    input_shape3d = train_array3d.shape
    input_timeseries = keras.Input(shape = input_shape3d[1:])
    input_flat = keras.Input(shape = train_array2d.shape[1:])

    ####Now make 1d conv net
    # This is the encoder (drawn from autoencoder thing) - set the shape to be the shape of the timeseries data
    x = layers.Conv1D(hp.Int("layer_size0", min_value=20, max_value=180, step=20, default = 120), 
                      hp.Int("filter_size0", min_value=5, max_value=60, step=5, default = 32), activation='relu', padding = 'same',  kernel_initializer = init,
                      kernel_regularizer= kernal_regulariser,
                      bias_regularizer= bias_regulariser,
                      activity_regularizer= activity_regulariser)(input_timeseries)
    x = layers.Conv1D(hp.Int("layer_size1", min_value=20, max_value=170, step=20, default = 80), 
                      hp.Int("filter_size1", min_value=5, max_value=50, step=5, default = 32), activation='relu', padding = 'same', kernel_initializer = init,
                      kernel_regularizer= kernal_regulariser,
                      bias_regularizer= bias_regulariser,
                      activity_regularizer= activity_regulariser)(x)
    
    x = layers.MaxPooling1D(hp.Int('pool_size', 1, 4, 1, default = 1), padding = 'same')(x)

    encoded = layers.Conv1D(hp.Int("layer_size2", min_value=20, max_value=160, step=20, default = 40), 
                      hp.Int("filter_size2", min_value=5, max_value=40, step=5, default = 16), activation='relu', padding = 'same', kernel_initializer = init,
                      kernel_regularizer= kernal_regulariser,
                      bias_regularizer= bias_regulariser,
                      activity_regularizer= activity_regulariser)(x)

    ##Now make the other head with input
    y = layers.Dense(hp.Int("layer_size2", min_value=10, max_value=50, step=5, default = 30), activation = 'relu', kernel_initializer = init,
                      kernel_regularizer= kernal_regulariser,
                      bias_regularizer= bias_regulariser,
                      activity_regularizer= activity_regulariser)(input_flat)

    #Now make the other head
    flattened = layers.Flatten()(encoded)
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
    death_head = layers.Dense(3, activation = 'softmax', use_bias = True)(dense2)
    time_head = layers.Dense(3, activation = 'softmax', use_bias = True)(dense2)
    PEWS_head = layers.Dense(3, activation = 'softmax', use_bias = True)(dense2)

    #This is the full model with death and LOS as the outcome
    full_model = keras.Model([input_timeseries, input_flat], [death_head, time_head, PEWS_head])
    full_model.compile(optimizer = hp.Choice('optimiser', ['adam', 'rmsprop'], default = 'adam'), loss='categorical_crossentropy',  metrics=['accuracy', 
                        'mse', tf.keras.metrics.MeanAbsoluteError(), 
                        tf.keras.metrics.AUC()])               

    return full_model


class MyTuner(kt.tuners.Hyperband):
  def run_trial(self, trial, *args, **kwargs):
    # You can add additional HyperParameters for preprocessing and custom training loops
    # via overriding `run_trial`
    kwargs['batch_size'] = trial.hyperparameters.Int('batch_size', 32, 256, step=32)
    kwargs['epochs'] = trial.hyperparameters.Int('epochs', 10, 50)
    super(MyTuner, self).run_trial(trial, *args, **kwargs)# Uses same arguments as the BayesianOptimization Tuner.

tuner = MyTuner(
    temporal_convolutional_net,
    objective="val_loss",
    max_epochs=100,
    executions_per_trial=2,
    overwrite=True,
    directory="/store/DAMTP/dfs28/PICU_data",
    project_name="convnet_log",
)

#Callbacks
my_callbacks = [tf.keras.callbacks.EarlyStopping(patience=5),
                    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.001)]

tuner.search(
    [train_array3d, train_array2d], [train_outcomes[:, 2:5], train_outcomes[:, 5:8], train_outcomes[:, 8:11]],
    validation_data=([test_array3d, test_array2d], [test_outcomes[:, 2:5], test_outcomes[:, 5:8], test_outcomes[:, 8:11]]),
    callbacks=[tf.keras.callbacks.EarlyStopping(patience=4)],
)



model = KerasClassifier(build_fn=temporal_convolutional_net)
optimizers = ['rmsprop', 'adam']
init = ['glorot_uniform', 'normal', 'uniform']
epochs = [50, 100, 150]
batch_size = [range(50, 200, 20)]
layer_size0 = [range(20, 180, 10)]
layer_size1 = [range(15, 179, 10)]
layer_size2 = [range(10, 178, 10)]
layer_size3 = [range(5, 170, 10) ] 
layer_size4 = [range(5, 165, 10)]
layer_size5 = [range(2, 160, 10)]
filter_size = (range(5, 50, 10))
pool_size = [range(1, 4)]
dropout = [False, True]
reg_type = ['l1', 'l2', 'l1_l2', 'none']
reg_type_bias = ['l1', 'l2', 'l1_l2', 'none'] 
reg_type_activity = ['l1', 'l2', 'l1_l2', 'none'] 
reg_weight = [1e-5, 1e-4, 1e-3] 

param_grid = dict(optimizer=optimizers, epochs=epochs, batch_size=batch_size, init=init, layer_size0 = layer_size0,
                 layer_size1 = layer_size1, layer_size2 = layer_size2, layer_size3 = layer_size3, layer_size4 = layer_size4, 
                 layer_size5 = layer_size5, filter_size1 = filter_size, filter_size0 = filter_size, filter_size2 = filter_size, 
                 dropout = dropout, reg_type_activity = reg_type, reg_type_kernel = reg_type, reg_type_bias = reg_type, 
                 reg_weight1 = reg_weight, reg_weight2 = reg_weight)

scan_object = talos.Scan(x=[train_array3d, train_array2d],
                         y=[train_outcomes[:, 2:5], train_outcomes[:, 5:8], train_outcomes[:, 8:11]],
                         x_val=[test_array3d, test_array2d],
                         y_val=[test_outcomes[:, 2:5], test_outcomes[:, 5:8], test_outcomes[:, 8:11]],
                         params=param_grid,
                         model=temporal_convolutional_net, 
                         experiment_name = '1d_convnet')


#Think gridsearchcv takes 
random = RandomizedSearchCV(estimator=model, param_distributions=param_grid, cv = 10, random_state = 0)
random_result = random.fit([train_array3d, train_array2d], [train_outcomes[:, 2:5], train_outcomes[:, 5:8], train_outcomes[:, 8:11]], callbacks = my_callbacks)

#Now do crossvalidation with model
kf = KFold(n_splits=100, shuffle = True)
kf.get_n_splits(array3d)

for train_index, test_index in kf.split(array3d):
    print("TRAIN:", train_index, "TEST:", test_index)


    train_array3d, test_array3d = array3d[train_index], array3d[test_index]
    train_array2d, test_array2d = array2d[train_index], array2d[test_index]
    train_outcomes1, test_outcomes1 = outcomes[train_index, 5:8], outcomes[test_index, 5:8]
    train_outcomes2, test_outcomes2 = outcomes[train_index, 2:5], outcomes[test_index, 2:5]

    #Compile the model - should probably use sparse
    full_model.compile(optimizer='adam', loss='categorical_crossentropy',  metrics=['accuracy'])

    #Now fit the model
    full_model_history = full_model.fit([train_array3d, train_array2d], [train_outcomes1, train_outcomes2], 
                                    epochs = 20,
                                    batch_size = 100,
                                    shuffle = True, 
                                    validation_data = ([test_array3d, test_array2d], [test_outcomes1, test_outcomes2]), 
                                    callbacks = my_callbacks)

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
#Should then consider removing values with low shapley values and repeating network to see how it performs
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
ax2.imshow(ave_shap_values[1, :, :])
ax3.imshow(ave_shap_values[2, :, :])

#I think the naming is upside down but not sure how to fix this
for i, j in enumerate(cols):
    ax1.text(-5, 14.5 - i, j)
    ax2.text(-5, 14.5 - i, j)
    ax3.text(-5, 14.5 - i, j)

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