### XGBoost

import numpy as np
import sklearn 
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from itertools import cycle
from scipy import interp
import matplotlib.pyplot as plt
import os
import pandas as pd
import json
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score, mean_squared_error, mean_absolute_error, auc, confusion_matrix, roc_curve, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from skopt import BayesSearchCV

#Read in the data
#Consider binning unnecessary vars (r, std or just slopes in general)
data = np.load('/store/DAMTP/dfs28/PICU_data/np_arrays.npz')
array3d = data['d3']
array2d = data['d2']
outcomes = data['outcomes']
characteristics = data['chars']
splines = data['splines']
slopes = data['slopes']
r_values = data['r_values']


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
split_characteristics = test_trainsplit(characteristics, np.array([85, 15]))
split_array2d = test_trainsplit(array2d, np.array([85, 15]))
split_slopes = test_trainsplit(slopes, np.array([85, 15]))
split_R = test_trainsplit(r_values, np.array([85, 15]))
split_outcomes = test_trainsplit(outcomes, np.array([85, 15]))

#Training sets
train_characteristics = split_characteristics[0]
train_array2d = split_array2d[0]
train_slopes = split_slopes[0]
train_R = split_R[0]
train_outcomes = split_outcomes[0]

#Test sets
test_characteristics = split_characteristics[1]
test_array2d = split_array2d[1]
test_slopes = split_slopes[1]
test_R = split_R[1]
test_outcomes = split_outcomes[1]

#Make binary outcomes
#Make the binary values
binary_deterioration_train_outcomes = np.transpose(np.array([np.sum(train_outcomes[:, 8:9], axis = 1), np.sum(train_outcomes[:,9:11], axis = 1)]))
binary_deterioration_test_outcomes = np.transpose(np.array([np.sum(test_outcomes[:, 8:9], axis = 1), np.sum(test_outcomes[:,9:11], axis = 1)]))



#Set x and y
X = np.concatenate((train_array2d, train_characteristics, train_slopes, train_R), axis=1)
y = np.argmax(binary_deterioration_train_outcomes, axis = 1)

# grid search
model = XGBClassifier(objective='binary:logistic', eval_metric = 'logloss', use_label_encoder=False, n_jobs = 32)
kfold = StratifiedKFold(n_splits=4, shuffle=True, random_state=7)
param_grid = {"learning_rate"    : [0.001, 0.01, 0.10, 0.20, 0.30],
 				'max_depth' : [5, 10, 15, 20],
 					"min_child_weight" : [ 1, 5, 7, 15],
 					"gamma"            : [ 0.0, 0.2, 0.4 ],
 					"colsample_bytree" : [ 0.3, 0.5 , 0.7, 1], 
					 "subsample":[0.5, 0.75, 1], 
					 "scale_pos_weight" : [1, 2, 4], 
					 "n_estimators" : [50, 100, 150]}

if os.path.exists('/mhome/damtp/q/dfs28/Project/PICU_project/models/XGBoost_best_binary_slope.json') == False:                         
       clf = BayesSearchCV(model, param_grid, random_state=0, cv = kfold, iid = False)
       search = clf.fit(X, y)

       #Save the best hyperparameters
       best_hyperparameters = search.best_params_

       a_file = open("/mhome/damtp/q/dfs28/Project/PICU_project/models/XGBoost_best_binary_slope.json", "w")
       json.dump(best_hyperparameters, a_file)
       a_file.close()

else:
       f = open("/mhome/damtp/q/dfs28/Project/PICU_project/models/XGBoost_best_binary_slope.json", )
       param_grid = json.load(f)

#get colnames
point_cols = data['point_cols']
point_cols = [i for i in point_cols]
series_cols = data['series_cols']

#Combine the series cols objects and name them for labelling of axes
series_cols_mean = ['Mean ' + i for i in series_cols]
series_cols_std = ['STD ' + i for i in series_cols]
series_cols_slopes = ['Slopes ' + i for i in series_cols]
series_cols_R = ['R ' + i for i in series_cols]
all_cols = series_cols_mean
all_cols.extend(series_cols_std)
all_cols.extend(series_cols_slopes)
all_cols.extend(series_cols_R)

#Now run and get parameters
model1 = XGBClassifier(objective='binary:logistic', eval_metric = 'logloss', use_label_encoder=False, n_jobs = 32, **param_grid)
X_test = np.concatenate((test_array2d, test_characteristics, test_slopes, test_R), axis=1)
y_test = np.argmax(binary_deterioration_test_outcomes, axis = 1)

#Don't need to rerun 10 times 
accuracy = list()
MSE = list()
AUROC = list()
MAE = list()
Precision = list()
Recall = list()
F1 = list()
AUPRC = list()

#Need to one hot encode
onehot_encoder = sklearn.preprocessing.OneHotEncoder(sparse=False)
integer_encoded_test = y_test.reshape(len(y_test), 1)
onehot_encoded_test = onehot_encoder.fit_transform(integer_encoded_test)

#Run the model
clf1 = model1.fit(X, y)
y_pred = clf1.predict(X_test)
y_pred_proba = clf1.predict_proba(X_test)
integer_encoded_pred = y_pred.reshape(len(y_pred), 1)
onehot_encoded_pred = onehot_encoder.fit_transform(integer_encoded_pred)

#Save the outcomes
accuracy.append(accuracy_score(y_test, y_pred))
MSE.append(mean_squared_error(y_test, y_pred))
MAE.append(mean_absolute_error(y_test, y_pred))
AUROC.append(roc_auc_score(onehot_encoded_test, y_pred_proba,  average = 'macro'))
Recall.append(recall_score(onehot_encoded_test, onehot_encoded_pred, average = 'macro'))
Precision.append(precision_score(onehot_encoded_test, onehot_encoded_pred, average = 'macro', zero_division = 0))
F1.append(f1_score(onehot_encoded_test, onehot_encoded_pred, average = 'macro'))
AUPRC.append(average_precision_score(onehot_encoded_test, y_pred_proba,  average = 'macro'))

#Save to a json file
results = {'acc_PEWS' : np.mean(accuracy),
            'AUC_PEWS' : np.mean(AUROC),
            'MSE_PEWS' : np.mean(MSE),
            'MAE_PEWS' : np.mean(MAE), 
            'precision_PEWS' : np.mean(Precision), 
            'recall_PEWS' : np.mean(Recall), 
            'F1_PEWS' : np.mean(F1)}

a_file = open("/mhome/damtp/q/dfs28/Project/PICU_project/files/XGBoost_results_slope_binary", "w")
json.dump(results, a_file)
a_file.close()



conf_mat1 = confusion_matrix(y_test, y_pred)

#Tune to sensitivity
#Use ratio of outcome 3 to 1
y_pred_proba = clf1.predict_proba(X_test)
y_pred_ratio = y_pred_proba[:, 0]/ y_pred_proba[:, 1]
y_pred_ratio1 = y_pred_proba[:, 1]/ y_pred_proba[:, 0]

#Work out best threshold
fpr, tpr, thresholds = roc_curve(1- np.argmax(binary_deterioration_test_outcomes, axis = 1), y_pred_proba[:,1])
gmeans = np.sqrt(tpr * (1-fpr))
ix = np.argmax(gmeans)
print('Best Threshold=%f, G-Mean=%.3f' % (thresholds[ix], gmeans[ix]))

fpr1, tpr1, thresholds1 = roc_curve(1- np.argmax(binary_deterioration_test_outcomes, axis = 1), y_pred_proba[:,1])

#Get the tuned best
y_tuned = (y_pred_proba[:,1] > thresholds[ix]).astype(int)
confusion_matrix((y_pred_proba[:,1] > thresholds[ix]).astype(int), np.argmax(binary_deterioration_test_outcomes, axis = 1))

#Now calculate precision given a 90% sensitivity
precision = tpr/(fpr + tpr)
precision[tpr > 0.9][0]

y_tuned_sensitivity = (y_pred_ratio > thresholds[tpr > 0.9][0]).astype(int)
confusion_matrix((y_pred_proba[:,1] > thresholds[tpr > 0.9][0]).astype(int), np.argmax(binary_deterioration_test_outcomes, axis = 1))


#Importance plot
feature_importance1 = clf1.feature_importances_
#Get top 20 most important features
feature_order1 = np.argsort(-1*feature_importance1)
feature_names1 =  point_cols + all_cols
top_features_names1 = [feature_names1[i] for i in feature_order1[0:20]]
top_feature_values1 = [feature_importance1[i] for i in feature_order1[0:20]]

#Horizontal bar plot
fig, ax = plt.subplots(1, 1, figsize = (10, 7))
ax.barh(top_features_names1, top_feature_values1)
ax.set_yticks(np.arange(len(top_features_names1)))
ax.set_yticklabels(top_features_names1)
ax.invert_yaxis()  # labels read top-to-bottom
ax.set_xlabel('Gain (feature importance)')
ax.set_title('Feature importance for Deterioration')
plt.subplots_adjust(left=0.3)
fig.savefig('/mhome/damtp/q/dfs28/Project/PICU_project/figs/PICU/XGBoost/XGBoost_importance_deterioration_only_binary.png')





#### Do some SHAP plotting
import shap

# make sure the SHAP values add up to marginal predictions
#Summary plot
explainer = shap.TreeExplainer(clf1)
shap_values = explainer.shap_values(X_test)
np.abs(shap_values.sum(1) + explainer.expected_value - pred).max()
shap.summary_plot(shap_values, X_test, feature_names = feature_names1)
fig.savefig('/mhome/damtp/q/dfs28/Project/PICU_project/figs/PICU/XGBoost/SHAP_xgboost_binary_summary.png')

#Pure features
explainer.feature_names = feature_names1
shap_explainer = explainer(X_test)
shap_explainer.feature_names = feature_names1
shap.plots.bar(shap_explainer, max_display=30)

#Plot precision recall curve
from sklearn.metrics import precision_recall_curve
precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba[:, 1])
precision

recall

thresholds