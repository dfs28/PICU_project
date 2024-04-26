### XGBoost

import numpy as np
import sklearn 
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from itertools import cycle
from scipy import interp
import matplotlib.pyplot as plt
import os
import pandas as pd
import json
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score, mean_squared_error, mean_absolute_error, auc, confusion_matrix, roc_curve, precision_score, recall_score, f1_score
from sklearn.metrics import PrecisionRecallDisplay, precision_recall_curve
import matplotlib.pyplot as plt
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
import argparse 
import re
import random

#Make it so you can run this from the command line
parser = argparse.ArgumentParser(description="Allow running of XGBoost with different input parameters",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-l", "--length", default=6, type=int, help="Length of predictive target in hours")
parser.add_argument("-z", "--no_z", default=False, type=bool, help="Whether or not to use z-scores")
parser.add_argument("-i", "--input_length", default=3, type=int, help="Length of the input window in hours")
args = vars(parser.parse_args())

#Set up whether you want to use the no_z values - default value to be false - but will not override if given above
no_z = args['no_z']
input_length = args['input_length']
length = args['length']
    
print(f'Running with {length}h time target window, {input_length}h input window, using z-scores {no_z == False}')

#Pull out the data depending on whether no_z was used
if not no_z:
    #Read in the data
    file_start = '/store/DAMTP/dfs28/PICU_data/np_arrays_pSOFA'
else:
    file_start = '/store/DAMTP/dfs28/PICU_data/np_arrays_no_zscore_pSOFA'
    
if input_length == 3:
    file_name = file_start + '.npz' 
else:
    file_name = file_start + f'_{input_length}h.npz'

#Read in data    
data = np.load(file_name)
array3d = data['d3']
array2d = data['d2']
outcomes = data['outcomes']
characteristics = data['chars']
splines = data['splines']
slopes = data['slopes']
r_values = data['r_values']


def test_trainsplit(array, split, array3d = array3d, outcome_array = outcomes, seed = 1):
    """
    Function to split up 3d slices into test, train, validate
    split is an np.ndarray
    """

    #Flexibly choose dimension to split along
    shape = array3d.shape
    z_dim = np.max(shape)
    
    #Get the unique identifiers 
    unique_pts = np.unique(outcome_array[:, 0])
    
    #Now randomly shuffle the order of patients
    np.random.seed(seed)
    np.random.shuffle(unique_pts) #Annoyingly does this in-place - not sure how to change it
    
    #Set some initial parameters
    total_length = outcomes.shape[0]
    proportions = split / np.sum(split)
    total_proportion_length = np.round(np.cumsum(total_length * proportions))
    
    # Calculate lengths of each patient
    lengths = np.array([np.sum(outcomes[:, 0] == pt) for pt in unique_pts])
    
    # Set up some storage
    sampled_indices = np.array([])
    final_splits = list()
    
    # Iterate over split proportions
    for proportion_length in total_proportion_length:
        # Get the cumulative lengths and where they are less than the cumulative proportion
        cum_lengths = np.cumsum(lengths)
        indices = np.where(cum_lengths <= proportion_length)[0]
        
        # Exclude already sampled indices
        indices = np.setdiff1d(indices, sampled_indices)
        chosen_index = indices
        
        # Add chosen index to sampled indices
        final_splits.append(chosen_index)
        sampled_indices = np.concatenate((sampled_indices, chosen_index), axis=None)
    
    #Now add back so they are shuffled.
    random_split = [unique_pts[i] for i in final_splits]
    
    #Work through the array and pick out the outcome indices and add them to the split list
    split_array = list()
    for i in random_split:
        array_locs = pd.Series(outcomes[:, 0]).isin(i)
        split_array.append(array[array_locs, ])
    
    return split_array


#Split up testing and outcomes
split_characteristics = test_trainsplit(characteristics, np.array([85, 15]))
split_array2d = test_trainsplit(array2d, np.array([85, 15]))
split_array3d = test_trainsplit(array3d, np.array([85, 15]))
split_slopes = test_trainsplit(slopes, np.array([85, 15]))
split_R = test_trainsplit(r_values, np.array([85, 15]))
split_outcomes = test_trainsplit(outcomes, np.array([85, 15]))

#Training sets
train_characteristics = split_characteristics[0]
train_array3d = split_array3d[0]
train_array2d = split_array2d[0]
train_slopes = split_slopes[0]
train_R = split_R[0]
train_outcomes = split_outcomes[0]

#Test sets
test_characteristics = split_characteristics[1]
test_array3d = split_array3d[1]
test_array2d = split_array2d[1]
test_slopes = split_slopes[1]
test_R = split_R[1]
test_outcomes = split_outcomes[1]

#Make binary outcomes
#Make the binary values
binary_deterioration_train_hour_outcomes = np.array(train_outcomes[:, 12] < length, dtype = int)
binary_deterioration_train_outcomes = np.transpose(np.array([1- binary_deterioration_train_hour_outcomes, binary_deterioration_train_hour_outcomes]))
binary_deterioration_test_hour_outcomes = np.array(test_outcomes[:, 12] < length, dtype = int)
binary_deterioration_test_outcomes = np.transpose(np.array([1- binary_deterioration_test_hour_outcomes, binary_deterioration_test_hour_outcomes]))

#Set x and y
X = np.concatenate((train_array2d, train_characteristics, train_slopes, train_R), axis=1)
y = np.argmax(binary_deterioration_train_outcomes, axis = 1)

# grid search
model = XGBClassifier(objective='binary:logistic', eval_metric = 'aucpr', use_label_encoder=False, n_jobs = 32)
kfold = StratifiedKFold(n_splits=4, shuffle=True, random_state=7)
param_grid = {"learning_rate"    : [0.001, 0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
 				'max_depth' : [5, 10, 15, 20, 25, 30, 35, 40],
 					"min_child_weight" : [ 1, 5, 7, 15],
 					"gamma"            : [ 0.0, 0.2, 0.4 ],
 					"colsample_bytree" : [ 0.3, 0.5 , 0.7, 1], 
					 "subsample":[0.5, 0.75, 0.9, 1], 
					 "scale_pos_weight" : [0.1, 0.15, 0.2, 0.25, 0.5, 0.75, 1, 2, 4], 
					 "n_estimators" : [50, 100, 150, 200, 250]}

model_path_name = '/mhome/damtp/q/dfs28/Project/PICU_project/models/'
file_name = 'XGBoost_best_'
file_suffix = f'{input_length}h_{length}h_{"no_zscore_" if no_z else ""}pSOFA'

if os.path.exists(model_path_name + file_name + file_suffix + '.json') == False:                         
    clf = BayesSearchCV(model, param_grid, random_state=0, cv = kfold, iid = False)
    search = clf.fit(X, y)
    
    #Save the best hyperparameters
    best_hyperparameters = param_grid = search.best_params_

    a_file = open(model_path_name + file_name + file_suffix + '.json', "w")
    json.dump(best_hyperparameters, a_file)
    a_file.close()

else:
    f = open(model_path_name + file_name + file_suffix + '.json', )
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
model1 = XGBClassifier(objective='binary:logistic', eval_metric = 'aucpr', use_label_encoder=False, n_jobs = 32, **param_grid)
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
AUROC.append(roc_auc_score(y_test, y_pred_proba[:, 1]))
Recall.append(recall_score(y_test, y_pred))
Precision.append(precision_score(y_test, y_pred))
F1.append(f1_score(y_test, y_pred))
AUPRC.append(average_precision_score(y_test, y_pred_proba[:,1]))

#Calculate precision at recall 0.9
prec, recall, thresholds_prc = precision_recall_curve(y_test, y_pred_proba[:,1], pos_label=clf1.classes_[1])
pr_display = PrecisionRecallDisplay(precision=prec, recall=recall).plot()
thresholds_prc = np.append([0], thresholds_prc) #Thresholds is n-1 length of precision and recall
threshold_90 = thresholds_prc[recall > 0.9][-1]

#Now  calculate F1 at that level - should get precision at level of recall
y_pred_tuned = (y_pred_proba[:,1] > threshold_90).astype(int)
recall_90 = recall_score(y_test, y_pred_tuned)
i = 1
while recall_90 < 0.9:
    #Make sure that recall definitely above 0.9
    y_pred_tuned = (y_pred_proba[:,1] > threshold_90).astype(int)
    recall_90 = recall_score(y_test, y_pred_tuned)
    i += 1

f1_at_90 = f1_score(y_test, y_pred_tuned)
precision_at_90 = precision_score(y_test, y_pred_tuned)


#Save to a json file
results = {'acc_PEWS' : np.mean(accuracy),
            'AUC_PEWS' : np.mean(AUROC),
            'MSE_PEWS' : np.mean(MSE),
            'MAE_PEWS' : np.mean(MAE), 
            'precision_PEWS' : np.mean(Precision), 
            'recall_PEWS' : np.mean(Recall), 
            'F1_PEWS' : np.mean(F1), 
            'AUPRC_PEWS' : np.mean(AUPRC), 
            'precision_at_recall_90': precision_at_90,
            'f1_at_recall_90': f1_at_90, 
            'recall_score_closest_90': recall_90}

res_path_name = '/mhome/damtp/q/dfs28/Project/PICU_project/files/'

results_df = pd.DataFrame({'metrics': results.keys(), 'results': results.values()})
results_df.to_csv(res_path_name + 'XGBoost_' + file_suffix + '.csv')

### Save some relevant plots 
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
ax.set_title(f'Feature importance for XGBoost - {length}h model')
plt.subplots_adjust(left=0.3)

figs_path_name = '/mhome/damtp/q/dfs28/Project/PICU_project/figs/PICU/XGBoost/'
fig.savefig(figs_path_name + 'XGBoost_importance_' + file_suffix + '.pdf', format="pdf")


#### Do some SHAP plotting
import shap

# make sure the SHAP values add up to marginal predictions
#Summary plot
explainer = shap.TreeExplainer(clf1)
shap_values = explainer.shap_values(X_test, check_additivity=False)
fig, ax = plt.subplots(1, 1, figsize = (15, 11))
shap_plot1 = shap.summary_plot(shap_values, X_test, feature_names = feature_names1, title = f'SHAP values for model predicting deterioration within {length}h', show=False)

plt.savefig(figs_path_name + 'SHAP_xgboost_' + file_suffix + '.pdf', format="pdf", bbox_inches = 'tight')

#Pure features
explainer.feature_names = feature_names1
shap_explainer = explainer(X_test, check_additivity=False)
shap_explainer.feature_names = feature_names1
fig, ax = plt.subplots(1, 1, figsize = (15, 11))
shap_plot2 = shap.plots.bar(shap_explainer, max_display=30, show = False)
plt.savefig(figs_path_name + 'SHAP_importances_xgboost_' + file_suffix + '.pdf', bbox_inches = 'tight', format="pdf")


#Plot precision recall curve
fig, ax = plt.subplots(1, 1, figsize = (15, 11))
pr_display1 = PrecisionRecallDisplay(precision=prec, recall=recall)
pr_display1.plot().figure_.savefig(figs_path_name + 'PRC_xgboost_' + file_suffix + '.pdf', bbox_inches = 'tight', format="pdf")

#Matched precision_recall curves:
clf_pSOFA = LogisticRegression(random_state=0, max_iter= 100, multi_class='multinomial', solver='lbfgs', penalty = 'l2')
clf_pSOFA.fit(np.max(train_array3d[:, series_cols == 'pSOFA', :], axis = 2), np.argmax(binary_deterioration_train_outcomes, axis = 1))
y_predict_pSOFA = clf_pSOFA.predict_proba(np.max(test_array3d[:, series_cols == 'pSOFA', :], axis = 2))
average_precision_score(binary_deterioration_test_outcomes[:,1], y_predict_pSOFA[:,1])
prec_pSOFA, recall_pSOFA, thresholds_prc = precision_recall_curve(binary_deterioration_test_outcomes[:,1], y_predict_pSOFA[:,1], pos_label=clf1.classes_[1])
pr_display_pSOFA = PrecisionRecallDisplay(precision=prec_pSOFA, recall=recall_pSOFA)
pr_display1 = PrecisionRecallDisplay(precision=prec, recall=recall)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
ax1.set_ylim((0.7, 1))
ax2.set_ylim((0.7, 1))
ax1.title.set_text('PRC for XGBoost Model')
ax2.title.set_text('PRC for pSOFA Model')
pr_display1.plot(ax=ax1)
pr_display_pSOFA.plot(ax=ax2)

plt.savefig(figs_path_name + 'Paired_PRC_' + file_suffix + '.pdf', bbox_inches = 'tight', format="pdf")