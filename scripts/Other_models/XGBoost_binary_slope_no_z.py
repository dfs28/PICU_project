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
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score, mean_squared_error, mean_absolute_error, auc, confusion_matrix, roc_curve, precision_score, recall_score, f1_score, precision_recall_curve
import matplotlib.pyplot as plt
from skopt import BayesSearchCV

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, cross_validate, cross_val_predict

import re

#Read in the data
#Consider binning unnecessary vars (r, std or just slopes in general)
data = np.load('/store/DAMTP/dfs28/PICU_data/np_arrays_no_zscore.npz')
array3d = data['d3']
array2d = data['d2']
outcomes = data['outcomes']
characteristics = data['chars']
splines = data['splines']
slopes = data['slopes']
r_values = data['r_values']

#With z score
data_z = np.load('/store/DAMTP/dfs28/PICU_data/np_arrays.npz')
array2d_z = data_z['d2']
characteristics_z = data_z['chars']
slopes_z = data_z['slopes']
r_values_z = data_z['r_values']

#Combine the point and series cols objects to pull out where the z score data we want to use is
series_cols_z = data_z['series_cols']
series_cols = data['series_cols']
KG_values_z = ['O2Flow.kg', 'Inotropes_kg', 'Urine_output_kg']
KG_locations_z = np.where([i in KG_values_z for i in series_cols_z])[0]
KG_values = ['O2Flow', 'Inotropes', 'Urine_output']
KG_locations = np.where([i in KG_values for i in series_cols])[0]

point_cols_z = data_z['point_cols']
point_cols = data['point_cols']
weight_value_z = ['Weight_z_scores']
weight_location_z = np.where([i in weight_value_z for i in point_cols_z])[0]
weight_value = ['interpolated_wt_kg']
weight_location = np.where([i in weight_value for i in point_cols])[0]

#Substitute no_z values into characteristics, slopes, R
characteristics[:,[KG_values, KG_values*2]] = characteristics_z[:,[KG_values_z, KG_locations_z*2]]
slopes[:, KG_values] = slopes_z[:, KG_locations_z]
r_values[:, KG_locations] = r_values_z[:, KG_locations_z]

#Adjust series colnames
series_cols = data['series_cols']
series_cols[KG_locations] = series_cols_z[KG_locations_z]

#Adjust the point values
array2d[:, weight_location] = array2d_z[:, weight_location_z]
point_cols[weight_location[0]] = point_cols_z[weight_location_z[0]]


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

binary_deterioration_outcomes = np.transpose(np.array([np.sum(outcomes[:, 8:9], axis = 1), np.sum(outcomes[:,9:11], axis = 1)]))



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
#point_cols = data['point_cols']
point_cols = [i for i in point_cols]
#series_cols = data['series_cols']

#Combine the series cols objects and name them for labelling of axes
series_cols_mean = ['Mean ' + i for i in series_cols]
series_cols_std = ['STD ' + i for i in series_cols]
series_cols_slopes = ['Slopes ' + i for i in series_cols]
series_cols_R = ['R ' + i for i in series_cols]
all_cols = series_cols_mean
all_cols.extend(series_cols_std)
all_cols.extend(series_cols_slopes)
all_cols.extend(series_cols_R)

#Rename the column names so they are human readable
cleaned_all_cols = [re.sub('(.*)(\_missing).*', 'Missingness in \\1', i) for i in all_cols]
cleaned_all_cols = [re.sub('.*(Mean{1})(.*)', '3-hour\\2 average', i) for i in cleaned_all_cols]
cleaned_all_cols = [re.sub('.*( R {1})(.*)', '3-hour\\2 trend goodness of fit', i) for i in cleaned_all_cols]
cleaned_all_cols = [re.sub('(R {1})(.*)', '3-hour\\2 trend goodness of fit', i) for i in cleaned_all_cols]
cleaned_all_cols = [re.sub('.*(STD{1})(.*)', '3-hour\\2 variability', i) for i in cleaned_all_cols]
cleaned_all_cols = [re.sub('.*(Slopes{1})(.*)', '3-hour\\2 trend', i) for i in cleaned_all_cols]
cleaned_all_cols = [re.sub('\A(.*)( [A-Za-z]{2,3})(_zscore{1})(.*)', '\\1 age-normalised\\2 \\4', i) for i in cleaned_all_cols]
cleaned_all_cols = [re.sub('\A(.*3-hour{1})([A-Za-z]{1})(.*)', '\\1 \\2\\3', i) for i in cleaned_all_cols]


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
            'F1_PEWS' : np.mean(F1),
            'AUPRC_PEWS' : np.mean(AUPRC)}

a_file = open("/mhome/damtp/q/dfs28/Project/PICU_project/files/XGBoost_results_slope_binary_no_zscore_with_weight_norm", "w")
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

from sklearn.metrics import PrecisionRecallDisplay

prec, recall, thresholds = precision_recall_curve(y_test, y_pred_proba[:,1], pos_label=clf1.classes_[1])
pr_display = PrecisionRecallDisplay(precision=prec, recall=recall).plot()

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


#Now repeat only for patients where they had inotropes
inotropes_missing = [i for i, j in enumerate(feature_names1) if j == 'Mean Inotropes_missing']
on_inotropes = X[:, inotropes_missing[0]] != 1
X_on_inotropes = X[on_inotropes, :]
y_on_inotropes = y[on_inotropes]
test_on_inotropes = X_test[:, inotropes_missing[0]] != 1
X_test_on_inotropes = X_test[test_on_inotropes, :]
y_test_on_inotropes = y_test[test_on_inotropes]

#Run the model
clf2 = model1.fit(X_on_inotropes, y_on_inotropes)

#Importance plot
feature_importance2 = clf2.feature_importances_
#Get top 20 most important features
feature_order2 = np.argsort(-1*feature_importance2)
top_features_names2 = [feature_names1[i] for i in feature_order2[0:20]]
top_feature_values2 = [feature_importance2[i] for i in feature_order2[0:20]]

#Horizontal bar plot
fig, ax = plt.subplots(1, 1, figsize = (10, 7))
ax.barh(top_features_names2, top_feature_values2)
ax.set_yticks(np.arange(len(top_features_names2)))
ax.set_yticklabels(top_features_names2)
ax.invert_yaxis()  # labels read top-to-bottom
ax.set_xlabel('Gain (feature importance)')
ax.set_title('Feature importance for Deterioration')
plt.subplots_adjust(left=0.3)
fig.savefig('/mhome/damtp/q/dfs28/Project/PICU_project/figs/PICU/XGBoost/XGBoost_importance_deterioration_only_binary_on_inotropes.png')



#Redo best thresholds using PRC function
precision_calc, recall_calc, thresholds_prc = precision_recall_curve(y_test, y_pred_proba[:,1], pos_label=clf1.classes_[1])
precision_calc[recall_calc > 0.9][-1]

y_tuned_sensitivity = (y_pred_ratio > thresholds_prc[np.where(recall_calc > 0.9)[0][-1]]).astype(int)
tuned_conf_mat = confusion_matrix((y_pred_proba[:,1] > thresholds_prc[np.where(recall_calc > 0.9)[0][-1]]).astype(int), np.argmax(binary_deterioration_test_outcomes, axis = 1))

f1_score((y_pred_proba[:,1] > thresholds_prc[np.where(recall_calc > 0.9)[0][-1]]).astype(int), np.argmax(binary_deterioration_test_outcomes, axis = 1))



#### Do some SHAP plotting
import shap

# make sure the SHAP values add up to marginal predictions
#Summary plot
explainer = shap.TreeExplainer(clf1)
shap_values = explainer.shap_values(X_test)
np.abs(shap_values.sum(1) + explainer.expected_value - pred).max()


fig, ax = plt.subplots(1, 1, figsize = (15, 11))
shap.summary_plot(shap_values, X_test, feature_names = feature_names1, title = 'SHAP values for all patients model')
fig.savefig('/mhome/damtp/q/dfs28/Project/PICU_project/figs/PICU/XGBoost/SHAP_xgboost_binary_summary_no_z_weight.png', bbox_inches = 'tight')

#Summary plot for not on inotropes
explainer2 = shap.TreeExplainer(clf2)
shap_values2 = explainer2.shap_values(X_test_on_inotropes)
np.abs(shap_values2.sum(1) + explainer2.expected_value - pred).max()
fig, ax = plt.subplots(1, 1, figsize = (15, 11))
shap.summary_plot(shap_values2, X_test_on_inotropes, feature_names = feature_names1, title = 'SHAP values for only inotropes model')
fig.savefig('/mhome/damtp/q/dfs28/Project/PICU_project/figs/PICU/XGBoost/SHAP_xgboost_binary_summary_inotropes_no_z_weight.png', bbox_inches = 'tight')

#Pure features
explainer.feature_names = feature_names1
shap_explainer = explainer(X_test)
shap_explainer.feature_names = feature_names1
shap.plots.bar(shap_explainer, max_display=30)

#Pure features not on inotropes
explainer2.feature_names = feature_names1
shap_explainer2 = explainer2(X_test_on_inotropes)
shap_explainer2.feature_names = feature_names1
shap.plots.bar(shap_explainer2, max_display=30)

#Plot precision recall curve
from sklearn.metrics import precision_recall_curve
precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba[:, 1])
precision

recall

thresholds






### Look at just using PEWS for the logistic regression
## Input PEWS into logistic regression
series_cols = data['series_cols']
PEWS = array3d[:, series_cols == 'PEWS', :]
PEWS = PEWS.reshape(PEWS.shape[0], PEWS.shape[2])
last_PEWS = PEWS[:, -1:]
clf_PEWS = LogisticRegression(random_state=0, max_iter= 100, multi_class='multinomial', solver='lbfgs', penalty = 'l2')
scores1_last_PEWS_PEWS = cross_validate(clf_PEWS, last_PEWS, np.argmax(binary_deterioration_outcomes, axis = 1) , scoring = ['roc_auc_ovr', 'accuracy', 'precision', 'recall'], cv = 10)

y_predict = cross_val_predict(clf_PEWS, last_PEWS, np.argmax(binary_deterioration_outcomes, axis = 1), cv=10, method = 'predict_proba')

means3_last_PEWS_PEWS = [np.mean(scores1_last_PEWS_PEWS[i]) for i in scores1_last_PEWS_PEWS.keys()]
stds3_last_PEWS_PEWS = [np.std(scores1_last_PEWS_PEWS[i]) for i in scores1_last_PEWS_PEWS.keys()]

#Tune to sensitivity
#Use ratio of outcome 3 to 1
y_pred_ratio = y_predict[:, 0]/ y_predict[:, 1]
y_pred_ratio1 = y_predict[:, 1]/ y_predict[:, 0]

#Work out best threshold
fpr, tpr, thresholds = roc_curve(1- np.argmax(binary_deterioration_outcomes, axis = 1), y_predict[:,1])
gmeans = np.sqrt(tpr * (1-fpr))
ix = np.argmax(gmeans)
print('Best Threshold=%f, G-Mean=%.3f' % (thresholds[ix], gmeans[ix]))

fpr1, tpr1, thresholds1 = roc_curve(1- np.argmax(binary_deterioration_outcomes, axis = 1), y_predict[:,1])

#Get the tuned best
y_tuned = (y_predict[:,1] > thresholds[ix]).astype(int)
confusion_matrix((y_predict[:,1] > thresholds[ix]).astype(int), np.argmax(binary_deterioration_outcomes, axis = 1))

#Now calculate precision given a 90% sensitivity
precision = tpr/(fpr + tpr)
precision[tpr > 0.9][0]

y_tuned_sensitivity = (y_pred_ratio > thresholds[tpr > 0.9][0]).astype(int)
confusion_matrix((y_pred_proba[:,1] > thresholds[tpr > 0.9][0]).astype(int), np.argmax(binary_deterioration_test_outcomes, axis = 1))

average_precision_score(binary_deterioration_outcomes, y_predict)
f1_score(np.round(1- np.argmax(binary_deterioration_outcomes, axis = 1)), np.round(y_predict[:,1]))

prec, recall, _ = precision_recall_curve(binary_deterioration_outcomes[:,1], y_predict[:,1])
pr_display1 = PrecisionRecallDisplay(precision=prec, recall=recall).plot()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
ax1.set_ylim((0, 1))
ax2.set_ylim((0, 1))
ax1.title.set_text('PRC for XGBoost Model')
ax2.title.set_text('PRC for PEWS Model')
pr_display.plot(ax=ax1)
pr_display1.plot(ax=ax2)

plt.show()


#We can have a look at their flowsheet and other events



#### Plot predictions against actual patient data for plotting in the paper
#We can have a look at their flowsheet and other events
flowsheet = pd.read_csv('/mhome/damtp/q/dfs28/Project/Project_data/files/caboodle_patient_selected_flowsheetrows_main_pivot.csv', sep= ',', parse_dates=['taken_datetime'])
demographics = pd.read_csv('/mhome/damtp/q/dfs28/Project/Project_data/files/caboodle_patient_demographics.csv', sep = ',', parse_dates = ['birth_date', 'death_date'])
#medications = pd.read_csv('/mhome/damtp/q/dfs28/Project/Project_data/files/caboodle_patient_selected_medication_admins_main.csv', sep = ',', parse_dates = ['start_datetime', 'end_datetime'])
#labs = pd.read_csv('/store/DAMTP/dfs28/updated_files/caboodle_patient_selected_lab_components_main_pivot.csv', sep = ',', parse_dates = ['collected_datetime', 'received_datetime', 'verified_datetime'])


#Work out predictions on individual patients
#Where did patients deteriorate within 6h
outcomes[outcomes[:,0]==1534.0,8:9] 

#Where did patients die
#Note outcomes[:,0] is the location of the project ID
died_test = np.intersect1d(demographics.death_date.dropna().index + 1, outcomes[binary_deterioration_train_outcomes.shape[0]:,0]) #Start from after train outcomes
died_locs = {i:j for i, j in enumerate(outcomes[:,0]) if j in died_test}

#Now the predictions for the patients who died
died_predictions = y_pred_proba[np.array([i for i in died_locs], dtype = int) - binary_deterioration_train_outcomes.shape[0],1]
died_predictions_patient = [died_locs[i] for i in died_locs]
died_predictions_1533 = y_pred_proba[np.array(np.where(outcomes[:,0]==1534.0), dtype = int) - binary_deterioration_train_outcomes.shape[0],1]
died_predictions_patient = [died_locs[i] for i in died_locs]


#These are the predictions for the first patient who died
patient_died_predictions = died_predictions[died_predictions_patient == died_predictions_patient[0]]
flowsheet_not_na = flowsheet.loc[flowsheet.project_id == np.unique(flowsheet.project_id)[int(died_predictions_patient[0])], :].dropna(how = 'all', axis = 1)

############# Now match time to the time of the predictions
start_time = flowsheet_not_na.taken_datetime[flowsheet_not_na.taken_datetime.index[0]]
points_of_prediction = [start_time + pd.Timedelta(r'%sh' % 3*(i+1)) for i in range(patient_died_predictions.shape[0])]

#Make a dataframe with predictions and when they were made, and then merge that on so you can see the predictions alongside the data
prediction_time = pd.DataFrame({'taken_datetime': points_of_prediction, 'prediction': patient_died_predictions})
flowsheet_not_na = pd.merge(flowsheet_not_na, prediction_time, left_on = 'taken_datetime', right_on = 'taken_datetime', how = 'outer')
flowsheet_not_na = flowsheet_not_na.dropna(subset = ['project_id'])

#Make some plots
#Make locations where a reasonable amount and interesting data
shapes = np.array([flowsheet_not_na.loc[:,i].dropna().shape[0] for i in flowsheet_not_na.columns])
interesting_cols = np.intersect1d(np.where(flowsheet_not_na.dtypes == float), np.where(shapes > 9))[1:]

#Choose only the columns we're interested in
interesting_column_names = [
       'R PEDS CALCULATED URINE_mL', 
        'R GOSH COMFORT SCORE',
        'R VENT ETCO2_kPa',
       'R AN EXPIRED MINUTE VOLUME_L/min', 'R VENT EXP TIDAL VOLUME_mL',
       'R IP VENT FLOW OBS_L/min', 
       'R GLASGOW COMA SCALE SCORE', 
       'R GOSH IP HEART RATE ECG_beats per minute',
       'R FIO2_%', 'R GOSH IP INSPIRED TIDAL VOLUME_mL',
       'R GOSH VENT RATE OBSERVED', 'R MAP', 'R VENT MAP_cm H2O',
       'R VENT MINUTE VENTILATION_L/min', 
       'R VENT PEEP_cm H2O', 'R GOSH IP HEART RATE PLETHYSMOGRAM',
       'RESPIRATIONS', 'R RICHMOND AGITATION SEDATION SCALE (RASS)',
       'PULSE OXIMETRY_%', 'R GOSH SPO2/FIO2 RATIO',
       'GOSH IP SPONTANEOUS RESP RATE_Breaths per minute']

interesting_cols = np.intersect1d(flowsheet_not_na.columns[interesting_cols], np.array(interesting_column_names))
      
##Plot all of these columns
#Make figure with subplots so that it formats the right size
square_root_length = int(np.ceil(np.sqrt(len(interesting_cols)))) #So we can make the right sized plot
fig, axs = plt.subplots(square_root_length, square_root_length)

#Loop through the different columns we're interested in
for i, j in enumerate(interesting_cols):
    
    #Instantiate the axis and twin axis so that we can plot predictions and variable on same x axis
    ax = axs[i % square_root_length, int(np.floor(i/square_root_length))]
    ax_twin = ax.twinx()
    
    #Plot
    ax.plot(flowsheet_not_na.loc[:, ['taken_datetime', j]].dropna().loc[:,['taken_datetime']],
            flowsheet_not_na.loc[:, ['taken_datetime', j]].dropna().loc[:,[j]], 'b-', label='Original')
    ax_twin.plot(flowsheet_not_na.loc[:, ['taken_datetime', 'prediction']].dropna().loc[:,['taken_datetime']],
            flowsheet_not_na.loc[:, ['taken_datetime', 'prediction']].dropna().loc[:,['prediction']], 'r-', label='Twin')

    #Set the xlim so that are start to finish and match all the way along - will probably want to narrow this down to look at a place where there was a big change
    ax.set_xlim(flowsheet_not_na.taken_datetime[0], flowsheet_not_na.taken_datetime[flowsheet_not_na.index[-1]])
    #ax.get_legend().remove()
    ax_twin.set_xlim(flowsheet_not_na.taken_datetime[0], flowsheet_not_na.taken_datetime[flowsheet_not_na.index[-1]])
    #ax_twin.get_legend().remove()
    
    #Set x and y label. Will probably want to add back title
    ax.set_ylabel(j)
    ax_twin.set_ylabel('prediction')
    
fig.tight_layout()
plt.show()    







# Plan to calculate importance when patients aren't on inotropes
