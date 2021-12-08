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
from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error, mean_absolute_error, auc, confusion_matrix, roc_curve, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

#Read in the data
data = np.load('/store/DAMTP/dfs28/PICU_data/np_arrays.npz')
array3d = data['d3']
array2d = data['d2']
outcomes = data['outcomes']
characteristics = data['chars']
splines = data['splines']


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
split_outcomes = test_trainsplit(outcomes, np.array([85, 15]))
train_characteristics = split_characteristics[0]
train_array2d = split_array2d[0]
train_outcomes = split_outcomes[0]
test_characteristics = split_characteristics[1]
test_array2d = split_array2d[1]
test_outcomes = split_outcomes[1]


#
X = np.concatenate((train_array2d, train_characteristics), axis=1)
y_outcomes = train_outcomes[:, 8:11]
y = np.argmax(train_outcomes[:, 8:11], axis = 1)

# grid search
model = XGBClassifier()
kfold = StratifiedKFold(n_splits=4, shuffle=True, random_state=7)
param_grid = {"learning_rate"    : [0.001, 0.01, 0.10, 0.20, 0.30],
 				'max_depth' : [5, 10, 15, 20],
 					"min_child_weight" : [ 1, 5, 7, 15],
 					"gamma"            : [ 0.0, 0.2, 0.4 ],
 					"colsample_bytree" : [ 0.3, 0.5 , 0.7, 1], 
					 "subsample":[0.5, 0.75, 1], 
					 "scale_pos_weight" : [1, 2, 4], 
					 "n_estimators" : [50, 100, 150]}

if os.path.exists('/mhome/damtp/q/dfs28/Project/PICU_project/models/XGBoost_best.json') == False:                         
       clf = RandomizedSearchCV(model, param_grid, random_state=0, cv = kfold)
       search = clf.fit(X, y)

       #Save the best hyperparameters
       best_hyperparameters = search.best_params_

       a_file = open("/mhome/damtp/q/dfs28/Project/PICU_project/models/XGBoost_best.json", "w")
       json.dump(best_hyperparameters, a_file)
       a_file.close()

else:
       f = open("/mhome/damtp/q/dfs28/Project/PICU_project/models/XGBoost_best.json", )
       param_grid = json.load(f)

#get colnames
point_cols = data['point_cols']
point_cols = [i for i in point_cols]
series_cols = data['series_cols']

#Combine the series cols objects and name them for labelling of axes
series_cols_mean = ['Mean ' + i for i in series_cols]
series_cols_std = ['STD ' + i for i in series_cols]
for i in series_cols_std:
    series_cols_mean.append(i)

#Now run and get parameters
model1 = XGBClassifier(param_dict = param_grid)
X_test = np.concatenate((test_array2d, test_characteristics), axis=1)
y_test = np.argmax(test_outcomes[:, 8:11], axis = 1)

#Don't need to rerun 10 times 
accuracy = list()
MSE = list()
AUROC = list()
MAE = list()
Precision = list()
Recall = list()
F1 = list()

#Need to one hot encode
onehot_encoder = sklearn.preprocessing.OneHotEncoder(sparse=False)
integer_encoded_test = y_test.reshape(len(y_test), 1)
onehot_encoded_test = onehot_encoder.fit_transform(integer_encoded_test)

#Run the model
clf1 = model1.fit(X, y)
y_pred = clf1.predict(X_test)
integer_encoded_pred = y_pred.reshape(len(y_pred), 1)
onehot_encoded_pred = onehot_encoder.fit_transform(integer_encoded_pred)

#Save the outcomes
accuracy.append(accuracy_score(y_pred, y_test))
MSE.append(mean_squared_error(y_pred, y_test))
MAE.append(mean_absolute_error(y_pred, y_test))
AUROC.append(roc_auc_score(onehot_encoded_pred, onehot_encoded_test, multi_class = 'ovr', average = 'macro'))
Recall.append(recall_score(onehot_encoded_pred, onehot_encoded_test, average = 'macro'))
Precision.append(precision_score(onehot_encoded_pred, onehot_encoded_test, average = 'macro'))
F1.append(f1_score(onehot_encoded_pred, onehot_encoded_test, average = 'macro'))

#Save to a json file
results = {'acc_PEWS' : np.mean(accuracy),
            'AUC_PEWS' : np.mean(AUROC),
            'MSE_PEWS' : np.mean(MSE),
            'MAE_PEWS' : np.mean(MAE), 
            'precision_PEWS' : np.mean(Precision), 
            'recall_PEWS' : np.mean(Recall), 
            'F1_PEWS' : np.mean(F1)}

a_file = open("/mhome/damtp/q/dfs28/Project/PICU_project/files/XGBoost_results", "w")
json.dump(results, a_file)
a_file.close()

conf_mat1 = confusion_matrix(y_pred, y_test)

#Importance plot
feature_importance1 = clf1.feature_importances_
#Get top 20 most important features
feature_order1 = np.argsort(-1*feature_importance1)
feature_names1 =  point_cols + series_cols_mean
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
fig.savefig('/mhome/damtp/q/dfs28/Project/PICU_project/figs/PICU/XGBoost/XGBoost_importance_deterioration_only.png')

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize = (20, 10))
ax1.set_title('Deterioration')
ax1.bar(top_features_names1, top_feature_values1, color='green')
plt.xticks(rotation='vertical')


#Now for mortality
param_binary = param_grid
param_binary['objective'] = 'binary:logistic'
model2 = XGBClassifier(param_dict = param_binary, use_label_encoder=False)
y_outcomes1 = train_outcomes[:, 2:5]
y1 = np.argmax(train_outcomes[:, 2:5], axis = 1)/2
X_test = np.concatenate((test_array2d, test_characteristics), axis=1)
y_test1 = np.argmax(test_outcomes[:, 2:5], axis = 1)/2


accuracy1 = list()
MSE1 = list()
AUROC1 = list()
MAE1 = list()
onehot_encoder = sklearn.preprocessing.OneHotEncoder(sparse=False)
integer_encoded_test = y_test1.reshape(len(y_test1), 1)
onehot_encoded_test = onehot_encoder.fit_transform(integer_encoded_test)
for i in range(10):
       clf2 = model2.fit(X, y1)
       y_pred1 = clf2.predict(X_test)
       y_pred_proba1 = clf2.predict(X_test)
       integer_encoded_pred = y_pred.reshape(len(y_pred1), 1)
       onehot_encoded_pred = onehot_encoder.fit_transform(integer_encoded_pred)
       accuracy1.append(accuracy_score(y_pred1, y_test1))
       MSE1.append(mean_squared_error(y_pred1, y_test1))
       MAE1.append(mean_absolute_error(y_pred1, y_test1))
       AUROC1.append(roc_auc_score(onehot_encoded_pred, onehot_encoded_test, multi_class = 'ovr', average = 'macro'))

#Use ratio of outcome 3 to 1
y_pred_bin1 = y_pred1[:, (0, 2)]
y_pred_ratio = y_pred1[:, 0]/ y_pred1[:, 2]

#Work out best threshold
fpr, tpr, thresholds = roc_curve(np.argmax(all_test_outcomes[:, (2, 4)], axis = 1), y_pred_ratio)
gmeans = np.sqrt(tpr * (1-fpr))
ix = np.argmax(gmeans)
print('Best Threshold=%f, G-Mean=%.3f' % (thresholds[ix], gmeans[ix]))

#
y_tuned = (y_pred_ratio > thresholds[ix]).astype(int)
confusion_matrix((y_pred_ratio > thresholds[ix]).astype(int), np.argmax(all_test_outcomes[:, 2:5], axis = 1)/2)

feature_importance2 = clf2.feature_importances_
#Get top 20 most important features
feature_order2 = np.argsort(-1*feature_importance2)
feature_names2 =  point_cols + series_cols_mean
top_features_names2 = [feature_names2[i] for i in feature_order2[0:20]]
top_feature_values2 = [feature_importance2[i] for i in feature_order2[0:20]]

ax2.set_title('Mortality')
ax2.bar(top_features_names2, top_feature_values2, color='red')
plt.xticks(rotation='vertical')


#Finally for LOS
model3 = XGBClassifier(param_dict = param_grid)
y_outcomes2 = train_outcomes[:, 5:8]
y2 = np.argmax(train_outcomes[:, 5:8], axis = 1)
X_test = np.concatenate((test_array2d, test_characteristics), axis=1)
y_test2 = np.argmax(test_outcomes[:, 5:8], axis = 1)

accuracy2 = list()
MSE2 = list()
AUROC2 = list()
MAE2 = list()
onehot_encoder = sklearn.preprocessing.OneHotEncoder(sparse=False)
integer_encoded_test = y_test2.reshape(len(y_test2), 1)
onehot_encoded_test = onehot_encoder.fit_transform(integer_encoded_test)
for i in range(1):
       clf3 = model3.fit(X, y2)
       y_pred3 = clf3.predict(X_test)
       integer_encoded_pred = y_pred.reshape(len(y_pred3), 1)
       onehot_encoded_pred = onehot_encoder.fit_transform(integer_encoded_pred)
       accuracy2.append(accuracy_score(y_pred3, y_test2))
       MSE2.append(mean_squared_error(y_pred3, y_test2))
       MAE2.append(mean_absolute_error(y_pred3, y_test2))
       AUROC2.append(roc_auc_score(onehot_encoded_pred, onehot_encoded_test, multi_class = 'ovr', average = 'macro'))

feature_importance3 = clf3.feature_importances_
#Get top 20 most important features
feature_order3 = np.argsort(-1*feature_importance3)
feature_names3 =  point_cols + series_cols_mean
top_features_names3 = [feature_names3[i] for i in feature_order3[0:20]]
top_feature_values3 = [feature_importance3[i] for i in feature_order3[0:20]]

ax3.set_title('Length of stay')
ax3.bar(top_features_names3, top_feature_values3, color='blue')

ax1.set_xticklabels(top_features_names1, rotation = 'vertical')
ax2.set_xticklabels(top_features_names2, rotation = 'vertical')
ax3.set_xticklabels(top_features_names3, rotation = 'vertical')
plt.subplots_adjust(bottom = 0.3)
fig.savefig('/mhome/damtp/q/dfs28/Project/PICU_project/figs/PICU/XGBoost_importance_deterioration.png')


# summarize results
print("Best: %f using %s" % (search.best_score_, search.best_params_))
means = search.cv_results_['mean_test_score']
stds = search.cv_results_['std_test_score']
params = search.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
	print("%f (%f) with: %r" % (mean, stdev, param))

conf_mat1 = confusion_matrix(y_pred, y_test)
conf_mat2 = confusion_matrix(y_pred1, y_test1)
conf_mat3 = confusion_matrix(y_pred3, y_test2)


#Function to do plotting
def plot_ROCs(y_outcomes, x_predicted, classes, title, filename):
       """
       Function to plot ROC for multiclass output
       """

       n_classes = len(classes)

       # Compute ROC curve and ROC area for each class
       fpr = dict()
       tpr = dict()
       roc_auc = dict()
       for i in range(n_classes):
              fpr[i], tpr[i], _ = roc_curve(y_outcomes[:, i], x_predicted[:, i])
              roc_auc[i] = auc(fpr[i], tpr[i])

       # Compute micro-average ROC curve and ROC area
       fpr["micro"], tpr["micro"], _ = roc_curve(y_outcomes.ravel(), x_predicted.ravel())
       roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

       # First aggregate all false positive rates
       all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

       # Then interpolate all ROC curves at this points
       mean_tpr = np.zeros_like(all_fpr)
       for i in range(n_classes):
              mean_tpr += interp(all_fpr, fpr[i], tpr[i])

       # Finally average it and compute AUC
       mean_tpr /= n_classes

       fpr["macro"] = all_fpr
       tpr["macro"] = mean_tpr
       roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

       # Plot all ROC curves
       plt.figure()
       plt.plot(fpr["micro"], tpr["micro"],
              label='micro-average ROC curve (area = {0:0.2f})'
                     ''.format(roc_auc["micro"]),
              color='deeppink', linestyle=':', linewidth=4)

       plt.plot(fpr["macro"], tpr["macro"],
              label='macro-average ROC curve (area = {0:0.2f})'
                     ''.format(roc_auc["macro"]),
              color='navy', linestyle=':', linewidth=4)

       colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
       classes = cycle(classes)
       lw = 2
       for i, color, class_name in zip(range(n_classes), colors, classes):
              plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                     label='{0}: (area = {1:0.2f})'
                     ''.format(class_name, roc_auc[i]))

       plt.plot([0, 1], [0, 1], 'k--', lw=lw)
       plt.xlim([0.0, 1.0])
       plt.ylim([0.0, 1.05])
       plt.xlabel('False Positive Rate')
       plt.ylabel('True Positive Rate')
       plt.title(title)
       plt.legend(loc="lower right")
       filepath = '/mhome/damtp/q/dfs28/Project/PICU_project/figs/PICU/Logistic_regression/' + filename 
       plt.savefig(filepath)



clf.predict(X)
clf.predict_proba(X)
clf.score(X, y)
x_predicted = clf.predict_proba(X)

"""
plot_ROCs(y_outcomes, x_predicted, ["Deterioration in 6h", 'Deterioration 6-24h', 'Deterioration >24h'], 'Plotting ROC for XGBoost: Deterioration', 'XGBoost_deterioration_ROC')

"""