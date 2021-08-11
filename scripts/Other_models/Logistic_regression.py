## Script for running logistic regression on the data

import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve, auc, confusion_matrix
from itertools import cycle
from scipy import interp
import matplotlib.pyplot as plt
import os
import pandas as pd
from sklearn.model_selection import KFold, cross_validate, cross_val_predict

#Read in the data
data = np.load('/store/DAMTP/dfs28/PICU_data/np_arrays.npz')
array3d = data['d3']
array2d = data['d2']
outcomes = data['outcomes']
characteristics = data['chars']
splines = data['splines']
n_classes = 3


binary_deterioration = np.array((outcomes[:, 9], np.argmax(outcomes[:, 9:11], axis = 1)))
y4 = np.argmax(binary_deterioration, axis = 0)

"""
To do: include splines, lasso/ l1 penalised logit, ROC, accuracy for everything
Possibly for fairness need results just from the test set
Changing thresholds thing
Somehow get some sort of confidence that s.thing is going to happen (for usefulness sake)
Use pews score - just regress PEWS against outcomes?
"""

 
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


### Logistic regression for mean, sd, and 2d array
#Combine inputs:
X = np.concatenate((array2d, characteristics), axis=1)
X_splines = np.concatenate((X, splines), axis = 1)
y1_outcomes = outcomes[:, 2:5]
y1 = np.argmax(outcomes[:, 2:5], axis = 1)
y1_test = test_trainsplit(y1, np.array([85, 15]))[1]
y1_train = test_trainsplit(y1, np.array([85, 15]))[0]
X_test = test_trainsplit(X, np.array([85, 15]))[1]
X_train = test_trainsplit(X, np.array([85, 15]))[0]

y2_outcomes = outcomes[:, 5:8]
y2 = np.argmax(outcomes[:, 5:8], axis = 1)
y2_test = test_trainsplit(y2, np.array([85, 15]))[1]
y2_train = test_trainsplit(y2, np.array([85, 15]))[0]

y3_outcomes = outcomes[:, 8:11]
y3 = np.argmax(outcomes[:, 8:11], axis = 1)
y3_test = test_trainsplit(y3, np.array([85, 15]))[1]
y3_train = test_trainsplit(y3, np.array([85, 15]))[0]

kfold = KFold(n_splits=10, shuffle=False)
clf_none = LogisticRegression(random_state=0, max_iter= 100, multi_class='multinomial', solver='lbfgs', penalty = 'none')
scores1_none = cross_validate(clf_none, X, y1, scoring = ['roc_auc_ovr', 'accuracy', 'neg_mean_squared_error', 'neg_mean_absolute_error'], cv = kfold)
scores2_none = cross_validate(clf_none, X, y2, scoring = ['roc_auc_ovr', 'accuracy', 'neg_mean_squared_error', 'neg_mean_absolute_error'], cv = kfold)
scores3_none = cross_validate(clf_none, X, y3, scoring = ['roc_auc_ovr', 'accuracy', 'neg_mean_squared_error', 'neg_mean_absolute_error'], cv = kfold)
scores4_none = cross_validate(clf_none, X, y4, scoring = ['roc_auc_ovr', 'accuracy', 'neg_mean_squared_error', 'neg_mean_absolute_error'], cv = kfold)

means1_none = [np.mean(scores1_none[i]) for i in scores1_none.keys()]
stds1_none = [np.std(scores1_none[i]) for i in scores1_none.keys()]
means2_none = [np.mean(scores2_none[i]) for i in scores2_none.keys()]
stds2_none = [np.std(scores2_none[i]) for i in scores2_none.keys()]
means3_none = [np.mean(scores3_none[i]) for i in scores3_none.keys()]
stds3_none = [np.std(scores3_none[i]) for i in scores3_none.keys()]
means4_none = [np.mean(scores4_none[i]) for i in scores4_none.keys()]
stds4_none = [np.std(scores4_none[i]) for i in scores4_none.keys()]

#Now do Ridge
clf_l2 = LogisticRegression(random_state=0, max_iter= 1000, multi_class='multinomial', solver='lbfgs', penalty = 'l2')
clf_l2_1 = LogisticRegression(random_state=0, max_iter= 1000, multi_class='multinomial', solver='lbfgs', penalty = 'l2')
clf_l2_2 = LogisticRegression(random_state=0, max_iter= 1000, multi_class='multinomial', solver='lbfgs', penalty = 'l2')
clf_l2_3 = LogisticRegression(random_state=0, max_iter= 1000, multi_class='multinomial', solver='lbfgs', penalty = 'l2')
clf_l2_1.fit(X_train, y1_train)
clf_l2_2.fit(X_train, y2_train)
clf_l2_3.fit(X_train, y3_train)
scores1_l2 = cross_validate(clf_l2 , X, y1, scoring = ['roc_auc_ovr', 'accuracy', 'neg_mean_squared_error', 'neg_mean_absolute_error', 'f1_score', 'precision', ''], cv = kfold, return_estimator =True)
scores2_l2 = cross_validate(clf_l2 , X, y2, scoring = ['roc_auc_ovr', 'accuracy', 'neg_mean_squared_error', 'neg_mean_absolute_error'], cv = kfold, return_estimator =True)
scores3_l2 = cross_validate(clf_l2 , X, y3, scoring = ['roc_auc_ovr', 'accuracy', 'neg_mean_squared_error', 'neg_mean_absolute_error'], cv = kfold, return_estimator =True)
means1_l2 = [np.mean(scores1_l2[i]) for i in scores1_l2.keys()]
stds1_l2 = [np.std(scores1_l2[i]) for i in scores1_l2.keys()]
means2_l2 = [np.mean(scores2_l2[i]) for i in scores2_l2.keys()]
stds2_l2 = [np.std(scores2_l2[i]) for i in scores2_l2.keys()]
means3_l2 = [np.mean(scores3_l2[i]) for i in scores3_l2.keys()]
stds3_l2 = [np.std(scores3_l2[i]) for i in scores3_l2.keys()]
means4_l2 = [np.mean(scores4_l2[i]) for i in scores4_l2.keys()]
stds4_l2 = [np.std(scores4_l2[i]) for i in scores4_l2.keys()]

#Now get the optimised confusion matrix
#Use ratio of outcome 3 to 1
y_pred1 = clf_l2_1.predict_proba(X_test)
y_pred_bin1 = y_pred1[:, (0, 2)]
y_pred_ratio = y_pred1[:, 0]/ y_pred1[:, 1]

#Work out best threshold
fpr, tpr, thresholds = roc_curve(np.argmax(all_test_outcomes[:, (2, 4)], axis = 1), y_pred_ratio)
gmeans = np.sqrt(tpr * (1-fpr))
ix = np.argmax(gmeans)
print('Best Threshold=%f, G-Mean=%.3f' % (thresholds[ix], gmeans[ix]))

#
y_tuned = (y_pred_ratio > thresholds[ix]).astype(int)
confusion_matrix((y_pred_ratio > thresholds[ix]).astype(int), np.argmax(all_test_outcomes[:, 2:5], axis = 1)/2)

#Now do it with no measure of variability
scores1_l2_nostd = cross_validate(clf_l2 , X[:,:113], y1, scoring = ['roc_auc_ovr', 'accuracy', 'neg_mean_squared_error', 'neg_mean_absolute_error'], cv = kfold, return_estimator =True)
scores2_l2_nostd = cross_validate(clf_l2 , X[:,:113], y2, scoring = ['roc_auc_ovr', 'accuracy', 'neg_mean_squared_error', 'neg_mean_absolute_error'], cv = kfold, return_estimator =True)
scores3_l2_nostd = cross_validate(clf_l2 , X[:,:113], y3, scoring = ['roc_auc_ovr', 'accuracy', 'neg_mean_squared_error', 'neg_mean_absolute_error'], cv = kfold, return_estimator =True)
y_predict1 = cross_val_predict(clf_l2, X_test, y1, cv=10)
y_predict2 = cross_val_predict(clf_l2, X, y2, cv=10)
y_predict3 = cross_val_predict(clf_l2, X, y3, cv=10)

#Ridge without stds?
means1_l2_nostd = [np.mean(scores1_l2_nostd[j]) for i, j in enumerate(scores1_l2_nostd.keys()) if i in range(3, 7)]
stds1_l2_nostd = [np.std(scores1_l2_nostd[j]) for i, j in enumerate(scores1_l2_nostd.keys()) if i in range(3, 7)]
means2_l2_nostd = [np.mean(scores2_l2_nostd[j]) for i, j in enumerate(scores2_l2_nostd.keys()) if i in range(3, 7)]
stds2_l2_nostd = [np.std(scores2_l2_nostd[j]) for i, j in enumerate(scores2_l2_nostd.keys()) if i in range(3, 7)]
means3_l2_nostd = [np.mean(scores3_l2_nostd[j]) for i, j in enumerate(scores3_l2_nostd.keys()) if i in range(3, 7)]
stds3_l2_nostd = [np.std(scores3_l2_nostd[j]) for i, j in enumerate(scores3_l2_nostd.keys()) if i in range(3, 7)]


#Survival
feature_importance1 = clf_l2_1.coef_
sum_squared_features1 = np.sum(feature_importance1**2, axis = 0)
feature_order1 = np.argsort(-1*sum_squared_features1)
feature_names1 =  point_cols + series_cols_mean
top_features_names1 = [feature_names1[i] for i in feature_order1[0:20]]
top_feature_values1 = np.take(feature_importance1, feature_order1[0:20], axis = 1)
feature_importance = pd.DataFrame(top_features_names1, columns = ['names'])
feature_importance['0'] = top_feature_values1[0, :]
feature_importance['1'] = top_feature_values1[1, :]
feature_importance['2'] = top_feature_values1[2, :]
feature_importance = feature_importance.melt(id_vars=["names"], var_name = 'Outcome')
plt.figure(figsize=(10, 10))
sns.barplot(x="names", hue="Outcome", y="value", data=feature_importance)
plt.xticks(rotation=90)
plt.xlabel('Feature')
plt.ylabel('Beta value')
plt.title('Logistic regression features')
plt.subplots_adjust(bottom = 0.44)
plt.savefig('/mhome/damtp/q/dfs28/Project/PICU_project/figs/PICU/L2_features.png')

fig, ax1 = plt.subplots(1, 1, figsize = (20, 10))
plt.xticks(rotation=90)
#Survival
feature_importance1 = clf_l2_1.coef_
#Get top 20 most important features
sum_squared_features1 = np.sum(feature_importance1**2, axis = 0)
feature_order1 = np.argsort(-1*sum_squared_features1)
feature_names1 =  point_cols + series_cols_mean
top_features_names1 = [feature_names1[i] for i in feature_order1[0:20]]
top_feature_values1 = np.take(feature_importance1, feature_order1[0:20], axis = 1)
x_values = np.linspace(1, 20, num = 20)
ax1.bar(x_values, top_feature_values1[1, ], width=0.2, color='g', align='center')
ax1.set_xticklabels(top_features_names1, rotation = 'vertical')
ax1.bar(x_values-0.2, top_feature_values1[0, ], width=0.2, color='b', align='center')
ax1.bar(x_values+0.2, top_feature_values1[2, ], width=0.2, color='r', align='center')
ax1.set_title('Logistic regression features')

fig.savefig('/mhome/damtp/q/dfs28/Project/PICU_project/figs/PICU/L2_features.png')


#Confusion matrices just with test set
y_predict_test1 = clf_l2_1.predict(X_test)
y_predict_test2 = clf_l2_2.predict(X_test)
y_predict_test3 = clf_l2_3.predict(X_test)
conf_mat1 = confusion_matrix(y_predict_test1*2, y1_test)
conf_mat2 = confusion_matrix(y_predict_test2, y2_test)
conf_mat3 = confusion_matrix(y_predict_test3, y3_test)

means1_l2 = [np.mean(scores1_l2[i]) for i in scores1_l2.keys()]
stds1_l2 = [np.std(scores1_l2[i]) for i in scores1_l2.keys()]
means2_l2 = [np.mean(scores2_l2[i]) for i in scores2_l2.keys()]
stds2_l2 = [np.std(scores2_l2[i]) for i in scores2_l2.keys()]
means3_l2 = [np.mean(scores3_l2[i]) for i in scores3_l2.keys()]
stds3_l2 = [np.std(scores3_l2[i]) for i in scores3_l2.keys()]

#Now do the above with the splines
clf_splines = LogisticRegression(random_state=0, max_iter= 100, multi_class='multinomial', solver='lbfgs', penalty = 'none')
scores1_splines = cross_validate(clf_splines, X_splines, y1, scoring = ['roc_auc_ovr', 'accuracy', 'neg_mean_squared_error', 'neg_mean_absolute_error'], cv = kfold)
scores2_splines = cross_validate(clf_splines, X_splines, y2, scoring = ['roc_auc_ovr', 'accuracy', 'neg_mean_squared_error', 'neg_mean_absolute_error'], cv = kfold)
scores3_splines = cross_validate(clf_splines, X_splines, y3, scoring = ['roc_auc_ovr', 'accuracy', 'neg_mean_squared_error', 'neg_mean_absolute_error'], cv = kfold)

means1_splines = [np.mean(scores1_splines[i]) for i in scores1_splines.keys()]
stds1_splines = [np.std(scores1_splines[i]) for i in scores1_splines.keys()]
means2_splines = [np.mean(scores2_splines[i]) for i in scores2_splines.keys()]
stds2_splines = [np.std(scores2_splines[i]) for i in scores2_splines.keys()]
means3_splines = [np.mean(scores3_splines[i]) for i in scores3_splines.keys()]
stds3_splines = [np.std(scores3_splines[i]) for i in scores3_splines.keys()]

#Now do Ridge
clf_splines_l2 = LogisticRegression(random_state=0, max_iter= 100, multi_class='multinomial', solver='lbfgs', penalty = 'l2')
scores1_splines_l2 = cross_validate(clf_splines_l2, X_splines, y1, scoring = ['roc_auc_ovr', 'accuracy', 'neg_mean_squared_error', 'neg_mean_absolute_error'], cv = kfold)
scores2_splines_l2 = cross_validate(clf_splines_l2, X_splines, y2, scoring = ['roc_auc_ovr', 'accuracy', 'neg_mean_squared_error', 'neg_mean_absolute_error'], cv = kfold)
scores3_splines_l2 = cross_validate(clf_splines_l2, X_splines, y3, scoring = ['roc_auc_ovr', 'accuracy', 'neg_mean_squared_error', 'neg_mean_absolute_error'], cv = kfold)

means1_splines_l2 = [np.mean(scores1_splines_l2[i]) for i in scores1_splines_l2.keys()]
stds1_splines_l2 = [np.std(scores1_splines_l2[i]) for i in scores1_splines_l2.keys()]
means2_splines_l2 = [np.mean(scores2_splines_l2[i]) for i in scores2_splines_l2.keys()]
stds2_splines_l2 = [np.std(scores2_splines_l2[i]) for i in scores2_splines_l2.keys()]
means3_splines_l2 = [np.mean(scores3_splines_l2[i]) for i in scores3_splines_l2.keys()]
stds3_splines_l2 = [np.std(scores3_splines_l2[i]) for i in scores3_splines_l2.keys()]

#Should probably build predicted using composite
plot_ROCs(y_outcomes, x_predicted, ["Discharge in <2 days", 'Discharge in 2-7days', 'Discharge >7 days'], 'Plotting ROC for Logistic regression: Discharge', 'Logit_discharge_ROC')

#Logit for death
X = np.concatenate((array2d, characteristics), axis=1)
y_outcomes = outcomes[:, (2, 4)]
y = np.argmax(outcomes[:, 2:5], axis = 1)

#do cv
clf = LogisticRegression(random_state=0, max_iter= 100, multi_class='multinomial', solver='lbfgs').fit(X, y)
clf.predict(X)
clf.predict_proba(X)
clf.score(X, y)
x_predicted = clf.predict_proba(X)


# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(2):
    fpr[i], tpr[i], _ = roc_curve(y_outcomes[:, i], x_predicted[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_outcomes[:, i].ravel(), x_predicted[:, i].ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

plt.figure()
lw = 2
plt.plot(fpr[2], tpr[2], color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[2])
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Plotting ROC for Logistic regression: Death')
plt.legend(loc="lower right")
plt.savefig('/mhome/damtp/q/dfs28/Project/PICU_project/figs/PICU/Logistic_regression/ROC_death')


#Logit for Deterioration
X = np.concatenate((array2d, characteristics), axis=1)
y_outcomes = outcomes[:, 8:11]
y = np.argmax(outcomes[:, 8:11], axis = 1)

clf = LogisticRegression(random_state=0, max_iter= 100, multi_class='multinomial', solver='lbfgs').fit(X, y)
clf.predict(X)
clf.predict_proba(X)
clf.score(X, y)
x_predicted = clf.predict_proba(X)

plot_ROCs(y_outcomes, x_predicted, ["Deterioration in <6h", 'Deterioration 6-24h', 'No deterioration <24H'], 'Plotting ROC for Logistic regression: Death', 'Logit_death_ROC')

#### Should think about precision recall curves

## Input PEWS into logistic regression
series_cols = data['series_cols']
PEWS = array3d[:, series_cols == 'PEWS', :]
PEWS = PEWS.reshape(PEWS.shape[0], PEWS.shape[2])
last_PEWS = PEWS[:, -1:]
clf_PEWS = LogisticRegression(random_state=0, max_iter= 100, multi_class='multinomial', solver='lbfgs', penalty = 'l2')
scores1_last_PEWS_death = cross_validate(clf_PEWS, last_PEWS, np.argmax(outcomes[:, 2:5], axis = 1) , scoring = ['roc_auc_ovr', 'accuracy', 'neg_mean_squared_error', 'neg_mean_absolute_error'], cv = 10)
scores1_last_PEWS_LOS = cross_validate(clf_PEWS, last_PEWS, np.argmax(outcomes[:, 5:8], axis = 1) , scoring = ['roc_auc_ovr', 'accuracy', 'neg_mean_squared_error', 'neg_mean_absolute_error'], cv = 10)
scores1_last_PEWS_PEWS = cross_validate(clf_PEWS, last_PEWS, np.argmax(outcomes[:, 8:11], axis = 1) , scoring = ['roc_auc_ovr', 'accuracy', 'neg_mean_squared_error', 'neg_mean_absolute_error'], cv = 10)

y_predict1 = cross_val_predict(clf_l2, X_test, y1, cv=10)

means1_last_PEWS_death = [np.mean(scores1_last_PEWS_death[i]) for i in scores1_last_PEWS_death.keys()]
stds1_last_PEWS_death = [np.std(scores1_last_PEWS_death[i]) for i in scores1_last_PEWS_death.keys()]
means2_last_PEWS_LOS = [np.mean(scores1_last_PEWS_LOS[i]) for i in scores1_last_PEWS_LOS.keys()]
stds2_last_PEWS_LOS = [np.std(scores1_last_PEWS_LOS[i]) for i in scores1_last_PEWS_LOS.keys()]
means3_last_PEWS_PEWS = [np.mean(scores1_last_PEWS_PEWS[i]) for i in scores1_last_PEWS_PEWS.keys()]
stds3_last_PEWS_PEWS = [np.std(scores1_last_PEWS_PEWS[i]) for i in scores1_last_PEWS_PEWS.keys()]