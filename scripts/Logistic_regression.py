## Script for running logistic regression on the data

import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


#Read in the data
data = np.load('/store/DAMTP/dfs28/PICU_data/np_arrays.npz')
array2d = data['d2']
outcomes = data['outcomes']
characteristics = data['chars']
splines = data['splines']

#Combine inputs:
X = np.concatenate((array2d, characteristics), axis=1)
y = np.argmax(outcomes[:, 2:5], axis=1)

clf = LogisticRegression(random_state=0).fit(X, y)
clf.predict(X)
clf.predict_proba(X[:2, :])
array([[9.8...e-01, 1.8...e-02, 1.4...e-08],
       [9.7...e-01, 2.8...e-02, ...e-08]])
clf.score(X, y)
0.97...
