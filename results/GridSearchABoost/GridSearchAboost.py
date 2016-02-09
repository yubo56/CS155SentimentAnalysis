#!/usr/bin/env python
# uses GridSearchCV to optimize SVM
import numpy as np
import sklearn.ensemble as ens
import sklearn.tree as tree
import sklearn.svm as svm
import sklearn.grid_search as gs
import lib.loader as ld
import sklearn.feature_extraction.text as tfidf

# training data
trainx, trainy = ld.loadtrain('data/trainingdata.txt')
trainx2 = tfidf.TfidfTransformer().fit_transform(trainx)
parameters = {'learning_rate': np.arange(0.01,0.21,0.02)}
mdl = ens.AdaBoostClassifier(n_estimators=100,
        base_estimator=tree.DecisionTreeClassifier())
clf = gs.GridSearchCV(mdl, parameters, n_jobs=-1, cv=5)
clf.fit(trainx2, trainy)

# print results
print(clf.best_score_)              # best score
print(clf.best_params_)             # best params
