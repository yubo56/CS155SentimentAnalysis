#!/usr/bin/env python
# uses GridSearchCV to optimize SVM
import numpy as np
import sklearn.svm as svm
import sklearn.grid_search as gs
import lib.loader as ld
import sklearn.feature_extraction.text as tfidf

# training data
trainx, trainy = ld.loadtrain('data/trainingdata.txt')
trainx2 = tfidf.TfidfTransformer().fit_transform(trainx)
parameters = {'C': np.arange(1, 101, 5),
        'penalty': ['l2', 'l1']}
mdl = svm.LinearSVC(dual=False)
clf = gs.GridSearchCV(mdl, parameters, n_jobs=-1, cv=5)
clf.fit(trainx2, trainy)

# print results
print(clf.best_score_)              # best score
print(clf.best_params_)             # best params
