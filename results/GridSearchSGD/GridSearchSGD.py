#!/usr/bin/env python
# uses GridSearchCV to optimize SVM
import numpy as np
import sklearn.linear_model as lm
import sklearn.grid_search as gs
import lib.loader as ld
import sklearn.feature_extraction.text as tfidf

# training data
trainx, trainy = ld.loadtrain('data/trainingdata.txt')
trainx2 = tfidf.TfidfTransformer().fit_transform(trainx)
parameters = {'alpha': [10**i for i in np.arange(-5, -2, 0.2)], 
        'loss': ['hinge', 'log']}
mdl = lm.SGDClassifier()
clf = gs.GridSearchCV(mdl, parameters, n_jobs=-1, cv=5)
clf.fit(trainx2.toarray(), trainy)

# print results
print(clf.best_score_)              # best score
print(clf.best_params_)             # best params

parameters = {'alpha': [10**i for i in np.arange(-3, 0, 0.2)], 
        'loss': ['hinge', 'log']}
mdl = lm.SGDClassifier()
clf = gs.GridSearchCV(mdl, parameters, n_jobs=-1, cv=5)
clf.fit(trainx2.toarray(), trainy)

# print results
print(clf.best_score_)              # best score
print(clf.best_params_)             # best params
