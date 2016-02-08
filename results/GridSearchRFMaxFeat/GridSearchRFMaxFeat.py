#!/usr/bin/env python
# uses GridSearchCV to optimize RandomForestClassifier
import numpy as np
import sklearn.ensemble as ens
import sklearn.grid_search as gs
import lib.loader as ld
import sklearn.feature_extraction.text as tfidf

# training data
trainx, trainy = ld.loadtrain('data/trainingdata.txt')
trainx2 = tfidf.TfidfTransformer().fit_transform(trainx)
parameters = {'max_features': np.arange(0.04, 0.051, 0.0005),
        'min_samples_split': np.arange(5, 12, 1),
        'n_estimators': np.arange(100, 501, 50)}
mdl = ens.RandomForestClassifier(n_jobs=-1)
clf = gs.GridSearchCV(mdl, parameters)
clf.fit(trainx2, trainy)

# print results
print(clf.best_score_)              # best score
print(clf.best_params_)             # best params
print(abs(clf.best_score_ - i.mean_validation_score) < 0.00001 for i in
    clf.grid_scores_)               # should print out best grid score?
