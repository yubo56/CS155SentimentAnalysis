#!/usr/bin/env python
# uses GridSearchCV to optimize SVM
import numpy as np
import sklearn.ensemble as ens
import sklearn.grid_search as gs
import lib.loader as ld
import sklearn.feature_extraction.text as tfidf

# training data
trainx, trainy = ld.loadtrain('data/trainingdata.txt')
trainx2 = tfidf.TfidfTransformer().fit_transform(trainx)
parameters = {'max_depth': np.arange(2,7,1), 
        'subsample': np.arange(0.1,1.0,0.1)}
mdl = ens.GradientBoostingClassifier(n_estimators=100, min_samples_split=7)
clf = gs.GridSearchCV(mdl, parameters, n_jobs=-1, cv=5)
clf.fit(trainx2.toarray(), trainy)

# print results
print(clf.best_score_)              # best score
print(clf.best_params_)             # best params
