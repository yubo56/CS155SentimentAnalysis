#!/usr/bin/env python
import numpy as np
from pickle import Pickler
from random import randint

import lib.loader as ld
import lib.composite as c

import sklearn.ensemble as ens
import sklearn.tree as tree
import sklearn.svm as svm
import sklearn.linear_model as lm
import sklearn.feature_extraction.text as tfidf

if __name__ == '__main__':
    min_changes = 5
    trainx, trainy = ld.loadtrain('data/trainingdata.txt')
    testx = ld.loadtest('data/testingdata.txt')
    trainx = tfidf.TfidfTransformer().fit_transform(trainx).toarray()
    testx = tfidf.TfidfTransformer().fit_transform(testx).toarray()
    comp = c.Composite(trainx, trainy, testx)
    p = Pickler(open('predictions/comp.learn', 'wb'), -1)

    # loop until few changes
    numChange = len(testx)
    while numChange > min_changes:
        rand = randint(0, 8)
        if rand == 0:
            numChange = comp.append(ens.AdaBoostClassifier(n_estimators=100,
                learning_rate=0.03))
        elif rand == 1:
            numChange = comp.append(ens.BaggingClassifier(n_estimators=100,
                max_features=0.5, max_samples=0.7, n_jobs=-1))
        elif rand == 2:
            numChange =\
            comp.append(tree.DecisionTreeClassifier(max_features=0.64,
                min_samples_split=5))
        elif rand == 3:
            numChange =\
            comp.append(ens.GradientBoostingClassifier(n_estimators=100,
                min_samples_split=7, max_depth=6, subsample=0.7))
        elif rand == 4:
            numChange = comp.append(ens.RandomForestClassifier(n_estimators=450,
                max_features=0.0420, min_samples_split=7, njobs=-1))
        elif rand == 5:
            numChange = comp.append(lm.SGDClassifier(loss='hinge',
                alpha=0.000631))
        elif rand == 6:
            numChange = comp.append(svm.LinearSVC(dual=False, C=0.01))
        elif rand == 7:
            numChange = comp.append(ens.RandomForestClassifier(n_estimators=100,
                max_features=0.320, min_samples_split=7, n_jobs=-1))
        elif rand == 8:
            numChange = comp.append(svm.LinearSVC(dual=False, C=1000))
        ld.write('predictions/Prediction6.txt', comp.predict())
        print('Number Changes: ' + str(numChange))
        p.dump(comp)

