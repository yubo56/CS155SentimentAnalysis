#!/usr/bin/env python
import numpy as np
from pickle import Pickler
import random as r

import lib.loader as ld
import lib.composite as c

import sklearn.ensemble as ens
import sklearn.tree as tree
import sklearn.svm as svm
import sklearn.linear_model as lm
import sklearn.feature_extraction.text as tfidf

if __name__ == '__main__':
    min_changes = 3
    trainx, trainy = ld.loadtrain('data/trainingdata.txt')
    testx = ld.loadtest('data/testingdata.txt')
    trainx = tfidf.TfidfTransformer().fit_transform(trainx).toarray()
    testx = tfidf.TfidfTransformer().fit_transform(testx).toarray()
    comp = c.Composite(trainx, trainy, testx)
    p = Pickler(open('predictions/comp.learn', 'wb'), -1)

    # loop until few changes
    numChange = len(testx)
    while True:
        rand = r.randint(0, 8)
        if rand == 0:
            numChange = comp.append(ens.AdaBoostClassifier(n_estimators=100,
                learning_rate=0.01 + 0.04 * r.random())) # 0.03 +- 0.02
        elif rand == 1:
            numChange = comp.append(ens.BaggingClassifier(n_estimators=100,
                max_features=0.3 + 0.4 * r.random(), max_samples=0.5 + 0.4 *
                r.random(), n_jobs=-1))
                # 0.5 +- 0.2, 0.7 +- 0.2
        elif rand == 2:
            numChange =\
            comp.append(tree.DecisionTreeClassifier(max_features=0.44 + 0.4 *
                r.random(), min_samples_split=5))
                # 0.64 +- 0.2
        elif rand == 3:
            numChange =\
            comp.append(ens.GradientBoostingClassifier(n_estimators=100,
                min_samples_split=7, max_depth=6, subsample=0.4 + 0.6 *
                r.random()))
                # 0.7 +- 0.3
        elif rand == 4:
            numChange = comp.append(ens.RandomForestClassifier(n_estimators=450,
                max_features=0.0020 + 0.2 * r.random(), min_samples_split=7,
                n_jobs=-1))
                # 0.0420 + 0.16 - 0.04
        elif rand == 5:
            numChange = comp.append(lm.SGDClassifier(loss='hinge',
                alpha=0.000631 + 0.01 * r.random()))
                # 0.000631 + 0.01
        elif rand == 6:
            numChange = comp.append(svm.LinearSVC(dual=False, C=0.01 + 0.1 *
                r.random()))
                # 0.01 + 0.1
        elif rand == 7:
            numChange = comp.append(ens.RandomForestClassifier(n_estimators=100,
                max_features=0.120 + 0.4 * r.random(), min_samples_split=7,
                n_jobs=-1))
                # 0.32 +- 0.2
        elif rand == 8:
            numChange = comp.append(svm.LinearSVC(dual=False, C=100 + 900 *
                r.random()))
                # 1000 - 900
        ld.write('predictions/Prediction6.txt', comp.predict())
        print('Number Changes: ' + str(numChange))
        p.dump(comp.clfs)

# 2289 added to comp before out of memory
# didn't pickle properly, haven't rerun, ref. Prediction7.py
