#!/usr/bin/env python
# runs cross validation with various built-in sklearn models
import lib.loader as ld
import sklearn.linear_model as lm           # SGDClassifier (hinge, log),
import sklearn.tree as tree                 # DecisionTreeClassifier
import sklearn.ensemble as ens              # RandomForestClassifier
                                            # BaggingClassifier
                                            # AdaBoostClassifier
import sklearn.svm as svm                   # SVC
                                            # LinearSVC
import sklearn.cross_validation as cv       # cross validation
import numpy as np                          # np.mean()

if __name__ == '__main__':
    # load data
    trainx, trainy = ld.loadtrain('data/trainingdata.txt')

    # set up classifiers
    clfs = [ lm.SGDClassifier(loss='hinge'),            # hinge loss
        lm.SGDClassifier(loss='log'),                   # log loss
        tree.DecisionTreeClassifier(min_samples_leaf=3),# decision tree
        ens.RandomForestClassifier(criterion='entropy'),# random forest
        ens.BaggingClassifier(),                        # bagging
        ens.AdaBoostClassifier(),
        svm.SVC(),
        svm.LinearSVC()
        ]

    # get cv scores
    scores = [cv.cross_val_score(i, trainx, trainy, cv=5) for i in clfs]

    # print scores
    print("Hinge CV average: " + str(np.mean(scores[0])))
    print("Log CV average: " + str(np.mean(scores[1])))
    print("Decision Tree average: " + str(np.mean(scores[2])))
    print("Random Forest average: " + str(np.mean(scores[3])))
    print("Bagging average: " + str(np.mean(scores[4])))
    print("AdaBoost average: " + str(np.mean(scores[5])))
    print("SVM average: " + str(np.mean(scores[6])))
    print("Linear SVM average: " + str(np.mean(scores[6])))
