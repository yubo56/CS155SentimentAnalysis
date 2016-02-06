#!/usr/bin/env python
# runs cross validation with various built-in sklearn models
import lib.loader as ld
import sklearn.linear_model as lm           # SGDClassifier (hinge, log),
import sklearn.cross_validation as cv       # cross validation
import numpy as np                          # np.mean()

if __name__ == '__main__':
    # load data
    trainx, trainy = ld.loadtrain('data/trainingdata.txt')
    testx = ld.loadtest('data/trainingdata.txt')

    # set up classifier(s)
    clfhinge = lm.SGDClassifier(loss='hinge')    # default hinge
    clflog = lm.SGDClassifier(loss='log')    # default hinge

    # get cv scores
    scoreshinge = cv.cross_val_score(clfhinge, trainx, trainy, cv=5)
    scoreslog = cv.cross_val_score(clflog, trainx, trainy, cv=5)
    # print(scoreshinge)
    # print(scoreslog)
    print("Hinge CV average: " + str(np.mean(scoreshinge)) + "\t Log CV average: " + str(np.mean(scoreslog)))
