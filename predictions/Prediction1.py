#!/usr/bin/env python
# First prediction, using sklearn's RandomForestClassifier
import lib.loader as ld
import sklearn.ensemble as ens

if __name__ == '__main__':
    trainx, trainy = ld.loadtrain('data/trainingdata.txt')
    testx = ld.loadtest('data/testingdata.txt')
    clf = ens.RandomForestClassifier(criterion='entropy', n_estimators=20)
    clf.fit(trainx, trainy)
    ld.write('predictions/Prediction1.txt', clf.predict(testx))

# Score: 0.52515
