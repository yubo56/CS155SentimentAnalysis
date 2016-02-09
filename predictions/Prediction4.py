#!/usr/bin/env python
# Test submission with LinearSVC using parameter seen online, since cv searching
# isn't producing the right ballpark of parameters...
import numpy as np
import sklearn.svm as svm
import lib.loader as ld
import sklearn.feature_extraction.text as tfidf

if __name__ == '__main__':
    trainx, trainy = ld.loadtrain('data/trainingdata.txt')
    testx = ld.loadtest('data/testingdata.txt')
    trainx2 = tfidf.TfidfTransformer().fit_transform(trainx)
    testx2 = tfidf.TfidfTransformer().fit_transform(testx)

    clf = svm.LinearSVC( C=0.1)
    clf.fit(trainx2, trainy)
    ld.write('predictions/Prediction4.txt', clf.predict(testx2))

# score = 0.51
