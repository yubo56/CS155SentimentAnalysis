#!/usr/bin/env python
# prediction 3, with gridsearch'd parameters

import lib.loader as ld
import sklearn.ensemble as ens
import sklearn.feature_extraction.text as tfidf

if __name__ == '__main__':
    ens.RandomForestClassifier(n_jobs=-1)
    trainx, trainy = ld.loadtrain('data/trainingdata.txt')
    testx = ld.loadtest('data/testingdata.txt')
    trainx2 = tfidf.TfidfTransformer().fit_transform(trainx)
    testx2 = tfidf.TfidfTransformer().fit_transform(testx)

    clf = ens.RandomForestClassifier(max_features=0.0420, criterion='entropy',
            n_estimators=5000, min_samples_split=7)
    clf.fit(trainx2, trainy)
    ld.write('predictions/Prediction3.txt', clf.predict(testx2))

# score 0.51
