#!/usr/bin/env python
# Second prediction, after a bit of tweaking, optimized over max_fetaures,
# min_samples_split, implemented rudimentary idf
import lib.loader as ld
import sklearn.cross_validation as cv
import sklearn.ensemble as ens
import sklearn.feature_extraction.text as tfidf

if __name__ == '__main__':
    trainx, trainy = ld.loadtrain('data/trainingdata.txt')
    testx = ld.loadtest('data/testingdata.txt')
    trainx2 = tfidf.TfidfTransformer().fit_transform(trainx)

    clf = ens.RandomForestClassifier(criterion='entropy', n_estimators=100,
            max_features=0.38, min_samples_split=30)
    print(cv.cross_val_score(clf, trainx2, y=trainy, cv=5))
    print(cv.cross_val_score(clf, trainx, y=trainy, cv=5))
    # clf.fit(trainx2, trainy)
    # ld.write('predictions/Prediction2.txt', clf.predict(testx))
