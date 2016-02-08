#!/usr/bin/env python
# tests RandomForest for max_features in [0.1,1]
import numpy as np
import matplotlib.pyplot as plt
import sklearn.cross_validation as cv
import sklearn.ensemble as ens
import lib.loader as ld

numtests = 30
numavg = 5
numcv = 5
vals = np.linspace(5, 2 * numtests, numtests)
results = np.zeros([len(vals), numavg, numcv])

# classifiers
clfs = [ ens.RandomForestClassifier(n_estimators=50, min_samples_split = i) for i in vals ]
trainx, trainy = ld.loadtrain('data/trainingdata.txt')

for i in range(numtests):
    for j in range(numavg):
        results[i][j] = cv.cross_val_score(clfs[i], trainx, y=trainy, cv=numcv)

np.save('results/testRFSplit', results)
means = list()
stdevs = list()
for i in range(numtests):
    means.append(np.mean(np.concatenate([j for j in results[i]])))
    stdevs.append(np.std(np.concatenate([j for j in results[i]])))
    print('Val: ' + str(vals[i]) + '\t Mean: ' + str(means[i]) + '\tStDev: ' 
            + str(stdevs[i]))

plt.plot(vals, means)
plt.title('Means of CV scores')
plt.savefig('results/testRFSplitMeans.png')
plt.clf()
plt.plot(vals, stdevs)
plt.title('StDevs of CV scores')
plt.savefig('results/testRFSplitStDevs.png')
