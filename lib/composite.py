#!/usr/bin/env python
import numpy as np
class Composite(object):
    """
    Composite predictor
    """
    def __init__(self, trainx, trainy, testx):
        super(Composite, self).__init__()
        assert len(trainx) == len(trainy)
        assert len(trainx[0]) == len(testx[0])
        self.__trainx = trainx
        self.__trainy = trainy
        self.__testx = testx
        self.__preds = np.array([0] * len(testx))       # store private copy
        self.clfs = list()
        self.preds = np.array([0] * len(testx))         # public copy, in [0,1]

    def append(self, clf):
        """
        trains and updates score
        appends model to Composite
        :return: number of changes
        """
        clf.fit(self.__trainx, self.__trainy)

        # change predictions and append
        self.clfs.append(clf)

        old_preds = list(self.preds)
        print("\tNumber classifiers: " + str(len(self.clfs)) + "\t Last type: "
                + type(clf).__name__)
        self.__preds += clf.predict(self.__testx)
        self.preds = self.__preds // (len(self.clfs) // 2 + 1)
        # print('\tMax: ' + str(max(self.preds)) + '\tMin: ' +
        #         str(min(self.preds)))

        new_preds = list(self.preds)
        return sum([old_preds[i] != new_preds[i] for i in range(len(old_preds))])

    def predict(self):
        """
        returns own prediction
        """
        return self.preds # returns 1 where majority said 1
    
    def score(self, datax, datay):
        """
        scores self on new dataset
        """
        return np.mean([i.score(datax, datay) for i in self.clfs])
