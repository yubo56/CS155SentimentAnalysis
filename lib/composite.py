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


# this should really extend Composite but I'm lazy
class CompositeCV(object):
    """
    Composite predictor

    Only adds if improves cv score
    """
    def __init__(self, trainx, trainy, testx, cv=5):
        super(CompositeCV, self).__init__()
        assert len(trainx) == len(trainy)
        assert len(trainx[0]) == len(testx[0])
        self.__trainx = trainx
        self.__trainy = trainy
        self.__testx = testx
        self.__preds = np.array([0] * len(testx))       # store private copy
        self.clfs = list()
        self.preds = np.array([0] * len(testx))         # public copy, in [0,1]

        # set up to do cv
        self.__cv = list()
        lendat = len(self.__trainx)
        for i in range(cv):
            trnx = self.__trainx[lendat // cv * 0: lendat // cv * i]
            trnx = np.concatenate((trnx, 
                self.__trainx[lendat // cv * (i+1):]))
            trny = self.__trainy[lendat // cv * 0: lendat // cv * i]
            trny = np.concatenate((trny, 
                self.__trainy[lendat // cv * (i+1):]))
            valx = self.__trainx[lendat // cv * i: lendat // cv * (i+1)]
            self.__cv.append((trnx, trny, valx))
        self.__cvpreds = np.array([0] * (lendat // cv * cv)) # cv predictions

    @staticmethod
    def diff(x, y):
        """
        helper, counts differences between x, y
        truncates to len x
        """
        return sum([x[i] != y[i] for i in range(len(x))])

    def append(self, clf):
        """
        trains and updates score
        appends model to Composite
        :return: agreement with __trainy of cv
        """
        print("\tNumber classifiers: " + str(len(self.clfs)) + "\t Last type: "
                + type(clf).__name__)
        cvpreds = list()            # store current predictions
        # generate cv predictions
        for i in range(len(self.__cv)):
            clf.fit(self.__cv[i][0], self.__cv[i][1])
            cvpreds.extend(clf.predict(self.__cv[i][2]))
        cvpreds = np.array(cvpreds).astype(int) # allow vectorizing
        # check whether cv improves
        old_preds = self.__cvpreds\
                // (len(self.clfs) // 2 + 1) # old cv preds
        new_preds = (self.__cvpreds + cvpreds)\
                // ((len(self.clfs) + 1) // 2 + 1) # new cv preds
        if self.diff(new_preds, self.__trainy) < \
                self.diff(old_preds, self.__trainy): # if num differences decreases
            # change cv, refit and append
            self.__cvpreds += cvpreds
            clf.fit(self.__trainx, self.__trainy)
            self.clfs.append(clf)

            self.__preds += clf.predict(self.__testx)
            self.preds = self.__preds // (len(self.clfs) // 2 + 1)

            return self.diff(new_preds, self.__trainy) # return cv agreement
        else:
            return -1 # do not add

    def predict(self):
        """
        returns own prediction
        """
        assert min(self.preds) >= 0 and max(self.preds) <= 1
        return self.preds # returns 1 where majority said 1
    
    def score(self, datax, datay):
        """
        scores self on new dataset
        """
        return np.mean([i.score(datax, datay) for i in self.clfs])
