#!/usr/bin/env python
def load(FN, delim='|', hasY=True):
    """
    loads data from a file into X, Y variables. Let N be number of rows and M
    be number of features, then X[0:N][0:M] and Y[0:N] are the ranges of
    permitted indicies

    Input: FN - filename to be read
        delim - delimiter (default:'|')
        first - flag defining whether y values first or last (default:False)
        
    Output: n x m array of floats (x values)
            n x 1 array of floats (y values)
    """
    lines = open(FN, 'r')
    xs = []
    if hasY == True:
        ys = []
    lines.readline() # get rid of the 'x1, x2..., y' line
    for i in lines.readlines():
        i.strip()
        i = i.split(delim)
        if hasY == True:
            xs.append([ int(j) for j in i[0:len(i) - 1] ])
            ys.append(int(i[len(i) - 1]))
        else:
            xs.append([ int(j) for j in i[0:len(i)] ])
    if hasY == True:
        return xs, ys
    return xs

def loadtest(FN, delim='|'):
    """
    alias for load() for testing data
    """
    return load(FN, delim, hasY=False)

def loadtrain(FN, delim='|'):
    """
    alias for load() for training data
    """
    return load(FN, delim, hasY=True)

def write(FN, res, delim=','):
    """
    Prints data to desired file format

    Input: FN - filename to print to
          res - array of prediction results
        delim - delimiter to split results, default: ','
    """
    f = open(FN, 'w')
    f.write("Id,Prediction\n")
    for i in range(len(res)):
        f.write(delim.join([str(i + 1), str(res[i])]) + '\n')
