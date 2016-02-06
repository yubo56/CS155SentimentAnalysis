#!/usr/bin/env python
def load(FN, delim='|'):
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
    ys = []
    lines.readline() # get rid of the 'x1, x2..., y' line
    for i in lines.readlines():
        i.strip()
        i = i.split(delim)
        xs.append([ int(j) for j in i[0:len(i) - 1] ])
        ys.append(int(i[len(i) - 1]))
    return xs, ys
