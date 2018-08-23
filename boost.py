import numpy as np
import math
def stumpClassify(dataMatrix, dimen, threshVal, threshIneq):
    retArray = np.ones(dataMatrix.shape()[0], 1)
    if threshIneq == 'lt':
        retArray[dataMatrix[:, dimen] <= threshVal] = -1
    else:
        retArray[dataMatrix[:, dimen] > threshVal] = 1
    return retArray

def buildStump(dataArr, classLabels, D):
    dataMatrix = dataArr
    labelMat = classLabels.T
    m, n = np.shape(dataMatrix)
    numSteps = 10.0
    bestStump = {}
    bestClassEst = np.zeros(m,1)
    minError = math.inf
    for i in range(n):
        rangeMin = dataMatrix[:, i].min()
        rangeMax = dataMatrix[:, i].max()
        stepSize = (rangeMax - rangeMin) / numSteps
        for j in range(-1, int(numSteps)+1):
            for inequal in ['lt', 'gt']:
                threshVal = (rangeMin + np.float(j) * stepSize)
                predictedVals = stumpClassify(dataMatrix, i, threshVal, inequal)
                errArr = np.ones(m, 1)
                errArr[predictedVals == labelMat] = 0
                weightedError = D.T*errArr
                print("split: dim %d\nthresh %.2f\nthresh inequal: %s\nthe weighted error is %.3f"
                      %(i, threshVal, inequal, weightedError))
                if weightedError < minError:
                    minError = weightedError
                bestClassEst = predictedVals.copy()
                bestStump['dim'] = i
                b


