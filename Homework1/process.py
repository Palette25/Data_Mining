from sklearn.preprocessing import MinMaxScaler
from scipy.special import *
from numpy import *
import numpy
import csv

def correctVector(inX, weights):
    prob = expit(sum(inX * weights))
    if prob > 0.5:
        return 1.0
    else:
        return 0.0

def loadDataSet(path):
    dataMat = []
    labelmat = []
    fr = csv.reader(open(path, 'r'))
    index = -1
    for line in fr:
        index += 1
        if index == 0:
            continue
        temp = []
        for index in range(32):
            temp.append(float(line[index]))

        dataMat.append(temp)
        labelmat.append(float(line[32]))
    return dataMat, labelmat
 
 
def stocGradAscent(dataMat, labelMat):
    dataMatrix = mat(dataMat)
    labelMatrix = mat(labelMat).transpose()
    m, n = shape(dataMatrix)

    alpha = 0.0005
    maxCycles = 500
    weights = ones((n, 1))
    for k in range(maxCycles):
        H = expit(dataMatrix * weights)
        error = labelMatrix - H
        weights = weights + alpha * dataMatrix.transpose() * error

    return weights
 

if __name__ == '__main__':
    trainPath = 'trainSet.csv'
    testPath = 'trainSet.csv'
 
    dataMat, labelMat = loadDataSet(trainPath)
    
    dataMat = MinMaxScaler().fit_transform(dataMat)

    n = shape(dataMat)[0]

    total_accuracy = 0

    print("Checking Batch Accuracy....")
    # Divide into ten batch
    for k in range(10):
        begin = int(n / 10 * k)
        train_size = 15000
        
        trainData = dataMat[begin:begin+train_size]
        bias = ones((shape(trainData)[0], 1))
        numpy.append(trainData, bias, 1)

        weight = stocGradAscent(array(trainData), labelMat[begin:begin+train_size])
        # Check test set accuracy
        test_begin = int(begin + train_size)
        test_size = int(n / 10 - train_size)

        count = 0
        for index in range(test_size):
            if int(correctVector(array(dataMat[test_begin+index]), weight)) == int(labelMat[test_begin+index]):
                count += 1
        accuracy = (float(count) / test_size)
        total_accuracy += accuracy

        print("The accuracy rate of batch %d is : %f" % (k, accuracy))

    print("Final accuracy rate : %f" % float(total_accuracy / 10))

    print("Checking Half-Fall Accuracy....")
    half = int(n/2)
    weight = stocGradAscent(array(dataMat[:half]), labelMat[:half])
    count = 0
    for index in range(half):
        if int(correctVector(array(dataMat[half+index]), weight)) == int(labelMat[half+index]):
                count += 1

    accuracy = (float(count) / half)
    print("Half-Fall Accuracy : %f" % accuracy)