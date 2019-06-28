from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from numpy import *
import csv

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

def HalfFall(dataMat, labelMat):
	trainSet = dataMat[:half]
	trainLabel = labelMat[:half]

	testSet = dataMat[half:]
	testLabel = labelMat[half:]

	clf = SVC(kernel='rbf')
	clf.fit(trainSet, trainLabel)

	count = 0
	for index in range(half):
		if clf.predict(numpy.array(testSet[index])) == testLabel[index]:
			count += 1

	accuracy = (float(count) / half)
	print('Accuracy : %f' % accuracy)

if __name__ == '__main__':
	dataMat, labelMat = loadDataSet('trainSet.csv')
	scaler = MinMaxScaler(feature_range=(0,1))
	dataMat = scaler.fit_transform(dataMat)
	n = shape(dataMat)[0]
	unit = int(n / 10)

	# Perform Half-Fall checking
	# HalfFall(dataMat, labelMat)
	
	# Perform K-Fall checking
	k_parts_data = []
	for k in range(10):
		k_parts_data.append(dataMat[k * unit : (k + 1) * unit])

	
