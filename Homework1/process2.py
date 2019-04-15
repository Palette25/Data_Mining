from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPClassifier
from numpy import *
import numpy
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

def loadTestSet(path):
	dataMat = []
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
	return dataMat

def findBestRate(rate, k_parts_data, k_parts_label):
	total = 0
	for k in range(10):
		trainData = []
		trainLabel = []
		testData = []
		testLabel = []
		# Fill in Train Data and Train Label
		for index in range(10):
			if index == k:
				testData = k_parts_data[index]
				testLabel = k_parts_label[index]
			else:
				if shape(trainData)[0] != 0:
					numpy.append(trainData, k_parts_data[index])
					numpy.append(trainLabel, k_parts_label[index])
				else:			
					trainData = k_parts_data[index]
					trainLabel = k_parts_label[index]

		classifier = MLPClassifier(activation='logistic',alpha=1e-7,max_iter=150,learning_rate_init=rate).fit(trainData, trainLabel)
		accuracy = classifier.score(testData, testLabel)

		print('Batch %d : Accuracy %f' % (k, accuracy))
		total += accuracy

	total = total / 10
	print('Final Accuracy : %f, Rate : %f' % (total, rate))
	return total

def findBestAlpha(maxRate, alpha, k_parts_data, k_parts_label):
	total = 0
	for k in range(10):
		trainData = []
		trainLabel = []
		testData = []
		testLabel = []
		# Fill in Train Data and Train Label
		for index in range(10):
			if index == k:
				testData = k_parts_data[index]
				testLabel = k_parts_label[index]
			else:
				if shape(trainData)[0] != 0:
					numpy.append(trainData, k_parts_data[index])
					numpy.append(trainLabel, k_parts_label[index])
				else:			
					trainData = k_parts_data[index]
					trainLabel = k_parts_label[index]

		classifier = MLPClassifier(activation='logistic',alpha=alpha,max_iter=150,learning_rate_init=maxRate).fit(trainData, trainLabel)
		accuracy = classifier.score(testData, testLabel)

		print('Batch %d : Accuracy %f' % (k, accuracy))
		total += accuracy

	total = total / 10
	print('Final Accuracy : %f, Alpha : %f' % (total, alpha))
	return total

if __name__ == '__main__':
	dataMat, labelMat = loadDataSet('trainSet.csv')
	realTestDataMat = loadTestSet('testSet.csv')

	scaler = MinMaxScaler()
	dataMat = scaler.fit_transform(dataMat)
	realTestDataMat = scaler.transform(realTestDataMat)
	n = shape(dataMat)[0]
	unit = int(n / 10)
	total = 0

	# Perform K-Fall checking
	k_parts_data = []
	k_parts_label = []
	for k in range(10):
		k_parts_data.append(dataMat[k * unit : (k + 1) * unit])
		k_parts_label.append(labelMat[k * unit : (k + 1) * unit])

	rate = 0.001
	maxRate = 0
	res = 0
	while rate <= 0.01:
		temp = findBestRate(rate, k_parts_data, k_parts_label)
		if temp > res:
			res = temp
			maxRate = rate
		rate += 0.001

	print('Max Rate : %f' % maxRate)

	alpha = 1e-3
	bestAlpha = 0
	res = 0
	while alpha >= 1e-8:
		temp = findBestAlpha(maxRate, alpha, k_parts_data, k_parts_label)
		if temp > res:
			res = temp
			bestAlpha = alpha
		alpha /= 10

	print('Best Alpha : %f' % bestAlpha)
	
	# Real test set prediction
	classifier = MLPClassifier(activation='logistic',alpha=bestAlpha,max_iter=150,learning_rate_init=maxRate).fit(dataMat, labelMat)
	predict_result = classifier.predict(realTestDataMat)

	csv_writer = csv.writer(open('result1.csv', 'w', newline=''))
	csv_writer.writerow(['ID', 'Predicted'])
	for index in range(len(predict_result)):
		csv_writer.writerow([index+1, int(predict_result[index])])
