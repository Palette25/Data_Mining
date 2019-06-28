from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import RandomizedSearchCV
from scipy.sparse import coo_matrix
from sklearn.utils import shuffle
from sklearn.model_selection import KFold

from numpy import *
import matplotlib.pyplot as plt
import numpy as np
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


if __name__ == '__main__':
	dataMat, labelMat = loadDataSet('trainSet.csv')
	realTestDataMat = loadTestSet('testSet.csv')

	scaler = MinMaxScaler((-1, 1))
	dataMat = scaler.fit_transform(dataMat)
	realTestDataMat = scaler.transform(realTestDataMat)

	dataMat, dataMat_sparse, labelMat = shuffle(dataMat, coo_matrix(dataMat), labelMat)

	'''
	# Print Corrcoef factors
	y = []
	for index in range(32):
		y.append(np.corrcoef(dataMat[:, index], labelMat)[0][1])

	print(y)
	plt.plot(range(32), y)

	plt.show()

	'''

	'''
	model = LogisticRegression(max_iter=200)
	param_test = {
		'C' : [1, 2, 3, 4, 5]
	}
	Search = RandomizedSearchCV(model, param_test, scoring='accuracy', n_iter=5, n_jobs=3, cv=5, verbose=100, iid=False)
	Search.fit(dataMat, labelMat)

	print(Search.best_params_)

	'''
	kf = KFold(n_splits=10)

	index = 0
	for train_index, test_index in kf.split(dataMat):
		index += 1
		trainData, testData = dataMat[train_index], dataMat[test_index]
		trainLabel, testLabel = np.array(labelMat)[train_index], np.array(labelMat)[test_index]

		classifier = LogisticRegression(max_iter=500)
		classifier.fit(trainData, trainLabel)

		score = classifier.score(testData, testLabel)

		print("Batch %d, Accuracy : %f" % (index, score))

	'''
	classifier = MLPClassifier(activation='logistic', max_iter=500)
	classifier.fit(dataMat, labelMat)

	predict_result = classifier.predict(realTestDataMat)

	csv_writer = csv.writer(open('result1.csv', 'w', newline=''))
	csv_writer.writerow(['ID', 'Predicted'])
	for index in range(len(predict_result)):
		csv_writer.writerow([index+1, int(predict_result[index])])
	'''