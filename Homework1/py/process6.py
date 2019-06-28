from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from sklearn import metrics

from numpy import *
import xgboost as xgb
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

def xgbModelCV(model, dataMat, labelMat, cv_folds=5, early_stopping_rounds=50):
	xgb_params = model.get_xgb_params()
	xgTrain = xgb.DMatrix(dataMat, label=labelMat)
	cvResult = xgb.cv(xgb_params, xgTrain, num_boost_round=model.get_params()['n_estimators'], nfold=cv_folds, metrics='auc', early_stopping_rounds=early_stopping_rounds)
	# Set result params
	model.set_params(n_estimators=cvResult.shape[0])

	# Prediction
	cvPrediction = model.predict(dataMat)
	cvPreprob = model.predict_proba(dataMat)[:, 1]

	# Print model accuracy report
	print("Accuracy : %.4g" % metrics.accuracy_score(labelMat, cvPrediction))
	print("AUC Score (Train) : %f" & metrics.roc_auc_score(labelMat, cvPreprob))


if __name__ == '__main__':
	dataMat, labelMat = loadDataSet('trainSet.csv')
	realTestDataMat = loadTestSet('testSet.csv')

	scaler = MinMaxScaler()
	dataMat = scaler.fit_transform(dataMat)
	realTestDataMat = scaler.transform(realTestDataMat)
	n = shape(dataMat)[0]
	unit = int(n / 10)
	total = 0

	model = XGBClassifier(
				learning_rate=0.1,
				n_estimators=500,
				max_depth=3,
				min_child_weight=4,
				gamma=1.0,
				reg_alpha=0.001,
				reg_lambda=0.001,
				subsample=0.8,
				colsample_btree=0.8,
				objective='binary:logistic',
				scale_pos_weight=1,
				seed=27,
				nthread=4
		)

	# xgbModelCV(model, dataMat, labelMat)
	
	# Perform K-Fall checking
	k_parts_data = []
	k_parts_label = []
	for k in range(10):
		k_parts_data.append(dataMat[k * unit : (k + 1) * unit])
		k_parts_label.append(labelMat[k * unit : (k + 1) * unit])

	# Do best parameters grid searching
	'''
	param_test = {
		'max_depth' : [3, 4, 5, 6, 7, 8, 9]
	}
	gSearch = RandomizedSearchCV(model, param_test, scoring='accuracy', n_iter=7, n_jobs=3, cv=5, verbose=100, iid=False)
	gSearch.fit(dataMat, labelMat)

	print(gSearch.best_params_)

	param_test = {
		'min_child_weight' : [3, 4, 5, 6, 7, 8, 9]
	}

	gSearch = RandomizedSearchCV(model, param_test, scoring='accuracy',n_iter=7, n_jobs=3, iid=False, cv=5, verbose=100)
	gSearch.fit(dataMat, labelMat)

	print(gSearch.best_params_)

	param_test = {
		'gamma' : [i/10.0 for i in range(11)]
	}

	gSearch = RandomizedSearchCV(model, param_test, scoring='accuracy',n_iter=10, n_jobs=3, iid=False, cv=5, verbose=100)
	gSearch.fit(dataMat, labelMat)

	print(gSearch.best_params_)

	

	param_test = {
		'subsample' : [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
	}

	gSearch = RandomizedSearchCV(model, param_test, scoring='accuracy',n_iter=6, n_jobs=3, iid=False, cv=5, verbose=100)
	gSearch.fit(dataMat, labelMat)

	print(gSearch.best_params_)

	param_test = {
		'colsample_btree' : [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
	}

	gSearch = RandomizedSearchCV(model, param_test, scoring='accuracy',n_iter=6, n_jobs=3, iid=False, cv=5, verbose=100)
	gSearch.fit(dataMat, labelMat)

	print(gSearch.best_params_)

	param_test = {
		'reg_alpha' : [0.0001, 0.001, 0.01, 0.1, 1]
	}

	gSearch = RandomizedSearchCV(model, param_test, scoring='accuracy',n_iter=5, n_jobs=3, iid=False, cv=5, verbose=100)
	gSearch.fit(dataMat, labelMat)

	print(gSearch.best_params_)

	param_test = {
		'reg_lambda' : [0.0001, 0.001, 0.01, 0.1, 1]
	}

	gSearch = RandomizedSearchCV(model, param_test, scoring='accuracy',n_iter=5, n_jobs=3, iid=False, cv=5, verbose=100)
	gSearch.fit(dataMat, labelMat)

	print(gSearch.best_params_)
	
	'''
	# Perform K-Fold Cross Validation
	total = 0
	for k in range(10):
		model = XGBClassifier(
				learning_rate=0.1,
				n_estimators=500,
				max_depth=3,
				min_child_weight=4,
				gamma=1.0,
				reg_alpha=0.001,
				reg_lambda=0.001,
				subsample=0.8,
				colsample_btree=0.8,
				objective='binary:logistic',
				scale_pos_weight=1,
				seed=27,
				nthread=4
		)

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

		model.fit(trainData, trainLabel)
		accuracy = accuracy_score(model.predict(testData), testLabel)

		print('Batch %d : Accuracy %f' % (k, accuracy))
		total += accuracy

	total = total / 10
	print('Final Accuracy : %f' % total)

	model = XGBClassifier(
				learning_rate=0.015,
				n_estimators=2000,
				max_depth=4,
				min_child_weight=1,
				gamma=0,
				reg_alpha=0.001,
				reg_lambda=0.001,
				subsample=0.8,
				colsample_btree=0.55,
				objective='binary:logistic',
				scale_pos_weight=1,
				n_jobs=8
		)
	# Real test set prediction
	'''
	model.fit(dataMat, labelMat)
	predict_result = model.predict(realTestDataMat)

	csv_writer = csv.writer(open('result1.csv', 'w', newline=''))
	csv_writer.writerow(['ID', 'Predicted'])
	for index in range(len(predict_result)):
		csv_writer.writerow([index+1, int(predict_result[index])])
	'''