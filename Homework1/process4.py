# encoding utf-8
from sklearn.metrics import accuracy_score,recall_score,f1_score
from sklearn.semi_supervised import LabelPropagation
from sklearn.preprocessing import MinMaxScaler
from numpy import *
import xgboost as xgb
import numpy as np
import pandas as pd
import csv

#train xgb
config = {
	'rounds': 10000,
	'folds': 5
}

params = {
	'booster':'gbtree',
	'objective':'binary:logistic',
	'stratified':True,
	'max_depth':5,
	'min_child_weight':1,
	'gamma':3,
	'subsample':0.8,#0.7
	'colsample_bytree':0.6, 
	'lambda':3, 
	'eta':0.05,
	'seed':20,
	'silent':1,
	'eval_metric':'auc'
}

def loadTrainDataSet(path):
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

def loadTestDataSet(path):
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

def customedscore1(preds, dtrain):
	label = dtrain.get_label()
	d = pd.DataFrame()
	d['prob'] = list(preds)
	d['y'] = list(label)
	d = d.sort_values(['prob'], ascending=[0])
	y = d.y
	PosAll = pd.Series(y).value_counts()[1]
	NegAll = pd.Series(y).value_counts()[0]
	
	pCumsum = d['y'].cumsum()
	nCumsum = np.arange(len(y)) - pCumsum + 1
	pCumsumPer = pCumsum / PosAll
	nCumsumPer = nCumsum / NegAll
	
	TR1 = pCumsumPer[abs(nCumsumPer-0.001).idxmin()]
	TR2 = pCumsumPer[abs(nCumsumPer-0.005).idxmin()]
	TR3 = pCumsumPer[abs(nCumsumPer-0.01).idxmin()]
	score = 0.4 * TR1 + 0.3 * TR2 + 0.3 * TR3
	return 'SCORE',float(score)

def xgbPredict(trainFeature,trainLabel,testFeature,rounds,params):
	zero_len = 0
	one_len = 0
	for index in range(len(trainLabel)):
		if trainLabel[index] == 0:
			zero_len += 1
		else:
			one_len += 1

	params['scale_pos_weights '] = float(zero_len)/(one_len)
	
	dtrain = xgb.DMatrix(trainFeature, label = trainLabel)
	dtest = xgb.DMatrix(testFeature)

	watchlist  = [(dtrain,'train')]
	num_round = rounds
	
	model = xgb.train(params, dtrain, num_round, watchlist, verbose_eval = 50,feval = customedscore1)
	predict = model.predict(dtest)
	return model,predict

def findBestLR(rate, k_parts_data, k_parts_label):
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
					np.append(trainData, k_parts_data[index])
					np.append(trainLabel, k_parts_label[index])
				else:			
					trainData = k_parts_data[index]
					trainLabel = k_parts_label[index]

		model,predict = xgbPredict(trainData,trainLabel,testData,700,params)


if __name__ == '__main__':
	dataMat, labelMat = loadTrainDataSet('trainSet.csv')
	realTestDataMat = loadTestDataSet('testSet.csv')

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

	rate = 0.01
	while rate <= 0.00:
		findBestLR(rate, k_parts_data, k_parts_label)
		rate += 0.01
	
	model,predict = xgbPredict(dataMat,labelMat,realTestDataMat,700,params)

	csv_writer = csv.writer(open('result1.csv', 'w', newline=''))
	csv_writer.writerow(['ID', 'Predicted'])
	for index in range(len(predict)):
		csv_writer.writerow([index+1, int(predict[index])])