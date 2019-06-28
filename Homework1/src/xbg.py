import warnings
import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import f1_score

warnings.filterwarnings('ignore')
# Set features name
trainSetName = []
testSetName = []
for index in range(32):
    trainSetName.append('F' + str(index+1))
    testSetName.append('F' + str(index+1))
    
trainSetName.append('label')

dataPath = '../dataSet/'
# Load DataSet
trainData = pd.read_csv(dataPath+'trainSet.csv', names=trainSetName)
# Drop header
trainData = trainData.drop([0])

trainLabel = trainData['label'].astype(int)
trainLabel = np.array(trainLabel.tolist())
del trainData['label']

testData = pd.read_csv(dataPath+'testSet.csv', names=testSetName)
testData = testData.drop([0])

# Scale datas
scaler = MinMaxScaler(feature_range=(-1,1))
trainData = scaler.fit_transform(trainData)
testData = scaler.transform(testData)

# Randomized CV for XGBoost parameters tuning
param_test = {
    'max_depth': range(3, 10, 1)
}

XGBModel = XGBClassifier(
    learning_rate = 0.01,
    n_estimators = 2000,
    objective = 'binary:logistic',
    min_child_weight = 1,
    gamma = 0,
    subsample = 0.8,
    colsample_bytree = 0.8,
    scale_pos_weight = 1,
    nthread = 4
)

trainSet = xgb.DMatrix(trainData, label=trainLabel)

rSearch = RandomizedSearchCV(XGBModel, param_test, n_iter=7, scoring='f1', iid=False, cv=5, verbose=50)
rSearch.fit(trainData, trainLabel)
print(rSearch.best_params_)

XGBModel.set_params(max_depth= rSearch.best_params_['max_depth'])

param_test = {
    'min_child_weight': range(1, 8, 1)
}

rSearch = RandomizedSearchCV(XGBModel, param_test, n_iter=7, scoring='f1', iid=False, cv=5, verbose=50)
rSearch.fit(trainData, trainLabel)
print(rSearch.best_params_)

XGBModel.set_params(min_child_weight= rSearch.best_params_['min_child_weight'])

param_test = {
    'gamma': [i/10.0 for i in range(0, 5)]
}

rSearch = RandomizedSearchCV(XGBModel, param_test, n_iter=5, scoring='f1', iid=False, cv=5, verbose=50)
rSearch.fit(trainData, trainLabel)
print(rSearch.best_params_)

XGBModel.set_params(gamma= rSearch.best_params_['gamma'])

param_test = {
    'subsample': [i/10.0 for i in range(6, 10)]
}

rSearch = RandomizedSearchCV(XGBModel, param_test, n_iter=4, scoring='f1', iid=False, cv=5, verbose=50)
rSearch.fit(trainData, trainLabel)
print(rSearch.best_params_)

XGBModel.set_params(subsample= rSearch.best_params_['subsample'])

param_test = {
    'colsample_bytree': [i/10.0 for i in range(6, 10)]
}

rSearch = RandomizedSearchCV(XGBModel, param_test, n_iter=4, scoring='f1', iid=False, cv=5, verbose=50)
rSearch.fit(trainData, trainLabel)
print(rSearch.best_params_)

XGBModel.set_params(colsample_bytree= rSearch.best_params_['colsample_bytree'])

param_test = {
    'reg_alpha': [1e-4, 1e-3, 1e-2, 1e-1, 1, 10]
}

rSearch = RandomizedSearchCV(XGBModel, param_test, n_iter=6, scoring='f1', iid=False, cv=5, verbose=50)
rSearch.fit(trainData, trainLabel)
print(rSearch.best_params_)

XGBModel.set_params(reg_alpha= rSearch.best_params_['reg_alpha'])

cvResult = xgb.cv(XGBModel.get_xgb_params(), trainSet, num_boost_round=XGBModel.get_params()['n_estimators'], nfold=5,
                    metrics='auc', early_stopping_rounds=10)
print(cvResult)
XGBModel.set_params(n_estimators=cvResult.shape[0])

print(XGBModel.get_xgb_params())

params = XGBModel.get_xgb_params()
'''
params = {
    'learning_rate': 0.01,
    'booster': 'gblinear',
    'objective': 'binary:logistic',
    'min_child_weight': 4,
    'max_depth': 3,
    'gamma': 1.0,
    'reg_alpha': 0.001,
    'reg_lambda': 0.001,
    'nthread': 4
}
'''

n_folds = 10
rounds = 2000
kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=None)
total_accuracy = 0

for index, (trainIndex, testIndex) in enumerate(kf.split(trainData, trainLabel)):
    tr_x = trainData[trainIndex]
    tr_y = trainLabel[trainIndex]
    te_x = trainData[testIndex]
    te_y = trainLabel[testIndex]
    
    trainSet = xgb.DMatrix(tr_x, label=tr_y)
    testSet = xgb.DMatrix(te_x)
    
    model = xgb.train(params, trainSet, rounds, verbose_eval=10, feval=F1_Score)
    preds = model.predict(testSet)
    pred = []
    for ele in preds:
        if ele < 0.5:
            pred.append(0)
        else:
            pred.append(1)
    
    accuracy = f1_score(pred, te_y)
    total_accuracy += accuracy
    # Decide KFold scores
    print('Iteration %d, F1_Score: %.7f' % (index, accuracy))

print('Mean F1_Score: %.7f' % (total_accuracy / n_folds))

# Output final prediction
trainSet = xgb.DMatrix(trainData, label=trainLabel)
testSet = xgb.DMatrix(testData)

model = xgb.train(params, trainSet, rounds, verbose_eval=10, feval=F1_Score)

preds = model.predict(testSet)
pred = []
for ele in preds:
    if ele < 0.5:
        pred.append(0)
    else:
        pred.append(1)

result = pd.DataFrame(columns=['ID', 'Predicted'])
result['ID'] = [x+1 for x in range(len(testData))]
result['Predicted'] = pred
print(result)

result.to_csv('../result/submission.csv', index=False)