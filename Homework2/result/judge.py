import pandas as pd

gbdtPath = 'gbdt_result.csv'
lightPath = 'lightgbm_result.csv'
test_csv = ['test1.csv', 'test2.csv', 'test3.csv', 'test4.csv', 'test5.csv', 'test6.csv']

def loadTestCsv():
	result = []

	for i in range(6):
		path = '../dataSet/' + test_csv[i]
		df_temp = pd.read_csv(path)
		result.append(df_temp.values)

	return result


def loadResults():
	df_gbdt = pd.read_csv(gbdtPath)
	df_light = pd.read_csv(lightPath)

	return df_gbdt.values, df_light.values


if __name__ == '__main__':
	gbdtRes, lightgbmRes = loadResults()
	result = loadTestCsv()

	print(len(gbdtRes))
	print(len(lightgbmRes))
	for ele in result:
		print(len(ele))

	print(result[0])


	for index in range(6):
		print(gbdtRes[index])

	for index in range(6):
		print(lightgbmRes[index])