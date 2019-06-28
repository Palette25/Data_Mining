import pandas as pd

result = pd.read_csv('../result/submission.csv')
sample = pd.read_csv('../result/result1.csv')

vals = [0, 1]

print(result.groupby(['Predicted']).count())
print(sample.groupby(['Predicted']).count())