import pandas as pd
DIRECTORY = '/kaggle/input/playground-series-s4e5/'
train = pd.read_csv(DIRECTORY + 'train.csv').drop(columns='id')
test = pd.read_csv(DIRECTORY + 'test.csv').set_index('id')


meanX_to_meanP = train.groupby(train.drop(columns='FloodProbability').mean(axis=1))['FloodProbability'].mean()
test.mean(axis=1).map(meanX_to_meanP).to_csv('./submission.csv', header=['FloodProbability'])

