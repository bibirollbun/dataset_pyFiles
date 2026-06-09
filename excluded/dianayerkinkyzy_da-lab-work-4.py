import pandas as pd
dataframe = pd.read_csv('/kaggle/input/titanic-dataset-25/train.csv')
dataframe


dataframe.isnull().sum()


from sklearn.impute import SimpleImputer
i = SimpleImputer(strategy='mean')
dataframe['Age'] = i.fit_transform(dataframe[['Age']])
print(dataframe.isnull().sum())  
print(dataframe)


import pandas as pd
dataframe2 = pd.read_csv('/kaggle/input/time-series-dataset/city_data.csv')
dataframe2


dataframe2.dtypes


dataframe2['Date'] = pd.to_datetime(dataframe2['Date'])
dataframe2.set_index('Date', inplace=True)
dataframe2.sort_index()


dataframe2.isnull().sum()


dataframe2['Xylene'] = dataframe2['Xylene'].ffill()


dataframe2['Xylene'].isnull().sum()

