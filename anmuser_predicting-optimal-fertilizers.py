import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import LabelEncoder
from autogluon.tabular import TabularPredictor


!pip install autogluon


train = pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv')
sample = pd.read_csv('/kaggle/input/playground-series-s5e6/sample_submission.csv')


X = train.drop(['Fertilizer Name'],axis = 1)
y = train['Fertilizer Name']


# Standardize 'Soil Type' in both datasets
X['Soil Type'] = X['Soil Type'].str.strip().str.lower()
test['Soil Type'] = test['Soil Type'].str.strip().str.lower()


from sklearn.preprocessing import LabelEncoder

# Fit the encoder on training data
label_encoder = LabelEncoder()
X['Soil Type'] = label_encoder.fit_transform(X['Soil Type'])

# Transform the test data
test['Soil Type'] = label_encoder.transform(test['Soil Type'])


train.isna().sum()


train['Fertilizer Name'].value_counts()


categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
label_encoder = LabelEncoder()
for column in categorical_cols:
    X[column] = label_encoder.fit_transform(X[column])
categorical_cols_test = test.select_dtypes(include=['object', 'category']).columns.tolist()
for column in categorical_cols_test:
    test[column] = label_encoder.transform(test[column])


X['Fertilizer Name'] = y


X.head(3)


predictor = TabularPredictor(label='Fertilizer Name',
            eval_metric='accuracy').fit(train_data=X,
            presets= 'best_quality',
            time_limit=1200)# 600)


predictor.fit_summary()


predictions = predictor.predict(test)#.evaluate(test)
predictions.head()


sample.head()


sample["Fertilizer Name"] = predictions 
sample.to_csv("submission.csv", index=False)


sample.head()




