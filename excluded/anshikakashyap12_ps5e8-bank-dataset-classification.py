# importing libraries
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


# loading datasets
train = pd.read_csv("/kaggle/input/playground-series-s5e8/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e8/test.csv")


# seperating features and target variable
X_train = train.drop(['id', 'y'], axis=1)
Y_train = train['y']
X_test = test.drop('id', axis=1)
id_col = test['id']


# data preprocessing
num_col = X_train.select_dtypes(exclude='object').columns
cat_col = X_train.select_dtypes(include='object').columns

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', Pipeline(steps=[
            ('onehotencode', OneHotEncoder(handle_unknown='ignore')) ]), cat_col)
    ])


# model training
model_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                 ('regressor', LinearRegression())])

model_pipeline.fit(X_train, Y_train)


# predicting using model created
predictions = model_pipeline.predict(X_test)


# creating submission file
predictions = predictions.round(1)

submission = pd.DataFrame({'id': id_col, 'y': predictions})
submission.to_csv('submission.csv', index=False)

print("Submission file created successfully!")
print(submission.head())


# # to clear kaggle working
# !rm -rf /kaggle/working/*

