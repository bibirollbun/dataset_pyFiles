# importing libraries
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


# loading datasets
train = pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv")


# seperating features and target variable
X_train = train.drop(['id', 'Personality'], axis=1)
Y_train = train['Personality']
X_test = test.drop('id', axis=1)
id_col = test['id']


# data preprocessing
num_col = X_train.select_dtypes(exclude='object').columns
cat_col = X_train.select_dtypes(include='object').columns

preprocessor = ColumnTransformer(
    transformers=[
        ('num', SimpleImputer(strategy='mean'), num_col),
        ('cat', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehotencode', OneHotEncoder(handle_unknown='ignore')) ]), cat_col)
    ])


# label encoding target variable
label_encoder = LabelEncoder()
Y_train_encoded = label_encoder.fit_transform(Y_train)


# model training
model_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                 ('classifier', RandomForestClassifier(random_state=67, n_estimators = 200))])

model_pipeline.fit(X_train, Y_train_encoded)


# predicting using model created
predictions_encoded = model_pipeline.predict(X_test)
predictions = label_encoder.inverse_transform(predictions_encoded)


# creating submission file
submission = pd.DataFrame({'id': id_col, 'Personality': predictions})
submission.to_csv('submission.csv', index=False)

print("Submission file created successfully!")
print(submission.head())


# # to clear kaggle working
# !rm -rf /kaggle/working/*

