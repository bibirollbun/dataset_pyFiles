### This is My First Notebook :) I hope it helps. I am a beginner myself.

# Importing all the dependencies
from pyexpat import features

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score,mean_absolute_error
from sklearn.preprocessing import OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import LogisticRegression
print('Importing Complete')


#Reading CSV file
df = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv', index_col=0)


df.head() # Taking a sneek at the data to get the idea of columns


y = df.Personality # Label


X = df.drop('Personality', axis=1) # Features


#Splitting the Data to train set and validation set to measure the performance of the model
x_train, x_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=0)


y_train


# Separating the numerical and categorical columns for preprocessing pipeline
num_cols = x_train.select_dtypes(include=np.number).columns


num_cols


cat_cols = x_train.select_dtypes(include='object').columns


cat_cols


### Setting up numerical and categorical pipeline 
num_pipeline = Pipeline([('imputer', SimpleImputer(strategy='mean'))])
categorical_pipeline = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')),('encoder',OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))])


### Using column transformer to apply the respective pipeline to their columns
preprocessor = ColumnTransformer([('numerical', num_pipeline, num_cols),
                                  ('categorical', categorical_pipeline, cat_cols)])


### Select a model for the data
### I first used only one model and then later added another to compare the preformance
model = RandomForestClassifier()
new_model = LogisticRegression()


### Make the pipeline for the data and model
my_pipeline = Pipeline([('preprocessor', preprocessor),('model', model)])


new_pipline = Pipeline([('preprocessor', preprocessor),('model', new_model)])



label_encoder = LabelEncoder()


features = x_train.columns.tolist()
features


# exp,_ = y_train.factorize()
# exp = pd.DataFrame(exp)


# x_train[features].corrwith(exp)


### Encoding my y
y_train_encoded = label_encoder.fit_transform(y_train)


### Fitting my pipeline with features and label to train it
my_pipeline.fit(x_train, y_train_encoded)


new_pipline.fit(x_train, y_train_encoded)


logistic_pred = new_pipline.predict(x_val)


preds = my_pipeline.predict(x_val)


y_val_encoded = label_encoder.transform(y_val)


### Now after predicting with the validation set . I tried finding the accuracy


accuracy_score(y_val_encoded, logistic_pred)


mean_absolute_error(y_val_encoded, preds)
accuracy_score(y_val_encoded, preds)


test_df = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv', index_col=0)


test_df


preds = my_pipeline.predict(test_df)


final_preds = new_pipline.predict(test_df)


og_preds = label_encoder.inverse_transform(preds)


conv_pred = label_encoder.inverse_transform(final_preds)


res = pd.DataFrame({'id':test_df.index,'personality': conv_pred})


res.to_csv('submission_final.csv', index=False)

