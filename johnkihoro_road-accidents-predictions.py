import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
from lightgbm import LGBMRegressor
import warnings
warnings.filterwarnings("ignore")



for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))



df = pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv').drop('id', axis=1)
print(df.shape)
df


print(df.duplicated().sum())
df.info()


df.drop_duplicates(inplace=True)
df.describe()


df.describe(include='object')


for column in df.select_dtypes(exclude=np.number).columns:
    print(column, '--'*10, df[column].unique())


sns.pairplot(df);


X = df.drop('accident_risk', axis=1)
y = df['accident_risk']


sns.histplot(y);


print(y.skew())
sns.boxplot(x=y)


numerical_columns = X.select_dtypes(include='number').columns
categorical_columns = X.select_dtypes(exclude='number').columns
print('numerical columns are:\n', numerical_columns)
print('--'*50)
print('\nCategorical columns include:\n', categorical_columns)


df.head()


X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=7)
X_train


onehot = preprocessing.OneHotEncoder(handle_unknown='ignore', sparse_output=False)
onehot.fit(X_train[categorical_columns])

X_train_cat = pd.DataFrame(onehot.transform(X_train[categorical_columns]), columns=onehot.get_feature_names_out()).reset_index(drop=True)
X_train = pd.concat([X_train.drop(categorical_columns, axis=1).reset_index(drop=True), X_train_cat], axis=1)
X_train


X_valid_cat = pd.DataFrame(onehot.transform(X_valid[categorical_columns]), columns=onehot.get_feature_names_out()).reset_index(drop=True)
X_valid = pd.concat([X_valid.drop(categorical_columns, axis=1).reset_index(drop=True), X_valid_cat], axis=1)
X_valid


X_categorical = pd.DataFrame(onehot.transform(X[categorical_columns]), columns=onehot.get_feature_names_out())
X = pd.concat([X.drop(categorical_columns, axis=1).reset_index(drop=True), X_categorical.reset_index(drop=True)], axis=1)
X


lgbm = LGBMRegressor(random_state=27, verbose=0)
params = {'n_estimators':[1000, 1500, 2000],
         'num_leaves':[50, 180, 150],
         'learning_rate':[0.001, 0.01, 1]}
grid = GridSearchCV(estimator=lgbm, param_grid=params, cv=5)
grid.fit(X, y)


optimal_parameters = grid.best_params_
print(optimal_parameters)


lgbm = LGBMRegressor(**optimal_parameters, random_state=27, verbose=0)
cv_scores = -1*cross_val_score(lgbm, X, y, scoring='neg_mean_squared_error', cv=5)
cv_scores


lgbm.fit(X_train, y_train)


training_r2 = lgbm.score(X_train, y_train)
prediction_r2 = lgbm.score(X_valid, y_valid)
print(f'R2 for training is {training_r2:.4f}')
print(f'R2 for prediction is {prediction_r2:.4f}')


y_preds = lgbm.predict(X_valid)
y_preds


mse = mean_squared_error(y_valid, y_preds)
rmse = np.sqrt(mse)
print(f'RMSE is {rmse:.4f}')


lgbm.fit(X, y)


full_training_r2 = lgbm.score(X, y)
print(f'R2 for training is {full_training_r2:.4f}')


test_data = pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv').drop('id', axis=1)
print(test_data.shape)
test_data.head()


test_categorical = pd.DataFrame(onehot.transform(test_data[categorical_columns]), columns=onehot.get_feature_names_out())
test = pd.concat([test_data.drop(categorical_columns, axis=1).reset_index(drop=True), test_categorical.reset_index(drop=True)], axis=1)
test.head()


test_preds = lgbm.predict(test)
test_preds


sub = pd.read_csv('/kaggle/input/playground-series-s5e10/sample_submission.csv')
sub['accident_risk'] = test_preds
sub


sub.to_csv('Sub file.csv', index=False)


importance = pd.Series(lgbm.feature_importances_, index=X_train.columns)
importance.sort_values(ascending=False, inplace=True)


features=X.columns
indices = np.argsort(importance)

plt.figure(figsize=(8, 6))
plt.barh(range(len(importance)), importance[indices], align="center")
plt.yticks(range(len(importance)), [features[i] for i in indices])
plt.xlabel("Feature Importance")
plt.ylabel("Features")
plt.title("Feature Importance Bar Plot")
plt.tight_layout()
plt.show();

