# Import prerequisites
import numpy as np
import pandas as pd

import seaborn as sns

from sklearn.experimental import enable_halving_search_cv

from sklearn.model_selection import train_test_split, KFold, cross_validate, HalvingGridSearchCV
from sklearn.metrics import mean_absolute_percentage_error, make_scorer


from sklearn.ensemble import RandomForestRegressor

import lightgbm as lgbm
from xgboost import XGBRegressor
from catboost import CatBoostRegressor, cv, Pool


data_dir = "/kaggle/input/playground-series-s5e1/"

train_set_loc = data_dir + "train.csv"

training_set = pd.read_csv(train_set_loc)

print("Total rows:", len(training_set))

training_set.head()


# Check NaN and remove
print("Number of NaN:", training_set.isna().sum())

training_set.dropna(inplace=True)

training_set


sns.scatterplot(x="date", y="num_sold", data=training_set[['date', 'num_sold']])


# Drop unnecessary id column
training_set.drop(columns=['id'], inplace=True)
training_set.head()


# See what unique values there are for each column
for col in list(training_set.columns):

    print(col+": "+str(training_set[col].unique()))


# See the number of unique values for each column
for col in list(training_set.columns):

    print(col+": "+str(training_set[col].nunique()))


# Select only discrete features to one hot encode
discreteFeatures = training_set[['date', 'country', 'store', 'product']]
encodedFeatures = pd.get_dummies(discreteFeatures)

encodedFeatures.head()


data_dir = "/kaggle/input/playground-series-s5e1/"

test_set_loc = data_dir + "test.csv"

test_set = pd.read_csv(test_set_loc)

test_set_ids = test_set['id']

print("Total rows:", len(test_set))

# Select only discrete features to one hot encode
discreteFeatures_test = test_set[['date', 'country', 'store', 'product']]
encodedFeatures_test = pd.get_dummies(discreteFeatures_test)


test_features_df = pd.DataFrame(columns=[col for col in encodedFeatures_test.columns if col not in encodedFeatures.columns])

merged_training_set = pd.concat([encodedFeatures, test_features_df], axis=1)
merged_training_set = merged_training_set.sort_index(axis=1)


# Check if all columns are now unique
(merged_training_set.columns.value_counts() == 1).all()


# Check for NaN
print("Number of NaN:", merged_training_set.isna().sum())


# Fill NAN in merged features
merged_training_set.fillna(False, inplace=True)
print("Number of NaN:", merged_training_set.isna().sum())


merged_training_set.isna().sum().value_counts()


# Safe to become new training set dataframe if all truely unique columns
encodedFeatures = merged_training_set
encodedFeatures.head()


training_features_df = pd.DataFrame(columns=[col for col in encodedFeatures.columns if col not in encodedFeatures_test.columns])

merged_test_set = pd.concat([encodedFeatures_test, training_features_df], axis=1)
merged_test_set = merged_test_set.sort_index(axis=1)


# Check if all columns are now unique
(merged_test_set.columns.value_counts() == 1).all()


# Check for NaN
print("Number of NaN:", merged_test_set.isna().sum())


# Fill NAN in merged features
merged_test_set.fillna(False, inplace=True)
print("Number of NaN:", merged_test_set.isna().sum())


merged_training_set.isna().sum().value_counts()


# Safe to become new testing set dataframe if all truely unique columns
X_test = merged_test_set
X_test.head()


num_rows_for_benchmark = 50000

X = encodedFeatures[0:num_rows_for_benchmark]
Y = training_set['num_sold'][0:num_rows_for_benchmark]

x_train, x_val, y_train, y_val = train_test_split(X, Y, test_size=0.2, random_state=42)


kFolds = KFold(n_splits=5, shuffle=True, random_state=42)


# MAPE scorer
mape_scorer = make_scorer(mean_absolute_percentage_error, greater_is_better=False)


# Random Forest
randomForest = RandomForestRegressor(random_state=42)

crossValidateRandomForest = cross_validate(randomForest, x_train, y_train, cv=kFolds, scoring=mape_scorer, return_estimator=True)

# Get best model from kfolds approach
bestRandomForestModel = crossValidateRandomForest['estimator'][np.argmax(crossValidateRandomForest['test_score'])]

y_pred = bestRandomForestModel.predict(x_val)
randomForest_mape = mean_absolute_percentage_error(y_val, y_pred)

randomForest_mape


print("Cross Validation MAPE Scores:")
print(crossValidateRandomForest['test_score'])

print("Avg MAPE:")
print(np.mean(crossValidateRandomForest['test_score']))

print("Best Model Eval MAPE:", randomForest_mape)


# Light GBM
lgbmModel = lgbm.LGBMRegressor(learning_rate=0.1, random_state=42, verbose=-1)

crossValidateLGBM = cross_validate(lgbmModel, x_train, y_train, cv=kFolds, scoring=mape_scorer, return_estimator=True)

# Get best model from kfolds approach
bestLGBMModel = crossValidateLGBM['estimator'][np.argmax(crossValidateLGBM['test_score'])]

y_pred = bestLGBMModel.predict(x_val)
lgbmModel_mape = mean_absolute_percentage_error(y_val, y_pred)

#lgbmModel_mse

print("Cross Validation MAPE Scores:")
print(crossValidateLGBM['test_score'])

print("Avg MAPE:")
print(np.mean(crossValidateLGBM['test_score']))

print("Best LGBM Model Eval MAPE:", lgbmModel_mape)


# XGBoost
xgbModel = XGBRegressor(learning_rate=0.3, random_state=42)
crossValidateXGB = cross_validate(xgbModel, x_train, y_train, cv=kFolds, scoring=mape_scorer, return_estimator=True)

# Get best model from kfolds approach

bestXGBModel = crossValidateXGB['estimator'][np.argmax(crossValidateXGB['test_score'])]


y_pred = bestXGBModel.predict(x_val)
xgbModel_mape = mean_absolute_percentage_error(y_val, y_pred)

xgbModel_mape

print("Cross Validation MAPE Scores:")
print(crossValidateXGB['test_score'])

print("Avg MAPE:")
print(np.mean(crossValidateXGB['test_score']))

print("Best XGB Model Eval MAPE:", xgbModel_mape)


# CatBoost
catboost_pool = Pool(data=x_train, label=y_train, cat_features=list(x_train.columns))

params = {
    'iterations': 100,
    'learning_rate': 0.03,
    'loss_function': 'MAPE'
}

crossValidateCatBoost = cv(params=params, pool=catboost_pool, folds=kFolds, return_models=True, verbose=False)

#crossValidateCatBoost[0].describe()

#bestCatBoostModel = crossValidateCatBoost[0].iloc[np.argmin(crossValidateCatBoost[0]['test-MAPE-mean'])]

crossValidateCatBoost


# Manually determine best model from Catboost and test on validation data
best_cat_select = 0 # Select based on best fold in previous cell output

bestCatBoostModel = crossValidateCatBoost[1][best_cat_select]

y_pred = bestCatBoostModel.predict(x_val)
catBoostModel_mape = mean_absolute_percentage_error(y_val, y_pred)

catBoostModel_mape


# Determine which model is best for generalized adaptability based on RMSE
print("MAPE Values For Considered Models")
print("================================")
print("Random Forest:", randomForest_mape)
print("LGBM:", lgbmModel_mape)
print("XGB:", xgbModel_mape)
print("Cat Boost:", catBoostModel_mape)


# Create a halving search to get the best hyperparameters to use
bestModel = RandomForestRegressor(random_state=42)

params = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}

hyperparameter_search = HalvingGridSearchCV(estimator=bestModel, param_grid=params, scoring=mape_scorer, random_state=42, cv=5)

hyperparameter_search.fit(x_train, y_train)

bestModel = hyperparameter_search.best_estimator_
bestParameters = hyperparameter_search.best_params_

print(hyperparameter_search.best_score_)
print(bestParameters)


y_pred = bestModel.predict(x_val)
bestModel_mape = mean_absolute_percentage_error(y_val, y_pred)

print("Best Parameters:", bestParameters)
print("Best Model MAPE:", bestModel_mape)


# Load full dataset as training set
X = encodedFeatures
Y = training_set['num_sold']

x_train, x_val, y_train, y_val = train_test_split(X, Y, test_size=0.2, random_state=42)


# Best Model Final Train
bestModel = RandomForestRegressor(**bestParameters, random_state=42)

# Get best model from kfolds approach
bestModel.fit(x_train, y_train)

y_pred = bestModel.predict(x_val)
best_mape = mean_absolute_percentage_error(y_val, y_pred)

best_mape


# Get encoded features from test set
X_test.head()


y_infer = bestModel.predict(X_test) # X_test is from the data import earlier in the notebook


# Create results dataframe
results = pd.DataFrame({
    'id': test_set_ids,
    'num_sold': y_infer
})

results


# Save results as results format (CSV)
results.to_csv('/kaggle/working/submission.csv', index=False)


pd.read_csv('/kaggle/working/submission.csv')

