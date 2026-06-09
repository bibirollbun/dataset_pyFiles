import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb

from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_log_error as msle
from sklearn.preprocessing import LabelEncoder


train = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv', index_col = 'id')
test = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv', index_col = 'id')
original = pd.read_csv('/kaggle/input/calories-burnt-prediction/calories.csv').drop(columns = ['User_ID']).rename(columns = {'Gender': 'Sex'})
sample = pd.read_csv('/kaggle/input/playground-series-s5e5/sample_submission.csv')

train = pd.concat([train, original], axis = 0, ignore_index = True).drop_duplicates().reset_index(drop = True)


print(f"{train[train.duplicated(keep = False)]}\n")
train.head()


train.info()


import warnings

# look at the numeric columns
def num_plotter(data, target = None):
    for col in data.select_dtypes(["int", "float"]):
        if col != target:
            plt.figure(figsize = (6,1))
            sns.boxplot(data = data, x = col, y = target)
            plt.show();

with warnings.catch_warnings(): # Disabling FutureWarning for the plots
    warnings.simplefilter(action = 'ignore', category = FutureWarning)
    num_plotter(train, target = 'Sex')


# look at the non-numeric columns
def cat_bar_plotter(df, normalize = False):
    for col in df.select_dtypes("object").columns:
        plt.figure(figsize = (6,3))
        df[col].value_counts(normalize = normalize, dropna = False).plot.bar()
        plt.show();

cat_bar_plotter(train, normalize = True)


def preprocess(df):
    df['BMI'] = df['Weight'] / (df['Height'] / 100) ** 2
    df['BMR_HarrisBenedict'] = np.where(df['Sex'] == 0,
                                (9.247 * df['Weight'] + 3.098 * df['Height'] - 4.33 * df['Age'] + 447.593),
                                (13.397 * df['Weight'] + 4.799 * df['Height'] - 5.677 * df['Age'] + 88.362)
                               )
    return df

train = preprocess(train)
test = preprocess(test)


def corr_mat(df):
    c_mat = df.corr(numeric_only = True)
    sns.heatmap(c_mat, mask = np.triu(np.ones_like(c_mat, dtype = bool)), vmin = -1, vmax = 1, annot = True, cmap = 'coolwarm');

corr_mat(train)


from itertools import combinations

obj_columns = ['Sex']
encode_columns = ['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp']
encoded_columns = []
pair_size = [2, 3, 4]

for r in pair_size:
    for col in list(combinations(encode_columns, r)):
        new_col_name = '_'.join(col)
        encoded_columns.append(new_col_name)
        if r == 2:
            train[new_col_name] = (train[col[0]] * train[col[1]]) ** 0.5
            test[new_col_name] = (test[col[0]] * test[col[1]]) ** 0.5
        if r == 3:
            train[new_col_name] = (train[col[0]] * train[col[1]] * train[col[2]]) ** 0.5
            test[new_col_name] = (test[col[0]] * test[col[1]] * test[col[2]]) ** 0.5
        if r == 4:
            train[new_col_name] = (train[col[0]] * train[col[1]] * train[col[2]] * train[col[3]]) ** 0.5
            test[new_col_name] = (test[col[0]] * test[col[1]] * test[col[2]] * test[col[3]]) ** 0.5

# Encoder for categorical data
label_encoders = {col: LabelEncoder() for col in obj_columns}

for col in obj_columns:
    train[col] = label_encoders[col].fit_transform(train[col])
    test[col] = label_encoders[col].transform(test[col])


def msle_objective(target, prediction):
    prediction = np.maximum(prediction, -1 + 1e-6)
    gradient = 2 * (np.log1p(prediction) - np.log1p(target)) / (prediction + 1)
    hessian = 2 * (-np.log1p(prediction) + np.log1p(target) + 1) / ((prediction + 1) ** 2)

    return gradient, hessian

def rmsle_eval(target, prediction):
    preds = np.maximum(prediction, -1 + 1e-6)
    squared_log_error = np.power((np.log1p(preds) - np.log1p(target)), 2)
    rmsle = np.sqrt(np.mean(squared_log_error))

    return "rmsle", rmsle, False

params = {
    'n_estimators': 10_000,
    'learning_rate': 0.02,
    'objective': msle_objective,
    'metric': 'custom',
    'max_depth': -1,
    'num_leaves': 512,
    'colsample_bytree': 0.7,
    'max_bin': 1024,
    'verbosity': -1,
}
fit_params = {
    'eval_metric': rmsle_eval,
    'callbacks': [lgb.early_stopping(50), lgb.log_evaluation(250)]
}

X = train.drop(columns = ['Calories'])
y = train['Calories']


def cross_val(X, y, n_splits):
    kf = KFold(n_splits = n_splits, shuffle = True, random_state = 404)
    y_pred = np.zeros(len(sample))
    cv_rmsle = []
    
    for train_ind, val_ind in kf.split(X, y):
        # Subset data based on CV folds
        X_train, y_train = X.iloc[train_ind], y.iloc[train_ind]
        X_val, y_val = X.iloc[val_ind], y.iloc[val_ind]
        # Fit the Model on fold's training data
        model = lgb.LGBMRegressor(**params).fit(**fit_params, X = X_train, y = y_train, eval_set = [(X_val, y_val)])
        cv_rmsle.append(msle(np.maximum(y_val, 0), np.maximum(model.predict(X_val), 0)) ** 0.5)
        y_pred += np.maximum(model.predict(test), 0)

    print(f"All Validation RMSLE: {[round(x, 5) for x in cv_rmsle]}")
    print(f"Cross Val RMSLE: {np.mean(cv_rmsle):.5f} +/- {np.std(cv_rmsle):.5f}")
        
    return y_pred / 5

final_pred = cross_val(X, y, 5)


# importance_types = ['split', 'gain']

# booster = model.booster_

# features = booster.feature_name()

# for itype in importance_types:
#     score = booster.feature_importance(importance_type=itype)
#     score_series = pd.Series(score, index=features).sort_values(ascending=False)

#     plt.figure(figsize=(10, 6))
#     score_series.head(30).plot(kind='bar')
#     plt.title(f"Feature Importance - {itype}")
#     plt.ylabel(itype)
#     plt.xlabel("Features")
#     plt.tight_layout()
#     plt.show()


sample['Calories'] = final_pred
sample.to_csv('submission.csv',index = False)




