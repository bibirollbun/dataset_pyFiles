Model = 'LGBM'
PROBLEM = 'Regression'


%load_ext cudf.pandas


import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import os
from glob import glob

import matplotlib.pyplot as plt
import seaborn as sns

# from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
# from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

# from xgboost import XGBClassifier
# from xgboost import XGBRegressor

# from sklearn.linear_model import LinearRegression

# from cuml.linear_model import LinearRegression as cuLinearRegression

# from cuml.linear_model import Ridge as cuRidge

# from cuml.linear_model import Lasso as cuLasso

# from cuml.linear_model import ElasticNet as cuElasticNet

import lightgbm as lgb
from lightgbm import early_stopping, log_evaluation

#from sklearn.neural_network import MLPRegressor

# --- set environment variables first ---
# os.environ["TF_DETERMINISTIC_OPS"] = "1"  # ensure deterministic GPU ops
# os.environ["PYTHONHASHSEED"] = "42"       # reproducible hashing
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # must be set before tf import
# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense

# from catboost import CatBoostRegressor

# from sklearn.ensemble import BaggingRegressor

# from sklearn.ensemble import RandomForestRegressor

# from sklearn.ensemble import ExtraTreesRegressor

# from cuml.ensemble import RandomForestRegressor as cuRF

# from sklearn.ensemble import AdaBoostRegressor

# from sklearn.ensemble import GradientBoostingRegressor

# from cuml.svm import LinearSVR

# from cuml.ensemble import RandomForestRegressor

# from cuml.neighbors import KNeighborsRegressor

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder, OneHotEncoder, PowerTransformer
from scipy.stats import zscore, boxcox, entropy
from sklearn.tree import DecisionTreeRegressor

from tqdm import tqdm

from math import comb
from itertools import combinations, combinations_with_replacement

from sklearn.preprocessing import StandardScaler, MinMaxScaler

#set random seed comment during  final model training
import random
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
#tf.random.set_seed(SEED)


train = pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv')

# Combining output from all models
# train = pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv')

# input_dir = '/kaggle/input/playground-s5-e9-ensemble/'

# csv_files = glob(os.path.join(input_dir, 'X*.csv'))
# dfs = []
# dfs_col_name = []
# for file in csv_files:
#     df = pd.read_csv(file)
#     col_name = os.path.splitext(os.path.basename(file))[0]
#     dfs.append(df)
#     dfs_col_name.append(col_name)
# X = pd.concat(dfs, axis=1)
# X.columns = dfs_col_name

# csv_files = glob(os.path.join(input_dir, 'test*.csv'))
# dfs = []
# for file in csv_files:
#     df = pd.read_csv(file)
#     df.drop(['id'], axis = 1, inplace = True)
#     col_name = os.path.splitext(os.path.basename(file))[0]
#     dfs.append(df)
# test = pd.concat(dfs, axis=1)
# test.columns = dfs_col_name


print('Train:', train.shape, '\nTest: ', test.shape)

# for ensemble
# print('X:', X.shape, '\nTest: ', test.shape)


train.head()


test.head()


train.info()


test.info()


train.describe()


test.describe()


for column in train.columns[train.dtypes == 'object']:
    print(train[column].value_counts(),'\n')


train.isna().sum()


test.isna().sum()


# Ensure all data are numeric floats
corr = train.select_dtypes(['int64', 'float64']).astype(float).corr()

# Plot heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, square=True)
plt.title("Correlation Heatmap", fontsize=16)
plt.show()


continuous_features = train.select_dtypes(include=['int64','float64']).columns
n = len(continuous_features)
n_cols = 2
n_rows = int(np.ceil(n / n_cols))

plt.figure(figsize=(16, n_rows * 4))

for i, feature in enumerate(continuous_features, 1):
    ax = plt.subplot(n_rows, n_cols, i)

    # Simple boxplot (no stat annotations)
    sns.boxplot(x=train[feature], orient="h", color="skyblue", whis=1.5, showfliers=True, ax=ax)
    ax.set_title(f'Boxplot of {feature}')
    ax.set_xlabel('Value')

    # Compute whisker positions (last inliers inside Tukey fences)
    s = train[feature].dropna().astype(float)
    q1 = s.quantile(0.25)
    q3 = s.quantile(0.75)
    iqr = q3 - q1
    lower_fence = q1 - 1.5 * iqr
    upper_fence = q3 + 1.5 * iqr
    lower_whisker = s[s >= lower_fence].min()
    upper_whisker = s[s <= upper_fence].max()

    # Draw lines at the actual whisker ends (these will align with the plot)
    ax.axvline(lower_whisker, linestyle="--", color="red", alpha=0.8, label=f"Lower whisker ({lower_whisker:.2f})")
    ax.axvline(upper_whisker, linestyle="--", color="red", alpha=0.8, label=f"Upper whisker ({upper_whisker:.2f})")
    ax.legend(loc="upper right", fontsize=9)

    # Count of the outliers
    print(f'Outliers count in {feature}:',train[train[feature]>upper_whisker].shape[0]+train[train[feature]<lower_whisker].shape[0])

plt.tight_layout()
plt.show()


outlier_features = ['num_reported_accidents', 'accident_risk']

for i, feature in enumerate(outlier_features, 1):
    # Compute whisker positions (last inliers inside Tukey fences)
    s = train[feature].dropna().astype(float)
    q1 = s.quantile(0.25)
    q3 = s.quantile(0.75)
    iqr = q3 - q1
    lower_fence = q1 - 1.5 * iqr
    upper_fence = q3 + 1.5 * iqr
    lower_whisker = s[s >= lower_fence].min()
    upper_whisker = s[s <= upper_fence].max()

    print(f'Outliers count in {feature}:',train[train[feature]>upper_whisker].shape[0]+train[train[feature]<lower_whisker].shape[0])

    # Find outliers
    outlier_mask = (train[feature] < lower_whisker) | (train[feature] > upper_whisker)
    outlier_indices = train.index[outlier_mask].tolist()

    # print(f'Outlier indices: {outlier_indices[:10]}')
    print(f'Whiskers are {lower_whisker} and {upper_whisker}')


train.iloc[101,:]


train.iloc[956,:]


# Select continuous (numeric) features
cont_features = train.select_dtypes(include=['int64', 'float64']).columns

n_features = len(cont_features)
n_cols = 2  # plots per row
n_rows = int(np.ceil(n_features / n_cols))

plt.figure(figsize=(14, n_rows * 4))

for i, feature in enumerate(cont_features, 1):
    plt.subplot(n_rows, n_cols, i)
    
    # Plot histogram with KDE
    sns.histplot(train[feature], bins=30, kde=True, color="skyblue")
    
    # Skewness
    skew_val = train[feature].skew()
    
    # Title with skewness
    plt.title(f"Distribution of {feature}\nSkewness = {skew_val:.2f}", fontsize=12)
    plt.xlabel("")
    plt.ylabel("Frequency")

plt.tight_layout()
plt.show()


def rmse(y_true,y_pred):
    return np.sqrt(mean_squared_error(y_true,y_pred))


def test_features(X_train, X_val, y_train, y_val,enable_categorical):
    X_train_copy=X_train.copy()
    X_val_copy=X_val.copy()
    for col in X_train_copy.select_dtypes(include=['object','bool']).columns:
        X_train_copy[col] = X_train_copy[col].astype('category')
        X_val_copy[col] = X_val_copy[col].astype('category')
    # model = XGBRegressor(random_state=42,enable_categorical=enable_categorical,tree_method='gpu_hist')
    # model = XGBRegressor(random_state=42,enable_categorical=enable_categorical)
    # model.fit(X_train_copy, y_train, eval_set=[(X_val_copy,y_val)], verbose=0)

    # model = LinearRegression()
    # model.fit(X_train_copy, y_train)

    # model = cuLinearRegression()
    # cat_features = [col for i, col in enumerate(X_train_copy.columns) if str(X_train_copy[col].dtype) == 'category']
    # X_train_copy.drop(cat_features, axis = 1, inplace = True)
    # X_val_copy.drop(cat_features, axis = 1, inplace = True)
    # X_train_copy.fillna(0, inplace=True)
    # X_val_copy.fillna(0, inplace=True)
    # model.fit(X_train_copy, y_train)
    
    # model = cuRidge(alpha = 10)
    # cat_features = [col for i, col in enumerate(X_train_copy.columns) if str(X_train_copy[col].dtype) == 'category']
    # X_train_copy.drop(cat_features, axis = 1, inplace = True)
    # X_val_copy.drop(cat_features, axis = 1, inplace = True)
    # X_train_copy.fillna(0, inplace=True)
    # X_val_copy.fillna(0, inplace=True)
    # model.fit(X_train_copy, y_train)
    
    # model = cuLasso()
    # cat_features = [col for i, col in enumerate(X_train_copy.columns) if str(X_train_copy[col].dtype) == 'category']
    # X_train_copy.drop(cat_features, axis = 1, inplace = True)
    # X_val_copy.drop(cat_features, axis = 1, inplace = True)
    # X_train_copy.fillna(0, inplace=True)
    # X_val_copy.fillna(0, inplace=True)
    # model.fit(X_train_copy, y_train)
    
    # model = cuElasticNet()
    # cat_features = [col for i, col in enumerate(X_train_copy.columns) if str(X_train_copy[col].dtype) == 'category']
    # X_train_copy.drop(cat_features, axis = 1, inplace = True)
    # X_val_copy.drop(cat_features, axis = 1, inplace = True)
    # X_train_copy.fillna(0, inplace=True)
    # X_val_copy.fillna(0, inplace=True)
    # model.fit(X_train_copy, y_train)

    model = lgb.LGBMRegressor(
        objective="regression",
        random_state=42,
        #device="gpu",
        verbose=-1
    )
    model.fit(
        X_train_copy, y_train,
        eval_set=[(X_val_copy, y_val)],
        eval_metric="rmse",
        callbacks=[log_evaluation(0)]
    )

    # model = MLPRegressor(
    #     hidden_layer_sizes=(2),   # 2 hidden layers: 100 and 50 neurons
    #     activation="relu",              # activation function
    #     solver="adam",                  # optimizer
    #     max_iter=300,
    #     learning_rate_init=0.001,
    #     random_state=42,
    #     early_stopping=True,   # ✅ enables early stopping
    #     n_iter_no_change=10,   # stop if no improvement for 10 epochs
    #     tol=1e-4       
    #     )
    # model.fit(X_train_copy, y_train)

    # model = Sequential([
    #     Dense(2, activation='relu', input_shape=(X_train.shape[1],)),
    #     #Dense(64, activation='relu'),
    #     Dense(1)  # single neuron for regression output
    # ])
    # model.compile(
    #     optimizer='adam',
    #     loss='mse',         # Mean Squared Error for regression
    #     metrics=['mse']     # Mean Absolute Error as an extra metric
    # )
    # history = model.fit(
    #     X_train_copy, y_train,
    #     validation_data=(X_val_copy, y_val),
    #     epochs=1,
    #     batch_size=512,
    #     verbose=-1,
    #     shuffle=False
    # )
    
    # model = CatBoostRegressor(
    #     loss_function='RMSE',
    #     iterations=1500,
    #     learning_rate=0.03,
    #     depth=6,
    #     l2_leaf_reg=3,
    #     random_seed=42,
    #     eval_metric='RMSE',
    #     verbose=False,
    #     early_stopping_rounds=100
    # )
    # cat_features = [i for i, col in enumerate(X_train_copy.columns) if str(X_train_copy[col].dtype) == 'category']
    # model.fit(
    #     X_train_copy, y_train,
    #     eval_set=[(X_val_copy, y_val)],
    #     cat_features=cat_features,
    #     verbose=False
    # )

    # model = BaggingRegressor()
    # cat_features = [col for i, col in enumerate(X_train_copy.columns) if str(X_train_copy[col].dtype) == 'category']
    # X_train_copy.drop(cat_features, axis = 1, inplace = True)
    # X_val_copy.drop(cat_features, axis = 1, inplace = True)
    # model.fit(X_train_copy, y_train)

    # model = RandomForestRegressor()
    # cat_features = [col for i, col in enumerate(X_train_copy.columns) if str(X_train_copy[col].dtype) == 'category']
    # X_train_copy.drop(cat_features, axis = 1, inplace = True)
    # X_val_copy.drop(cat_features, axis = 1, inplace = True)
    # model.fit(X_train_copy, y_train)

    # model = cuRF(
    #     n_estimators=32,
    #     max_depth=8,
    #     n_streams=1,
    #     bootstrap=False,      # to behave more like ExtraTrees
    #     #split_criterion="entropy"
    #     random_state = 42
    # )
    # cat_features = [col for i, col in enumerate(X_train_copy.columns) if str(X_train_copy[col].dtype) == 'category']
    # X_train_copy.drop(cat_features, axis = 1, inplace = True)
    # X_val_copy.drop(cat_features, axis = 1, inplace = True)
    # model.fit(X_train_copy, y_train)

    # model = AdaBoostRegressor(n_estimators = 1, random_state = 42)
    # cat_features = [col for i, col in enumerate(X_train_copy.columns) if str(X_train_copy[col].dtype) == 'category']
    # X_train_copy.drop(cat_features, axis = 1, inplace = True)
    # X_val_copy.drop(cat_features, axis = 1, inplace = True)
    # X_train_copy.fillna(0, inplace=True)
    # X_val_copy.fillna(0, inplace=True)
    # model.fit(X_train_copy, y_train)

    # model = GradientBoostingRegressor(n_estimators = 1, random_state = 42)
    # cat_features = [col for i, col in enumerate(X_train_copy.columns) if str(X_train_copy[col].dtype) == 'category']
    # X_train_copy.drop(cat_features, axis = 1, inplace = True)
    # X_val_copy.drop(cat_features, axis = 1, inplace = True)
    # X_train_copy.fillna(0, inplace=True)
    # X_val_copy.fillna(0, inplace=True)
    # model.fit(X_train_copy, y_train)

    # model = KNeighborsRegressor(verbose = 0)
    # cat_features = [col for i, col in enumerate(X_train_copy.columns) if str(X_train_copy[col].dtype) == 'category']
    # X_train_copy.drop(cat_features, axis = 1, inplace = True)
    # X_val_copy.drop(cat_features, axis = 1, inplace = True)
    # X_train_copy.fillna(0, inplace=True)
    # X_val_copy.fillna(0, inplace=True)
    # model.fit(X_train_copy, y_train)

    # model = LinearSVR(verbose = 0)
    # cat_features = [col for i, col in enumerate(X_train_copy.columns) if str(X_train_copy[col].dtype) == 'category']
    # X_train_copy.drop(cat_features, axis = 1, inplace = True)
    # X_val_copy.drop(cat_features, axis = 1, inplace = True)
    # X_train_copy.fillna(0, inplace=True)
    # X_val_copy.fillna(0, inplace=True)
    # model.fit(X_train_copy, y_train)

    # Decision Tree
    # model = RandomForestRegressor(n_estimators=1, bootstrap=False, random_state=42)
    # cat_features = [col for i, col in enumerate(X_train_copy.columns) if str(X_train_copy[col].dtype) == 'category']
    # X_train_copy.drop(cat_features, axis = 1, inplace = True)
    # X_val_copy.drop(cat_features, axis = 1, inplace = True)
    # X_train_copy.fillna(0, inplace=True)
    # X_val_copy.fillna(0, inplace=True)
    # model.fit(X_train_copy, y_train)

    # accuracy
    # y_pred = model.predict(X_val_copy)
    # print(accuracy_score(y_val,y_pred))
    # return accuracy_score(y_val,y_pred)

    # rmse
    y_pred = model.predict(X_val_copy)
    #print(rmse(y_val,y_pred))

    # y_pred.fillna(y_pred.mean(), inplace = True)
    return rmse(y_val,y_pred)
    
    # roc-auc
    # y_pred_proba = model.predict_proba(X_val_copy)[:, 1]
    # print(roc_auc_score(y_val, y_pred_proba))
    # return roc_auc_score(y_val, y_pred_proba)


X, y = train.drop(['accident_risk'], axis=1), train['accident_risk']#(train['accident_risk']=='Extrovert').astype(int) 

# for ensemble
# y = train['accident_risk'] # (train['accident_risk']=='Extrovert').astype(int) 


X_train, X_val, y_train, y_val = train_test_split(X, y, shuffle=True, random_state = 42)
baseline = test_features(X_train, X_val, y_train, y_val, True)
print(baseline)


X.drop(['id'],axis = 1, inplace=True)
test.drop(['id'],axis = 1, inplace=True)


X_train, X_val, y_train, y_val = train_test_split(X, y, shuffle=True, random_state = 42)
score = test_features(X_train, X_val, y_train, y_val, True)
if score<baseline:
    print(f'Improved baseline to {score} from {baseline}')
    baseline = score


train_original = pd.read_csv('/kaggle/input/simulated-roads-accident-data/synthetic_road_accidents_100k.csv')


X_original, y_original = train_original.drop(['accident_risk'], axis=1), train_original['accident_risk']#(train_original['accident_risk']=='Extrovert').astype(int)
X_temp = pd.concat([X, X_original])
y_temp = pd.concat([y, y_original])

X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, shuffle=True, random_state = 42)
score = test_features(X_train, X_val, y_train, y_val, True)
if score<baseline:
    print(f'Improved baseline from {baseline} to {score}')
    baseline = score
    X = X_temp
    y = y_temp


X.isna().sum()


# use value other than 1 as it affects skewness calculation
fill_values = [-1, 'mean', 'median', 'mode']

for column in X.select_dtypes(['float64']):
    print(column)
    X_prev = X.copy()
    agg_score = {}
    for agg in fill_values:
        print(agg)
        X_copy = X_prev.copy()
        if agg == -1:
            X_copy[column] = X_copy[column].fillna(-1)
        if agg == 'mean':
            X_copy[column] = X_copy[column].fillna(X_copy[column].mean())
        if agg == 'median':
            X_copy[column] = X_copy[column].fillna(X_copy[column].median())
        if agg == 'mode':
            X_copy[column] = X_copy[column].fillna(X_copy[column].mode()[0])
    
        X_train, X_val, y_train, y_val = train_test_split(X_copy, y, shuffle=True, random_state = 42)
        score_agg = test_features(X_train, X_val, y_train, y_val, True)
        agg_score[agg] = score_agg

    # below section will change depending on the metric used
    max_value = -1
    if agg_score:
        max_value = max(agg_score.values())
        max_keys = [k for k, v in agg_score.items() if v == max_value]

    if max_value>baseline:

        if X[column].skew() < 1 and X[column].skew() > -1 and 'mean' in max_keys:
            X[column] = X[column].fillna(X[column].mean())
            test[column] = test[column].fillna(test[column].mean())
            print(f'Improved baseline of {max_value} that {baseline} before using mean')
        elif (X[column].skew() >= 1 or X[column].skew() <= -1) and 'median' in max_keys:
            X[column] = X[column].fillna(X[column].median())
            test[column] = test[column].fillna(test[column].median())
            print(f'Improved baseline of {max_value} that {baseline} before using median')
        else:
            if 'mean' in max_keys:
                X[column] = X[column].fillna(X[column].mean())
                test[column] = test[column].fillna(test[column].mean())
                print(f'Improved baseline of {max_value} that {baseline} before using mean')
            elif 'median' in max_keys:
                X[column] = X[column].fillna(X[column].median())
                test[column] = test[column].fillna(test[column].median())
                print(f'Improved baseline of {max_value} that {baseline} before using median')
            elif 'mode' in max_keys:
                X[column] = X[column].fillna(X[column].mode()[0])
                test[column] = test[column].fillna(test[column].mode()[0])
                print(f'Improved baseline of {max_value} that {baseline} before using mode')
            elif -1 in max_keys:
                X[column] = X[column].fillna(-1)
                test[column] = test[column].fillna(-1)
                print(f'Improved baseline of {max_value} that {baseline} before using -1')   
        
        baseline=max_value


X.isna().sum()


for column in X.select_dtypes(['object']):
    print(column)
    X_copy = X.copy()
    mode_value = X_copy[column].mode(dropna=True)[0]
    X_copy[column] = X_copy[column].fillna(mode_value)
    
    X_train, X_val, y_train, y_val = train_test_split(X_copy, y, shuffle=True, random_state = 42)
    score = test_features(X_train, X_val, y_train, y_val, True)

    if score>baseline:
        print(f'Improved baseline of {score} that {baseline} before')
        baseline=score
        X[column].fillna(mode_value, inplace=True)
        test[column].fillna(mode_value, inplace=True)


for column in X.columns:
    print(column)
    X_copy = X.copy()
    if X[column].dtype == 'float64':
        if X[column].skew() > 0.5 or X[column].skew() < -0.5:
            X_copy[column].fillna('median', inplace=True)
            X_train, X_val, y_train, y_val = train_test_split(X_copy, y, shuffle=True, random_state = 42)
            score = test_features(X_train, X_val, y_train, y_val, True)
        
            if score>=baseline:
                print(f'Improved baseline of {score} that {baseline} before')
                baseline=score
                X[column].fillna('median', inplace=True)
                test[column].fillna('median', inplace=True)
        else:
            X_copy[column].fillna('mean', inplace=True)
            X_train, X_val, y_train, y_val = train_test_split(X_copy, y, shuffle=True, random_state = 42)
            score = test_features(X_train, X_val, y_train, y_val, True)
        
            if score>=baseline:
                print(f'Improved baseline of {score} that {baseline} before')
                baseline=score
                X[column].fillna('mean', inplace=True)
                test[column].fillna('mean', inplace=True)
    else:
        mode_value = X_copy[column].mode(dropna=True)[0]
        X_copy[column] = X_copy[column].fillna(mode_value)
        X_train, X_val, y_train, y_val = train_test_split(X_copy, y, shuffle=True, random_state = 42)
        score = test_features(X_train, X_val, y_train, y_val, True)
    
        if score>=baseline:
            print(f'Improved baseline of {score} that {baseline} before')
            baseline=score
            X[column].fillna(mode_value, inplace=True)
            test[column].fillna(mode_value, inplace=True)


stats = ["mean","std","count","nunique","median","min","max","skew"]
continuous_columns = X.select_dtypes(['float64','int64']).columns
categorical_columns = X.select_dtypes(['object','bool']).columns


X_copy = X.copy()
test_copy = test.copy()


baseline = 0.056306508018685725
X=X_copy.copy()
test=test_copy.copy()


replacements = ['mean', 'median', 'mode', 'cap']
columns = list(continuous_columns)
# columns = ['AudioLoudness', 'AcousticQuality', 'LivePerformanceLikelihood', 'MoodScore']

# [1] To keep the train data same for the next iteration
# X_reset_col = X.copy()
# # this will help with the final test
# X_testing = X.copy()

train_idx, val_idx = train_test_split(
    np.arange(len(X)), shuffle=True, random_state=42
)

for col in columns:

    # [1] To keep the train data same for the next iteration
    # X = X_reset_col.copy()
    
    if X[col].skew() > 1 or X[col].skew() < -1:
        s = X[col].dropna().astype(float)
        q1 = s.quantile(0.25)
        q3 = s.quantile(0.75)
        iqr = q3 - q1
        lower_fence = q1 - 1.5 * iqr
        upper_fence = q3 + 1.5 * iqr
        conditions = [
            X[col] > upper_fence,
            X[col] < lower_fence
        ]
        choices = ['over', 'under']
        outliers = np.select(conditions, choices, default='between')

        # for test
        s = test[col].dropna().astype(float)
        q1 = s.quantile(0.25)
        q3 = s.quantile(0.75)
        iqr = q3 - q1
        lower_fence = q1 - 1.5 * iqr
        upper_fence = q3 + 1.5 * iqr
        conditions = [
            test[col] > upper_fence,
            test[col] < lower_fence
        ]
        choices = ['over', 'under']
        outliers_test = np.select(conditions, choices, default='between')
        
    else:
        
        threshold = 3
        # df['is_outlier'] = np.abs(df['z_score']) > threshold
        conditions = [
            zscore(X[col]) > threshold,
            zscore(X[col]) < -threshold
        ]
        choices = ['over', 'under']
        outliers = np.select(conditions, choices, default='between')

        # test
        conditions = [
            zscore(test[col]) > threshold,
            zscore(test[col]) < -threshold
        ]
        choices = ['over', 'under']
        outliers_test = np.select(conditions, choices, default='between')

    if len(np.unique(outliers)) == 1:
        continue

    print(f'\nFor column {col}:')
    
    col_to_process = X[col]

    # [1] To keep the train data same for the next iteration
    # X_reset_rep = X.copy()
        
    replacement_score = {}
    original_dropped = {}
        
    for replacement in replacements:
        
        # [1] To keep the train data same for the next iteration
        # X = X_reset_rep.copy()
        
        print(f'For imputing strategy {replacement}:')
        
        if replacement == 'mean':
            mean_val = col_to_process.mean()
            X[f'{col}_imputed'] = np.where(outliers != 'between', mean_val, col_to_process)
        
        elif replacement == 'median':
            median_val = col_to_process.median()
            X[f'{col}_imputed'] = np.where(outliers != 'between', median_val, col_to_process)
    
        elif replacement == 'mode':
            mode_val = col_to_process.mode()[0]
            X[f'{col}_imputed'] = np.where(outliers != 'between', mode_val, col_to_process)
    
        elif replacement == 'cap':
            lower_cap, upper_cap = col_to_process.quantile(0.01), col_to_process.quantile(0.99)
            X[f'{col}_imputed'] = np.where(outliers == 'under', lower_cap,
                                   np.where(outliers == 'over', upper_cap, col_to_process))
            
        X_train, X_val, y_train, y_val = X.iloc[train_idx].copy(), X.iloc[val_idx].copy(), y.iloc[train_idx], y.iloc[val_idx]
        score_with_original = test_features(X_train, X_val, y_train, y_val, True)
        score_without_original = test_features(X_train.drop([col], axis = 1), X_val.drop([col], axis = 1), y_train, y_val, True)
        # will change depending on the error metric
        if score_without_original < score_with_original:
            replacement_score[replacement] = score_without_original
            original_dropped[replacement] = True
        else:
            replacement_score[replacement] = score_with_original
            original_dropped[replacement] = False
        X.drop([f'{col}_imputed'], axis = 1, inplace = True)
            
    # will change depending on the error metric
    max_value = 100
    if replacement_score:
        max_value = min(replacement_score.values())
        max_keys = [k for k, v in replacement_score.items() if v == max_value]
    
    if max_value<baseline:
        print(f'Improved baseline to {max_value} from {baseline}')
        baseline = max_value
        
        if Model in ['LGBM', 'XGB', 'AdaBoost'] and 'cap' in max_keys:
            lower_cap, upper_cap = col_to_process.quantile(0.01), col_to_process.quantile(0.99)
            X[f'{col}_imputed'] = np.where(outliers == 'under', lower_cap,
                                   np.where(outliers == 'over', upper_cap, col_to_process))
            lower_cap, upper_cap = test[col].quantile(0.01), test[col].quantile(0.99)
            test[f'{col}_imputed'] = np.where(outliers_test == 'under', lower_cap,
                                   np.where(outliers_test == 'over', upper_cap, test[col]))
            if original_dropped['cap']:
                X.drop([col], axis = 1, inplace = True)
                test.drop([col], axis = 1, inplace = True)
                # [1] To keep the train data same for the next iteration
                # lower_cap, upper_cap = X_testing[col].quantile(0.01), X_testing[col].quantile(0.99)
                # X_testing[f'{col}_imputed'] = np.where(outliers == 'under', lower_cap,
                #                        np.where(outliers == 'over', upper_cap, X_testing[col]))
    
        elif (X[col].skew() < 0.5 or X[col].skew() > -0.5) and 'mean' in max_keys:
            mean_val = col_to_process.mean()
            X[f'{col}_imputed'] = np.where(outliers != 'between', mean_val, col_to_process)
            mean_val = test[col].mean()
            test[f'{col}_imputed'] = np.where(outliers_test != 'between', mean_val, test[col])
            if original_dropped['mean']:
                X.drop([col], axis = 1, inplace = True)
                test.drop([col], axis = 1, inplace = True)
            # [1] To keep the train data same for the next iteration
            # mean_val = X_testing[col].mean()
            # X_testing[f'{col}_imputed'] = np.where(outliers != 'between', mean_val, X_testing[col])
            
        elif (X[col].skew() > 0.5 or X[col].skew() < -0.5) and 'median' in max_keys:
            median_val = col_to_process.median()
            X[f'{col}_imputed'] = np.where(outliers != 'between', median_val, col_to_process)
            median_val = test[col].median()
            test[f'{col}_imputed'] = np.where(outliers_test != 'between', median_val, test[col])
            if original_dropped['median']:
                X.drop([col], axis = 1, inplace = True)
                test.drop([col], axis = 1, inplace = True)
            # [1] To keep the train data same for the next iteration
            # median_val = X_testing[col].median()
            # X_testing[f'{col}_imputed'] = np.where(outliers != 'between', median_val, X_testing[col])
        
        else:
            if 'mode' in max_keys:
                mode_val = col_to_process.mode()[0]
                X[f'{col}_imputed'] = np.where(outliers != 'between', mode_val, col_to_process)
                mode_val = test[col].mode()[0]
                test[f'{col}_imputed'] = np.where(outliers_test != 'between', mode_val, test[col])
                if original_dropped['mode']:
                    X.drop([col], axis = 1, inplace = True)
                    test.drop([col], axis = 1, inplace = True)
                # [1] To keep the train data same for the next iteration
                # mode_val = X_testing[col].mode()[0]
                # X_testing[f'{col}_imputed'] = np.where(outliers != 'between', mode_val, X_testing[col])
        
            elif 'cap' in max_keys:
                lower_cap, upper_cap = col_to_process.quantile(0.01), col_to_process.quantile(0.99)
                X[f'{col}_imputed'] = np.where(outliers == 'under', lower_cap,
                                       np.where(outliers == 'over', upper_cap, col_to_process))
                lower_cap, upper_cap = test[col].quantile(0.01), test[col].quantile(0.99)
                test[f'{col}_imputed'] = np.where(outliers_test == 'under', lower_cap,
                                       np.where(outliers_test == 'over', upper_cap, test[col]))
                if original_dropped['cap']:
                    X.drop([col], axis = 1, inplace = True)
                    test.drop([col], axis = 1, inplace = True)
                    # [1] To keep the train data same for the next iteration
                    # lower_cap, upper_cap = X_testing[col].quantile(0.01), X_testing[col].quantile(0.99)
                    # X_testing[f'{col}_imputed'] = np.where(outliers == 'under', lower_cap,
                    #                        np.where(outliers == 'over', upper_cap, X_testing[col]))
            
            elif 'mean' in max_keys:
                mean_val = col_to_process.mean()
                X[f'{col}_imputed'] = np.where(outliers != 'between', mean_val, col_to_process)
                mean_val = test[col].mean()
                test[f'{col}_imputed'] = np.where(outliers_test != 'between', mean_val, test[col])
                if original_dropped['mean']:
                    X.drop([col], axis = 1, inplace = True)
                    test.drop([col], axis = 1, inplace = True)
                # [1] To keep the train data same for the next iteration
                # mean_val = X_testing[col].mean()
                # X_testing[f'{col}_imputed'] = np.where(outliers != 'between', mean_val, X_testing[col])
                
            elif 'median' in max_keys:
                median_val = col_to_process.median()
                X[f'{col}_imputed'] = np.where(outliers != 'between', median_val, col_to_process)
                median_val = test[col].median()
                test[f'{col}_imputed'] = np.where(outliers_test != 'between', median_val, test[col])
                if original_dropped['median']:
                    X.drop([col], axis = 1, inplace = True)
                    test.drop([col], axis = 1, inplace = True)
                # [1] To keep the train data same for the next iteration
                # median_val = X_testing[col].median()
                # X_testing[f'{col}_imputed'] = np.where(outliers != 'between', median_val, X_testing[col])

# [1] To keep the train data same for the next iteration              
# X_train, X_val, y_train, y_val = train_test_split(X_testing, y, shuffle=True, random_state = 42)
# score = test_features(X_train, X_val, y_train, y_val, True)
# print(f'Final score is {score}')


X.to_csv('X.csv', index=False)
y.to_csv('y.csv', index=False)
test.to_csv('test.csv', index=False)


if y.skew() > 1 or y.skew() < -1:
    s = y.dropna().astype(float)
    q1 = s.quantile(0.25)
    q3 = s.quantile(0.75)
    iqr = q3 - q1
    lower_fence = q1 - 1.5 * iqr
    upper_fence = q3 + 1.5 * iqr
    conditions = [
        y > upper_fence,
        y < lower_fence
    ]
    choices = ['over', 'under']
    outliers = np.select(conditions, choices, default='between')
    
else:  
    threshold = 3
    # df['is_outlier'] = np.abs(df['z_score']) > threshold
    conditions = [
        zscore(y) > threshold,
        zscore(y) < -threshold
    ]
    choices = ['over', 'under']
    outliers = np.select(conditions, choices, default='between')

train_idx, val_idx = train_test_split(
    np.arange(len(X)), shuffle=True, random_state=42
)

X_train, X_val, y_train, y_val = X.iloc[train_idx].copy(), X.iloc[val_idx].copy(), y.iloc[train_idx], y.iloc[val_idx]
outliers_train = outliers[train_idx].copy()

X_train = X_train[(outliers_train != 'over') & (outliers_train != 'under')]
y_train = y_train[(outliers_train != 'over') & (outliers_train != 'under')]
score = test_features(X_train, X_val, y_train, y_val, True)

if score<baseline:
    print(f'Improved baseline to {score} from {baseline}')
    baseline = score
    X = X[(outliers != 'over') & (outliers != 'under')]
    y = y[(outliers != 'over') & (outliers != 'under')]


X.to_csv('X.csv', index=False)
y.to_csv('y.csv', index=False)
test.to_csv('test.csv', index=False)


X.select_dtypes(include=['float64','int64']).columns


X = pd.read_csv('X.csv')
y = pd.read_csv('y.csv').squeeze()  # Converts back to Series if needed
test = pd.read_csv('test.csv')

# continuous_columns = ['num_lanes', 'curvature', 'speed_limit', 'num_reported_accidents_imputed']

baseline = 0.05637163848437093


columns = list(continuous_columns)
# columns = ['X_LGBM', 'X_linear', 'X_XGB_CPU', 'X_Ridge', 'X_XGB',
#        'X_Elastic']
skew_corc = ['log', 'sqrt', 'boxcox', 'yeo-johnson']

train_idx, val_idx = train_test_split(
    np.arange(len(X)), shuffle=True, random_state=42
)

for col in columns:

    if X[col].skew() < 0.5 and X[col].skew() > -0.5:
        continue
        
    print(f'\nFor column {col}:')
    
    corc_score = {}
    original_dropped = {}
    
    if X[col].isna().sum()>0:
        continue

    for corc in skew_corc:
        print(f'For correction strategy {corc}:')
        
        if (X[col] >= 0).all() and corc == 'log':
            X[f'{col}_cor'] = np.log1p(X[col])
        elif (X[col] >= 0).all() and corc == 'sqrt':
            X[f'{col}_cor'] = np.sqrt(X[col])
        elif (X[col] > 0).all() and corc == 'boxcox':
            X[f'{col}_cor'],_ = boxcox(X[col])
        elif corc == 'yeo-johnson':
            pt = PowerTransformer(method='yeo-johnson')
            X[f'{col}_cor'] = pt.fit_transform(X[[col]]).flatten()
        else:
            continue
    
        X_train, X_val, y_train, y_val = X.iloc[train_idx].copy(), X.iloc[val_idx].copy(), y.iloc[train_idx], y.iloc[val_idx]
        
        score_with_original = test_features(X_train, X_val, y_train, y_val, True)
        score_without_original = test_features(X_train.drop([col], axis = 1), X_val.drop([col], axis = 1), y_train, y_val, True)
        # will change depending on the error metric
        if score_without_original < score_with_original:
            corc_score[corc] = score_without_original
            original_dropped[corc] = True
        else:
            corc_score[corc] = score_with_original
            original_dropped[corc] = False

        X.drop([f'{col}_cor'], axis=1, inplace=True)

    # will change depending on the error metric
    max_value = 100
    if corc_score:
        max_value = min(corc_score.values())
        max_keys = [k for k, v in corc_score.items() if v == max_value]
        
    if max_value<baseline:

        if X[col].skew() >= 1 and 'log' in max_keys:
            X[f'{col}_cor'] = np.log1p(X[col])
            test[f'{col}_cor'] = np.log1p(test[col])
            if original_dropped['log']:
                X.drop([col], axis = 1, inplace = True)
                test.drop([col], axis = 1, inplace = True)
            print(f'Improved baseline to {max_value} from {baseline} using log')
        elif X[col].skew() < 1 and X[col].skew() > 0.5 and 'sqrt' in max_keys:
            X[f'{col}_cor'] = np.sqrt(X[col])
            test[f'{col}_cor'] = np.sqrt(test[col])
            if original_dropped['sqrt']:
                X.drop([col], axis = 1, inplace = True)
                test.drop([col], axis = 1, inplace = True)
            print(f'Improved baseline to {max_value} from {baseline} using sqrt')
        else:
            if 'boxcox' in max_keys:
                X[f'{col}_cor'],_ = boxcox(X[col])
                test[f'{col}_cor'],_ = boxcox(test[col])
                if original_dropped['boxcox']:
                    X.drop([col], axis = 1, inplace = True)
                    test.drop([col], axis = 1, inplace = True)
                print(f'Improved baseline to {max_value} from {baseline} using boxcox')
            elif 'yeo-johnson' in max_keys:
                pt = PowerTransformer(method='yeo-johnson')
                # combine test and train set here
                X[f'{col}_cor'] = pt.fit_transform(X[[col]]).flatten()
                test[f'{col}_cor'] = pt.fit_transform(test[[col]]).flatten()
                if original_dropped['yeo-johnson']:
                    X.drop([col], axis = 1, inplace = True)
                    test.drop([col], axis = 1, inplace = True)
                print(f'Improved baseline to {max_value} from {baseline} using yeo-johnson')
            elif 'log' in max_keys:
                X[f'{col}_cor'] = np.log1p(X[col])
                test[f'{col}_cor'] = np.log1p(test[col])
                if original_dropped['log']:
                    X.drop([col], axis = 1, inplace = True)
                    test.drop([col], axis = 1, inplace = True)
                print(f'Improved baseline to {max_value} from {baseline} using log')
            else:
                X[f'{col}_cor'] = np.sqrt(X[col])
                test[f'{col}_cor'] = np.sqrt(test[col])
                if original_dropped['sqrt']:
                    X.drop([col], axis = 1, inplace = True)
                    test.drop([col], axis = 1, inplace = True)
                print(f'Improved baseline to {max_value} from {baseline} using sqrt')
        
        baseline=max_value
                


continuous_columns = X.select_dtypes(['int64','float64']).columns
baseline = 26.452648986971848


X = pd.read_csv('X.csv')
y = pd.read_csv('y.csv').squeeze()  # Converts back to Series if needed
test = pd.read_csv('test.csv')

baseline = 0.05637163848437093


# columns = list(continuous_columns)
# columns = ['X_LGBM', 'X_linear', 'X_XGB_CPU', 'X_Ridge', 'X_XGB',
#        'X_Elastic']
encoding_methods = ['label', 'one-hot', 'freq']

train_idx, val_idx = train_test_split(
    np.arange(len(X)), shuffle=True, random_state=42
)

for col in X.columns:

    if X[col].dtype not in ['object','bool']:
        continue
        
    print(f'\nFor column {col}:')
    
    encode_score = {}
    original_dropped = {}
    label_encoders = {}
    one_hot_encoders = {}
    freq_encoders = {}
    
    for encode in encoding_methods:
        print(f'For encoding strategy {encode}:')
        
        if encode == 'label':
            le = LabelEncoder()
            X[f'{col}_encoding'] = le.fit_transform(X[col])
            label_encoders[col] = le
        elif encode == 'one-hot':
            ohe = OneHotEncoder(sparse=False, handle_unknown='ignore')
            ohe_array = ohe.fit_transform(X[[col]])   # double brackets -> DataFrame
            ohe_df = pd.DataFrame(ohe_array, 
                                  columns=[f"{col}_{cat}" for cat in ohe.categories_[0]],
                                  index=X.index)
            X = pd.concat([X, ohe_df], axis=1)
            one_hot_encoders[col] = ohe
        elif encode == 'freq':
            freq_map = X[col].value_counts(normalize=True)
            X[f'{col}_encoding'] = X[col].map(freq_map)
            freq_encoders[col] = freq_map
    
        X_train, X_val, y_train, y_val = X.iloc[train_idx].copy(), X.iloc[val_idx].copy(), y.iloc[train_idx], y.iloc[val_idx]
        
        score_with_original = test_features(X_train, X_val, y_train, y_val, True)
        score_without_original = test_features(X_train.drop([col], axis = 1), X_val.drop([col], axis = 1), y_train, y_val, True)
        
        # will change depending on the error metric
        if score_without_original < score_with_original:
            encode_score[encode] = score_without_original
            original_dropped[encode] = True
        else:
            encode_score[encode] = score_with_original
            original_dropped[encode] = False

        if encode == 'label' or encode == 'freq':
            X.drop([f'{col}_encoding'], axis=1, inplace=True)
        elif encode == 'one-hot':
            X.drop([f"{col}_{cat}" for cat in ohe.categories_[0]], axis = 1, inplace = True)

    # will change depending on the error metric
    max_value = 100
    if encode_score:
        max_value = min(encode_score.values())
        max_keys = [k for k, v in encode_score.items() if v == max_value]
        
    if max_value<baseline:

        if 'label' in max_keys:
            X[f'{col}_encoding'] = label_encoders[col].transform(X[col])
            test[f'{col}_encoding'] = label_encoders[col].transform(test[col])
            if original_dropped['label']:
                X.drop([col], axis = 1, inplace = True)
                test.drop([col], axis = 1, inplace = True)
            print(f"Improved baseline to {max_value} from {baseline} using Label and original columns was {'dropped' if original_dropped['label'] else 'kept'}")
        elif 'one-hot' in max_keys:
            ohe_array = one_hot_encoders[col].transform(X[[col]])   # double brackets -> DataFrame
            ohe_df = pd.DataFrame(ohe_array, 
                                  columns=[f"{col}_{cat}" for cat in ohe.categories_[0]],
                                  index=X.index)
            X = pd.concat([X, ohe_df], axis=1)
            ohe_array = one_hot_encoders[col].transform(test[[col]])   # double brackets -> DataFrame
            ohe_df = pd.DataFrame(ohe_array, 
                                  columns=[f"{col}_{cat}" for cat in ohe.categories_[0]],
                                  index=test.index)
            test = pd.concat([test, ohe_df], axis=1)
            if original_dropped['one-hot']:
                X.drop([col], axis = 1, inplace = True)
                test.drop([col], axis = 1, inplace = True)
            print(f"Improved baseline to {max_value} from {baseline} using One-hot and original columns was {'dropped' if original_dropped['one-hot'] else 'kept'}")
        elif 'freq' in max_keys:
            X[f'{col}_encoding'] = X[col].map(freq_encoders[col])
            test[f'{col}_encoding'] = test[col].map(freq_encoders[col])
            if original_dropped['freq']:
                X.drop([col], axis = 1, inplace = True)
                test.drop([col], axis = 1, inplace = True)
            print(f"Improved baseline to {max_value} from {baseline} using Freq and original columns was {'dropped' if original_dropped['freq'] else 'kept'}")
        
        baseline=max_value
                


X.to_csv('X.csv', index=False)
y.to_csv('y.csv', index=False)
test.to_csv('test.csv', index=False)


X.columns


encoded_columns = ['road_type_encoding','lighting_encoding']


categorical_columns


X.month.value_counts


le = LabelEncoder()

month_encoded = le.fit_transform(X['month'])

#dropping month
month_original = X['month']
X.drop('month', axis=1, inplace = True)

X['month_sin'] = np.sin(2*np.pi * month_encoded / 12)
#X['month_cos'] = np.cos(2*np.pi * month_encoded / 12)

X_train, X_val, y_train, y_val = train_test_split(X, y, shuffle=True, random_state = 42)
score = test_features(X_train, X_val, y_train, y_val, True)
    
if score>baseline:
    print(f'Improved baseline of {score} that {baseline} before')
    baseline = score
    month_encoded = le.fit_transform(test['month'])
    test['month_sin'] = np.sin(2*np.pi * month_encoded / 12)
    #test['month_cos'] = np.cos(2*np.pi * month_encoded / 12)
    test.drop('month', axis=1, inplace = True)
else:
    X.drop(['month_sin'], axis = 1, inplace = True)
    #X.drop(['month_cos'], axis = 1, inplace = True)
    X['month'] = month_original


X.columns


features = ['duration']
for feature in features:
    largest_num = X[feature].astype(str).max()
    largest_num_len = len(str(largest_num))-1
    num_digits_round = X[feature].astype(int).astype(str).apply(lambda x: len(x)).max()
    num_digits_total = X[feature].astype(str).apply(lambda x: len(x)).max()
    for i in range(1, num_digits_total):
        X[f'duration_digit_{i}'] = ((X[feature] * 10.0**(i-num_digits_round)) % 10).fillna(0).astype("int8")

    X_train, X_val, y_train, y_val = train_test_split(X, y, shuffle=True, random_state = 42)
    score = test_features(X_train, X_val, y_train, y_val, True)
    
    if score>baseline:
        print(f'Improved baseline of {score} that {baseline} before')
        
        largest_num = test[feature].astype(str).max()
        largest_num_len = len(str(largest_num))-1
        num_digits_round = test[feature].astype(int).astype(str).apply(lambda x: len(x)).max()
        num_digits_total = test[feature].astype(str).apply(lambda x: len(x)).max()
        for i in range(1, num_digits_total):
            test[f'duration_digit_{i}'] = ((test[feature] * 10.0**(i-num_digits_round)) % 10).fillna(0).astype("int8")
    
    else:
        X.drop([f"duration_digit_{i}" for i in range(1, num_digits_total)], axis = 1, inplace = True)


continuous_columns = X.select_dtypes(['int64','float64']).columns
baseline = 0.96647606551439


X = pd.read_csv('X.csv')
y = pd.read_csv('y.csv').squeeze()  # Converts back to Series if needed
test = pd.read_csv('test.csv')

continuous_columns = ['num_lanes', 'curvature', 'speed_limit', 'num_reported_accidents']
# continuous_columns = X.select_dtypes(['int64', 'float64']).columns

baseline = 0.05635848425311839


train_idx, val_idx = train_test_split(
    np.arange(len(X)), shuffle=True, random_state=42
)

for feature in continuous_columns:
    print(f'\nFor column {feature}:')
    
    num_digits_round = max(X[feature].astype(int).astype(str).apply(lambda x: len(x)).max(),
                            test[feature].astype(int).astype(str).apply(lambda x: len(x)).max())
    if X[feature].dtype == 'float64':
        max_decimal_len = max(X[feature].astype(str).apply(lambda x: len(x.split('.')[-1]) if '.' in x else 0).max(),
                                 test[feature].astype(str).apply(lambda x: len(x.split('.')[-1]) if '.' in x else 0).max())
    else:
        max_decimal_len = 0
    
    for i in range(1, num_digits_round+max_decimal_len+1):
        X[f'{feature}_digit_{i}'] = ((X[feature] * 10.0**(i-num_digits_round)) % 10).fillna(0).astype("int8")
        
        X_train, X_val, y_train, y_val = X.iloc[train_idx].copy(), X.iloc[val_idx].copy(), y.iloc[train_idx], y.iloc[val_idx]
        score = test_features(X_train, X_val, y_train, y_val, True)

        if score<baseline:
            print(f'Improved baseline to {score} from {baseline} including digit {i}')
            baseline = score
            test[f'{feature}_digit_{i}'] = ((test[feature] * 10.0**(i-num_digits_round)) % 10).fillna(0).astype("int8")        
        else:
            X.drop([f"{feature}_digit_{i}"], axis = 1, inplace = True)


X.to_csv('X.csv', index=False)
y.to_csv('y.csv', index=False)
test.to_csv('test.csv', index=False)


X = pd.read_csv('X.csv')
y = pd.read_csv('y.csv').squeeze()  # Converts back to Series if needed
test = pd.read_csv('test.csv')

baseline = 0.05635848425311838


for column in X.select_dtypes('float64'):
    print(f'\nFor column {column}:')

    best_score = baseline
    best_decimals = None

    for decimals in [0, 1, 2]:
        X_copy = X.copy()
        X_copy[column] = X_copy[column].round(decimals)

        X_train, X_val, y_train, y_val = train_test_split(X_copy, y, shuffle=True, random_state=42)
        score = test_features(X_train, X_val, y_train, y_val, True)

        if score < best_score:
            print(f'Improved baseline to {score} from {best_score} with rounding to {decimals} decimals')
            best_score = score
            best_decimals = decimals

    if best_decimals is not None:
        baseline = best_score
        X[column] = X[column].round(best_decimals)
        test[column] = test[column].round(best_decimals)



X.to_csv('X.csv', index=False)
y.to_csv('y.csv', index=False)
test.to_csv('test.csv', index=False)


X.select_dtypes(['int64','float64']).columns


X = pd.read_csv('X.csv')
y = pd.read_csv('y.csv').squeeze()  # Converts back to Series if needed
test = pd.read_csv('test.csv')

continuous_columns = ['num_lanes', 'curvature', 'speed_limit', 'num_reported_accidents']

baseline = 0.05635848425311838


train_idx, val_idx = train_test_split(
    np.arange(len(X)), shuffle=True, random_state=42
)

#continuous_columns = X.select_dtypes(['int64','float64']).columns
columns = continuous_columns

for column in tqdm(columns):
    #print(f'\nFor column {column}:')
    skew = X[column].skew()
    kurt = X[column].kurt()
    uniq = X[column].nunique()
    corr = abs(np.corrcoef(X[column], y)[0,1]) if y is not None else 0
    
    # if uniq < 5 or np.isnan(skew) or corr < 0.05:
    #     continue

    col_min, col_max = X[column].min(), X[column].max()
        
    if col_min == col_max:
        # Skip binning constant column
        continue

    no_of_bins = range(2, 11)

    max_score = baseline
    nbins = None
    bin_method = None

    for n in no_of_bins:

        if abs(skew) > 1.5:
            bin_methods = ['supervised', 'frequency']
        elif 0.75 < abs(skew) <= 1.5:
            bin_methods = ['width', 'frequency', 'supervised']
        else:
            bin_methods = ['width', 'supervised']
    
        for bin_tech in bin_methods:
            # print(f'For binning technique equal {bin_tech}:')
            if bin_tech == 'width' and kurt > 3:
                continue  # avoid width for heavy-tailed distributions
            if bin_tech == 'frequency' and abs(skew) < 0.75:
                continue  # quantile binning less helpful when symmetric

            if bin_tech == 'width':
                bins = np.linspace(col_min, col_max, n + 1)  # n+1 edges = n bins
                labels = range(n)
                bucket_labels = pd.cut(
                    pd.concat([X,test])[column],
                    bins=bins,
                    labels=labels,
                    include_lowest=True,  # include min value
                    right=True            # right edge inclusive
                )
                X[f"{column}_bucket_{n}bins_{bin_tech}"] = bucket_labels[:X.shape[0]].astype('int8')
                X_train, X_val, y_train, y_val = X.iloc[train_idx].copy(), X.iloc[val_idx].copy(), y.iloc[train_idx], y.iloc[val_idx]
                score = test_features(X_train, X_val, y_train, y_val, True)
            
            elif bin_tech == 'frequency':
                bucket_labels, actual_bins = pd.qcut(
                    pd.concat([X,test])[column],
                    q=n,
                    retbins=True,
                    duplicates='drop'
                )
                    
                actual_num_bins = len(actual_bins) - 1
                actual_labels = list(range(actual_num_bins))
                
                # Re-bin with appropriate labels
                bucket_labels = pd.qcut(
                    pd.concat([X,test])[column],
                    q=n,
                    labels=actual_labels,
                    duplicates='drop'
                )
                X[f"{column}_bucket_{n}bins_{bin_tech}"] = bucket_labels[:X.shape[0]].astype('int8')
                X_train, X_val, y_train, y_val = X.iloc[train_idx].copy(), X.iloc[val_idx].copy(), y.iloc[train_idx], y.iloc[val_idx]
                score = test_features(X_train, X_val, y_train, y_val, True)
                
            elif bin_tech == 'supervised':
                if PROBLEM == 'Regression':
                    tree = DecisionTreeRegressor(
                                max_leaf_nodes=n, 
                                min_samples_leaf=0.05,  # avoid overfitting small bins
                                random_state=42
                            )
                else:
                    tree = DecisionTreeClassifier(
                                max_leaf_nodes=n, 
                                min_samples_leaf=0.05,  # avoid overfitting small bins
                                random_state=42
                            )
                
                X_train, X_val, y_train, y_val = X.iloc[train_idx].copy(), X.iloc[val_idx].copy(), y.iloc[train_idx], y.iloc[val_idx]
                tree.fit(X_train[[column]], y_train)
                 
                # Extract split thresholds
                thresholds = sorted(tree.tree_.threshold[tree.tree_.threshold != -2])
                if not thresholds:
                    continue  # no valid splits
                     
                # Define bins using thresholds
                bin_edges = [X_train[column].min()] + thresholds + [X_train[column].max()]
                edges = np.unique(bin_edges)
                if edges is None:
                    continue
                X_train[f"{column}_bucket_{n}bins_{bin_tech}"] = pd.cut(X_train[column], bins=edges, labels=False, include_lowest=True)
                X_val[f"{column}_bucket_{n}bins_{bin_tech}"] = pd.cut(X_val[column], bins=edges, labels=False, include_lowest=True)
                score = test_features(X_train, X_val, y_train, y_val, True)

            if bin_tech in ['width', 'frequency']:
                X.drop([f"{column}_bucket_{n}bins_{bin_tech}"], axis = 1, inplace = True)
                
            if score<max_score:
                max_score = score
                nbins = n
                bin_method = bin_tech
            
    if max_score<baseline:
        print(f'Improved baseline to {max_score} from {baseline}')
        
        if bin_method == 'width':
            bins = np.linspace(col_min, col_max, nbins + 1)
            labels = range(nbins)
            bucket_labels = pd.cut(
                pd.concat([X,test])[column],
                bins=bins,
                labels=labels,
                include_lowest=True,  # include min value
                right=True            # right edge inclusive
            )
            X[f"{column}_bucket_{nbins}bins_{bin_method}"] = bucket_labels[:X.shape[0]].astype('int8')
            test[f"{column}_bucket_{nbins}bins_{bin_method}"] = bucket_labels[X.shape[0]:].astype('int8')
        elif bin_method == 'frequency':
            bucket_labels, actual_bins = pd.qcut(
                pd.concat([X,test])[column],
                q=nbins,
                retbins=True,
                duplicates='drop'
            )
                
            actual_num_bins = len(actual_bins) - 1
            actual_labels = list(range(actual_num_bins))
            
            # Re-bin with appropriate labels
            bucket_labels = pd.qcut(
                pd.concat([X,test])[column],
                q=nbins,
                labels=actual_labels,
                duplicates='drop'
            )
            X[f"{column}_bucket_{nbins}bins_{bin_method}"] = bucket_labels[:X.shape[0]].astype('int8')
            test[f"{column}_bucket_{nbins}bins_{bin_method}"] = bucket_labels[X.shape[0]:].astype('int8')
        elif bin_method == 'supervised':
            tree = DecisionTreeRegressor(
                        max_leaf_nodes=nbins, 
                        min_samples_leaf=0.05,  # avoid overfitting small bins
                        random_state=42
                    )
            
            tree.fit(X[[column]], y)
             
            # Extract split thresholds
            thresholds = sorted(tree.tree_.threshold[tree.tree_.threshold != -2])
            if not thresholds:
                continue  # no valid splits
                 
            # Define bins using thresholds
            bin_edges = [X[column].min()] + thresholds + [X[column].max()]
            edges = np.unique(bin_edges)
            if edges is None:
                continue
            X[f"{column}_bucket_{nbins}bins_{bin_method}"] = pd.cut(X[column], bins=edges, labels=False, include_lowest=True)
            test[f"{column}_bucket_{nbins}bins_{bin_method}"] = pd.cut(test[column], bins=edges, labels=False, include_lowest=True)
        
        baseline = max_score


X.to_csv('X.csv', index=False)
y.to_csv('y.csv', index=False)
test.to_csv('test.csv', index=False)


X = pd.read_csv('X.csv')
y = pd.read_csv('y.csv').squeeze()  # Converts back to Series if needed
test = pd.read_csv('test.csv')

baseline = 0.05635848425311838

stats = ["mean","std","count","nunique","median","min","max","skew"]
#stats = ["mean"]

columns = ['num_lanes', 'curvature', 'speed_limit', 'num_reported_accidents']


train_idx, val_idx = train_test_split(
    np.arange(len(X)), shuffle=True, random_state=42
)

# continuous_columns = X.select_dtypes(['int64','float64']).columns
# columns = continuous_columns
# columns = ['InstrumentalScore', 'LivePerformanceLikelihood_imputed_mean', 'MoodScore_imputed_mean']
for idx, column in enumerate(columns):
    skew = X[column].skew()
    kurt = X[column].kurt()
    uniq = X[column].nunique()
    corr = abs(np.corrcoef(X[column], y)[0,1]) if y is not None else 0
    
    if uniq < 5 or np.isnan(skew) or corr < max(0.05, 0.5 * np.mean(np.abs(X[columns].corrwith(y)))):
        continue

    col_min, col_max = X[column].min(), X[column].max()
        
    if col_min == col_max:
        # Skip binning constant column
        continue
        
    print(f'\nFor column {idx} out of {len(columns)}:')

    no_of_bins = range(2, 11)

    max_score = baseline
    nbins = None
    bin_method = None
    best_stat = None

    for n in tqdm(no_of_bins):

        if uniq <= n:
            break

        if abs(skew) > 1.5:
            bin_methods = ['supervised', 'frequency']
        elif 0.75 < abs(skew) <= 1.5:
            bin_methods = ['width', 'frequency', 'supervised']
        else:
            bin_methods = ['width', 'supervised']
    
        for bin_tech in bin_methods:
            # print(f'For binning technique equal {bin_tech}:')
            if bin_tech == 'width' and kurt > 3:
                continue  # avoid width for heavy-tailed distributions
            if bin_tech == 'frequency' and abs(skew) < 0.75:
                continue  # quantile binning less helpful when symmetric
        
            for stat in stats:
                # print(f'For stat {stat}:')

                if bin_tech == 'width':
                    bins = np.linspace(col_min, col_max, n + 1)  # n+1 edges = n bins
                    labels = range(n)
                    bucket_labels = pd.cut(
                        pd.concat([X,test])[column],
                        bins=bins,
                        labels=labels,
                        include_lowest=True,  # include min value
                        right=True            # right edge inclusive
                    )
                    X[f'{column}_bucket'] = bucket_labels[:X.shape[0]].astype('int8')
                    mean_encoded = X.groupby(f'{column}_bucket')[column].agg(stat)
                    X[f'{column}_bucket_{n}_{bin_tech}_{stat}'] = X[f'{column}_bucket'].map(mean_encoded).astype('float64')
                    X.drop([f'{column}_bucket'], axis = 1, inplace = True)
                    X = X.fillna(0)
                    X_train, X_val, y_train, y_val = X.iloc[train_idx].copy(), X.iloc[val_idx].copy(), y.iloc[train_idx], y.iloc[val_idx]
                    score = test_features(X_train, X_val, y_train, y_val, True)
                
                elif bin_tech == 'frequency':
                    bucket_labels, actual_bins = pd.qcut(
                        X[column],
                        q=n,
                        retbins=True,
                        duplicates='drop'
                    )
                    actual_num_bins = len(actual_bins) - 1
                    actual_labels = list(range(actual_num_bins))
                    # Re-bin with appropriate labels
                    bucket_labels = pd.qcut(
                        X[column],
                        q=n,
                        labels=actual_labels,
                        duplicates='drop'
                    )
                    X[f'{column}_bucket'] = bucket_labels[:X.shape[0]].astype('int8')
                    mean_encoded = X.groupby(f'{column}_bucket')[column].agg(stat)
                    X[f'{column}_bucket_{n}_{bin_tech}_{stat}'] = X[f'{column}_bucket'].map(mean_encoded).astype('float64')
                    X.drop([f'{column}_bucket'], axis = 1, inplace = True)
                    X = X.fillna(0)
                    X_train, X_val, y_train, y_val = X.iloc[train_idx].copy(), X.iloc[val_idx].copy(), y.iloc[train_idx], y.iloc[val_idx]
                    score = test_features(X_train, X_val, y_train, y_val, False)

                elif bin_tech == 'supervised':
                    if PROBLEM == 'Regression':
                        tree = DecisionTreeRegressor(
                                    max_leaf_nodes=n, 
                                    min_samples_leaf=0.05,  # avoid overfitting small bins
                                    random_state=42
                                )
                    else:
                        tree = DecisionTreeClassifier(
                                    max_leaf_nodes=n, 
                                    min_samples_leaf=0.05,  # avoid overfitting small bins
                                    random_state=42
                                )
                    
                    X_train, X_val, y_train, y_val = X.iloc[train_idx].copy(), X.iloc[val_idx].copy(), y.iloc[train_idx], y.iloc[val_idx]
                    tree.fit(X_train[[column]], y_train)
                     
                    # Extract split thresholds
                    thresholds = sorted(tree.tree_.threshold[tree.tree_.threshold != -2])
                    if not thresholds:
                        continue  # no valid splits
                         
                    # Define bins using thresholds
                    bin_edges = [X_train[column].min()] + thresholds + [X_train[column].max()]
                    edges = np.unique(bin_edges)
                    if edges is None:
                        continue
                    X_train[f'{column}_bucket'] = pd.cut(X_train[column], bins=edges, labels=False, include_lowest=True)
                    X_val[f'{column}_bucket'] = pd.cut(X_val[column], bins=edges, labels=False, include_lowest=True)
                    mean_encoded = pd.concat([X_train,X_val]).groupby(f'{column}_bucket')[column].agg(stat)
                    X_train[f'{column}_bucket_{n}_{bin_tech}_{stat}'] = X_train[f'{column}_bucket'].map(mean_encoded).astype('float64')
                    X_val[f'{column}_bucket_{n}_{bin_tech}_{stat}'] = X_val[f'{column}_bucket'].map(mean_encoded).astype('float64')
                    X_train.drop([f'{column}_bucket'], axis = 1, inplace = True)
                    X_val.drop([f'{column}_bucket'], axis = 1, inplace = True)
                    X_train = X_train.fillna(0)
                    X_val = X_val.fillna(0)
                    score = test_features(X_train, X_val, y_train, y_val, True)

                if bin_tech in ['width', 'frequency']:
                    X.drop([f'{column}_bucket_{n}_{bin_tech}_{stat}'], axis = 1, inplace = True)
                    
                if score<max_score:
                    max_score = score
                    nbins = n
                    bin_method = bin_tech
                    best_stat = stat
        
    if max_score<baseline:
    
        print(f'Improved baseline to {max_score} from {baseline} with {nbins} bins {bin_method} binning and {best_stat} aggregator')
        baseline = max_score
    
        X_total_con_cat = pd.concat([X[[column]],test[[column]]])
        
        if bin_method == 'width':
            bins = np.linspace(col_min, col_max, nbins + 1)  # n+1 edges = n bins
            labels = range(nbins)
            bucket_labels = pd.cut(
                X_total_con_cat[column],
                bins=bins,
                labels=labels,
                include_lowest=True,  # include min value
                right=True            # right edge inclusive
            )
            X_total_con_cat[f'{column}_bucket'] = bucket_labels
            X[f'{column}_bucket'] = bucket_labels[:X.shape[0]].astype('int8')
            test[f'{column}_bucket'] = bucket_labels[X.shape[0]:].astype('int8')
            mean_encoded = X_total_con_cat.groupby(f'{column}_bucket')[column].agg(best_stat)
            X[f'{column}_bucket_{nbins}_{bin_method}_{best_stat}'] = X[f'{column}_bucket'].map(mean_encoded)
            test[f'{column}_bucket_{nbins}_{bin_method}_{best_stat}'] = test[f'{column}_bucket'].map(mean_encoded)
            X.drop([f'{column}_bucket'], axis = 1, inplace = True)
            test.drop([f'{column}_bucket'], axis = 1, inplace = True)
            
        elif bin_method == 'frequency':
            bucket_labels, actual_bins = pd.qcut(
                X_total_con_cat[column],
                q=nbins,
                retbins=True,
                duplicates='drop'
            )
            actual_num_bins = len(actual_bins) - 1
            actual_labels = list(range(actual_num_bins))
            bucket_labels = pd.qcut(
                X_total_con_cat[column],
                q=nbins,
                labels=actual_labels,
                duplicates='drop'
            )
            X_total_con_cat[f'{column}_bucket'] = bucket_labels
            X[f'{column}_bucket'] = bucket_labels[:X.shape[0]].astype('int8')
            test[f'{column}_bucket'] = bucket_labels[X.shape[0]:].astype('int8')
            mean_encoded = X_total_con_cat.groupby(f'{column}_bucket')[column].agg(best_stat)
            X[f'{column}_bucket_{nbins}_{bin_method}_{best_stat}'] = X[f'{column}_bucket'].map(mean_encoded)
            test[f'{column}_bucket_{nbins}_{bin_method}_{best_stat}'] = test[f'{column}_bucket'].map(mean_encoded)
            X.drop([f'{column}_bucket'], axis = 1, inplace = True)
            test.drop([f'{column}_bucket'], axis = 1, inplace = True)

        elif bin_method == 'supervised':
            if PROBLEM == 'Regression':
                tree = DecisionTreeRegressor(
                            max_leaf_nodes=nbins, 
                            min_samples_leaf=0.05,  # avoid overfitting small bins
                            random_state=42
                        )
            else:
                tree = DecisionTreeClassifier(
                            max_leaf_nodes=nbins, 
                            min_samples_leaf=0.05,  # avoid overfitting small bins
                            random_state=42
                        )
            
            tree.fit(X[[column]], y)
             
            # Extract split thresholds
            thresholds = sorted(tree.tree_.threshold[tree.tree_.threshold != -2])
            if not thresholds:
                continue  # no valid splits
                 
            # Define bins using thresholds
            bin_edges = [X[column].min()] + thresholds + [X[column].max()]
            edges = np.unique(bin_edges)
            if edges is None:
                continue
            X_total_con_cat[f'{column}_bucket'] = pd.cut(X_total_con_cat[column], bins=edges, labels=False, include_lowest=True)
            X[f'{column}_bucket'] = pd.cut(X[column], bins=edges, labels=False, include_lowest=True)
            test[f'{column}_bucket'] = pd.cut(test[column], bins=edges, labels=False, include_lowest=True)
            mean_encoded = X_total_con_cat.groupby(f'{column}_bucket')[column].agg(best_stat)
            X[f'{column}_bucket_{nbins}_{bin_method}_{best_stat}'] = X[f'{column}_bucket'].map(mean_encoded).astype('float64')
            test[f'{column}_bucket_{nbins}_{bin_method}_{best_stat}'] = test[f'{column}_bucket'].map(mean_encoded).astype('float64')
            X.drop([f'{column}_bucket'], axis = 1, inplace = True)
            test.drop([f'{column}_bucket'], axis = 1, inplace = True)
            X = X.fillna(0)
            test = test.fillna(0)


X.to_csv('X.csv', index=False)
y.to_csv('y.csv', index=False)
test.to_csv('test.csv', index=False)


X = pd.read_csv('X.csv')
y = pd.read_csv('y.csv').squeeze()  # Converts back to Series if needed
test = pd.read_csv('test.csv')

continuous_columns = ['X_LGBM', 'X_linear', 'X_XGB_CPU', 'X_Lasso', 'X_Ridge', 'X_XGB', 'X_Elastic']
baseline = 26.16759139656384


conversions = [1000, 60]

factor = 1

for conversion in conversions:
    factor*=conversion

    X[f'TrackDurationMs_{factor}'] = X['TrackDurationMs']/factor

    X_train, X_val, y_train, y_val = train_test_split(X, y, shuffle=True, random_state = 42)
    score = test_features(X_train, X_val, y_train, y_val, False)
    
    if score<baseline:
        print(f'Improved baseline to {score} from {baseline}')
        test[f'TrackDurationMs_{factor}'] = test['TrackDurationMs']/factor
        baseline = score
        
    else:
        X.drop([f'TrackDurationMs_{factor}'], axis = 1, inplace = True)


X = pd.read_csv('X.csv')
y = pd.read_csv('y.csv').squeeze()  # Converts back to Series if needed
test = pd.read_csv('test.csv')

baseline =  0.056353460588877864

stats_set = [["mean","median"],["std"],["count","nunique"],["min","max"],["skew"]]
#stats = ["mean"]

continuous_columns = ['num_lanes', 'curvature', 'speed_limit', 'num_reported_accidents']
categorical_columns = list(X.select_dtypes(['bool','object']).columns) + ['road_type_encoding','lighting_encoding']


train_idx, val_idx = train_test_split(
    np.arange(len(X)), shuffle=True, random_state=42
)

for idx, countinuous_column in enumerate(continuous_columns):
    print(f'countinuous_column Iteration {idx+1} out of {len(continuous_columns)}')
        
    for categorical_column in tqdm(categorical_columns):
        #print(f'Iteration {index+1} out of {len(categorical_columns)}')
    
        for stats in stats_set:
            
            max_score = baseline
            best_stat = None
            for idx1, stat in enumerate(stats):
                #print(f'stats Iteration {idx1+1} out of {len(stats)}')
                
                mean_encoded = X.groupby(categorical_column)[countinuous_column].agg(stat)
                X[f'{categorical_column}_{countinuous_column}_{stat}'] = X[categorical_column].map(mean_encoded)
                
                X_train, X_val, y_train, y_val = X.iloc[train_idx].copy(), X.iloc[val_idx].copy(), y.iloc[train_idx], y.iloc[val_idx]
                score = test_features(X_train, X_val, y_train, y_val, True)

                X.drop([f'{categorical_column}_{countinuous_column}_{stat}'], axis=1, inplace=True)

                if score<max_score:
                    max_score = score
                    best_stat = stat
                
            if max_score<baseline:
                print(f'Improved baseline of {max_score} that {baseline} before')
                baseline=max_score
                
                # if baseline is imporved we will use the combiantion of train and test to find the aggregate and then assign to test
                X_total_con_cat = pd.concat([X[[countinuous_column,categorical_column]],test[[countinuous_column,categorical_column]]])
                mean_encoded = X_total_con_cat.groupby(categorical_column)[countinuous_column].agg(best_stat)
                X[f'{categorical_column}_{countinuous_column}_{best_stat}'] = X[categorical_column].map(mean_encoded)
                test[f'{categorical_column}_{countinuous_column}_{best_stat}'] = test[categorical_column].map(mean_encoded)


X.to_csv('X.csv', index=False)
y.to_csv('y.csv', index=False)
test.to_csv('test.csv', index=False)


X = pd.read_csv('X.csv')
y = pd.read_csv('y.csv').squeeze()  # Converts back to Series if needed
test = pd.read_csv('test.csv')

baseline = 0.056353460588877864

stats_set = [['count', 'nunique'], ['dominance', 'entropy'], ['proportion', 'cross_counts']]

categorical_columns = list(X.select_dtypes(['bool','object']).columns) + ['road_type_encoding','lighting_encoding']


train_idx, val_idx = train_test_split(
    np.arange(len(X)), shuffle=True, random_state=42
)

for idx1, categorical_column in enumerate(categorical_columns):
    print(f'categorical_column Iteration {idx1+1} out of {len(categorical_columns)}')
        
    for idx2,categorical_column_group in tqdm(enumerate(categorical_columns), total = len(categorical_columns)):
        #print(f'Iteration {idx2+1} out of {len(categorical_columns)}')

        if idx1 == idx2:
            continue
    
        for stats in stats_set:
            
            max_score = baseline
            best_stat = None
            for idx3, stat in enumerate(stats):
                #print(f'stats Iteration {idx3+1} out of {len(stats)}')

                group = X.groupby(categorical_column_group)[categorical_column]

                if stat == 'count':
                    freq = group.value_counts(normalize=False).unstack().fillna(0)
                    freq.columns = [f"{categorical_column_group}_{categorical_column}_count_{c}" for c in freq.columns]
                    X_temp = X.join(freq, on=categorical_column_group)
                    X_train, X_val, y_train, y_val = X_temp.iloc[train_idx].copy(), X_temp.iloc[val_idx].copy(), y.iloc[train_idx], y.iloc[val_idx]
                    score = test_features(X_train, X_val, y_train, y_val, True)
                elif stat == 'nunique':
                    nunique_col = group.nunique()
                    X[f"{categorical_column_group}_{categorical_column}_nunique"] = X[categorical_column_group].map(nunique_col)
                    X_train, X_val, y_train, y_val = X.iloc[train_idx].copy(), X.iloc[val_idx].copy(), y.iloc[train_idx], y.iloc[val_idx]
                    score = test_features(X_train, X_val, y_train, y_val, True)
                    X.drop([f'{categorical_column_group}_{categorical_column}_nunique'], axis=1, inplace=True)
                elif stat == 'dominance':
                    dominance = (
                        group
                         .apply(lambda s: s.value_counts(normalize=True).iloc[0])
                    )
                    X[f'{categorical_column_group}_{categorical_column}_dominance'] = X[categorical_column_group].map(dominance)
                    X_train, X_val, y_train, y_val = X.iloc[train_idx].copy(), X.iloc[val_idx].copy(), y.iloc[train_idx], y.iloc[val_idx]
                    score = test_features(X_train, X_val, y_train, y_val, True)
                    X.drop([f'{categorical_column_group}_{categorical_column}_dominance'], axis=1, inplace=True)
                elif stat == 'entropy':
                    ent = (
                        group
                        .apply(lambda s: entropy(s.value_counts(normalize=True), base=2))
                    )
                    X[f'{categorical_column_group}_{categorical_column}_entropy'] = X[categorical_column_group].map(ent)
                    X_train, X_val, y_train, y_val = X.iloc[train_idx].copy(), X.iloc[val_idx].copy(), y.iloc[train_idx], y.iloc[val_idx]
                    score = test_features(X_train, X_val, y_train, y_val, True)
                    X.drop([f'{categorical_column_group}_{categorical_column}_entropy'], axis=1, inplace=True)
                elif stat == 'proportion':
                    prop = (
                        X.groupby([categorical_column_group, categorical_column]).size() /
                        X.groupby(categorical_column_group).size()
                    ).unstack().fillna(0)
                    prop.columns = [f"{categorical_column_group}_{categorical_column}_proportion_{c}" for c in prop.columns]
                    X_temp = X.join(prop, on=categorical_column_group)
                    X_train, X_val, y_train, y_val = X_temp.iloc[train_idx].copy(), X_temp.iloc[val_idx].copy(), y.iloc[train_idx], y.iloc[val_idx]
                    score = test_features(X_train, X_val, y_train, y_val, True)
                elif stat == 'cross_counts':
                    cross = pd.crosstab(X[categorical_column_group], X[categorical_column])
                    cross.columns = [f"{categorical_column_group}_{categorical_column}_cross_{c}" for c in cross.columns]
                    X_temp = X.join(cross, on=categorical_column_group)
                    X_train, X_val, y_train, y_val = X_temp.iloc[train_idx].copy(), X_temp.iloc[val_idx].copy(), y.iloc[train_idx], y.iloc[val_idx]
                    score = test_features(X_train, X_val, y_train, y_val, True)
                    
                if score<max_score:
                    max_score = score
                    best_stat = stat
                
            if max_score<baseline:
                print(f'Improved baseline to {max_score} from {baseline}')
                baseline=max_score

                X_total_con_cat = pd.concat([X[[categorical_column, categorical_column_group]],test[[categorical_column, categorical_column_group]]])
                group = X_total_con_cat.groupby(categorical_column_group)[categorical_column]

                if best_stat == 'count':
                    freq = group.value_counts(normalize=False).unstack().fillna(0)
                    freq.columns = [f"{categorical_column_group}_{categorical_column}_count_{c}" for c in freq.columns]
                    X = X.join(freq, on=categorical_column_group)
                    test = test.join(freq, on=categorical_column_group)
                elif best_stat == 'nunique':
                    nunique_col = group.nunique()
                    X[f"{categorical_column_group}_{categorical_column}_nunique"] = X[categorical_column_group].map(nunique_col)
                    test[f"{categorical_column_group}_{categorical_column}_nunique"] = test[categorical_column_group].map(nunique_col)
                elif best_stat == 'dominance':
                    dominance = (
                        group
                         .apply(lambda s: s.value_counts(normalize=True).iloc[0])
                    )
                    X[f'{categorical_column_group}_{categorical_column}_dominance'] = X[categorical_column_group].map(dominance)
                    test[f'{categorical_column_group}_{categorical_column}_dominance'] = test[categorical_column_group].map(dominance)
                elif best_stat == 'entropy':
                    ent = (
                        group
                        .apply(lambda s: entropy(s.value_counts(normalize=True), base=2))
                    )
                    X[f'{categorical_column_group}_{categorical_column}_entropy'] = X[categorical_column_group].map(ent)
                    test[f'{categorical_column_group}_{categorical_column}_entropy'] = test[categorical_column_group].map(ent)
                elif best_stat == 'proportion':
                    prop = (
                        X_total_con_cat.groupby([categorical_column_group, categorical_column]).size() /
                        X_total_con_cat.groupby(categorical_column_group).size()
                    ).unstack().fillna(0)
                    prop.columns = [f"{categorical_column_group}_{categorical_column}_proportion_{c}" for c in prop.columns]
                    X = X.join(prop, on=categorical_column_group)
                    test = test.join(prop, on=categorical_column_group)
                elif best_stat == 'cross_counts':
                    cross = pd.crosstab(X_total_con_cat[categorical_column_group], X_total_con_cat[categorical_column])
                    cross.columns = [f"{categorical_column_group}_{categorical_column}_cross_{c}" for c in cross.columns]
                    X = X.join(cross, on=categorical_column_group)
                    test = test.join(cross, on=categorical_column_group)


X.to_csv('X.csv', index=False)
y.to_csv('y.csv', index=False)
test.to_csv('test.csv', index=False)


X = pd.read_csv('X.csv')
y = pd.read_csv('y.csv').squeeze()  # Converts back to Series if needed
test = pd.read_csv('test.csv')

baseline = 0.056353460588877864

stats_set = [["mean","median"],["std"],["count","nunique"],["min","max"],["skew"]]

continuous_columns = ['num_lanes', 'curvature', 'speed_limit', 'num_reported_accidents']


train_idx, val_idx = train_test_split(
    np.arange(len(X)), shuffle=True, random_state=42
)
#continuous_columns = X.select_dtypes(['int64','float64']).columns

#for idx1, countinuous_column in tqdm(enumerate(continuous_columns), total=len(continuous_columns)):
for idx1, countinuous_column in enumerate(continuous_columns):
    print(f'1. countinuous_column Iteration {idx1+1} out of {len(continuous_columns)}')
        
    for idx2,countinuous_column_sec in tqdm(enumerate(continuous_columns), total=len(continuous_columns)):
        if(idx1 == idx2):
            continue
                
        col_min, col_max = X[countinuous_column_sec].min(), X[countinuous_column_sec].max()
        if col_min == col_max:
            # Skip binning constant column
            continue
        
        #print(f'3. Inner Iteration {idx2+1} out of {len(continuous_columns)}')
        skew = X[countinuous_column_sec].skew()
        kurt = X[countinuous_column_sec].kurt()
        uniq = X[countinuous_column_sec].nunique()
        corr = abs(np.corrcoef(X[countinuous_column_sec], y)[0,1]) if y is not None else 0
    
        if np.isnan(skew) or corr < max(0.05, 0.5 * np.mean(np.abs(X[continuous_columns].corrwith(y)))):
            continue
        
        # for idx3, stat_group in tqdm(enumerate(stats_set), total=len(stats_set)):
        for idx3, stat_group in enumerate(stats_set):
            #print(f'2. stats Iteration {idx3+1} out of {len(stats_set)}\n')
            max_score = baseline
            nbins = None
            bin_method = None
            best_stat = None
            # for idx4, stat in tqdm(enumerate(stat_group), total=len(stat_group)):
            for idx4, stat in enumerate(stat_group):
                #print(f'2. stats Iteration {idx4+1} out of {len(stat_group)}\n')
                
                no_of_bins = range(2, 11)
                
                for n in no_of_bins:
                    # print(f'4. For {n} bins:')
                    
                    if uniq <= n:
                        break
                        
                    if abs(skew) > 1.5:
                        bin_methods = ['supervised', 'frequency']
                    elif 0.75 < abs(skew) <= 1.5:
                        bin_methods = ['width', 'frequency', 'supervised']
                    else:
                        bin_methods = ['width', 'supervised']
                    
                    for bin_tech in bin_methods:
                        # print(f'5. For binning technique equal {bin_tech}:')
                        
                        if bin_tech == 'width':
                            # Auto-generate n bins between min and max
                            bins = np.linspace(col_min, col_max, n + 1)  # n+1 edges = n bins
                            labels = range(n)  # 0, 1, ..., n-1
                            bucket_labels = pd.cut(
                                X[countinuous_column_sec],
                                bins=bins,
                                labels=labels,
                                include_lowest=True,  # include min value
                                right=True            # right edge inclusive
                            )
                            
                        elif bin_tech == 'frequency':
                            bucket_labels, actual_bins = pd.qcut(
                                X[countinuous_column_sec],
                                q=n,
                                retbins=True,
                                duplicates='drop'
                            )
                            actual_num_bins = len(actual_bins) - 1
                            actual_labels = list(range(actual_num_bins))
                            # Re-bin with appropriate labels
                            bucket_labels = pd.qcut(
                                X[countinuous_column_sec],
                                q=n,
                                labels=actual_labels,
                                duplicates='drop'
                            )
                            
                        elif bin_tech == 'supervised':
                            if PROBLEM == 'Regression':
                                tree = DecisionTreeRegressor(
                                            max_leaf_nodes=n, 
                                            min_samples_leaf=0.05,  # avoid overfitting small bins
                                            random_state=42
                                        )
                            else:
                                tree = DecisionTreeClassifier(
                                            max_leaf_nodes=n, 
                                            min_samples_leaf=0.05,  # avoid overfitting small bins
                                            random_state=42
                                        )
                            X_train, X_val, y_train, y_val = X.iloc[train_idx].copy(), X.iloc[val_idx].copy(), y.iloc[train_idx], y.iloc[val_idx]
                            tree.fit(X_train[[countinuous_column_sec]], y_train)
                            # Extract split thresholds
                            thresholds = sorted(tree.tree_.threshold[tree.tree_.threshold != -2])
                            if not thresholds:
                                continue  # no valid splits
                            # Define bins using thresholds
                            bin_edges = [X_train[countinuous_column_sec].min()] + thresholds + [X_train[countinuous_column_sec].max()]
                            edges = np.unique(bin_edges)
                            if edges is None:
                                continue
                            bucket_labels = pd.cut(X[countinuous_column_sec], bins=edges, labels=False, include_lowest=True)
                        
                        X[f'{countinuous_column_sec}_bucket'] = bucket_labels.astype('int8')
                        aggregate = X.groupby(f'{countinuous_column_sec}_bucket')[countinuous_column].agg(stat)
                        X[f'{countinuous_column_sec}_{countinuous_column}_{stat}_{n}_{bin_tech}'] = X[f'{countinuous_column_sec}_bucket'].map(aggregate)
                        X.drop([f'{countinuous_column_sec}_bucket'], axis = 1, inplace = True)
                        
                        X_train, X_val, y_train, y_val = X.iloc[train_idx].copy(), X.iloc[val_idx].copy(), y.iloc[train_idx], y.iloc[val_idx]
                        score = test_features(X_train, X_val, y_train, y_val, True)
                        
                        if score < max_score:
                            max_score = score
                            nbins = n
                            bin_method = bin_tech
                            best_stat = stat
                        
                        X.drop([f'{countinuous_column_sec}_{countinuous_column}_{stat}_{n}_{bin_tech}'], axis=1, inplace=True)
                        
            if max_score<baseline:
                print(f'Improved baseline to {max_score} from {baseline}')
                baseline=max_score
                
                # if baseline is imporved we will use the combiantion of train and test to find the aggregate and then assign to test
                X_total_con_cat = pd.concat([X[[countinuous_column,countinuous_column_sec]],test[[countinuous_column,countinuous_column_sec]]])
                
                if bin_method == 'width':
                    bins = np.linspace(col_min, col_max, nbins + 1)  # n+1 edges = n bins
                    labels = range(nbins)  # 0, 1, ..., n-1
                    bucket_labels = pd.cut(
                        X_total_con_cat[countinuous_column_sec],
                        bins=bins,
                        labels=labels,
                        include_lowest=True,  # include min value
                        right=True            # right edge inclusive
                    )
                elif bin_method == 'frequency':
                    bucket_labels, actual_bins = pd.qcut(
                        X_total_con_cat[countinuous_column_sec],
                        q=nbins,
                        retbins=True,
                        duplicates='drop'
                    )                
                    actual_num_bins = len(actual_bins) - 1
                    actual_labels = list(range(actual_num_bins))
                    # Re-bin with appropriate labels
                    bucket_labels = pd.qcut(
                        X_total_con_cat[countinuous_column_sec],
                        q=nbins,
                        labels=actual_labels,
                        duplicates='drop'
                    )
                elif bin_method == 'supervised':
                    if PROBLEM == 'Regression':
                        tree = DecisionTreeRegressor(
                                    max_leaf_nodes=nbins, 
                                    min_samples_leaf=0.05,  # avoid overfitting small bins
                                    random_state=42
                                )
                    else:
                        tree = DecisionTreeClassifier(
                                    max_leaf_nodes=nbins, 
                                    min_samples_leaf=0.05,  # avoid overfitting small bins
                                    random_state=42
                                )
                    tree.fit(X[[countinuous_column_sec]], y)
                    # Extract split thresholds
                    thresholds = sorted(tree.tree_.threshold[tree.tree_.threshold != -2])
                    if not thresholds:
                        continue  # no valid splits
                    # Define bins using thresholds
                    bin_edges = [X[countinuous_column_sec].min()] + thresholds + [X[countinuous_column_sec].max()]
                    edges = np.unique(bin_edges)
                    if edges is None:
                        continue
                    bucket_labels = pd.cut(X_total_con_cat[countinuous_column_sec], bins=edges, labels=False, include_lowest=True)
                    
                            
                X_total_con_cat[f'{countinuous_column_sec}_bucket'] = bucket_labels.astype('int8')
                X[f'{countinuous_column_sec}_bucket'] = bucket_labels[:X.shape[0]].astype('int8')
                test[f'{countinuous_column_sec}_bucket'] = bucket_labels[X.shape[0]:].astype('int8')
                aggregate = X_total_con_cat.groupby(f'{countinuous_column_sec}_bucket')[countinuous_column].agg(best_stat)
                X[f'{countinuous_column_sec}_{countinuous_column}_{best_stat}_{nbins}_{bin_method}'] = X[f'{countinuous_column_sec}_bucket'].map(aggregate)
                test[f'{countinuous_column_sec}_{countinuous_column}_{best_stat}_{nbins}_{bin_method}'] = test[f'{countinuous_column_sec}_bucket'].map(aggregate)
                X.drop([f'{countinuous_column_sec}_bucket'], axis = 1, inplace = True)
                test.drop([f'{countinuous_column_sec}_bucket'], axis = 1, inplace = True)


X.to_csv('X.csv', index=False)
y.to_csv('y.csv', index=False)
test.to_csv('test.csv', index=False)


X = pd.read_csv('X.csv')
y = pd.read_csv('y.csv').squeeze()  # Converts back to Series if needed
test = pd.read_csv('test.csv')

baseline = 0.05634491969406081

stats_set = [['count', 'nunique'], ['dominance', 'entropy'], ['proportion', 'cross_counts']]

categorical_columns = list(X.select_dtypes(['bool','object']).columns) + ['road_type_encoding','lighting_encoding']
continuous_columns = ['num_lanes', 'curvature', 'speed_limit', 'num_reported_accidents']


train_idx, val_idx = train_test_split(
    np.arange(len(X)), shuffle=True, random_state=42
)
#continuous_columns = X.select_dtypes(['int64','float64']).columns

# for idx1, categorical_column in tqdm(enumerate(categorical_columns), total=len(categorical_columns)):
for idx1, categorical_column in enumerate(categorical_columns):
    print(f'1. categorical_column Iteration {idx1+1} out of {len(categorical_columns)}')
        
    for idx2,continuous_column in tqdm(enumerate(continuous_columns), total=len(continuous_columns)):
        if(idx1 == idx2):
            continue
                
        col_min, col_max = X[continuous_column].min(), X[continuous_column].max()
        if col_min == col_max:
            # Skip binning constant column
            continue
        
        #print(f'3. Inner Iteration {idx2+1} out of {len(continuous_columns)}')
        skew = X[continuous_column].skew()
        kurt = X[continuous_column].kurt()
        uniq = X[continuous_column].nunique()
        corr = abs(np.corrcoef(X[continuous_column], y)[0,1]) if y is not None else 0
    
        if np.isnan(skew) or corr < max(0.05, 0.5 * np.mean(np.abs(X[continuous_columns].corrwith(y)))):
            continue
        
        # for idx3, stat_group in tqdm(enumerate(stats_set), total=len(stats_set)):
        for idx3, stat_group in enumerate(stats_set):
            #print(f'2. stats Iteration {idx3+1} out of {len(stats_set)}\n')
            max_score = baseline
            nbins = None
            bin_method = None
            best_stat = None
            
            # for idx4, stat in tqdm(enumerate(stat_group), total=len(stat_group)):
            for idx4, stat in enumerate(stat_group):
                #print(f'2. stats Iteration {idx4+1} out of {len(stat_group)}\n')
                
                no_of_bins = range(2, 11)
                
                for n in no_of_bins:
                    # print(f'4. For {n} bins:')
                    
                    if uniq <= n:
                        break
                        
                    if abs(skew) > 1.5:
                        bin_methods = ['supervised', 'frequency']
                    elif 0.75 < abs(skew) <= 1.5:
                        bin_methods = ['width', 'frequency', 'supervised']
                    else:
                        bin_methods = ['width', 'supervised']
                    
                    for bin_tech in bin_methods:
                        # print(f'5. For binning technique equal {bin_tech}:')
                        
                        if bin_tech == 'width':
                            # Auto-generate n bins between min and max
                            bins = np.linspace(col_min, col_max, n + 1)  # n+1 edges = n bins
                            labels = range(n)  # 0, 1, ..., n-1
                            bucket_labels = pd.cut(
                                X[continuous_column],
                                bins=bins,
                                labels=labels,
                                include_lowest=True,  # include min value
                                right=True            # right edge inclusive
                            )
                            
                        elif bin_tech == 'frequency':
                            bucket_labels, actual_bins = pd.qcut(
                                X[continuous_column],
                                q=n,
                                retbins=True,
                                duplicates='drop'
                            )
                            actual_num_bins = len(actual_bins) - 1
                            actual_labels = list(range(actual_num_bins))
                            # Re-bin with appropriate labels
                            bucket_labels = pd.qcut(
                                X[continuous_column],
                                q=n,
                                labels=actual_labels,
                                duplicates='drop'
                            )
                            
                        elif bin_tech == 'supervised':
                            if PROBLEM == 'Regression':
                                tree = DecisionTreeRegressor(
                                            max_leaf_nodes=n, 
                                            min_samples_leaf=0.05,  # avoid overfitting small bins
                                            random_state=42
                                        )
                            else:
                                tree = DecisionTreeClassifier(
                                            max_leaf_nodes=n, 
                                            min_samples_leaf=0.05,  # avoid overfitting small bins
                                            random_state=42
                                        )
                            X_train, X_val, y_train, y_val = X.iloc[train_idx].copy(), X.iloc[val_idx].copy(), y.iloc[train_idx], y.iloc[val_idx]
                            tree.fit(X_train[[continuous_column]], y_train)
                            # Extract split thresholds
                            thresholds = sorted(tree.tree_.threshold[tree.tree_.threshold != -2])
                            if not thresholds:
                                continue  # no valid splits
                            # Define bins using thresholds
                            bin_edges = [X_train[continuous_column].min()] + thresholds + [X_train[continuous_column].max()]
                            edges = np.unique(bin_edges)
                            if edges is None:
                                continue
                            bucket_labels = pd.cut(X[continuous_column], bins=edges, labels=False, include_lowest=True)
                        
                        X[f'{continuous_column}_bucket'] = bucket_labels.astype('int8')

                        group = X.groupby(f'{continuous_column}_bucket')[categorical_column]

                        if stat == 'count':
                            freq = group.value_counts(normalize=False).unstack().fillna(0)
                            freq.columns = [f"{continuous_column}_{categorical_column}_count_{c}" for c in freq.columns]
                            X_temp = X.join(freq, on=f'{continuous_column}_bucket')
                            X_temp.drop([f'{continuous_column}_bucket'], axis = 1, inplace = True)
                            X.drop([f'{continuous_column}_bucket'], axis = 1, inplace = True)
                            X_train, X_val, y_train, y_val = X_temp.iloc[train_idx].copy(), X_temp.iloc[val_idx].copy(), y.iloc[train_idx], y.iloc[val_idx]
                            score = test_features(X_train, X_val, y_train, y_val, True)
                        elif stat == 'nunique':
                            nunique_col = group.nunique()
                            X[f"{continuous_column}_{categorical_column}_nunique"] = X[f'{continuous_column}_bucket'].map(nunique_col)
                            X.drop([f'{continuous_column}_bucket'], axis = 1, inplace = True)
                            X_train, X_val, y_train, y_val = X.iloc[train_idx].copy(), X.iloc[val_idx].copy(), y.iloc[train_idx], y.iloc[val_idx]
                            score = test_features(X_train, X_val, y_train, y_val, True)
                            X.drop([f'{continuous_column}_{categorical_column}_nunique'], axis=1, inplace=True)
                        elif stat == 'dominance':
                            dominance = (
                                group
                                 .apply(lambda s: s.value_counts(normalize=True).iloc[0])
                            )
                            X[f'{continuous_column}_{categorical_column}_dominance'] = X[f'{continuous_column}_bucket'].map(dominance)
                            X.drop([f'{continuous_column}_bucket'], axis = 1, inplace = True)
                            X_train, X_val, y_train, y_val = X.iloc[train_idx].copy(), X.iloc[val_idx].copy(), y.iloc[train_idx], y.iloc[val_idx]
                            score = test_features(X_train, X_val, y_train, y_val, True)
                            X.drop([f'{continuous_column}_{categorical_column}_dominance'], axis=1, inplace=True)
                        elif stat == 'entropy':
                            ent = (
                                group
                                .apply(lambda s: entropy(s.value_counts(normalize=True), base=2))
                            )
                            X[f'{continuous_column}_{categorical_column}_entropy'] = X[f'{continuous_column}_bucket'].map(ent)
                            X.drop([f'{continuous_column}_bucket'], axis = 1, inplace = True)
                            X_train, X_val, y_train, y_val = X.iloc[train_idx].copy(), X.iloc[val_idx].copy(), y.iloc[train_idx], y.iloc[val_idx]
                            score = test_features(X_train, X_val, y_train, y_val, True)
                            X.drop([f'{continuous_column}_{categorical_column}_entropy'], axis=1, inplace=True)
                        elif stat == 'proportion':
                            prop = (
                                X.groupby([f'{continuous_column}_bucket', categorical_column]).size() /
                                X.groupby(f'{continuous_column}_bucket').size()
                            ).unstack().fillna(0)
                            prop.columns = [f"{continuous_column}_{categorical_column}_proportion_{c}" for c in prop.columns]
                            X_temp = X.join(prop, on=f'{continuous_column}_bucket')
                            X_temp.drop([f'{continuous_column}_bucket'], axis = 1, inplace = True)
                            X.drop([f'{continuous_column}_bucket'], axis = 1, inplace = True)
                            X_train, X_val, y_train, y_val = X_temp.iloc[train_idx].copy(), X_temp.iloc[val_idx].copy(), y.iloc[train_idx], y.iloc[val_idx]
                            score = test_features(X_train, X_val, y_train, y_val, True)
                        elif stat == 'cross_counts':
                            cross = pd.crosstab(X[f'{continuous_column}_bucket'], X[categorical_column])
                            cross.columns = [f"{continuous_column}_{categorical_column}_cross_{c}" for c in cross.columns]
                            X_temp = X.join(cross, on=f'{continuous_column}_bucket')
                            X_temp.drop([f'{continuous_column}_bucket'], axis = 1, inplace = True)
                            X.drop([f'{continuous_column}_bucket'], axis = 1, inplace = True)
                            X_train, X_val, y_train, y_val = X_temp.iloc[train_idx].copy(), X_temp.iloc[val_idx].copy(), y.iloc[train_idx], y.iloc[val_idx]
                            score = test_features(X_train, X_val, y_train, y_val, True)

                        
                        if score < max_score:
                            max_score = score
                            nbins = n
                            bin_method = bin_tech
                            best_stat = stat
                        
            if max_score<baseline:
                print(f'Improved baseline to {max_score} from {baseline}')
                baseline=max_score
                
                # if baseline is imporved we will use the combiantion of train and test to find the aggregate and then assign to test
                X_total_con_cat = pd.concat([X[[categorical_column,continuous_column]],test[[categorical_column,continuous_column]]])
                
                if bin_method == 'width':
                    bins = np.linspace(col_min, col_max, nbins + 1)  # n+1 edges = n bins
                    labels = range(nbins)  # 0, 1, ..., n-1
                    bucket_labels = pd.cut(
                        X_total_con_cat[continuous_column],
                        bins=bins,
                        labels=labels,
                        include_lowest=True,  # include min value
                        right=True            # right edge inclusive
                    )
                elif bin_method == 'frequency':
                    bucket_labels, actual_bins = pd.qcut(
                        X_total_con_cat[continuous_column],
                        q=nbins,
                        retbins=True,
                        duplicates='drop'
                    )                
                    actual_num_bins = len(actual_bins) - 1
                    actual_labels = list(range(actual_num_bins))
                    # Re-bin with appropriate labels
                    bucket_labels = pd.qcut(
                        X_total_con_cat[continuous_column],
                        q=nbins,
                        labels=actual_labels,
                        duplicates='drop'
                    )
                elif bin_method == 'supervised':
                    if PROBLEM == 'Regression':
                        tree = DecisionTreeRegressor(
                                    max_leaf_nodes=nbins, 
                                    min_samples_leaf=0.05,  # avoid overfitting small bins
                                    random_state=42
                                )
                    else:
                        tree = DecisionTreeClassifier(
                                    max_leaf_nodes=nbins, 
                                    min_samples_leaf=0.05,  # avoid overfitting small bins
                                    random_state=42
                                )
                    tree.fit(X[[continuous_column]], y)
                    # Extract split thresholds
                    thresholds = sorted(tree.tree_.threshold[tree.tree_.threshold != -2])
                    if not thresholds:
                        continue  # no valid splits
                    # Define bins using thresholds
                    bin_edges = [X[continuous_column].min()] + thresholds + [X[continuous_column].max()]
                    edges = np.unique(bin_edges)
                    if edges is None:
                        continue
                    bucket_labels = pd.cut(X_total_con_cat[continuous_column], bins=edges, labels=False, include_lowest=True)
                    
                            
                X_total_con_cat[f'{continuous_column}_bucket'] = bucket_labels.astype('int8')
                X[f'{continuous_column}_bucket'] = bucket_labels[:X.shape[0]].astype('int8')
                test[f'{continuous_column}_bucket'] = bucket_labels[X.shape[0]:].astype('int8')

                group = X_total_con_cat.groupby(f'{continuous_column}_bucket')[categorical_column]
                
                if stat == 'count':
                    freq = group.value_counts(normalize=False).unstack().fillna(0)
                    freq.columns = [f"{continuous_column}_{categorical_column}_{best_stat}_{nbins}_{bin_method}_{c}" for c in freq.columns]
                    X = X.join(freq, on=f'{continuous_column}_bucket')
                    X.drop([f'{continuous_column}_bucket'], axis = 1, inplace = True)
                    test = test.join(freq, on=f'{continuous_column}_bucket')
                    test.drop([f'{continuous_column}_bucket'], axis = 1, inplace = True)
                elif stat == 'nunique':
                    nunique_col = group.nunique()
                    X[f"{continuous_column}_{categorical_column}_{best_stat}_{nbins}_{bin_method}"] = X[f'{continuous_column}_bucket'].map(nunique_col)
                    X.drop([f'{continuous_column}_bucket'], axis = 1, inplace = True)
                    test[f"{continuous_column}_{categorical_column}_{best_stat}_{nbins}_{bin_method}"] = test[f'{continuous_column}_bucket'].map(nunique_col)
                    test.drop([f'{continuous_column}_bucket'], axis = 1, inplace = True)
                elif stat == 'dominance':
                    dominance = (
                        group
                         .apply(lambda s: s.value_counts(normalize=True).iloc[0])
                    )
                    X[f'{continuous_column}_{categorical_column}_{best_stat}_{nbins}_{bin_method}'] = X[f'{continuous_column}_bucket'].map(dominance)
                    X.drop([f'{continuous_column}_bucket'], axis = 1, inplace = True)
                    test[f'{continuous_column}_{categorical_column}_{best_stat}_{nbins}_{bin_method}'] = test[f'{continuous_column}_bucket'].map(dominance)
                    test.drop([f'{continuous_column}_bucket'], axis = 1, inplace = True)
                elif stat == 'entropy':
                    ent = (
                        group
                        .apply(lambda s: entropy(s.value_counts(normalize=True), base=2))
                    )
                    X[f'{continuous_column}_{categorical_column}_{best_stat}_{nbins}_{bin_method}'] = X[f'{continuous_column}_bucket'].map(ent)
                    X.drop([f'{continuous_column}_bucket'], axis = 1, inplace = True)
                    test[f'{continuous_column}_{categorical_column}_{best_stat}_{nbins}_{bin_method}'] = test[f'{continuous_column}_bucket'].map(ent)
                    test.drop([f'{continuous_column}_bucket'], axis = 1, inplace = True)
                elif stat == 'proportion':
                    prop = (
                        X_total_con_cat.groupby([f'{continuous_column}_bucket', categorical_column]).size() /
                        X_total_con_cat.groupby(f'{continuous_column}_bucket').size()
                    ).unstack().fillna(0)
                    prop.columns = [f"{continuous_column}_{categorical_column}_{best_stat}_{nbins}_{bin_method}_{c}" for c in prop.columns]
                    X = X.join(prop, on=f'{continuous_column}_bucket')
                    X.drop([f'{continuous_column}_bucket'], axis = 1, inplace = True)
                    test = test.join(prop, on=f'{continuous_column}_bucket')
                    test.drop([f'{continuous_column}_bucket'], axis = 1, inplace = True)
                elif stat == 'cross_counts':
                    cross = pd.crosstab(X_total_con_cat[f'{continuous_column}_bucket'], X_total_con_cat[categorical_column])
                    cross.columns = [f"{continuous_column}_{categorical_column}_{best_stat}_{nbins}_{bin_method}_{c}" for c in cross.columns]
                    X = X.join(cross, on=f'{continuous_column}_bucket')
                    X.drop([f'{continuous_column}_bucket'], axis = 1, inplace = True)
                    test = test.join(cross, on=f'{continuous_column}_bucket')
                    test.drop([f'{continuous_column}_bucket'], axis = 1, inplace = True)


X.to_csv('X.csv', index=False)
y.to_csv('y.csv', index=False)
test.to_csv('test.csv', index=False)


X = pd.read_csv('X.csv')
y = pd.read_csv('y.csv').squeeze()  # Converts back to Series if needed
test = pd.read_csv('test.csv')

baseline = 0.056338999171908606

stats_set = [["mean","median"],["std"],["count","nunique"],["min","max"],["skew"],["rank", "zscore"]]

continuous_columns = ['num_lanes', 'curvature', 'speed_limit', 'num_reported_accidents']


train_idx, val_idx = train_test_split(
    np.arange(len(X)), shuffle=True, random_state=42
)

#columns = X.select_dtypes(['int64','float64']).columns
columns = continuous_columns
X_train, X_val, y_train, y_val = X.iloc[train_idx].copy(), X.iloc[val_idx].copy(), y.iloc[train_idx], y.iloc[val_idx]

# for index,column in tqdm(enumerate(columns), total = len(columns)):
for index,column in enumerate(columns):
    print(f'Iteration {index+1} out of {len(columns)}')

    col_min, col_max = X[column].min(), X[column].max()
    if col_min == col_max:
        # Skip binning constant column
        continue

    skew = X[column].skew()
    kurt = X[column].kurt()
    uniq = X[column].nunique()
    corr = abs(np.corrcoef(X[column], y)[0,1]) if y is not None else 0
    
    for stat_group in tqdm(stats_set):
        # print(f'For {stat}\n\n')
        max_score = baseline
        nbins = None
        bin_method = None
        best_stat = None

        for stat in stat_group:
            
            no_of_bins = range(2, 11)
            
            for n in no_of_bins:
                # print(f'4. For {n} bins:')

                if uniq <= n:
                    break
                    
                if abs(skew) > 1.5:
                    bin_methods = ['supervised', 'frequency']
                elif 0.75 < abs(skew) <= 1.5:
                    bin_methods = ['width', 'frequency', 'supervised']
                else:
                    bin_methods = ['width', 'supervised']
                
                for bin_tech in bin_methods:
                    # print(f'5. For binning technique equal {bin_tech}:')
                    
                    if bin_tech == 'width':
                        # Auto-generate n bins between min and max
                        bins = np.linspace(col_min, col_max, n + 1)  # n+1 edges = n bins
                        labels = range(n)  # 0, 1, ..., n-1
                        bucket_labels = pd.cut(
                            X[column],
                            bins=bins,
                            labels=labels,
                            include_lowest=True,  # include min value
                            right=True            # right edge inclusive
                        )
                    elif bin_tech == 'frequency':
                        bucket_labels, actual_bins = pd.qcut(
                            X[column],
                            q=n,
                            retbins=True,
                            duplicates='drop'
                        )
                        actual_num_bins = len(actual_bins) - 1
                        actual_labels = list(range(actual_num_bins))             
                        # Re-bin with appropriate labels
                        bucket_labels = pd.qcut(
                            X[column],
                            q=n,
                            labels=actual_labels,
                            duplicates='drop'
                        )
                    elif bin_tech == 'supervised':
                        if PROBLEM == 'Regression':
                            tree = DecisionTreeRegressor(
                                        max_leaf_nodes=n, 
                                        min_samples_leaf=0.05,  # avoid overfitting small bins
                                        random_state=42
                                    )
                        else:
                            tree = DecisionTreeClassifier(
                                        max_leaf_nodes=n, 
                                        min_samples_leaf=0.05,  # avoid overfitting small bins
                                        random_state=42
                                    )
                        tree.fit(X_train[[column]], y_train)
                        # Extract split thresholds
                        thresholds = sorted(tree.tree_.threshold[tree.tree_.threshold != -2])
                        if not thresholds:
                            continue  # no valid splits
                        # Define bins using thresholds
                        bin_edges = [X_train[column].min()] + thresholds + [X_train[column].max()]
                        edges = np.unique(bin_edges)
                        if edges is None:
                            continue
                        bucket_labels = pd.cut(X[column], bins=edges, labels=False, include_lowest=True)
                    
                    X_train[f'{column}_bucket'] = bucket_labels.iloc[train_idx].astype('int8')
                    X_val[f'{column}_bucket'] = bucket_labels.iloc[val_idx].astype('int8')        
                    
                    X_train['accident_risk'] = y_train

                    if stat == 'rank':
                        aggregate = X_train.groupby(f'{column}_bucket')['accident_risk'].mean().rank(method='dense', ascending=True)
                        aggregate = aggregate / aggregate.max()
                    elif stat == 'zscore':
                        mean = X_train.groupby(f'{column}_bucket')['accident_risk'].mean()
                        if mean.std() != 0:
                            aggregate = (mean - mean.mean()) / mean.std()
                        else:
                            aggregate = (mean - mean.mean())
                    else:
                        aggregate = X_train.groupby(f'{column}_bucket')['accident_risk'].agg(stat)
                    X_train[f'{column}_encoded_accident_risk_{stat}_{n}_{bin_tech}'] = X_train[f'{column}_bucket'].map(aggregate)
                    X_val[f'{column}_encoded_accident_risk_{stat}_{n}_{bin_tech}'] = X_val[f'{column}_bucket'].map(aggregate)
                    
                    X_train.drop(['accident_risk'],axis=1,inplace=True)
                    
                    X_train.drop([f'{column}_bucket'], axis = 1, inplace = True)
                    X_val.drop([f'{column}_bucket'], axis = 1, inplace = True)
                    
                    score = test_features(X_train, X_val, y_train, y_val, True)

                    if max_score < score:
                        max_score = score
                        nbins = n
                        bin_method = bin_tech
                        best_stat = stat

                    X_train.drop([f'{column}_encoded_accident_risk_{stat}_{n}_{bin_tech}'], axis=1, inplace=True)
                    X_val.drop([f'{column}_encoded_accident_risk_{stat}_{n}_{bin_tech}'], axis=1, inplace=True)
                    
        if max_score<baseline:
            print(f'Improved baseline to {max_score} from {baseline}')
            baseline=max_score

            X_test_concat = pd.concat([X[[column]],test[[column]]])
            
            if bin_method == 'width':
                # Auto-generate n bins between min and max
                bins = np.linspace(col_min, col_max, nbins + 1)  # n+1 edges = n bins
                labels = range(nbins)  # 0, 1, ..., n-1
                bucket_labels = pd.cut(
                    X_test_concat[column],
                    bins=bins,
                    labels=labels,
                    include_lowest=True,  # include min value
                    right=True            # right edge inclusive
                )
                
            elif bin_method == 'frequency':
                bucket_labels, actual_bins = pd.qcut(
                    X_test_concat[column],
                    q=nbins,
                    retbins=True,
                    duplicates='drop'
                )   
                actual_num_bins = len(actual_bins) - 1
                actual_labels = list(range(actual_num_bins))
                # Re-bin with appropriate labels
                bucket_labels = pd.qcut(
                    X_test_concat[column],
                    q=nbins,
                    labels=actual_labels,
                    duplicates='drop'
                )
                
            elif bin_method == 'supervised':
                if PROBLEM == 'Regression':
                    tree = DecisionTreeRegressor(
                                max_leaf_nodes=nbins, 
                                min_samples_leaf=0.05,  # avoid overfitting small bins
                                random_state=42
                            )
                else:
                    tree = DecisionTreeClassifier(
                                max_leaf_nodes=nbins, 
                                min_samples_leaf=0.05,  # avoid overfitting small bins
                                random_state=42
                            )
                tree.fit(X[[column]], y)
                # Extract split thresholds
                thresholds = sorted(tree.tree_.threshold[tree.tree_.threshold != -2])
                if not thresholds:
                    continue  # no valid splits
                # Define bins using thresholds
                bin_edges = [X[column].min()] + thresholds + [X[column].max()]
                edges = np.unique(bin_edges)
                if edges is None:
                    continue
                bucket_labels = pd.cut(X_test_concat[column], bins=edges, labels=False, include_lowest=True)
            
            X[f'{column}_bucket'] = bucket_labels[:X.shape[0]].astype('int8')
            test[f'{column}_bucket'] = bucket_labels[X.shape[0]:].astype('int8')
            
            X['accident_risk'] = y
            
            if best_stat == 'rank':
                aggregate = X.groupby(f'{column}_bucket')['accident_risk'].mean().rank(method='dense', ascending=True)
                aggregate = aggregate / aggregate.max()
            elif best_stat == 'zscore':
                mean = X.groupby(f'{column}_bucket')['accident_risk'].mean()
                if mean.std() != 0:
                    aggregate = (mean - mean.mean()) / mean.std()
                else:
                    aggregate = (mean - mean.mean())
            else:    
                aggregate = X.groupby(f'{column}_bucket')['accident_risk'].agg(best_stat)
            
            X[f'{column}_encoded_accident_risk_{best_stat}_{nbins}_{bin_method}'] = X[f'{column}_bucket'].map(aggregate)
            test[f'{column}_encoded_accident_risk_{best_stat}_{nbins}_{bin_method}'] = test[f'{column}_bucket'].map(aggregate)
            
            X.drop(['accident_risk'],axis=1,inplace=True)
            
            X.drop([f'{column}_bucket'], axis = 1, inplace = True)
            test.drop([f'{column}_bucket'], axis = 1, inplace = True)


X.to_csv('X.csv', index=False)
y.to_csv('y.csv', index=False)
test.to_csv('test.csv', index=False)


X = pd.read_csv('X.csv')
y = pd.read_csv('y.csv').squeeze()  # Converts back to Series if needed
test = pd.read_csv('test.csv')

baseline = 0.056338999171908606

stats_set = [["mean","median"],["std"],["count","nunique"],["min","max"],["skew"],["rank", "zscore"]]

categorical_columns = list(X.select_dtypes(['bool','object']).columns) + ['road_type_encoding','lighting_encoding']


train_idx, val_idx = train_test_split(
    np.arange(len(X)), shuffle=True, random_state=42
)

columns = list(categorical_columns)#+list(continuous_columns)
X_train, X_val, y_train, y_val = X.iloc[train_idx].copy(), X.iloc[val_idx].copy(), y.iloc[train_idx], y.iloc[val_idx]

for index,column in enumerate(columns):
    print(f'Iteration {index+1} out of {len(columns)}')
    
    for stat_group in tqdm(stats_set):
        max_score = baseline
        best_stat = None

        for stat in stat_group:
            # print(f'For {stat}')
            
            X_train['accident_risk'] = y_train

            if stat == 'rank':
                aggregate = X_train.groupby(column)['accident_risk'].mean().rank(method='dense', ascending=True)
                aggregate = aggregate / aggregate.max()
            elif stat == 'zscore':
                mean = X_train.groupby(column)['accident_risk'].mean()
                if mean.std() != 0:
                    aggregate = (mean - mean.mean()) / mean.std()
                else:
                    aggregate = (mean - mean.mean())
            else:
                aggregate = X_train.groupby(column)['accident_risk'].agg(stat)
            
            X_train.drop(['accident_risk'],axis=1,inplace=True)
            
            X_train[f'{column}_encoded_accident_risk_{stat}'] = X_train[column].map(aggregate)
            X_val[f'{column}_encoded_accident_risk_{stat}'] = X_val[column].map(aggregate)
            
            score = test_features(X_train, X_val, y_train, y_val, True)

            if score < max_score:
                max_score = score
                best_stat = stat
                
            X_train.drop([f'{column}_encoded_accident_risk_{stat}'], axis=1, inplace=True)
            X_val.drop([f'{column}_encoded_accident_risk_{stat}'], axis=1, inplace=True)
                
        if max_score<baseline:
            print(f'Improved baseline of {max_score} that {baseline} before')
            baseline=max_score

            X['accident_risk'] = y

            if best_stat == 'rank':
                aggregate = X.groupby(column)['accident_risk'].mean().rank(method='dense', ascending=True)
                aggregate = aggregate / aggregate.max()
            elif best_stat == 'zscore':
                mean = X.groupby(column)['accident_risk'].mean()
                if mean.std() != 0:
                    aggregate = (mean - mean.mean()) / mean.std()
                else:
                    aggregate = (mean - mean.mean())
            else:
                aggregate = X.groupby(column)['accident_risk'].agg(best_stat)
            
            X.drop(['accident_risk'],axis=1,inplace=True)

            X[f'{column}_encoded_accident_risk_{best_stat}'] = X[column].map(aggregate)
            test[f'{column}_encoded_accident_risk_{best_stat}'] = test[column].map(aggregate)


X = pd.read_csv('X.csv')
y = pd.read_csv('y.csv').squeeze()  # Converts back to Series if needed
test = pd.read_csv('test.csv')

continuous_columns = ['num_lanes', 'curvature', 'speed_limit', 'num_reported_accidents']

baseline = 0.056338999171908606


train_idx, val_idx = train_test_split(
    np.arange(len(X)), shuffle=True, random_state=42
)

# continuous_columns = X.select_dtypes(['int64','float64']).columns
#columns = list(label_encoded_columns)+list(continuous_columns)
columns = list(continuous_columns)
X_copy = X.copy()

degree = len(columns)
    
# Iterate over degrees 1 to `degree`
for d in range(2, degree + 1):
#for d in range(degree, degree + 1):
    # Get all combinations with replacement of columns
    n = len(columns)
    total_combinations = comb(n + d -1, d)  # combinations_with_replacement count
    for cols in tqdm(combinations_with_replacement(columns, d), total=total_combinations, desc=f'Degree {d}'):
        col_name = '*'.join(cols)
        X_copy[col_name] = X_copy[cols[0]]
        
        for idx in range(1,len(cols)):
            X_copy[col_name] = X_copy[col_name] * X_copy[cols[idx]]

        X_train, X_val, y_train, y_val = X_copy.iloc[train_idx].copy(), X_copy.iloc[val_idx].copy(), y.iloc[train_idx], y.iloc[val_idx]
        score = test_features(X_train, X_val, y_train, y_val, True)
        
        if score<baseline:
            print(f'Improved baseline to {score} from {baseline}')
            baseline=score
            
            X[col_name] = X[cols[0]]
            test[col_name] = test[cols[0]]
            for idx in range(1,len(cols)):
                X[col_name] = X[col_name] * X[cols[idx]]
                test[col_name] = test[col_name] * test[cols[idx]]
            
        else:
            X_copy.drop([col_name], axis=1, inplace=True)


X.to_csv('X.csv', index=False)
y.to_csv('y.csv', index=False)
test.to_csv('test.csv', index=False)


X = pd.read_csv('X.csv')
y = pd.read_csv('y.csv').squeeze()  # Converts back to Series if needed
test = pd.read_csv('test.csv')

baseline = 0.056338999171908606


train_idx, val_idx = train_test_split(
    np.arange(len(X)), shuffle=True, random_state=42
)

columns = X.columns
for col in tqdm(columns):
    
    X_copy = X.copy()
    X_copy.drop([col], axis=1, inplace = True)

    X_train, X_val, y_train, y_val = X_copy.iloc[train_idx], X_copy.iloc[val_idx], y.iloc[train_idx], y.iloc[val_idx]
    score = test_features(X_train, X_val, y_train, y_val, True)
        
    if score<baseline:
        print(f'Improved baseline to {score} from {baseline}')
        baseline=score
        X.drop([col], axis=1, inplace = True)
        test.drop([col], axis=1, inplace = True)


X.to_csv('X.csv', index=False)
y.to_csv('y.csv', index=False)
test.to_csv('test.csv', index=False)


X = pd.read_csv('X.csv')
y = pd.read_csv('y.csv').squeeze()  # Converts back to Series if needed
test = pd.read_csv('test.csv')

baseline = 0.05633860616081338

# columns = X.columns
columns = X.select_dtypes(['int64','float64']).columns


train_idx, val_idx = train_test_split(
    np.arange(len(X)), shuffle=True, random_state=42
)

for column in tqdm(columns):

    for scale in ['standard', 'minmax']:

        if scale == 'standard':
            scalar = StandardScaler()
        else:
            scalar = MinMaxScaler()

        col_original = X[column].copy()
        X[column] = scalar.fit_transform(X[[column]]).astype('float32')

        X_train, X_val, y_train, y_val = X.iloc[train_idx].copy(), X.iloc[val_idx].copy(), y.iloc[train_idx], y.iloc[val_idx]
        score = test_features(X_train, X_val, y_train, y_val, True)
        
        if score<baseline:
            print(f'Improved baseline to {score} from {baseline}')
            baseline=score
            test[column] = scalar.transform(test[[column]]).astype('float32')
        else:
            X[column] = col_original


X.to_csv('X.csv', index=False)
y.to_csv('y.csv', index=False)
test.to_csv('test.csv', index=False)


X = pd.read_csv('X.csv')
y = pd.read_csv('y.csv').squeeze()  # Converts back to Series if needed
test = pd.read_csv('test.csv')


print(type(X), type(y), type(test))


X = X.reindex(sorted(X.columns), axis=1)
test = test.reindex(sorted(test.columns), axis=1)


sample = pd.read_csv('/kaggle/input/playground-series-s5e10/sample_submission.csv')
sample.columns


def rmse(y_true,y_pred):
    return np.sqrt(mean_squared_error(y_true,y_pred))

def param_finetuning(X, y, test, bestScore):
    xgb_params = {
        
        'n_estimators': 85,
        'learning_rate': 0.28,
        
        # 'max_depth': 8,
        # 'min_child_weight': 3,
        # 'subsample': 0.8,
        # 'colsample_bytree': 0.8,
        # 'reg_lambda': 1.0,
        # 'reg_alpha': 0.1,
        # 'gamma': 0,
        # 'tree_method': 'gpu_hist',  # or 'hist' if no GPU
        # 'random_state': 42,
        'n_jobs': -1,
    }

    X = X.copy()
    test = test.copy()

    for col in X.select_dtypes(include=['object','bool']).columns:
        X[col] = X[col].astype('category')
        test[col] = test[col].astype('category')

    sample = pd.read_csv('/kaggle/input/playground-series-s5e10/sample_submission.csv')

    X_train, X_val, y_train, y_val = train_test_split(X,y, shuffle=True, random_state = 42)    
        
    # model = XGBRegressor(**xgb_params, enable_categorical=True, random_state=42)
    # model.fit(X_train, y_train, eval_set=[(X_val, y_val)],
    #              verbose=10000, early_stopping_rounds=100)

    # model = cuLinearRegression()
    # model.fit(X_train, y_train)

    # model = cuRidge(alpha = 250)
    # model.fit(X_train, y_train)

    # model = cuElasticNet(alpha = 0.3050, l1_ratio = 0.4990)
    # model.fit(X_train, y_train)

    # model = cuLasso(
    #     alpha = 0.58221,
    #     max_iter = 10000,
    #     tol = 0.0001,
    #     solver = 'qn'
    # )
    # cat_features = [col for i, col in enumerate(X_train.columns) if str(X_train[col].dtype) == 'category']
    # X_train.drop(cat_features, axis = 1, inplace = True)
    # X_val.drop(cat_features, axis = 1, inplace = True)
    # test.drop(cat_features, axis = 1, inplace = True)
    # model.fit(X_train, y_train)

    # model = cuElasticNet(
    #     alpha = 0.3811, 
    #     l1_ratio = 0.5
    # )
    # cat_features = [col for i, col in enumerate(X_train.columns) if str(X_train[col].dtype) == 'category']
    # X_train.drop(cat_features, axis = 1, inplace = True)
    # X_val.drop(cat_features, axis = 1, inplace = True)
    # test.drop(cat_features, axis = 1, inplace = True)
    # model.fit(X_train, y_train)

    model = lgb.LGBMRegressor(
        objective="regression",
        boosting_type="gbdt",
        n_estimators=957,
        num_leaves=31,
        learning_rate=0.05,
        random_state=42,
        #device="gpu",
        deterministic=True,
        verbose = -1,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="rmse",
        callbacks=[log_evaluation(-1)]
    )

    # mlp = MLPRegressor(
    #     hidden_layer_sizes=(100, 50),   # 2 hidden layers: 100 and 50 neurons
    #     activation="relu",              # activation function
    #     solver="adam",                  # optimizer
    #     max_iter=300,
    #     learning_rate_init=0.001,
    #     random_state=42
    # )
    # mlp.fit(X_train, y_train)

    # model = CatBoostRegressor(
    #     loss_function='RMSE',
    #     iterations=1500,
    #     learning_rate=0.03,
    #     depth=6,
    #     l2_leaf_reg=3,
    #     random_seed=42,
    #     eval_metric='RMSE',
    #     verbose=False,
    #     early_stopping_rounds=100
    # )
    # cat_features = [i for i, col in enumerate(X_train.columns) if str(X_train[col].dtype) == 'category']
    # model.fit(
    #     X_train, y_train,
    #     eval_set=[(X_val, y_val)],
    #     cat_features=cat_features,
    #     verbose=10000
    # )

    # model = cuRF(
    #     n_estimators=505,
    #     max_depth=12,
    #     #max_features=0.9,
    #     # min_samples_split=2,
    #     # min_samples_leaf = 2,
    #     n_streams=1,
    #     bootstrap=False,      # to behave more like ExtraTrees
    #     #split_criterion="entropy"
    # )
    # cat_features = [col for i, col in enumerate(X_train.columns) if str(X_train[col].dtype) == 'category']
    # X_train.drop(cat_features, axis = 1, inplace = True)
    # X_val.drop(cat_features, axis = 1, inplace = True)
    # test.drop(cat_features, axis = 1, inplace = True)
    # model.fit(X_train, y_train)

    # model = AdaBoostRegressor(n_estimators = 6, learning_rate=2.25,random_state = 42)
    # cat_features = [col for i, col in enumerate(X_train.columns) if str(X_train[col].dtype) == 'category']
    # X_train.drop(cat_features, axis = 1, inplace = True)
    # X_val.drop(cat_features, axis = 1, inplace = True)
    # test.drop(cat_features, axis = 1, inplace = True)
    # model.fit(X_train, y_train)

    # model = GradientBoostingRegressor(
    #     n_estimators = 110, 
    #     learning_rate=1,
    #     subsample=0.9,
    #     min_samples_leaf=2,
    #     max_depth=4,
    #     max_leaf_nodes=16,
    #     random_state = 42)
    # cat_features = [col for i, col in enumerate(X_train.columns) if str(X_train[col].dtype) == 'category']
    # X_train.drop(cat_features, axis = 1, inplace = True)
    # X_val.drop(cat_features, axis = 1, inplace = True)
    # test.drop(cat_features, axis = 1, inplace = True)
    # model.fit(X_train, y_train)

    # model = KNeighborsRegressor(
    #     n_neighbors = 45,
    #     metric = 'cosine',
    #     verbose = 0
    # )
    # cat_features = [col for i, col in enumerate(X_train.columns) if str(X_train[col].dtype) == 'category']
    # X_train.drop(cat_features, axis = 1, inplace = True)
    # X_val.drop(cat_features, axis = 1, inplace = True)
    # test.drop(cat_features, axis = 1, inplace = True)
    # model.fit(X_train, y_train)

    # model = LinearSVR(
    #     C = 1.38,
    #     penalty = 'l1',
    #     max_iter = 1200,
    #     verbose = 0
    # )
    # cat_features = [col for i, col in enumerate(X_train.columns) if str(X_train[col].dtype) == 'category']
    # X_train.drop(cat_features, axis = 1, inplace = True)
    # X_val.drop(cat_features, axis = 1, inplace = True)
    # test.drop(cat_features, axis = 1, inplace = True)
    # model.fit(X_train, y_train)

    # model = RandomForestRegressor(
    #     n_estimators=1, 
    #     bootstrap=False,
    #     split_criterion = 4,
    #     max_depth = 15,
    #     max_leaves = 10000,
    #     max_features = 1.0,
    #     n_bins = 512,
    #     min_samples_leaf = 4,
    #     random_state=42
    # )
    # cat_features = [col for i, col in enumerate(X_train.columns) if str(X_train[col].dtype) == 'category']
    # X_train.drop(cat_features, axis = 1, inplace = True)
    # X_val.drop(cat_features, axis = 1, inplace = True)
    # test.drop(cat_features, axis = 1, inplace = True)
    # model.fit(X_train, y_train)

    # will change depending on the score metric

    # accuracy
    # y_pred = model.predict(X_val)
    # score = accuracy_score(y_val,y_pred)

    # rmse
    y_pred = model.predict(X_val)
    score = rmse(y_val,y_pred)
    
    # roc-auc
    # y_pred_proba = model.predict_proba(X_val)[:, 1]
    # score = roc_auc_score(y_val, y_pred_proba)

    print(score)
    if score <= bestScore:
            print(f"Results Improved to {score} from {bestScore}")
        
            # roc auc
            # y_test = model.predict_proba(test)[:, 1]

            # accuracy and rmse
            y_test = model.predict(test)
        
            sample['accident_risk'] = y_test
            bestScore = score
    return score, sample


def rmse(y_true,y_pred):
    return np.sqrt(mean_squared_error(y_true,y_pred))

def param_finetuning(X, y, test, bestScore):
    X = X.copy()
    test = test.copy()

    for col in X.select_dtypes(include=['object','bool']).columns:
        X[col] = X[col].astype('category')
        test[col] = test[col].astype('category')

    sample = pd.read_csv('/kaggle/input/playground-series-s5e10/sample_submission.csv')

    X_train, X_val, y_train, y_val = train_test_split(X,y, shuffle=True, random_state = 42)    

    model = lgb.LGBMRegressor(
        learning_rate = 0.249881569,
        n_estimators = 100,
        num_leaves = 32,
        # max_depth = 16,
        min_data_in_leaf = 20,
        feature_fraction = 0.9,
        # lambda_l2 = 0.1,
        objective="regression",
        random_state=42,
        #device="gpu",
        verbose = -1,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="rmse",
        callbacks=[log_evaluation(-1)]
    )

    # rmse
    y_pred = model.predict(X_val)
    score = rmse(y_val,y_pred)
    
    print(score)
    if score <= bestScore:
            print(f"Results Improved to {score} from {bestScore}")
        
            y_test = model.predict(test)
        
            sample['accident_risk'] = y_test
            bestScore = score
    return score, sample


bestScore = 0.05629239067421117
bestScore,sample = param_finetuning(X,y,test,bestScore)
#sample.to_csv('/kaggle/working/submission.csv',index=False)


def rmse(y_true,y_pred):
    return np.sqrt(mean_squared_error(y_true,y_pred))
    
def model_training(X, y, test, n_splits, bestScore):
    xgb_params = {
        'n_estimators': 17,
        'learning_rate': 0.279123,
        'max_depth': 8,
        'min_child_weight': 1,
        'subsample': 1.0,
        'colsample_bytree': 1.0,
        'reg_lambda': 1.0,
        'reg_alpha': 0.0,
        'gamma': 0,
        # 'tree_method': 'gpu_hist',  # or 'hist' if no GPU
        # 'random_state': 42,
        'n_jobs': -1,
    }

    X = X.copy()
    test = test.copy()

    for col in X.select_dtypes(include=['object','bool']).columns:
        X[col] = X[col].astype('category')
        test[col] = test[col].astype('category')

    sample = pd.read_csv('/kaggle/input/playground-series-s5e10/sample_submission.csv')

    kfolds = KFold(n_splits=n_splits,shuffle=True)

    # Out-of-fold preds for CV evaluation
    oof_preds = np.zeros(len(X))
    test_preds = np.zeros((len(test), n_splits))  # one column per fold
    scores = []

    for fold, (train_idx, val_idx) in enumerate(kfolds.split(X, y)):
        print(f'Fold {fold + 1} of {n_splits}')
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        

        # model = XGBRegressor(**xgb_params, enable_categorical=True)
        # model.fit(X_train, y_train, eval_set=[(X_val, y_val)],
        #           verbose=10000, early_stopping_rounds=100)

        # model = cuLinearRegression()
        # model.fit(X_train, y_train)

        # model = cuRidge(alpha = 250)
        # model.fit(X_train, y_train)

        # model = cuLasso(
        #     alpha = 0.58221,
        #     max_iter = 10000,
        #     tol = 0.0001,
        #     solver = 'qn'
        # )
        # cat_features = [col for i, col in enumerate(X_train.columns) if str(X_train[col].dtype) == 'category']
        # X_train.drop(cat_features, axis = 1, inplace = True)
        # X_val.drop(cat_features, axis = 1, inplace = True)
        # model.fit(X_train, y_train)
        
        # model = cuElasticNet(
        #     alpha = 0.3811, 
        #     l1_ratio = 0.5
        # )
        # cat_features = [col for i, col in enumerate(X_train.columns) if str(X_train[col].dtype) == 'category']
        # X_train.drop(cat_features, axis = 1, inplace = True)
        # X_val.drop(cat_features, axis = 1, inplace = True)
        # model.fit(X_train, y_train)

        model = lgb.LGBMRegressor(
            learning_rate = 0.249881569,
            n_estimators = 100,
            num_leaves = 32,
            # max_depth = 16,
            min_data_in_leaf = 20,
            feature_fraction = 0.9,
            # lambda_l2 = 0.1,
            objective="regression",
            #device="gpu",
            verbose = -1,
        )
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric="rmse",
            callbacks=[early_stopping(50), log_evaluation(-1)]
        )

        # mlp = MLPRegressor(
        #     hidden_layer_sizes=(100, 50),   # 2 hidden layers: 100 and 50 neurons
        #     activation="relu",              # activation function
        #     solver="adam",                  # optimizer
        #     max_iter=300,
        #     learning_rate_init=0.001,
        #     random_state=42
        # )
        # mlp.fit(X_train, y_train)

        # model = CatBoostRegressor(
        #     loss_function='RMSE',
        #     iterations=1500,
        #     learning_rate=0.03,
        #     depth=6,
        #     l2_leaf_reg=3,
        #     random_seed=42,
        #     eval_metric='RMSE',
        #     verbose=False,
        #     early_stopping_rounds=100
        # )
        # cat_features = [i for i, col in enumerate(X_train.columns) if str(X_train[col].dtype) == 'category']
        # model.fit(
        #     X_train, y_train,
        #     eval_set=[(X_val, y_val)],
        #     cat_features=cat_features,
        #     verbose=10000
        # )

        # model = cuRF(
        #     n_estimators=505,
        #     max_depth=12,
        #     #max_features=0.9,
        #     # min_samples_split=2,
        #     # min_samples_leaf = 2,
        #     n_streams=1,
        #     bootstrap=False,      # to behave more like ExtraTrees
        #     #split_criterion="entropy"
        # )
        # cat_features = [col for i, col in enumerate(X_train.columns) if str(X_train[col].dtype) == 'category']
        # X_train.drop(cat_features, axis = 1, inplace = True)
        # X_val.drop(cat_features, axis = 1, inplace = True)
        # model.fit(X_train, y_train)
        
        # model = AdaBoostRegressor(n_estimators = 6, learning_rate=2.25)
        # cat_features = [col for i, col in enumerate(X_train.columns) if str(X_train[col].dtype) == 'category']
        # X_train.drop(cat_features, axis = 1, inplace = True)
        # X_val.drop(cat_features, axis = 1, inplace = True)
        # model.fit(X_train, y_train)
        
        # model = GradientBoostingRegressor(
        # n_estimators = 110, 
        # learning_rate=1,
        # subsample=0.9,
        # min_samples_leaf=2,
        # max_depth=4,
        # max_leaf_nodes=16
        # )
        # cat_features = [col for i, col in enumerate(X_train.columns) if str(X_train[col].dtype) == 'category']
        # X_train.drop(cat_features, axis = 1, inplace = True)
        # X_val.drop(cat_features, axis = 1, inplace = True)
        # model.fit(X_train, y_train)
        
        # model = KNeighborsRegressor(
        #     n_neighbors = 45,
        #     metric = 'cosine',
        #     verbose = 0
        # )
        # cat_features = [col for i, col in enumerate(X_train.columns) if str(X_train[col].dtype) == 'category']
        # X_train.drop(cat_features, axis = 1, inplace = True)
        # X_val.drop(cat_features, axis = 1, inplace = True)
        # model.fit(X_train, y_train)
        
        # model = LinearSVR(
        #     C = 1.38,
        #     penalty = 'l1',
        #     max_iter = 1200,
        #     verbose = 0
        # )
        # cat_features = [col for i, col in enumerate(X_train.columns) if str(X_train[col].dtype) == 'category']
        # X_train.drop(cat_features, axis = 1, inplace = True)
        # X_val.drop(cat_features, axis = 1, inplace = True)
        # model.fit(X_train, y_train)
        
        # model = RandomForestRegressor(
        #     n_estimators=1, 
        #     bootstrap=False,
        #     split_criterion = 4,
        #     max_depth = 15,
        #     max_leaves = 10000,
        #     max_features = 1.0,
        #     n_bins = 512,
        #     min_samples_leaf = 4
        # )
        # cat_features = [col for i, col in enumerate(X_train.columns) if str(X_train[col].dtype) == 'category']
        # X_train.drop(cat_features, axis = 1, inplace = True)
        # X_val.drop(cat_features, axis = 1, inplace = True)
        # model.fit(X_train, y_train)

        # will change depending on the score metric

        # accuracy
        # y_pred = model.predict(X_val)
        # oof_preds[val_idx] = y_pred
        # score = accuracy_score(y_val,y_pred)
        # scores.append(score)
        # test_preds[:, fold] = model.predict(test)

        # rmse
        y_pred = model.predict(X_val)
        oof_preds[val_idx] = y_pred
        score = rmse(y_val,y_pred)
        scores.append(score)
        test_preds[:, fold] = model.predict(test)
    
        # roc-auc
        # y_pred_proba = model.predict_proba(X_val)[:, 1]
        # oof_preds[val_idx] = y_pred_proba
        # score = roc_auc_score(y_val, y_pred_proba)
        # scores.append(score)
        # test_preds[:, fold] = model.predict_proba(test)[:, 1]

    # accuracy
    # score = accuracy_score(y, oof_preds)
    
    # rmse
    score = rmse(y, oof_preds)

    # roc-auc
    # score = roc_auc_score(y, oof_preds)
    print(f"Achieved Average CV score of {score}")
    if score <= bestScore:
        print(f"Results Improved to {score} from {bestScore}")
        
        # roc auc
        # y_test = model.predict_proba(test)[:, 1]
        
        # accuracy and rmse
        y_test = test_preds.mean(axis=1)
        X_Model = oof_preds
        
        sample['accident_risk'] = y_test
        bestScore = score

        sample.to_csv(f'/kaggle/working/test_{Model}.csv',index=False)
        pd.DataFrame(X_Model).to_csv(f'/kaggle/working/X_{Model}.csv', index=False)
            
    print(f"Best Accuracy: {bestScore}")
    return


bestScore = 0.056120007384603494
model_training(X,y,test,10,bestScore)


sample['Personality'] = sample['Personality'].map({1: 'Extrovert', 0: 'Introvert'})
sample.to_csv('/kaggle/working/submission.csv',index=False)


sample


X_Model = pd.read_csv(f'X_{Model}.csv')
test_Model = pd.read_csv(f'test_{Model}.csv')


X_Model.isna().sum()


test_Model.isna().sum()


test_Model.fillna(test_Model.mean(), inplace=True)
test_Model.to_csv(f'/kaggle/working/test_{Model}.csv',index=False)


test_Model


import numpy as np
import pandas as pd

import os
from glob import glob

from sklearn.preprocessing import StandardScaler

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam


# Combining output from all models
train = pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv')

input_dir = '/kaggle/working/'

csv_files = glob(os.path.join(input_dir, 'X_*.csv'))
dfs = []
dfs_col_name = []
for file in csv_files:
    df = pd.read_csv(file)
    col_name = os.path.splitext(os.path.basename(file))[0]
    dfs.append(df)
    dfs_col_name.append(col_name)
X = pd.concat(dfs, axis=1)
X.columns = dfs_col_name

csv_files = glob(os.path.join(input_dir, 'test_*.csv'))
dfs = []
for file in csv_files:
    df = pd.read_csv(file)
    df.drop(['id'], axis = 1, inplace = True)
    col_name = os.path.splitext(os.path.basename(file))[0]
    dfs.append(df)
test = pd.concat(dfs, axis=1)
test.columns = dfs_col_name

y = train['accident_risk'] # (train['accident_risk']=='Extrovert').astype(int)

sample = pd.read_csv('/kaggle/input/playground-series-s5e10/sample_submission.csv')


X=X[['X_XGB', 'X_LGBM']]
test=test[['X_XGB', 'X_LGBM']]


print(np.any(np.isnan(X)), np.any(np.isinf(X)))
print(np.any(np.isnan(y)), np.any(np.isinf(y)))
print(np.max(np.abs(X.values)))
print(np.percentile(y, [0, 25, 50, 75, 100]))


X = X.fillna(X.mean(numeric_only=True))


scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)
test = scaler.transform(test)


def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))

meta_nn = Sequential([
    Input(shape=(X.shape[1],)),
    Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
    BatchNormalization(),
    Dropout(0.5),
    Dense(16, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
    Dropout(0.3),
    Dense(1)
])

meta_nn.compile(optimizer=Adam(learning_rate=1e-3), loss='mse', metrics=[rmse])

early_stop = EarlyStopping(monitor='val_rmse', patience=10, restore_best_weights=True, mode='min')

history = meta_nn.fit(
    X, y,
    validation_split=0.2,  # works fine here since OOF already removes bias
    epochs=200,
    batch_size=64,
    callbacks=[early_stop],
    verbose=1
)

meta_preds = meta_nn.predict(test)



pd.DataFrame(meta_preds).isna().sum()


sample['accident_risk'] = meta_preds
sample.to_csv(f'/kaggle/working/NN_Meta.csv',index=False)




