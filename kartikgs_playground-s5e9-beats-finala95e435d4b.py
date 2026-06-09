Model = 'Lasso_Ensemble_Level1'


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

# import lightgbm as lgb
# from lightgbm import early_stopping, log_evaluation

#from sklearn.neural_network import MLPRegressor

# --- set environment variables first ---
# os.environ["TF_DETERMINISTIC_OPS"] = "1"  # ensure deterministic GPU ops
# os.environ["PYTHONHASHSEED"] = "42"       # reproducible hashing
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # must be set before tf import
# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder, OneHotEncoder, PowerTransformer
from scipy.stats import zscore, boxcox

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


train = pd.read_csv('/kaggle/input/playground-series-s5e9/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e9/test.csv')

# Combining output from all models
# train = pd.read_csv('/kaggle/input/playground-series-s5e9/train.csv')

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


int_features = train.select_dtypes(include=['int64','float64']).columns
n = len(int_features)
n_cols = 2
n_rows = int(np.ceil(n / n_cols))

plt.figure(figsize=(16, n_rows * 4))

for i, feature in enumerate(int_features, 1):
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
    for col in X_train_copy.select_dtypes(include='object').columns:
        X_train_copy[col] = X_train_copy[col].astype('category')
        X_val_copy[col] = X_val_copy[col].astype('category')
    
    # model = XGBRegressor(random_state=42,enable_categorical=enable_categorical,tree_method='gpu_hist')
    # model = XGBRegressor(random_state=42,enable_categorical=enable_categorical)
    # model.fit(X_train_copy, y_train, eval_set=[(X_val_copy,y_val)], verbose=0)

    # model = LinearRegression()
    # model.fit(X_train_copy, y_train)

    #model = cuLinearRegression()
    #model.fit(X_train_copy, y_train)
    
    model = cuRidge()
    model.fit(X_train_copy, y_train)
    
    # model = cuElasticNet()
    # model.fit(X_train_copy, y_train)
    
    # model = cuLasso()
    # model.fit(X_train_copy, y_train)

    # model = lgb.LGBMRegressor(
    #     objective="regression",
    #     random_state=42,
    #     device="gpu",
    #     verbose=-1
    # )
    # model.fit(
    #     X_train, y_train,
    #     eval_set=[(X_val, y_val)],
    #     eval_metric="rmse",
    #     callbacks=[log_evaluation(0)]
    # )

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
    # model.fit(X_train, y_train)

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
    #     X_train, y_train,
    #     validation_data=(X_val, y_val),
    #     epochs=1,
    #     batch_size=512,
    #     verbose=-1,
    #     shuffle=False
    # )

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


# X, y = train.drop(['BeatsPerMinute'], axis=1), train['BeatsPerMinute']#(train['BeatsPerMinute']=='Extrovert').astype(int) 

# for ensemble
y = train['BeatsPerMinute'] # (train['BeatsPerMinute']=='Extrovert').astype(int) 


X_train, X_val, y_train, y_val = train_test_split(X, y, shuffle=True, random_state = 42)
baseline = test_features(X_train, X_val, y_train, y_val, False)
print(baseline)


baseline = 26.339886414336178


X.drop(['id'],axis = 1, inplace=True)
test.drop(['id'],axis = 1, inplace=True)


X_train, X_val, y_train, y_val = train_test_split(X, y, shuffle=True, random_state = 42)
score = test_features(X_train, X_val, y_train, y_val, False)
if score<baseline:
    print(f'Improved baseline to {score} from {baseline}')
    baseline = score


baseline = 26.566437213592234


train_original = pd.read_csv('/kaggle/input/bpm-prediction-challenge/Train.csv')


X_original, y_original = train_original.drop(['BeatsPerMinute'], axis=1), train_original['BeatsPerMinute']#(train_original['BeatsPerMinute']=='Extrovert').astype(int)
X_temp = pd.concat([X, X_original])
y_temp = pd.concat([y, y_original])

X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, shuffle=True, random_state = 42)
score = test_features(X_train, X_val, y_train, y_val, False)
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
categorical_columns = X.select_dtypes(['object']).columns


le = LabelEncoder()

columns_to_encode = []

# X_LE = X.copy()

for idx, column in enumerate(categorical_columns,1):

    X_LE = X.copy()
    
    print(f'Encoding {column}, iteration {idx} out of {len(categorical_columns)}')
    
    X_LE[f'{column}_le'] = le.fit_transform(X[column])
    X_train, X_val, y_train, y_val = train_test_split(X_LE, y, shuffle=True, random_state = 42)
    score = test_features(X_train, X_val, y_train, y_val, True)
    
    if score>baseline:
        print(f'Improved baseline of {score} that {baseline} before')
        columns_to_encode.append(column)


le = LabelEncoder()

label_encoded_columns = []

X_LE = X.copy()

for idx, column in enumerate(categorical_columns,1):

    # X_LE = X.copy()
    
    print(f'Encoding {column}, iteration {idx} out of {len(categorical_columns)}')

    X_LE[f'{column}_le'] = le.fit_transform(X[column])
    col = X_LE[column]
    X_LE.drop([column], axis = 1, inplace = True)
    X_train, X_val, y_train, y_val = train_test_split(X_LE, y, shuffle=True, random_state = 42)
    score = test_features(X_train, X_val, y_train, y_val, True)
    
    if score>baseline:
        print(f'Improved baseline of {score} that {baseline} before')
    else:
        X_LE.drop([f'{column}_le'], axis = 1, inplace = True)
        X_LE[column] = col


le = LabelEncoder()

label_encoded_columns = []

for idx, column in enumerate(categorical_columns,1):
    print(f'Encoding {column}, iteration {idx} out of {len(categorical_columns)}')

    X[f'{column}_le'] = le.fit_transform(X[column])
    col = X[column]
    X.drop([column], axis = 1, inplace = True)
    X_train, X_val, y_train, y_val = train_test_split(X, y, shuffle=True, random_state = 42)
    score = test_features(X_train, X_val, y_train, y_val, True)
    
    if score>baseline:
        print(f'Improved baseline of {score} that {baseline} before')
        test[f'{column}_le'] = le.transform(test[column])
        test.drop([column], axis = 1, inplace = True)
        label_encoded_columns.append(f'{column}_le')
        #baseline = score
    else:
        X.drop([f'{column}_le'], axis = 1, inplace = True)
        X[column] = col


categorical_columns = X.select_dtypes(['object']).columns
baseline = 0.9662338915740757


ohe = OneHotEncoder(sparse=False, handle_unknown='ignore')

one_hot_encoded_columns = []

X_OHE = X.copy()

#[3] will help keep the original data for the next iteration and test included data in the final iteration
# X_OHE_final = X.copy()

for idx, column in enumerate(categorical_columns,1):

    print(f'Encoding {column}, iteration {idx} out of {len(categorical_columns)}')

    ohe_array = ohe.fit_transform(X[[column]])   # double brackets -> DataFrame
    ohe_df = pd.DataFrame(ohe_array, 
                          columns=[f"{column}_{cat}" for cat in ohe.categories_[0]],
                          index=X.index)
    
    # Temporarily add encoded columns
    X_OHE = pd.concat([X_OHE, ohe_df], axis=1)

    #[1] Droping the encocded column 
    # col = X_OHE[column]
    # X_OHE.drop([column], axis = 1, inplace = True)
    
    X_train, X_val, y_train, y_val = train_test_split(X_OHE, y, shuffle=True, random_state = 42)
    score = test_features(X_train, X_val, y_train, y_val, True)
    
    if score>baseline:
        print(f'Improved baseline of {score} that {baseline} before')

        #[2] Keep baseline the same
        # baseline = score 
        
        #[3] will help keep the original data for the next iteration
        # X_OHE_final = pd.concat([X_OHE_final, ohe_df], axis=1)
        # X_OHE.drop([f"{column}_{cat}" for cat in ohe.categories_[0]], axis = 1, inplace = True)
        
    else:
        X_OHE.drop([f"{column}_{cat}" for cat in ohe.categories_[0]], axis = 1, inplace = True)

        #[1] Inserting the dropped column
        #X_OHE[column] = col

    #[3] to test if all the included columns work well together
    # if idx == len(categorical_columns):
    #     X_train, X_val, y_train, y_val = train_test_split(X_OHE_final, y, shuffle=True, random_state = 42)
    #     score = test_features(X_train, X_val, y_train, y_val, True)
    #     print(f'The Final Score is {score}')





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


X_copy = X.copy()
test_copy = test.copy()


baseline = 26.339886414336178
X=X_copy.copy()
test=test_copy.copy()


replacements = ['mean', 'median', 'mode', 'cap']
# columns = list(label_encoded_columns)+list(continuous_columns)
columns = list(continuous_columns)
# columns = ['AudioLoudness', 'AcousticQuality', 'LivePerformanceLikelihood', 'MoodScore']

# [1] To keep the train data same for the next iteration
# X_reset_col = X.copy()
# # this will help with the final test
# X_testing = X.copy()

for col in columns:

    # [1] To keep the train data same for the next iteration
    # X = X_reset_col.copy()
    
    print(f'\nFor column {col}:')
    method = ''
    if X[col].skew() > 1 or X[col].skew() < -1:
        s = X[col].dropna().astype(float)
        q1 = s.quantile(0.25)
        q3 = s.quantile(0.75)
        iqr = q3 - q1
        lower_fence = q1 - 1.5 * iqr
        upper_fence = q3 + 1.5 * iqr
        method = 'IQR'
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
        method = 'IQR'
        conditions = [
            test[col] > upper_fence,
            test[col] < lower_fence
        ]
        choices = ['over', 'under']
        outliers_test = np.select(conditions, choices, default='between')
        
    else:
        
        threshold = 3
        # df['is_outlier'] = np.abs(df['z_score']) > threshold
        method = 'Zscore'
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

    col_to_process = X[col]


    # [1] To keep the train data same for the next iteration
    # X_reset_rep = X.copy()
    
    # [2] dropping column
    # drop_col = False
    # X.drop([col], axis = 1, inplace = True)
    
    for replacement in replacements:

        # [1] To keep the train data same for the next iteration
        # X = X_reset_rep.copy()
        
        print(f'For imputing strategy {replacement}:')
        if replacement == 'mean':
            mean_val = col_to_process.mean()
            X[f'{col}_imputed_mean'] = np.where(outliers != 'between', mean_val, col_to_process)
        
        elif replacement == 'median':
            median_val = col_to_process.median()
            X[f'{col}_imputed_median'] = np.where(outliers != 'between', median_val, col_to_process)
    
        elif replacement == 'mode':
            mode_val = col_to_process.mode()[0]
            X[f'{col}_imputed_mode'] = np.where(outliers != 'between', mode_val, col_to_process)
    
        elif replacement == 'cap':
            lower_cap, upper_cap = col_to_process.quantile(0.01), col_to_process.quantile(0.99)
            X[f'{col}_imputed_cap'] = np.where(outliers == 'under', lower_cap,
                                   np.where(outliers == 'over', upper_cap, col_to_process))

        X_train, X_val, y_train, y_val = train_test_split(X, y, shuffle=True, random_state = 42)
        score = test_features(X_train, X_val, y_train, y_val, False)
        
        if score<baseline:
            print(f'Improved baseline to {score} from {baseline}')
            
            if replacement == 'mean':
                mean_val = test[col].mean()
                test[f'{col}_imputed_mean'] = np.where(outliers_test != 'between', mean_val, test[col])
                # [1] To keep the train data same for the next iteration
                # mean_val = X_testing[col].mean()
                # X_testing[f'{col}_imputed_mean'] = np.where(outliers != 'between', mean_val, X_testing[col])
                
            elif replacement == 'median':
                median_val = test[col].median()
                test[f'{col}_imputed_median'] = np.where(outliers_test != 'between', median_val, test[col])
                # [1] To keep the train data same for the next iteration
                # median_val = X_testing[col].median()
                # X_testing[f'{col}_imputed_median'] = np.where(outliers != 'between', median_val, X_testing[col])
        
            elif replacement == 'mode':
                mode_val = test[col].mode()[0]
                test[f'{col}_imputed_mode'] = np.where(outliers_test != 'between', mode_val, test[col])
                # [1] To keep the train data same for the next iteration
                # mode_val = X_testing[col].mode()[0]
                # X_testing[f'{col}_imputed_mode'] = np.where(outliers != 'between', mode_val, X_testing[col])
        
            elif replacement == 'cap':
                lower_cap, upper_cap = test[col].quantile(0.01), test[col].quantile(0.99)
                test[f'{col}_imputed_cap'] = np.where(outliers_test == 'under', lower_cap,
                                       np.where(outliers_test == 'over', upper_cap, test[col]))
                # [1] To keep the train data same for the next iteration
                # lower_cap, upper_cap = X_testing[col].quantile(0.01), X_testing[col].quantile(0.99)
                # X_testing[f'{col}_imputed_cap'] = np.where(outliers == 'under', lower_cap,
                #                        np.where(outliers == 'over', upper_cap, X_testing[col]))

            #[2] dropping col
            # drop_col = True
            
            # [2`] dropping column
            # test.drop([col], axis = 1, inplace = True)
            # X_testing.drop([col], axis = 1, inplace = True)

            # [1`] comment if uncommenting [1]
            baseline = score

            #temporary
            # break
        
        else:
            X.drop([f'{col}_imputed_{replacement}'], axis = 1, inplace = True)
    
    # [2] reinserting dropped column
    # if not drop_col:
    #     X[col] = col_to_process
    # else:
    #     test.drop([col], axis = 1, inplace = True)

# [1] To keep the train data same for the next iteration              
# X_train, X_val, y_train, y_val = train_test_split(X_testing, y, shuffle=True, random_state = 42)
# score = test_features(X_train, X_val, y_train, y_val, True)
# print(f'Final score is {score}')


X.to_csv('X.csv', index=False)
y.to_csv('y.csv', index=False)
test.to_csv('test.csv', index=False)

X = pd.read_csv('X.csv')
y = pd.read_csv('y.csv').squeeze()  # Converts back to Series if needed
test = pd.read_csv('test.csv')


continuous_columns = X.select_dtypes(['int64','float64']).columns
baseline = 26.453630100717575


X = pd.read_csv('X.csv')
y = pd.read_csv('y.csv').squeeze()  # Converts back to Series if needed
test = pd.read_csv('test.csv')

continuous_columns = ['X_LGBM', 'X_linear', 'X_XGB_CPU', 'X_Lasso', 'X_Ridge', 'X_XGB',
       'X_Elastic']

baseline = 26.182618799178716


# columns = list(label_encoded_columns)+list(continuous_columns)
columns = list(continuous_columns)
# columns = ['X_LGBM', 'X_linear', 'X_XGB_CPU', 'X_Ridge', 'X_XGB',
#        'X_Elastic']
skew_corc = ['log', 'sqrt', 'boxcox', 'yeo-johnson']

for col in columns:
    print(f'\nFor column {col}:')
    
    corc_score = {}
    if X[col].isna().sum()>0:
        continue

    X_copy = X.copy()

    # [2] to iclude multple skew removal technique
    X_copy_multi = X.copy()
    
    for corc in skew_corc:
        print(f'For correction strategy {corc}:')
        
        if (X[col] >= 0).all() and corc == 'log':
            X_copy[f'{col}_cor'] = np.log1p(X[col])
            # [2] to iclude multple skew removal technique
            X_copy_multi[f'{col}_log'] = np.log1p(X[col])
        elif (X[col] >= 0).all() and corc == 'sqrt':
            X_copy[f'{col}_cor'] = np.sqrt(X[col])
            # [2] to iclude multple skew removal technique
            X_copy_multi[f'{col}_sqrt'] = np.sqrt(X[col])
        elif (X[col] > 0).all() and corc == 'boxcox':
            X_copy[f'{col}_cor'],_ = boxcox(X[col])
            # [2] to iclude multple skew removal technique
            X_copy_multi[f'{col}_boxcox'],_ = boxcox(X[col])
        elif corc == 'yeo-johnson':
            pt = PowerTransformer(method='yeo-johnson')
            X_copy[f'{col}_cor'] = pt.fit_transform(X[[col]]).flatten()
            # [2] to iclude multple skew removal technique
            X_copy_multi[f'{col}_yeo-johnson'] = pt.fit_transform(X[[col]]).flatten()
        else:
            continue
    
        X_train, X_val, y_train, y_val = train_test_split(X_copy, y, shuffle=True, random_state = 42)
        score_corc = test_features(X_train, X_val, y_train, y_val, True)
        corc_score[corc] = score_corc

    # will change depending on the error metric
    max_value = 100
    if corc_score:
        max_value = min(corc_score.values())
        max_keys = [k for k, v in corc_score.items() if v == max_value]
        
    if max_value<baseline:

        if X[col].skew() >= 1 and 'log' in max_keys:
            X[f'{col}_cor'] = np.log1p(X[col])
            test[f'{col}_cor'] = np.log1p(test[col])
            print(f'Improved baseline to {max_value} from {baseline} using log')
        elif X[col].skew() < 1 and X[col].skew() > 0.5 and 'sqrt' in max_keys:
            X[f'{col}_cor'] = np.sqrt(X[col])
            test[f'{col}_cor'] = np.sqrt(test[col])
            print(f'Improved baseline to {max_value} from {baseline} using sqrt')
        else:
            if 'boxcox' in max_keys:
                X[f'{col}_cor'],_ = boxcox(X[col])
                test[f'{col}_cor'],_ = boxcox(test[col])
                print(f'Improved baseline to {max_value} from {baseline} using boxcox')
            elif 'yeo-johnson' in max_keys:
                pt = PowerTransformer(method='yeo-johnson')
                # combine test and train set here
                X[f'{col}_cor'] = pt.fit_transform(X[[col]]).flatten()
                test[f'{col}_cor'] = pt.fit_transform(test[[col]]).flatten()
                print(f'Improved baseline to {max_value} from {baseline} using yeo-johnson')
            elif 'log' in max_keys:
                X[f'{col}_cor'] = np.log1p(X[col])
                test[f'{col}_cor'] = np.log1p(test[col])
                print(f'Improved baseline to {max_value} from {baseline} using log')
            else:
                X[f'{col}_cor'] = np.sqrt(X[col])
                test[f'{col}_cor'] = np.sqrt(test[col])
                print(f'Improved baseline to {max_value} from {baseline} using sqrt')
        
        baseline=max_value
                


continuous_columns = X.select_dtypes(['int64','float64']).columns
baseline = 26.452648986971848


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


for feature in continuous_columns:
    print(f'\nFor column {feature}:')
    
    largest_num = X[feature].astype(str).max()
    largest_num_len = len(str(largest_num))-1
    num_digits_round = X[feature].astype(int).astype(str).apply(lambda x: len(x)).max()
    num_digits_total = X[feature].astype(str).apply(lambda x: len(x)).max()
    for i in range(1, num_digits_total):
        X[f'{feature}_digit_{i}'] = ((X[feature] * 10.0**(i-num_digits_round)) % 10).fillna(0).astype("int8")

    X_train, X_val, y_train, y_val = train_test_split(X, y, shuffle=True, random_state = 42)
    score = test_features(X_train, X_val, y_train, y_val, False)
    
    if score<baseline:
        print(f'Improved baseline from {score} to {baseline}')
        baseline = score
        
        largest_num = test[feature].astype(str).max()
        largest_num_len = len(str(largest_num))-1
        num_digits_round = test[feature].astype(int).astype(str).apply(lambda x: len(x)).max()
        num_digits_total = test[feature].astype(str).apply(lambda x: len(x)).max()
        for i in range(1, num_digits_total):
            test[f'{feature}_digit_{i}'] = ((test[feature] * 10.0**(i-num_digits_round)) % 10).fillna(0).astype("int8")
    
    else:
        X.drop([f"{feature}_digit_{i}" for i in range(1, num_digits_total)], axis = 1, inplace = True)


for column in X.select_dtypes('float64'):
    print(f'\nFor column {column}:')

    X_copy = X.copy()

    X_copy[column] = X_copy[column].astype('int64')

    X_train, X_val, y_train, y_val = train_test_split(X_copy, y, shuffle=True, random_state = 42)
    score = test_features(X_train, X_val, y_train, y_val, False)

    if score<baseline:
        print(f'Improved baseline to {score} from {baseline}')
        baseline = score
        X[column] = X[column].astype('int64')
        test[column] = test[column].astype('int64')


X.to_csv('X.csv', index=False)
y.to_csv('y.csv', index=False)
test.to_csv('test.csv', index=False)


continuous_columns


X = pd.read_csv('X.csv')
y = pd.read_csv('y.csv').squeeze()  # Converts back to Series if needed
test = pd.read_csv('test.csv')

continuous_columns = ['X_LGBM', 'X_linear', 'X_XGB_CPU', 'X_Lasso', 'X_Ridge', 'X_XGB', 'X_Elastic']
baseline = 26.18041471090284

stats = ["mean","std","count","nunique","median","min","max","skew"]


X_copy = X.copy()
test_copy = test.copy()


baseline = 26.18041471090284
X=X_copy.copy()
test=test_copy.copy()


#continuous_columns = X.select_dtypes(['int64','float64']).columns
columns = continuous_columns
# columns = ['InstrumentalScore', 'LivePerformanceLikelihood_imputed_mean', 'MoodScore_imputed_mean']
for column in tqdm(columns):
    #print(f'\nFor column {column}:')

    col_min, col_max = X[column].min(), X[column].max()

    no_of_bins = range(2, 11)

    for n in no_of_bins:
        
        if col_min == col_max:
            # Skip binning constant column
            continue
            
        #print(f'For {n} bins:')
        # Auto-generate n bins between min and max
        bins = np.linspace(col_min, col_max, n + 1)  # n+1 edges = n bins
        labels = range(n)  # 0, 1, ..., n-1

        for bin_tech in ['width', 'frequency']:
            # print(f'For binning technique equal {bin_tech}:')

            if bin_tech == 'width':
                bucket_labels = pd.cut(
                    X[column],
                    bins=bins,
                    labels=labels,
                    include_lowest=True,  # include min value
                    right=True            # right edge inclusive
                )
            else:
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
            X[f"{column}_bucket_{n}bins_{bin_tech}"] = bucket_labels.astype('float64')
            
            X_train, X_val, y_train, y_val = train_test_split(X, y, shuffle=True, random_state = 42)
            score = test_features(X_train, X_val, y_train, y_val, False)
            
            if score<baseline:
                print(f'Improved baseline to {score} from {baseline}')
                
                if bin_tech == 'width':
                    test[f"{column}_bucket_{n}bins_{bin_tech}"] = pd.cut(
                        test[column],
                        bins=bins,
                        labels=labels,
                        include_lowest=True,  # include min value
                        right=True            # right edge inclusive
                    )
                else:
                    bucket_labels, actual_bins = pd.qcut(
                        test[column],
                        q=n,
                        retbins=True,
                        duplicates='drop'
                    )
                        
                    actual_num_bins = len(actual_bins) - 1
                    actual_labels = list(range(actual_num_bins))
                    
                    # Re-bin with appropriate labels
                    test[f"{column}_bucket_{n}bins_{bin_tech}"] = pd.qcut(
                        test[column],
                        q=n,
                        labels=actual_labels,
                        duplicates='drop'
                    )
                
                baseline = score
            else:
                X.drop([f"{column}_bucket_{n}bins_{bin_tech}"], axis = 1, inplace = True)
            
            # For bin counts assignment
            # bucket_counts = bucket_labels.value_counts().to_dict()
            # X[f"{column}_bucket_{n}bins_count"] = bucket_labels.map(bucket_counts)
                
            # X_train, X_val, y_train, y_val = train_test_split(X, y, shuffle=True, random_state = 42)
            # score = test_features(X_train, X_val, y_train, y_val, True)
            
            # if score<baseline:
            #     print(f'Improved baseline to {score} from {baseline}')
            #     bucket_labels = pd.cut(
            #     test[column],
            #     bins=bins,
            #     labels=labels,
            #     include_lowest=True,  # include min value
            #     right=True            # right edge inclusive
            #     )
            #     bucket_counts = bucket_labels.value_counts().to_dict()
            #     test[f"{column}_bucket_{n}bins_count"] = bucket_labels.map(bucket_counts)
            #     baseline = score
                
            # else:
            #     X.drop([f"{column}_bucket_{n}bins_count"], axis = 1, inplace = True)


X.to_csv('X.csv', index=False)
y.to_csv('y.csv', index=False)
test.to_csv('test.csv', index=False)


X = pd.read_csv('X.csv')
y = pd.read_csv('y.csv').squeeze()  # Converts back to Series if needed
test = pd.read_csv('test.csv')

continuous_columns = ['X_LGBM', 'X_linear', 'X_XGB_CPU', 'X_Lasso', 'X_Ridge', 'X_XGB', 'X_Elastic']
baseline = 26.175942747176382

stats = ["mean","std","count","nunique","median","min","max","skew"]
stats = ["mean"]


# continuous_columns = X.select_dtypes(['int64','float64']).columns
columns = continuous_columns
# columns = ['InstrumentalScore', 'LivePerformanceLikelihood_imputed_mean', 'MoodScore_imputed_mean']
for column in tqdm(columns):
    # print(f'\nFor column {column}:')

    col_min, col_max = X[column].min(), X[column].max()

    no_of_bins = range(2, 11)

    for n in tqdm(no_of_bins):

        if col_min == col_max:
            # Skip binning constant column
            continue
            
        # print(f'For {n} bins:')
        # Auto-generate n bins between min and max
        bins = np.linspace(col_min, col_max, n + 1)  # n+1 edges = n bins
        labels = range(n)  # 0, 1, ..., n-1

        for bin_tech in ['width', 'frequency']:
            # print(f'For binning technique equal {bin_tech}:')
        
            for stat in stats:
                # print(f'For stat {stat}:')

                if bin_tech == 'width':
                    bucket_labels = pd.cut(
                        X[column],
                        bins=bins,
                        labels=labels,
                        include_lowest=True,  # include min value
                        right=True            # right edge inclusive
                    )
                else:
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
                
                X[f'{column}_bucket'] = bucket_labels
                mean_encoded = X.groupby(f'{column}_bucket')[column].agg(stat)
                X[f'{column}_bucket_{n}_{bin_tech}_{stat}'] = X[f'{column}_bucket'].map(mean_encoded).astype('float64')
                X.drop([f'{column}_bucket'], axis = 1, inplace = True)
                
                X = X.fillna(0)
                X_train, X_val, y_train, y_val = train_test_split(X, y, shuffle=True, random_state = 42)
                score = test_features(X_train, X_val, y_train, y_val, False)
        
                if score<baseline:
                
                    print(f'Improved baseline to {score} from {baseline}')
                    baseline = score
                
                    X_total_con_cat = pd.concat([X[[column]],test[[column]]])

                    if bin_tech == 'width':
                        bucket_labels = pd.cut(
                            X_total_con_cat[column],
                            bins=bins,
                            labels=labels,
                            include_lowest=True,  # include min value
                            right=True            # right edge inclusive
                        )
                        bucket_labels_test = pd.cut(
                            test[column],
                            bins=bins,
                            labels=labels,
                            include_lowest=True,  # include min value
                            right=True            # right edge inclusive
                        )
                        
                    else:
                        bucket_labels, actual_bins = pd.qcut(
                            X_total_con_cat[column],
                            q=n,
                            retbins=True,
                            duplicates='drop'
                        )
                        actual_num_bins = len(actual_bins) - 1
                        actual_labels = list(range(actual_num_bins))
                        bucket_labels = pd.qcut(
                            X_total_con_cat[column],
                            q=n,
                            labels=actual_labels,
                            duplicates='drop'
                        )
                        
                        bucket_labels_test, actual_bins = pd.qcut(
                            test[column],
                            q=n,
                            retbins=True,
                            duplicates='drop'
                        )
                        actual_num_bins = len(actual_bins) - 1
                        actual_labels = list(range(actual_num_bins))
                        bucket_labels_test = pd.qcut(
                            test[column],
                            q=n,
                            labels=actual_labels,
                            duplicates='drop'
                        )
                        
                    X_total_con_cat[f'{column}_bucket'] = bucket_labels
                    test[f'{column}_bucket'] = bucket_labels_test
                    mean_encoded = X_total_con_cat.groupby(f'{column}_bucket')[column].agg(stat)
                    test[f'{column}_bucket_{n}_{bin_tech}_{stat}'] = test[f'{column}_bucket'].map(mean_encoded)
                    test.drop([f'{column}_bucket'], axis = 1, inplace = True)
                
                else:
                    X.drop([f'{column}_bucket_{n}_{bin_tech}_{stat}'], axis = 1, inplace = True)


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



for idx, countinuous_column in enumerate(continuous_columns):
    print(f'countinuous_column Iteration {idx+1} out of {len(continuous_columns)}')
    
    for idx1, stat in enumerate(stats):
        print(f'stats Iteration {idx1+1} out of {len(stats)}')
        
        for index,categorical_column in enumerate(categorical_columns):
            print(f'Iteration {index+1} out of {len(categorical_columns)}')
            
            mean_encoded = X.groupby(categorical_column)[countinuous_column].agg(stat)
            X[f'{categorical_column}_{countinuous_column}_{stat}'] = X[categorical_column].map(mean_encoded)

            X_train, X_val, y_train, y_val = train_test_split(X, y, shuffle=True, random_state = 42)
            score = test_features(X_train, X_val, y_train, y_val, True)
            if score>baseline:
                print(f'Improved baseline of {score} that {baseline} before')
                baseline=score

                # if baseline is imporved we will use the combiantion of train and test to find the aggregate and then assign to test
                X_total_con_cat = pd.concat([X[[countinuous_column,categorical_column]],test[[countinuous_column,categorical_column]]])
                mean_encoded = X_total_con_cat.groupby(categorical_column)[countinuous_column].agg(stat)
                test[f'{categorical_column}_{countinuous_column}_{stat}'] = test[categorical_column].map(mean_encoded)
                
            else:
                X.drop([f'{categorical_column}_{countinuous_column}_{stat}'], axis=1, inplace=True)


X = pd.read_csv('X.csv')
y = pd.read_csv('y.csv').squeeze()  # Converts back to Series if needed
test = pd.read_csv('test.csv')

stats = ["mean","std","count","nunique","median","min","max","skew"]
stats = ["mean"]

continuous_columns = ['X_LGBM', 'X_linear', 'X_XGB_CPU', 'X_Lasso', 'X_Ridge', 'X_XGB', 'X_Elastic']
baseline = 26.17313523600968


#continuous_columns = X.select_dtypes(['int64','float64']).columns

for idx, countinuous_column in tqdm(enumerate(continuous_columns), total=len(continuous_columns)):
    #print(f'1. countinuous_column Iteration {idx+1} out of {len(continuous_columns)}\n\n')
    
    for idx1, stat in tqdm(enumerate(stats), total=len(stats)):
        #print(f'2. stats Iteration {idx1+1} out of {len(stats)}\n')
        
        for index,countinuous_column_sec in tqdm(enumerate(continuous_columns), total=len(continuous_columns)):
            # helps avoiding same col name confict
            if(idx == index):
                continue
            #print(f'3. Inner Iteration {index+1} out of {len(continuous_columns)}')

            col_min, col_max = X[countinuous_column_sec].min(), X[countinuous_column_sec].max()

            no_of_bins = range(2, 6)

            for n in no_of_bins:
                # print(f'4. For {n} bins:')
                
                if col_min == col_max:
                    # Skip binning constant column
                    continue
                    
                # Auto-generate n bins between min and max
                bins = np.linspace(col_min, col_max, n + 1)  # n+1 edges = n bins
                labels = range(n)  # 0, 1, ..., n-1

                for bin_tech in ['width', 'frequency']:
                    # print(f'5. For binning technique equal {bin_tech}:')
                    
                    if bin_tech == 'width':
                        bucket_labels = pd.cut(
                            X[countinuous_column_sec],
                            bins=bins,
                            labels=labels,
                            include_lowest=True,  # include min value
                            right=True            # right edge inclusive
                        )
                    else:
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
                    
                    X[f'{countinuous_column_sec}_bucket'] = bucket_labels.astype('float64')
                    mean_encoded = X.groupby(f'{countinuous_column_sec}_bucket')[countinuous_column].agg(stat)
                    X[f'{countinuous_column_sec}_{countinuous_column}_{stat}_{n}_{bin_tech}'] = X[f'{countinuous_column_sec}_bucket'].map(mean_encoded)
                    X.drop([f'{countinuous_column_sec}_bucket'], axis = 1, inplace = True)
                    
                    X_train, X_val, y_train, y_val = train_test_split(X, y, shuffle=True, random_state = 42)
                    score = test_features(X_train, X_val, y_train, y_val, False)
                    
                    if score<baseline:
                        print(f'Improved baseline to {score} from {baseline}')
                        baseline=score
                    
                        # if baseline is imporved we will use the combiantion of train and test to find the aggregate and then assign to test
                        X_total_con_cat = pd.concat([X[[countinuous_column,countinuous_column_sec]],test[[countinuous_column,countinuous_column_sec]]])
                        
                        if bin_tech == 'width':
                            bucket_labels = pd.cut(
                                X_total_con_cat[countinuous_column_sec],
                                bins=bins,
                                labels=labels,
                                include_lowest=True,  # include min value
                                right=True            # right edge inclusive
                            )
                            #test
                            bucket_labels_test = pd.cut(
                                test[countinuous_column_sec],
                                bins=bins,
                                labels=labels,
                                include_lowest=True,  # include min value
                                right=True            # right edge inclusive
                            )
                        else:
                            bucket_labels, actual_bins = pd.qcut(
                                X_total_con_cat[countinuous_column_sec],
                                q=n,
                                retbins=True,
                                duplicates='drop'
                            )
                                
                            actual_num_bins = len(actual_bins) - 1
                            actual_labels = list(range(actual_num_bins))
                            
                            # Re-bin with appropriate labels
                            bucket_labels = pd.qcut(
                                X_total_con_cat[countinuous_column_sec],
                                q=n,
                                labels=actual_labels,
                                duplicates='drop'
                            )
                            #test
                            bucket_labels_test, actual_bins = pd.qcut(
                                test[countinuous_column_sec],
                                q=n,
                                retbins=True,
                                duplicates='drop'
                            )
                                
                            actual_num_bins = len(actual_bins) - 1
                            actual_labels = list(range(actual_num_bins))
                            
                            # Re-bin with appropriate labels
                            bucket_labels_test = pd.qcut(
                                test[countinuous_column_sec],
                                q=n,
                                labels=actual_labels,
                                duplicates='drop'
                            )
                        
                        X_total_con_cat[f'{countinuous_column_sec}_bucket'] = bucket_labels.astype('float64')
                        test[f'{countinuous_column_sec}_bucket'] = bucket_labels_test.astype('float64')
                        mean_encoded = X_total_con_cat.groupby(f'{countinuous_column_sec}_bucket')[countinuous_column].agg(stat)
                        test[f'{countinuous_column_sec}_{countinuous_column}_{stat}_{n}_{bin_tech}'] = test[f'{countinuous_column_sec}_bucket'].map(mean_encoded)
                        test.drop([f'{countinuous_column_sec}_bucket'], axis = 1, inplace = True)
                    
                    else:
                        X.drop([f'{countinuous_column_sec}_{countinuous_column}_{stat}_{n}_{bin_tech}'], axis=1, inplace=True)


X.to_csv('X.csv', index=False)
y.to_csv('y.csv', index=False)
test.to_csv('test.csv', index=False)


X = pd.read_csv('X.csv')
y = pd.read_csv('y.csv').squeeze()  # Converts back to Series if needed
test = pd.read_csv('test.csv')

#  stats = ["mean","std","count","nunique","median","min","max","skew"]
stats = ["mean"]
continuous_columns = ['X_LGBM', 'X_linear', 'X_XGB_CPU', 'X_Lasso', 'X_Ridge', 'X_XGB', 'X_Elastic']
baseline = 26.172987710229663


#columns = X.select_dtypes(['int64','float64']).columns
columns = continuous_columns
X_train, X_val, y_train, y_val = train_test_split(X, y, shuffle=True, random_state = 42)

for stat in tqdm(stats):
    # print(f'For {stat}\n\n')
    
    for index,column in tqdm(enumerate(columns), total = len(columns)):
        # print(f'Iteration {index+1} out of {len(columns)}\n')
        
        col_min, col_max = X[column].min(), X[column].max()

        no_of_bins = range(2, 11)
        
        for n in no_of_bins:
            # print(f'4. For {n} bins:')

            if col_min == col_max:
            # Skip binning constant column
                continue
            
            # Auto-generate n bins between min and max
            bins = np.linspace(col_min, col_max, n + 1)  # n+1 edges = n bins
            labels = range(n)  # 0, 1, ..., n-1
            
            for bin_tech in ['width', 'frequency']:
                # print(f'5. For binning technique equal {bin_tech}:')
                
                if bin_tech == 'width':
                    bucket_labels = pd.cut(
                        X_train[column],
                        bins=bins,
                        labels=labels,
                        include_lowest=True,  # include min value
                        right=True            # right edge inclusive
                    )
                    
                    #val
                    bucket_labels_val = pd.cut(
                        X_val[column],
                        bins=bins,
                        labels=labels,
                        include_lowest=True,  # include min value
                        right=True            # right edge inclusive
                    )
                else:
                    bucket_labels, actual_bins = pd.qcut(
                        X_train[column],
                        q=n,
                        retbins=True,
                        duplicates='drop'
                    )
                            
                    actual_num_bins = len(actual_bins) - 1
                    actual_labels = list(range(actual_num_bins))
                    
                    # Re-bin with appropriate labels
                    bucket_labels = pd.qcut(
                        X_train[column],
                        q=n,
                        labels=actual_labels,
                        duplicates='drop'
                    )
                    
                    #val
                    bucket_labels_val, actual_bins = pd.qcut(
                        X_val[column],
                        q=n,
                        retbins=True,
                        duplicates='drop'
                    )
                            
                    actual_num_bins = len(actual_bins) - 1
                    actual_labels = list(range(actual_num_bins))
                    
                    # Re-bin with appropriate labels
                    bucket_labels_val = pd.qcut(
                        X_val[column],
                        q=n,
                        labels=actual_labels,
                        duplicates='drop'
                    )
                
                X_train[f'{column}_bucket'] = bucket_labels.astype('float64')
                X_val[f'{column}_bucket'] = bucket_labels_val.astype('float64')        
                
                X_train['BeatsPerMinute'] = y_train
               
                mean_encoded = X_train.groupby(f'{column}_bucket')['BeatsPerMinute'].agg(stat)
                X_train[f'{column}_encoded_BeatsPerMinute_{stat}_{n}_{bin_tech}'] = X_train[f'{column}_bucket'].map(mean_encoded)
                X_val[f'{column}_encoded_BeatsPerMinute_{stat}_{n}_{bin_tech}'] = X_val[f'{column}_bucket'].map(mean_encoded)
                
                X_train.drop(['BeatsPerMinute'],axis=1,inplace=True)
                
                X_train.drop([f'{column}_bucket'], axis = 1, inplace = True)
                X_val.drop([f'{column}_bucket'], axis = 1, inplace = True)
        
                score = test_features(X_train, X_val, y_train, y_val, True)
                if score<baseline:
                    print(f'Improved baseline to {score} from {baseline}')
                    baseline=score



                    if bin_tech == 'width':
                        bucket_labels = pd.cut(
                            X[column],
                            bins=bins,
                            labels=labels,
                            include_lowest=True,  # include min value
                            right=True            # right edge inclusive
                        )
                        #test
                        bucket_labels_test = pd.cut(
                            test[column],
                            bins=bins,
                            labels=labels,
                            include_lowest=True,  # include min value
                            right=True            # right edge inclusive
                        )
                    else:
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
                        #test
                        bucket_labels_test, actual_bins = pd.qcut(
                            test[column],
                            q=n,
                            retbins=True,
                            duplicates='drop'
                        )
                                
                        actual_num_bins = len(actual_bins) - 1
                        actual_labels = list(range(actual_num_bins))
                        
                        # Re-bin with appropriate labels
                        bucket_labels_test = pd.qcut(
                            test[column],
                            q=n,
                            labels=actual_labels,
                            duplicates='drop'
                        )
                    
                    X[f'{column}_bucket'] = bucket_labels.astype('float64')
                    test[f'{column}_bucket'] = bucket_labels_test.astype('float64')
                    
                    X['BeatsPerMinute'] = y
                    
                    
                    mean_encoded = X.groupby(f'{column}_bucket')['BeatsPerMinute'].agg(stat)
                    
                    X[f'{column}_encoded_BeatsPerMinute_{stat}_{n}_{bin_tech}'] = X[f'{column}_bucket'].map(mean_encoded)
                    test[f'{column}_encoded_BeatsPerMinute_{stat}_{n}_{bin_tech}'] = test[f'{column}_bucket'].map(mean_encoded)
                    
                    X.drop(['BeatsPerMinute'],axis=1,inplace=True)
                    
                    X.drop([f'{column}_bucket'], axis = 1, inplace = True)
                    test.drop([f'{column}_bucket'], axis = 1, inplace = True)

                else:
                    X_train.drop([f'{column}_encoded_BeatsPerMinute_{stat}_{n}_{bin_tech}'], axis=1, inplace=True)
                    X_val.drop([f'{column}_encoded_BeatsPerMinute_{stat}_{n}_{bin_tech}'], axis=1, inplace=True)


X.to_csv('X.csv', index=False)
y.to_csv('y.csv', index=False)
test.to_csv('test.csv', index=False)


X.columns


baseline


columns = list(categorical_columns)#+list(continuous_columns)
X_train, X_val, y_train, y_val = train_test_split(X, y, shuffle=True, random_state = 42)

for stat in stats:
    print(f'For {stat}')
    
    for index,column in enumerate(columns):
        print(f'Iteration {index+1} out of {len(columns)}')
        
        X_train['BeatsPerMinute'] = y_train
        mean_encoded = X_train.groupby(column)['BeatsPerMinute'].agg(stat)
        X_train.drop(['BeatsPerMinute'],axis=1,inplace=True)
        
        X_train[f'{column}_encoded_Personality_{stat}'] = X_train[column].map(mean_encoded)
        X_val[f'{column}_encoded_Personality_{stat}'] = X_val[column].map(mean_encoded)
        
        score = test_features(X_train, X_val, y_train, y_val, True)
        if score<baseline:
            print(f'Improved baseline of {score} that {baseline} before')
            baseline=score

            X['BeatsPerMinute'] = y
            mean_encoded = X.groupby(column)['BeatsPerMinute'].agg(stat)
            X.drop(['BeatsPerMinute'],axis=1,inplace=True)

            X[f'{column}_encoded_Personality_{stat}'] = X[column].map(mean_encoded)
            test[f'{column}_encoded_Personality_{stat}'] = test[column].map(mean_encoded)
        else:
            X_train.drop([f'{column}_encoded_Personality_{stat}'], axis=1, inplace=True)
            X_val.drop([f'{column}_encoded_Personality_{stat}'], axis=1, inplace=True)


X = pd.read_csv('X.csv')
y = pd.read_csv('y.csv').squeeze()  # Converts back to Series if needed
test = pd.read_csv('test.csv')
continuous_columns = ['X_LGBM', 'X_linear', 'X_XGB_CPU', 'X_Lasso', 'X_Ridge', 'X_XGB', 'X_Elastic']
baseline = 26.15822235706407


# continuous_columns = X.select_dtypes(['int64','float64']).columns
#columns = list(label_encoded_columns)+list(continuous_columns)
columns = list(continuous_columns)
X_copy = X.copy()

degree = 3#len(columns)
    
# Iterate over degrees 1 to `degree`
# for d in range(2, degree + 1):
for d in range(degree, degree + 1):
    # Get all combinations with replacement of columns
    n = len(columns)
    total_combinations = comb(n + d -1, d)  # combinations_with_replacement count
    for cols in tqdm(combinations_with_replacement(columns, d), total=total_combinations, desc=f'Degree {d}'):
        col_name = '*'.join(cols)
        X_copy[col_name] = X_copy[cols[0]]
        
        for idx in range(1,len(cols)):
            X_copy[col_name] = X_copy[col_name] * X_copy[cols[idx]]

        X_train, X_val, y_train, y_val = train_test_split(X_copy, y, shuffle=True, random_state = 42)
        score = test_features(X_train, X_val, y_train, y_val, True)
        
        if score<baseline:
            print(f'Improved baseline of {score} that {baseline} before')
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

baseline = 26.151297340363598


columns = X.columns
for col in tqdm(columns):
    
    X_copy = X.copy()
    X_copy.drop([col], axis=1, inplace = True)

    X_train, X_val, y_train, y_val = train_test_split(X_copy, y, shuffle=True, random_state = 42)
    score = test_features(X_train, X_val, y_train, y_val, True)
        
    if score<baseline:
        print(f'Improved baseline to {score} from {baseline}')
        baseline=score
        X.drop([col], axis=1, inplace = True)
        test.drop([col], axis=1, inplace = True)


columns = X.columns

for column in tqdm(columns):

    for scale in ['standard', 'minmax']:

        if scale == 'standard':
            scalar = StandardScaler()
        else:
            scalar = MinMaxScaler()

        col_original = X[column].copy()
        X[column] = scalar.fit_transform(X[[column]]).astype('float32')

        X_train, X_val, y_train, y_val = train_test_split(X, y, shuffle=True, random_state = 42)
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


sample = pd.read_csv('/kaggle/input/playground-series-s5e9/sample_submission.csv')
sample.columns


def rmse(y_true,y_pred):
    return np.sqrt(mean_squared_error(y_true,y_pred))

def param_finetuning(X, y, test, bestScore):
    xgb_params = {
        'n_estimators': 2542,
        'max_depth': 4,
        'learning_rate': 0.01679322,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'tree_method': 'gpu_hist',
        'n_jobs': -1,
    }

    X = X.copy()
    test = test.copy()

    for col in X.select_dtypes(include='object').columns:
        X[col] = X[col].astype('category')
        test[col] = test[col].astype('category')

    sample = pd.read_csv('/kaggle/input/playground-series-s5e9/sample_submission.csv')

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

    model = cuLasso(alpha = 0.352447303762)
    model.fit(X_train, y_train)

    # model = lgb.LGBMRegressor(
    #     objective="regression",
    #     boosting_type="gbdt",
    #     n_estimators=500,
    #     num_leaves=31,
    #     learning_rate=0.05,
    #     random_state=42,
    #     device="gpu",
    #     deterministic=True,
    #     verbose = -1,
    # )
    # model.fit(
    #     X_train, y_train,
    #     eval_set=[(X_val, y_val)],
    #     eval_metric="rmse",
    #     callbacks=[log_evaluation(-1)]
    # )

    # mlp = MLPRegressor(
    #     hidden_layer_sizes=(100, 50),   # 2 hidden layers: 100 and 50 neurons
    #     activation="relu",              # activation function
    #     solver="adam",                  # optimizer
    #     max_iter=300,
    #     learning_rate_init=0.001,
    #     random_state=42
    # )
    # mlp.fit(X_train, y_train)

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
        
            sample['BeatsPerMinute'] = y_test
            bestScore = score
    return score, sample


bestScore = 26.395029294590337
bestScore,sample = param_finetuning(X,y,test,bestScore)
#sample.to_csv('/kaggle/working/submission.csv',index=False)


bestScore


def rmse(y_true,y_pred):
    return np.sqrt(mean_squared_error(y_true,y_pred))
    
def model_training(X, y, test, n_splits, bestScore):
    xgb_params = {
        'n_estimators': 2542,
        'max_depth': 4,
        'learning_rate': 0.01679322,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'tree_method': 'gpu_hist',
        'n_jobs': -1,
    }

    X = X.copy()
    test = test.copy()

    for col in X.select_dtypes(include='object').columns:
        X[col] = X[col].astype('category')
        test[col] = test[col].astype('category')

    sample = pd.read_csv('/kaggle/input/playground-series-s5e9/sample_submission.csv')

    kfolds = KFold(n_splits=n_splits,shuffle=True)

    # Out-of-fold preds for CV evaluation
    oof_preds = np.zeros(len(X))
    test_preds = np.zeros((len(test), n_splits))  # one column per fold
    X_preds = np.zeros((len(X), n_splits))
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

        # model = cuElasticNet(alpha = 0.3050, l1_ratio = 0.4990)
        # model.fit(X_train, y_train)

        # model = cuElasticNet(alpha = 0.2550)
        # model.fit(X_train, y_train)

        model = cuLasso(alpha = 0.3524473037625)
        model.fit(X_train, y_train)

        # model = lgb.LGBMRegressor(
        #     objective="regression",
        #     boosting_type="gbdt",
        #     num_leaves=31,
        #     learning_rate=0.05,
        #     n_estimators=500,
        #     random_state=42,
        #     device="gpu",
        #     verbose = -1
        # )
        # model.fit(
        #     X_train, y_train,
        #     eval_set=[(X_val, y_val)],
        #     eval_metric="rmse",
        #     callbacks=[early_stopping(50), log_evaluation(-1)]
        # )

        # mlp = MLPRegressor(
        #     hidden_layer_sizes=(100, 50),   # 2 hidden layers: 100 and 50 neurons
        #     activation="relu",              # activation function
        #     solver="adam",                  # optimizer
        #     max_iter=300,
        #     learning_rate_init=0.001,
        #     random_state=42
        # )
        # mlp.fit(X_train, y_train)

        # will change depending on the score metric

        # accuracy
        # y_pred = model.predict(X_val)
        # oof_preds[val_idx] = y_pred
        # score = accuracy_score(y_val,y_pred)
        # scores.append(score)
        # test_preds[:, fold] = model.predict(test)
        # X_preds[:, fold] = model.predict(X)

        # rmse
        y_pred = model.predict(X_val)
        oof_preds[val_idx] = y_pred
        score = rmse(y_val,y_pred)
        scores.append(score)
        test_preds[:, fold] = model.predict(test)
        X_preds[:, fold] = model.predict(X)
    
        # roc-auc
        # y_pred_proba = model.predict_proba(X_val)[:, 1]
        # oof_preds[val_idx] = y_pred_proba
        # score = roc_auc_score(y_val, y_pred_proba)
        # scores.append(score)
        # test_preds[:, fold] = model.predict_proba(test)[:, 1]
        # X_preds[:, fold] = model.predict_proba(X)[:, 1]

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
        X_Ridge = X_preds.mean(axis=1)
        
        sample['BeatsPerMinute'] = y_test
        bestScore = score

        sample.to_csv(f'/kaggle/working/test_{Model}.csv',index=False)
        pd.DataFrame(X_Ridge).to_csv(f'/kaggle/working/X_{Model}.csv', index=False)
            
    print(f"Best Accuracy: {bestScore}")
    return


bestScore = 26.404782018592634
model_training(X,y,test,10,bestScore)


sample


sample['Personality'] = sample['Personality'].map({1: 'Extrovert', 0: 'Introvert'})
sample.to_csv('/kaggle/working/submission.csv',index=False)


sample


X_Ridge = pd.read_csv(f'X_{Model}.csv')
test_Ridge = pd.read_csv(f'test_{Model}.csv')


X_Ridge.isna().sum()


test_Ridge.isna().sum()


test_Ridge.fillna(test_Ridge.mean(), inplace=True)
test_Ridge.to_csv(f'/kaggle/working/test_{Model}.csv',index=False)


test_Ridge




