import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from xgboost import XGBRegressor 

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder

import random

SEED = 42

np.random.seed(SEED)
random.seed(SEED)


train = pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv')


train.shape


test.shape


train.head()


train.columns


train.info()


train.isna().sum()


test.isna().sum()


train['id'].duplicated().sum()


train.drop(['id','Listening_Time_minutes'],axis=1).duplicated().sum()


train['Episode_Title'].duplicated().sum()


train['Episode_Title'].nunique()


train.describe()


test.describe()


train.select_dtypes(['float64']).corr()


float_features = train.select_dtypes(include='float64').columns

# Set up the figure size and layout
plt.figure(figsize=(15, len(float_features) * 4))

for i, feature in enumerate(float_features, 1):
    plt.subplot(len(float_features), 1, i)
    sns.boxplot(x=train[feature], color='skyblue')
    plt.title(f'Boxplot of {feature}')
    plt.xlabel('')

plt.tight_layout()
plt.show()


train[train['Episode_Length_minutes']>150]


float_features = test.select_dtypes(include='float64').columns

# Set up the figure size and layout
plt.figure(figsize=(15, len(float_features) * 4))

for i, feature in enumerate(float_features, 1):
    plt.subplot(len(float_features), 1, i)
    sns.boxplot(x=test[feature], color='skyblue')
    plt.title(f'Boxplot of {feature}')
    plt.xlabel('')

plt.tight_layout()
plt.show()


def rmse(y_true,y_pred):
    return np.sqrt(mean_squared_error(y_true,y_pred))


def test_feature(X,y, n_splits):
    xgb_params={
        'n_esitmator':560,
        'max_depth':14,
        'learning_rate':0.04222,
        'subsample':0.8,
        'colsample_bytree':0.8,
        'random_state':42,
        'tree_method':'gpu_hist',
        'njobs':-1,
    }
    
    for col in X.select_dtypes(include='object').columns:
        X[col] = X[col].astype('category')
        
    kfolds = KFold(n_splits=n_splits,shuffle=True, random_state=42)
    scores = []
    for fold, (train_idx, val_idx) in enumerate(kfolds.split(X,y)):
        print(f'Fold {fold+1} of {n_splits}')
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        model = XGBRegressor(**xgb_params,enable_categorical=True)
        model.fit(X_train, y_train, eval_set=[(X_val,y_val)], verbose=100, early_stopping_rounds=100)
        y_pred = model.predict(X_val)
        scores.append(rmse(y_val,y_pred))
    print(scores, np.mean(scores))
    return np.mean(scores)


def test_feature_efficient(X,y,enable_categorical):
    X_copy=X.copy()
    for col in X_copy.select_dtypes(include='object').columns:
        X_copy[col] = X_copy[col].astype('category')
    X_train, X_val, y_train, y_val = train_test_split(X_copy,y, shuffle=True, random_state = 42)    
    model = XGBRegressor(random_state=42,enable_categorical=enable_categorical,tree_method='gpu_hist')
    model.fit(X_train, y_train, eval_set=[(X_val,y_val)], verbose=0)
    y_pred = model.predict(X_val)
    print(rmse(y_val,y_pred))
    return rmse(y_val,y_pred)


def test_feature_efficient_val(X_train, X_val, y_train, y_val,enable_categorical):
    X_train_copy=X_train.copy()
    X_val_copy=X_val.copy()
    for col in X_train_copy.select_dtypes(include='object').columns:
        X_train_copy[col] = X_train_copy[col].astype('category')
        X_val_copy[col] = X_val_copy[col].astype('category')
    model = XGBRegressor(random_state=42,enable_categorical=enable_categorical,tree_method='gpu_hist')
    model.fit(X_train_copy, y_train, eval_set=[(X_val_copy,y_val)], verbose=0)
    y_pred = model.predict(X_val_copy)
    print(rmse(y_val,y_pred))
    return rmse(y_val,y_pred)


X, y = train.drop(['Listening_Time_minutes'], axis=1), train['Listening_Time_minutes']


baseline = test_feature_efficient(X, y,True)
columns_for_training = X.columns
columns_for_training


score = test_feature_efficient(X.drop(['id'],axis = 1),y,True)
if score<=baseline:
    print(f'Improved baseline of {score} than {baseline} before')
    baseline=score
    X.drop(['id'],axis = 1, inplace=True)
    test.drop(['id'],axis = 1, inplace=True)
    columns_for_training = X.columns
columns_for_training


extra_train = pd.read_csv('/kaggle/input/podcast-listening-time-prediction-dataset/podcast_dataset.csv')


extra_train.isna().sum()


extra_train = extra_train.dropna(subset=['Listening_Time_minutes'])


X_train, X_val, y_train, y_val = train_test_split(X, y, shuffle=True, random_state = 42)    
X_train = pd.concat([X_train,extra_train.iloc[:,:-1]],axis=0)
y_train = pd.concat([y_train, extra_train.iloc[:,-1:]],axis=0)


score = test_feature_efficient_val(X_train, X_val, y_train, y_val, True) 
if score<baseline:
    print(f'Improved baseline of {score} that {baseline} before')
    X = pd.concat([X,extra_train.iloc[:,:-1]],axis=0)
    y = pd.concat([y,extra_train.iloc[:,-1:]],axis=0)


score = test_feature_efficient_val(extra_train.iloc[:,:-1], X_val, extra_train.iloc[:,-1:], y_val, True)
if score<baseline:
    print(f'Improved baseline of {score} that {baseline} before')


cols = ['Episode_Length_minutes','Number_of_Ads']
multiplier=1.5
X_train, X_val, y_train, y_val = train_test_split(X,y, shuffle=True, random_state = 42)    
for col in cols:
    Q1 = X_train[col].quantile(0.25)
    Q3 = X_train[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - multiplier * IQR
    upper = Q3 + multiplier * IQR
    X_train[col] = X_train[col].clip(lower, upper)
    X_val[col] = X_val[col].clip(lower, upper)
    print(col,lower, upper)
    score = test_feature_efficient_val(X_train, X_val, y_train, y_val, True)
    if score<=baseline:
        print(f'Improved baseline of {score} that {baseline} before')
        baseline=score
        X[col] = X[col].clip(lower, upper)
        test[col] = test[col].clip(lower, upper)


for column in ['Episode_Length_minutes','Guest_Popularity_percentage']:
    score = test_feature_efficient(X.drop([column],axis = 1),y,True)
    if score<=baseline:
        print(f'Improved baseline of {score} than {baseline} before')


for column in ['Episode_Length_minutes', 'Guest_Popularity_percentage']:
    X_temp = X.copy()
    X_temp[column] = X_temp[column].fillna(0)
    score = test_feature_efficient(X_temp, y, True)
    if score <= baseline:
        print(f'Improved baseline of {score} by filling {column} with 0 (baseline was {baseline})')


for column in ['Episode_Length_minutes', 'Guest_Popularity_percentage']:
    X_temp = X.copy()
    X_temp[column] = X_temp[column].fillna('median')
    score = test_feature_efficient(X_temp, y, True)
    if score <= baseline:
        print(f'Improved baseline of {score} by filling {column} with 0 (baseline was {baseline})')


for column in ['Episode_Length_minutes', 'Guest_Popularity_percentage']:
    X_temp = X.copy()
    X_temp[column] = X_temp[column].fillna('mean')
    score = test_feature_efficient(X_temp, y, True)
    if score <= baseline:
        print(f'Improved baseline of {score} by filling {column} with 0 (baseline was {baseline})')


for column in ['Episode_Length_minutes', 'Guest_Popularity_percentage']:
    X_temp = X.copy()
    X_temp[column] = X_temp.groupby('Podcast_Name')[column].transform(lambda x:x.fillna(x.mean()))
    score = test_feature_efficient(X_temp, y, True)
    if score <= baseline:
        print(f'Improved baseline of {score} by filling {column} with 0 (baseline was {baseline})')


for column in ['Episode_Length_minutes', 'Guest_Popularity_percentage']:
    X_temp = X.copy()
    X_temp[column] = X_temp.groupby('Podcast_Name')[column].transform(lambda x:x.fillna(x.median()))
    score = test_feature_efficient(X_temp, y, True)
    if score <= baseline:
        print(f'Improved baseline of {score} by filling {column} with 0 (baseline was {baseline})')


X_temp = X.copy()
X_temp['Episode_Number'] = X['Episode_Title'].str.extract(r'(\d+)').astype(np.float64)
score = test_feature_efficient(X_temp,y,True)
if score<baseline:
    print(f'Improved baseline of {score} than {baseline} before')
    baseline=score
    X['Episode_Number']= X_temp['Episode_Number']
    columns_for_training = X.columns
columns_for_training


for feature in X.select_dtypes(['float64']).drop(['Episode_Length_minutes'],axis=1).columns:
    print(f'For Feature {feature}')
    X[f'{feature}*Episode_Length_minutes'] = X[feature]*X['Episode_Length_minutes']
    score = test_feature_efficient(X, y, True)
    if score<baseline:
        print(f'Improved baseline of {score} that {baseline} before')
        baseline=score
        columns_for_training=X.columns
    else:
        X.drop([f'{feature}*Episode_Length_minutes'], axis=1, inplace=True)
columns_for_training


X_temp = X.copy()
X_temp['Episode_Length_seconds'] = X_temp['Episode_Length_minutes']*60
score = test_feature_efficient(X_temp, y, True)
if score<baseline:
    print(f'Improved baseline of {score} that {baseline} before')


X_temp = X.copy()
X_temp['temp'] = X_temp.groupby('Podcast_Name')['Episode_Length_minutes'].transform(lambda x:x.fillna(x.median()))
X_temp['Episode_Length_seconds_int'] = X_temp['temp'].astype(int)
X_temp.drop(['temp'],axis=1, inplace = True)
score = test_feature_efficient(X_temp, y, True)
if score<baseline:
    print(f'Improved baseline of {score} that {baseline} before')


features = ['Episode_Length_minutes']
X_copy = X.copy()
for feature in features:
    largest_num = X_copy[feature].astype(str).max()
    largest_num_len = len(str(largest_num))-1
    X_copy['temp'] = X_copy.groupby('Podcast_Name')['Episode_Length_minutes'].transform(lambda x:x.fillna(x.median()))
    
    num_digits_round = X_copy['temp'].astype(int).astype(str).apply(lambda x: len(x)).max()
    num_digits_total = X_copy['temp'].astype(str).apply(lambda x: len(x)).max()
    X_copy.drop(['temp'],axis=1, inplace = True)

    
    for i in range(1, num_digits_total):
        X_copy[f'{feature}_digit_{i}'] = ((X_copy[feature] * 10**(i-num_digits_round)) % 10).fillna(0).astype("int8")
        score = test_feature_efficient(X_copy,y,True)
        # if score<baseline:
        #     print(f'Improved baseline of {score} than {baseline} before')
        #     baseline=score
        #     X[f'{feature}_digit_{i}'] = ((X[feature] * 10**(i-num_digits_round)) % 10).fillna(0).astype("int8")
        #     test[f'{feature}_digit_{i}'] = ((test[feature] * 10**(i-num_digits_round)) % 10).fillna(0).astype("int8")
        #     columns_for_training = X.columns
        # else:
        #     X_copy.drop([f'{feature}_digit_{i}'],axis=1, inplace=True)


categorical_columns = X.select_dtypes(['object']).columns
for index,categorical_column in enumerate(categorical_columns):
    print(f'Iteration {index+1} out of {len(categorical_columns)}')
    le = LabelEncoder()
    X[f'{categorical_column}_encoded'] = le.fit_transform(X[categorical_column])
    score = test_feature_efficient(X, y, True)
    if score<baseline:
        print(f'Improved baseline of {score} that {baseline} before')
        baseline=score
        columns_for_training = X.columns
    else:
        X.drop([f'{categorical_column}_encoded'], axis=1, inplace=True)
columns_for_training


stats = ["mean","std","count","nunique","median","min","max","skew"]
for stat in stats:
    print(f'Feature {stat}')
    categorical_columns = X.select_dtypes(['object']).columns
    for index,categorical_column in enumerate(categorical_columns):
        print(f'Iteration {index+1} out of {len(categorical_columns)}')
        mean_encoded = X.groupby(categorical_column)['Episode_Length_minutes'].agg(stat)
        X[f'{categorical_column}_encoded_ELM_{stat}'] = X[categorical_column].map(mean_encoded)
        score = test_feature_efficient(X, y, True)
        if score<baseline:
            print(f'Improved baseline of {score} that {baseline} before')
            baseline=score
            columns_for_training = X.columns
        else:
            X.drop([f'{categorical_column}_encoded_ELM_{stat}'], axis=1, inplace=True)
columns_for_training


def test_feature_efficient(X_train, X_val, y_train, y_val,enable_categorical):
    X_train_copy=X_train.copy()
    X_val_copy=X_val.copy()
    for col in X_train_copy.select_dtypes(include='object').columns:
        X_train_copy[col] = X_train_copy[col].astype('category')
        X_val_copy[col] = X_val_copy[col].astype('category')
    model = XGBRegressor(random_state=42,enable_categorical=enable_categorical,tree_method='gpu_hist')
    model.fit(X_train_copy, y_train, eval_set=[(X_val_copy,y_val)], verbose=0)
    y_pred = model.predict(X_val_copy)
    print(rmse(y_val,y_pred))
    return rmse(y_val,y_pred)


X_train, X_val, y_train, y_val = train_test_split(X,y, shuffle=True, random_state = 42)    


stats = ["mean","std","count","nunique","median","min","max","skew"]
for stat in stats:
    print(f'For {stat}')
    categorical_columns = X_train.select_dtypes(['object']).columns
    for index,categorical_column in enumerate(categorical_columns):
        print(f'Iteration {index+1} out of {len(categorical_columns)}')
        X_train['Listening_Time_minutes'] = y_train
        mean_encoded = X_train.groupby(categorical_column)['Listening_Time_minutes'].agg(stat)
        X_train.drop(['Listening_Time_minutes'],axis=1,inplace=True)
        X_train[f'{categorical_column}_encoded_LTM_{stat}'] = X_train[categorical_column].map(mean_encoded)
        X_val[f'{categorical_column}_encoded_LTM_{stat}'] = X_val[categorical_column].map(mean_encoded)
        score = test_feature_efficient(X_train, X_val, y_train, y_val, True)
        if score<baseline:
            print(f'Improved baseline of {score} that {baseline} before')
            baseline=score
            columns_for_training = X_train.columns
        else:
            X_train.drop([f'{categorical_column}_encoded_LTM_{stat}'], axis=1, inplace=True)
            X_val.drop([f'{categorical_column}_encoded_LTM_{stat}'], axis=1, inplace=True)
columns_for_training


X_train.describe()


test.describe()


stats = ["mean","std","count","nunique","median","min","max","skew"]
continuous_columns = ['Episode_Length_minutes','Host_Popularity_percentage','Guest_Popularity_percentage','Number_of_Ads']
bin_values = [[0,400,10],[0,150,10],[0,150,10],[0,5,1]]

for stat in stats:
    print(f'For {stat}')
    for index,continuous_column in enumerate(continuous_columns):
        print(f'Iteration {index+1} out of {len(continuous_columns)}')
        
        bins = [i for i in range(bin_values[index][0],bin_values[index][1],bin_values[index][2])]
        X_train[f'{continuous_column}_bin']= np.searchsorted(bins, X_train[continuous_column].values)
        X_train['Listening_Time_minutes'] = y_train
        mean_encoded = X_train.groupby(f'{continuous_column}_bin')['Listening_Time_minutes'].agg(stat)
        X_train.drop(['Listening_Time_minutes'],axis=1,inplace=True)
        X_train.drop([f'{continuous_column}_bin'],axis=1,inplace=True)
        X_train[f'{continuous_column}_encoded_LTM_{stat}'] = X_train[continuous_column].map(mean_encoded)
        X_val[f'{continuous_column}_encoded_LTM_{stat}'] = X_val[continuous_column].map(mean_encoded)
        score = test_feature_efficient(X_train, X_val, y_train, y_val, True)
        if score<baseline:
            print(f'Improved baseline of {score} that {baseline} before')
            baseline=score
            X[f'{continuous_column}_encoded_LTM_{stat}'] = X[continuous_column].map(mean_encoded)
            columns_for_training = X.columns
        else:
            X_train.drop([f'{continuous_column}_encoded_LTM_{stat}'], axis=1, inplace=True)
            X_val.drop([f'{continuous_column}_encoded_LTM_{stat}'], axis=1, inplace=True)
columns_for_training


def test_feature_efficient(X,y,enable_categorical):
    X_copy=X.copy()
    for col in X_copy.select_dtypes(include='object').columns:
        X_copy[col] = X_copy[col].astype('category')
    X_train, X_val, y_train, y_val = train_test_split(X_copy,y, shuffle=True, random_state = 42)    
    model = XGBRegressor(random_state=42,enable_categorical=enable_categorical,tree_method='gpu_hist')
    model.fit(X_train, y_train, eval_set=[(X_val,y_val)], verbose=0)
    y_pred = model.predict(X_val)
    print(rmse(y_val,y_pred))
    return rmse(y_val,y_pred)


score = test_feature_efficient(X.drop(X.select_dtypes('object').columns, axis=1),y,False)
if score<=baseline:
    print(f'Improved baseline of {score} than {baseline} before')
    #baseline=score
    #X.drop([X.select_dtypes('object').columns], axis=1, inplace=True)
    #columns_for_training = X.columns
#columns_for_training


for col in X.columns:
    score = test_feature_efficient(X.drop([col],axis = 1),y,True)
    if score<=baseline:
        print(f'Improved baseline of {score} than {baseline} before')
        baseline=score
        X.drop([col],axis = 1, inplace=True)
        columns_for_training = X.columns
columns_for_training


sample = pd.read_csv('/kaggle/input/playground-series-s5e4/sample_submission.csv')
sample.columns


def param_finetuning(X, y, test, bestScore):
    xgb_params = {
        'n_estimators': 425,
        'max_depth': 14,
        'learning_rate': 0.01899,
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

    sample = pd.read_csv('/kaggle/working/submission.csv')

    X_train, X_val, y_train, y_val = train_test_split(X,y, shuffle=True, random_state = 42)    
        
    model = XGBRegressor(**xgb_params, enable_categorical=True, random_state=42)
    
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)],
                  verbose=100, early_stopping_rounds=100)
    
    y_pred = model.predict(X_val)
    score = rmse(y_val,y_pred)
    if score < bestScore:
            print(f"Results Improved from {bestScore} to {score}")
            y_test = model.predict(test)
            sample['Listening_Time_minutes'] = y_test
            bestScore = score
    return score, sample


bestScore =12.723957394927524
bestScore,sample = param_finetuning(X,y,test,bestScore)
#sample.to_csv('/kaggle/working/submission.csv',index=False)


26.53428


def model_training(X, y, test, n_splits, bestScore):
    xgb_params = {
        'n_estimators': 425,
        'max_depth': 14,
        'learning_rate': 0.01899,
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

    sample = pd.read_csv('/kaggle/working/submission.csv')

    kfolds = KFold(n_splits=n_splits,shuffle=True)

    for fold, (train_idx, val_idx) in enumerate(kfolds.split(X, y)):
        print(f'Fold {fold + 1} of {n_splits}')
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        

        model = XGBRegressor(**xgb_params, enable_categorical=True)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)],
                  verbose=100, early_stopping_rounds=100)

        y_pred = model.predict(X_val)
        score = mean_squared_error(y_val, y_pred, squared=False)  # RMSE

        if score < bestScore:
            print(f"Results Improved from {bestScore} to {score}")
            y_test = model.predict(test)
            sample['Listening_Time_minutes'] = y_test
            bestScore = score

    print(f"Best RMSE: {bestScore}")
    return bestScore, sample


bestScore = 12.723957394927524
bestScore,sample = model_training(X,y,test,10,bestScore)
sample.to_csv('/kaggle/working/submission.csv',index=False)




