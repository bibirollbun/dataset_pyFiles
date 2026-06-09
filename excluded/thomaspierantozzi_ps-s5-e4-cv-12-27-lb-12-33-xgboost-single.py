import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import root_mean_squared_error
from sklearn.preprocessing import StandardScaler, TargetEncoder
from sklearn.model_selection import KFold
from tqdm.auto import tqdm
from itertools import combinations
from time import time
import xgboost as xgb
from lightgbm import LGBMRegressor
import optuna

#setting the default output for sklearn transformers to pandas df
from sklearn import set_config
set_config(transform_output='pandas')



train_dataset = pd.read_csv('./Datasets/train.csv', index_col=0)
test_dataset = pd.read_csv('./Datasets/test.csv', index_col=0)
original_dataset = pd.read_csv('./Datasets/original.csv') #the original dataset used by the competition hosts to make the new synthetic data
y = train_dataset['Listening_Time_minutes']
y_original = original_dataset['Listening_Time_minutes']
train_dataset.drop(columns=['Listening_Time_minutes'], inplace=True)
original_dataset.drop(columns=['Listening_Time_minutes'], inplace=True)


train_dataset.sample(10)


print(f'We are dealing with a train and test dataset respectively shaped as: '
      f'\n\tTrain: {train_dataset.shape}'
      f'\n\tTest:  {test_dataset.shape}'
      f'\n\tOriginal: {original_dataset.shape}')


object_columns = train_dataset.select_dtypes(include='object').columns.to_list()
numeric_columns = train_dataset.select_dtypes(include='number').columns.to_list()
print(f'Here we report the object columns: \n\t{object_columns}')
print(f'Here we report the numeric columns: \n\t{numeric_columns}')


columns_with_nan_train = {column: train_dataset[column].isna().sum(axis=0) for column in train_dataset.columns if train_dataset[column].isna().sum(axis=0) != 0}
columns_with_nan_test = {column: test_dataset[column].isna().sum(axis=0) for column in test_dataset.columns if test_dataset[column].isna().sum(axis=0) != 0}


print(f'The folliwing is a dictionary reporting the NaN values for the different columns:\n\tTRAIN DATAFRAME: {columns_with_nan_train}\n\tTEST DATAFRAME: {columns_with_nan_test}')


fig = plt.figure(figsize=(24, 8))
for index,key in enumerate(columns_with_nan_train.keys()):
      fig.add_subplot(1, len(columns_with_nan_train.keys()), index + 1)
      sns.scatterplot(
            x=train_dataset[key],
            y=y
      )


#we must merge together again the target values and the train dataset if we want to delete some rows and keep out the outliers
train_dataset = train_dataset.merge(y, left_index=True, right_index=True)
train_dataset.drop(index=train_dataset.loc[train_dataset['Episode_Length_minutes'] > 130].index, inplace=True)
train_dataset.drop(index=train_dataset.loc[train_dataset['Guest_Popularity_percentage'] > 100].index, inplace=True)
train_dataset.drop(index=train_dataset.loc[train_dataset['Number_of_Ads'] > 10].index, inplace=True)

#back again with having the targets in a separate df
y = train_dataset['Listening_Time_minutes']
train_dataset.drop(columns=['Listening_Time_minutes'], inplace=True)


fig = plt.figure(figsize=(24, 8))
for index, key in enumerate(columns_with_nan_train.keys()):
      fig.add_subplot(1, len(columns_with_nan_train.keys()), index + 1)
      sns.scatterplot(
            x=train_dataset[key],
            y=y
      )


sns.boxplot(
      x=train_dataset['Number_of_Ads'],
      y=y
)


from sklearn.impute import SimpleImputer
mean_imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
mode_imputer = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=train_dataset['Number_of_Ads'].mode()[0])

train_dataset['Episode_Length_minutes'] = mean_imputer.fit_transform(train_dataset[['Episode_Length_minutes']])
test_dataset['Episode_Length_minutes'] = mean_imputer.transform(test_dataset[['Episode_Length_minutes']]) #to avoid data leakages we fit the data on the train set and use the same parameters to impute the value on the test dataset

train_dataset['Number_of_Ads'] = mode_imputer.fit_transform(train_dataset[['Number_of_Ads']])


train_dataset['Guest_presence'] = ~train_dataset['Guest_Popularity_percentage'].isna()
test_dataset['Guest_presence'] = ~test_dataset['Guest_Popularity_percentage'].isna()
original_dataset['Guest_presence'] = ~original_dataset['Guest_Popularity_percentage'].isna()


sns.boxplot(
      x=train_dataset['Guest_presence'],
      y=y
)


train_dataset.fillna({'Guest_Popularity_percentage': 0}, inplace=True)
test_dataset.fillna({'Guest_Popularity_percentage': 0}, inplace=True)


train_dataset['Guest_Host_Mean_Popularity_binned'] = ((train_dataset['Guest_Popularity_percentage'] + train_dataset['Host_Popularity_percentage']) / 2 ) // 10 * 10
test_dataset['Guest_Host_Mean_Popularity_binned'] = ((test_dataset['Guest_Popularity_percentage'] + test_dataset['Host_Popularity_percentage']) / 2 ) // 10 * 10
original_dataset['Guest_Host_Mean_Popularity_binned'] = ((original_dataset['Guest_Popularity_percentage'] + original_dataset['Host_Popularity_percentage']) / 2 ) // 10 * 10
sns.boxplot(
      x=train_dataset['Guest_Host_Mean_Popularity_binned'],
      y=y
)
plt.xticks(rotation=60)


train_dataset.sample(10)


category_columns = [
      'Genre',
      'Publication_Day',
      'Publication_Time',
      'Episode_Sentiment'
]

category_dict = {key: list(train_dataset[key].unique()) for key in category_columns}
category_dict


weekdays_map = {
      'Sunday': 3,
      'Monday': 4,
      'Tuesday': 5,
      'Wednesday': 6,
      'Thursday': 0,
      'Friday': 1,
      'Saturday': 2
} #this funny mappign is due to the fact that a trend has been already catched giving a look at the boxplots

day_phases = {
      'Morning': 0,
      'Afternoon': 2,
      'Evening': 1, #this funny mappign is due to the fact that a trend has been already catched giving a look at the boxplots
      'Night': 3
}

sentiment_map = {
      'Positive': 1,
      'Neutral': 0,
      'Negative': -1
}
train_dataset['Publication_Day'] = train_dataset['Publication_Day'].map(arg=weekdays_map)
test_dataset['Publication_Day'] = test_dataset['Publication_Day'].map(arg=weekdays_map)
original_dataset['Publication_Day'] = original_dataset['Publication_Day'].map(arg=weekdays_map)
train_dataset['Publication_Time'] = train_dataset['Publication_Time'].map(arg=day_phases)
test_dataset['Publication_Time'] = test_dataset['Publication_Time'].map(arg=day_phases)
original_dataset['Publication_Time'] = original_dataset['Publication_Time'].map(arg=day_phases)
train_dataset['Episode_Sentiment'] = train_dataset['Episode_Sentiment'].map(arg=sentiment_map)
test_dataset['Episode_Sentiment'] = test_dataset['Episode_Sentiment'].map(arg=sentiment_map)
original_dataset['Episode_Sentiment'] = original_dataset['Episode_Sentiment'].map(arg=sentiment_map)


fig = plt.figure(figsize=(24, 8))
for index, feature in enumerate(category_columns):
      fig.add_subplot(1, 4, index + 1)
      sns.boxplot(
            x=train_dataset[feature],
            y=y
      )
      plt.xticks(rotation=60)
      plt.title(f'{feature} vs. TargetValue')


#let's start with understanding how many different podcasts we're dealing with
print(f'There are {len(train_dataset['Podcast_Name'].unique())} different podcasts in this dataset')


fig = plt.figure(figsize=(24, 8))
sns.barplot(
      x=train_dataset['Podcast_Name'],
      y=y,
)
plt.xticks(rotation=60)


train_dataset['Episode_Number'] = train_dataset['Episode_Title'].apply(lambda x: x.split(' ')[1]).astype(int)
test_dataset['Episode_Number'] = test_dataset['Episode_Title'].apply(lambda x: x.split(' ')[1]).astype(int)
original_dataset['Episode_Number'] = original_dataset['Episode_Title'].apply(lambda x: x.split(' ')[1]).astype(int)

#binning the data to reduce cardinality
train_dataset['Episode_Number_binned_10'] = train_dataset['Episode_Number'] // 10 * 10
test_dataset['Episode_Number_binned_10'] = test_dataset['Episode_Number'] // 10 * 10
original_dataset['Episode_Number_binned_10'] = original_dataset['Episode_Number'] // 10 * 10

train_dataset.drop(columns=['Episode_Title', 'Episode_Number'], inplace=True)
test_dataset.drop(columns=['Episode_Title', 'Episode_Number'], inplace=True)
original_dataset.drop(columns=['Episode_Title', 'Episode_Number'], inplace=True)


numeric_columns = train_dataset.select_dtypes(include='number').columns.to_list()
category_columns = train_dataset.select_dtypes(include=['category', 'object']).columns.to_list()
base_features = numeric_columns + category_columns
highly_corr_cols = ['Episode_Number_binned_10', 'Episode_Sentiment' , 'Publication_Time' , 'Publication_Day']


#these will be the set of features onto which grouping the data and engineer some new features
groupby_by = category_columns + highly_corr_cols
groupby_on = numeric_columns


original_dataset = pd.concat([original_dataset, y_original], axis=1)

te_orig_features = []

#we start with the TE vs. the original data
for col in base_features:
      print(f'Processing feature: {col}')
      new_feature_name = f'{col}_TE_ORIG'
      te_orig_features.append(new_feature_name)
      train_dataset[new_feature_name] = train_dataset[col].map(
            arg=original_dataset.groupby(col)['Listening_Time_minutes'].mean()
      )
      test_dataset[new_feature_name] = test_dataset[col].map(
            arg=original_dataset.groupby(col)['Listening_Time_minutes'].mean()
      )
      nan_values = train_dataset[col].isna().sum()
      print(f'\tNew feature NaN values: {nan_values}')
      train_dataset[new_feature_name] = train_dataset[new_feature_name].fillna(original_dataset['Listening_Time_minutes'].mean())
      test_dataset[new_feature_name] = test_dataset[new_feature_name].fillna(original_dataset['Listening_Time_minutes'].mean())


print(f'The new features are: \n{te_orig_features}')


#first a rough grouping based on the categorical features + the most highly correlated features, on the base of the most common aggregate functions
grouping_stats = [
      'mean',
      'max',
      'min',
      'std',
]
from itertools import product

agg_features = []
for by, on in product(groupby_by, groupby_on):
      if by != on:
            print('Engineering features with the following combo: ', by, ' ---> ', on)
            tmp = train_dataset.groupby(by=by)[on].agg(grouping_stats)
            tmp.columns = [f'Agg_by_{by}_on_{on}_{s}' for s in grouping_stats]
            agg_features.append(tmp.columns)
            train_dataset = train_dataset.merge(tmp, on=by, how='left', suffixes=('', ''))
            test_dataset = test_dataset.merge(tmp, on=by, how='left', suffixes=('', ''))
print(f'We engineered {len(agg_features) * len(grouping_stats)} features')



#Now let's repeat the process above by adding the combinations of the features from the first list, in order to have a multi-index groupby of combined features aggregated

agg2_features = []
for by, on in tqdm(product(combinations(groupby_by, 2), groupby_on)):
      if by != on:
            print('Engineering features with the following combo: ', by, ' ---> ', on)
            
            tmp = train_dataset.groupby(by=list(by))[on].agg(grouping_stats)
            tmp.columns = [f'Agg_by_{list(by)}_on_{on}_{s}' for s in grouping_stats]
            agg2_features.append(tmp.columns)
            train_dataset = train_dataset.merge(tmp, on=list(by), how='left', suffixes=('', ''))
            test_dataset = test_dataset.merge(tmp, on=list(by), how='left', suffixes=('', ''))
print(f'We engineered {len(agg2_features) * len(grouping_stats)} features')


#let's clean the names of the faetures from '[' ']' ',' since the xgboost algorithms cannot accept feature names containing those chars
train_dataset.columns = [name.replace('[', '').replace(']', '').replace(',', '_') for name in train_dataset.columns]
test_dataset.columns = [name.replace('[', '').replace(']', '').replace(',', '_') for name in test_dataset.columns]


train_dataset[train_dataset.select_dtypes(include='float64').columns] = train_dataset.select_dtypes(include='float64').astype('float32')
test_dataset[test_dataset.select_dtypes(include='float64').columns] = test_dataset.select_dtypes(include='float64').astype('float32')


import warnings
warnings.simplefilter('ignore')

concat_features = []

for elem1, elem2 in combinations(base_features, 2):
    feature_name = f'{elem1}_{elem2}'
    print(f'Adding feature {feature_name}')
    concat_features.append(feature_name)
    train_dataset[feature_name] = train_dataset[elem1].astype('str') + '_' + train_dataset[elem2].astype('str')
    test_dataset[feature_name] = test_dataset[elem1].astype('str') + '_' + test_dataset[elem2].astype('str')
print(f'{len (concat_features)} features added to dataset')  
    
concat_features2 = []

for elem1, elem2, elem3 in combinations(base_features, 3):
    feature_name = f'{elem1}_{elem2}_{elem3}'
    print(f'Adding feature {feature_name}')
    concat_features2.append(feature_name)
    concat_features.append(feature_name)
    train_dataset[feature_name] = train_dataset[elem1].astype('str') + '_' + train_dataset[elem2].astype('str') + '_' + train_dataset[elem3].astype('str')
    test_dataset[feature_name] = test_dataset[elem1].astype('str') + '_' + test_dataset[elem2].astype('str') + '_' + test_dataset[elem3].astype('str')
print(f'{len (concat_features2)} features added to dataset')



train_dataset = train_dataset.copy() #de-fragmenting the train_dataset
test_dataset = test_dataset.copy() #de-fragmenting the train_dataset


test_dataset.index = np.arange(750_000, 1_000_000)


from pathlib import  Path
path_to_save = Path('./Datasets')
train_dataset.to_csv(path_to_save / 'train_ready.csv')
if train_dataset.shape[1] == test_dataset.shape[1]:
      print(f'Saved the train dataset to {path_to_save / 'train_ready.csv'}')
      test_dataset.to_csv(path_to_save / 'test_ready.csv')
      print(f'Saved the test dataset to {path_to_save / 'test_ready.csv'}')
      y.to_csv(path_to_save / 'targets_ready.csv')
      print(f'Saved the target dataset to {path_to_save / 'target_ready.csv'}')
else:
      print('The shapes of the train and test dataset do not match...')


#defragging and forcing the deletion of the datasets processed so far
y.index = np.arange(y.shape[0])
X = train_dataset.copy()
y = y.copy()
X_test = test_dataset.copy()
del(train_dataset, test_dataset, original_dataset)


features_to_TE = X.select_dtypes(include='object').columns.to_list()


kfold = KFold(n_splits=5, shuffle=True)

for index, (train_index, cv_index) in enumerate(kfold.split(X)):
    print(f'{(' FOLD' + str(index + 1) + ' '):*^50}')
    
    train_set = X.loc[train_index].copy()
    train_target_set = y.loc[train_index].copy()
    cv_set = X.loc[cv_index].copy()
    cv_target_set = y.loc[cv_index].copy()
    
    te_time_start = time()
    print(f'\tTarget Encoding the data...')
    target_encoder = TargetEncoder()
    train_set[features_to_TE] = target_encoder.fit_transform(train_set[features_to_TE], train_target_set)
    cv_set[features_to_TE] = target_encoder.transform(cv_set[features_to_TE])
    te_time_end = time()
    print(f'\tEncoding performed succesfully in {te_time_end-te_time_start:.2f} seconds')
    
    std_enc_time_start = time()
    print('\tStandard Encoding the data...')
    std_scaler = StandardScaler()
    features_to_scale = train_set.select_dtypes(include='number').columns.to_list()
    train_set[features_to_scale] = std_scaler.fit_transform(train_set[features_to_scale]).astype(dtype='float32') #reduce the size of the dataframe
    cv_set[features_to_scale] = std_scaler.transform(cv_set[features_to_scale]).astype(dtype='float32') #reduce the size of the dataframe
    std_enc_time_end = time()
    print(f'\tEncoding performed successfully in {std_enc_time_end-std_enc_time_start:.2f} seconds')
    
    train_time_start = time()
    
    model_xgb = xgb.XGBRegressor(n_estimators=1_000,
                                 eta=5e-2,
                                 max_depth=12,
                                 subsample=0.75,
                                 colsample_bytree=0.75,
                                 colsample_bylevel=0.75,
                                 colsample_bynode=0.75,
                                 reg_alpha=1.5,
                                 gamma=0.8,
                                 reg_lambda=0.8,
                                 early_stopping_rounds=10,
                                 )

    print(f'\tTraining the model...')
    model_xgb.fit(
        train_set, 
        train_target_set,
        eval_set=[(cv_set, cv_target_set)],
        verbose=25,
    )
    train_time_end = time()
    print(f'\tModel trained in {train_time_end-train_time_start:.2f} seconds')
    
    eval_time_start = time()
    y_hat_train = model_xgb.predict(train_set)
    y_hat_cv = model_xgb.predict(cv_set)
    
    loss_train = root_mean_squared_error(train_target_set, y_hat_train)
    loss_cv = root_mean_squared_error(cv_target_set, y_hat_cv)
    eval_time_end = time()
    print(f'\tEvaluation performed in {eval_time_end-eval_time_start:.2f} seconds'
          f'\tTrain loss RMSE: {loss_train:.4f} | CV loss RMSE: {loss_cv:.4f}\n')



from xgboost import plot_importance

plot_importance(model_xgb, max_num_features=25)

