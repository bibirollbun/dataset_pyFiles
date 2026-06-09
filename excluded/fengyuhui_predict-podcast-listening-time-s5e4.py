import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')


train_df = pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv')

train_df.set_index('id', inplace=True)
test_df.set_index('id', inplace=True)


train_df.shape


train_df.head()


train_df.describe()


fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(20,3))

# 1. plot original distribution
sns.histplot(x=train_df.Listening_Time_minutes, ax=axes[0])
plt.title('Listening_Time_minutes')

# 2. plot percentage listening time (subset to length not missing and 5)
train_clean_idx = (
    ~train_df.Listening_Time_minutes.isna()) & (
        train_df.Listening_Time_minutes>0) & (
        train_df.Listening_Time_minutes <= train_df.Episode_Length_minutes*1.1
        )

train_df['listen_percent'] = train_df['Listening_Time_minutes']/train_df['Episode_Length_minutes']
sns.histplot(x=train_df.loc[train_clean_idx,'listen_percent'], ax=axes[1])
plt.title('Percentage Time')

# 3. plot percentage listening time in log scale
train_df['log_percent'] = np.log(train_df.listen_percent)
sns.histplot(x=train_df.loc[train_clean_idx,'log_percent'], ax=axes[2])
plt.title('Log Percentage Time')


train_df.isna().sum()


test_df.isna().sum()


# 1. Episode_Length_minutes
train_df['ep_len_missing'] = 0
train_invalid_idx = (train_df['Episode_Length_minutes'].isna()) | (
    train_df['Episode_Length_minutes'] <= 0)
train_df.loc[train_invalid_idx,'ep_len_missing'] = 1
train_df.loc[train_invalid_idx,'Episode_Length_minutes'] = train_df.Episode_Length_minutes.median()

test_invalid_idx = (test_df['Episode_Length_minutes'].isna()) | (
    test_df['Episode_Length_minutes'] <= 0)
test_df['ep_len_missing'] = 0
test_df.loc[test_invalid_idx,'ep_len_missing'] = 1
test_df.loc[test_invalid_idx,'Episode_Length_minutes'] = test_df.Episode_Length_minutes.median()

# 2. Guest_Popularity_percentage_missing
train_df['Guest_Popularity_percentage_missing'] = 0
train_df.loc[train_df['Guest_Popularity_percentage'].isna(),'Guest_Popularity_percentage_missing'] = 1
train_df.loc[train_df.Guest_Popularity_percentage.isna(),'Guest_Popularity_percentage'] = train_df.Guest_Popularity_percentage.median()

test_df['Guest_Popularity_percentage_missing'] = 0
test_df.loc[test_df['Guest_Popularity_percentage'].isna(),'Guest_Popularity_percentage_missing'] = 1
test_df.loc[test_df.Guest_Popularity_percentage.isna(),'Guest_Popularity_percentage'] = test_df.Guest_Popularity_percentage.median()


# 3. Fill missing values with mode
train_df.loc[train_df.Number_of_Ads.isna(),'Number_of_Ads']=0

# 4. Convert categorical to numeric
train_df['ep_num']=train_df['Episode_Title'].str.replace('Episode ','').astype('int64')
test_df['ep_num']=test_df['Episode_Title'].str.replace('Episode ','').astype('int64')


cat_features = ['Podcast_Name','Genre','Publication_Day',
                'Publication_Time','Episode_Sentiment']

num_features = ['Episode_Length_minutes','Host_Popularity_percentage',
               'Guest_Popularity_percentage','ep_num']


# 1. Episode_Length_minutes
train_df.loc[train_df.Episode_Length_minutes>150, 'ep_len_missing'] = 1
train_df.loc[train_df.Episode_Length_minutes>150, 'Episode_Length_minutes'] = train_df.Episode_Length_minutes.median()

test_df.loc[test_df.Episode_Length_minutes>150, 'ep_len_missing'] = 1
test_df.loc[test_df.Episode_Length_minutes>150, 'Episode_Length_minutes'] = test_df.Episode_Length_minutes.median()

# 2. Host_Popularity_percentage
train_df.loc[train_df.Host_Popularity_percentage>100, 'Host_Popularity_percentage'] = train_df.Host_Popularity_percentage.median()
test_df.loc[test_df.Host_Popularity_percentage>100, 'Host_Popularity_percentage'] = test_df.Host_Popularity_percentage.median()

# 3. Guest_Popularity_percentage
train_df.loc[train_df.Guest_Popularity_percentage>100, 'Guest_Popularity_percentage'] = train_df.Guest_Popularity_percentage.median()
test_df.loc[test_df.Guest_Popularity_percentage>100, 'Guest_Popularity_percentage'] = test_df.Guest_Popularity_percentage.median()

# 4. Number of Ads > 10
train_df.loc[train_df.Number_of_Ads>10, 'Number_of_Ads'] = 0
test_df.loc[test_df.Number_of_Ads>10, 'Number_of_Ads'] = 0


# Numeric Features
def explore_num_features(num_features = num_features):

    fig, axes = plt.subplots(nrows=len(num_features),
                             ncols=4,figsize=(20, 3*len(num_features)))

    for i in range(len(num_features)):
        feature = num_features[i]
        
        # 1. compare distribution between train and test (histogram)
        train_df_plot = train_df.loc[:,[feature]]
        train_df_plot['source'] = 'train'

        test_df_plot = test_df.loc[:,[feature]]
        test_df_plot['source'] = 'test'

        all_df = pd.concat([train_df_plot, test_df_plot]).dropna()

        sns.kdeplot(all_df, x=feature, hue='source', fill=True, ax=axes[i][0])

        # 2. association between feature and target (scatter plot)
        sns.scatterplot(train_df, x=feature, y='Listening_Time_minutes', 
                        ax=axes[i][1], s=1,alpha=0.3)

        # 3. association between feature and target (as percent)
        sns.scatterplot(train_df.loc[(
            train_df.ep_len_missing==0) & (
                train_df.listen_percent<=1.5),:],
                        x=feature, y='listen_percent',
                        ax=axes[i][2], s=1, alpha=0.3)

        # 4. asspciation between feature and target (as log percent)
        sns.scatterplot(train_df.loc[(
            train_df.ep_len_missing==0) & (
                train_df.listen_percent<=1.5),:],
                        x=feature, y='log_percent',
                        ax=axes[i][3], s=1, alpha=0.3)


explore_num_features()


# Explore Categorical Features
def explore_cat_features(cat_features = cat_features):

    fig, axes = plt.subplots(nrows=len(cat_features), ncols=4,
                             figsize=(20, 3*len(cat_features)))

    for i in range(len(cat_features)):
        feature = cat_features[i]
        
        # 1. compare distribution between train and test (barchart)
        train_sum = train_df.groupby(feature).ep_num.count().reset_index().rename(columns={"ep_num":'percent'})
        train_sum['percent']=train_sum.percent/train_df.shape[0]
        train_sum['source']='train'

        test_sum = test_df.groupby(feature).ep_num.count().reset_index().rename(columns={"ep_num":'percent'})
        test_sum['percent']=test_sum.percent/test_df.shape[0]
        test_sum['source']='test'

        result_sum = pd.concat([train_sum, test_sum])

        sns.barplot(result_sum, x=feature, hue='source', y='percent', ax=axes[i][0])
        plt.xlabel(feature)
        plt.title('train vs test')

        # 2. association between feature and target (kde plot)
        sns.kdeplot(train_df, x='Listening_Time_minutes', hue=feature, ax=axes[i][1])

        # 3. association between feature and percent (kde plot)
        sns.kdeplot(train_df.loc[(train_df.ep_len_missing==0) & (
                train_df.listen_percent<=1.5),:],
                        x='listen_percent',hue=feature, ax=axes[i][2])

        # 4. association between feature and target (box plot)
        sns.boxplot(train_df, y='Listening_Time_minutes', x=feature, hue=feature, ax=axes[i][3])


explore_cat_features(
    cat_features=['Publication_Day','ep_len_missing',
                'Publication_Time','Episode_Sentiment','Number_of_Ads'])


from sklearn.preprocessing import OneHotEncoder

OH_encoder = OneHotEncoder(min_frequency=0.01, sparse=False)
OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(train_df[cat_features]))
OH_cols_test = pd.DataFrame(OH_encoder.transform(test_df[cat_features]))

OH_cols_train.columns = OH_encoder.get_feature_names_out(cat_features)
OH_cols_test.columns = OH_encoder.get_feature_names_out(cat_features)

OH_cols_train.index = train_df.index
OH_cols_test.index = test_df.index

train_df_X = pd.concat([train_df, OH_cols_train], axis=1)
test_df_X = pd.concat([test_df, OH_cols_test], axis=1)

OH_cols=OH_encoder.get_feature_names_out(cat_features).tolist()


# 1. add quatratic term for ep_length
train_df_X['ep_len_2'] = train_df_X['Episode_Length_minutes'] ** 2
test_df_X['ep_len_2'] = test_df_X['Episode_Length_minutes'] ** 2

train_df_X['log_ep_len'] = np.log(train_df_X['Episode_Length_minutes'])
test_df_X['log_ep_len'] = np.log(test_df_X['Episode_Length_minutes'])

# 2. ad_density
train_df_X['ad_density'] = train_df_X['Number_of_Ads']/train_df_X['Episode_Length_minutes']
test_df_X['ad_density'] = test_df_X['Number_of_Ads']/test_df_X['Episode_Length_minutes']

# 3. max guest and host popularity
train_df_X['max_popularity'] = np.maximum(train_df_X['Host_Popularity_percentage'],
                                   train_df_X['Guest_Popularity_percentage'])

test_df_X['max_popularity'] = np.maximum(test_df_X['Host_Popularity_percentage'],
                                  test_df_X['Guest_Popularity_percentage'])

train_df_X['popularity_diff'] = train_df_X['Host_Popularity_percentage']- train_df_X['Guest_Popularity_percentage']

test_df_X['popularity_diff'] = test_df_X['Host_Popularity_percentage']-test_df_X['Guest_Popularity_percentage']


# Train Validation Split

from sklearn.model_selection import train_test_split

X = train_df_X.copy()
y = X.pop('Listening_Time_minutes')
test_X = test_df_X.copy()

train_X, valid_X, train_y, valid_y = train_test_split(X, y, random_state = 1)


# Calculate RMSE

from sklearn.metrics import mean_squared_error

def calculate_RMSE(pred_train, pred_valid,
                   train_y=train_y, valid_y=valid_y,
                   evaluate_subcat=False,
                   valid_X = valid_X,
                   OH_features=OH_cols):
    
    print(f'Train RMSE is {mean_squared_error(pred_train, train_y)}')
    print(f'Valid RMSE is {mean_squared_error(pred_valid, valid_y)}')

    if evaluate_subcat:
        valid_pred_data=pd.DataFrame({
            'pred': pred_valid,
            'true': valid_y
        }, index=valid_X.index)

        valid_data=valid_X.merge(valid_pred_data, on='id')

        for var in OH_features:
            n_sample = valid_data.loc[valid_data[var]==1,"pred"].shape[0]
            if n_sample>=1:
                mse = mean_squared_error(
                    valid_data.loc[valid_data[var]==1,"pred"],
                    valid_data.loc[valid_data[var]==1,"true"]
                )
                print(f'Valid RMSE for {var} is {mse}, sample size {n_sample}')


# Plot predict vs Features and residual vs Features

def resid_plot_feature_num(num_features, pred_valid_y,
                           valid_X=valid_X,
                           valid_y=valid_y
                          ):

    fig, axes = plt.subplots(nrows=len(num_features), ncols=2,
                             figsize=(20, 3*len(num_features)))

    for i in range(len(num_features)):
        feature = num_features[i]

        # 1. valid, pred vs feature
        sns.scatterplot(x=valid_X[feature], y=pred_valid_y, 
                        s=1, alpha=0.3, ax=axes[i][0])
        plt.title(f"valid pred for {feature}")

        # 2. valid, resid vs feature
        sns.scatterplot(x=valid_X[feature], y=valid_y - pred_valid_y, 
                        s=1, alpha=0.5, ax=axes[i][1])
        plt.title(f"valid resid for {feature}")


from sklearn.linear_model import LinearRegression

reg_features = ['Episode_Length_minutes',
              'Host_Popularity_percentage','Guest_Popularity_percentage',
              'ep_len_missing','Guest_Popularity_percentage_missing',
              'ep_num',
              'Genre_Business','Genre_Comedy','Genre_Education','Genre_Health',
              'Genre_Lifestyle','Genre_Music','Genre_News','Genre_Sports',
              'Genre_Technology','Genre_True Crime',
              'Publication_Day_Friday', 'Publication_Day_Monday',
              'Publication_Day_Saturday', 'Publication_Day_Sunday',
              'Publication_Day_Thursday', 'Publication_Day_Tuesday',
              'Publication_Day_Wednesday',
              'Publication_Time_Afternoon','Publication_Time_Evening',
              'Publication_Time_Morning','Publication_Time_Night',
              'Episode_Sentiment_Negative','Episode_Sentiment_Neutral',
              'Episode_Sentiment_Positive'
             ]

reg = LinearRegression()
reg.fit(train_X.loc[:,reg_features], train_y)

pred_train = reg.predict(train_X.loc[:,reg_features])
pred_valid= reg.predict(valid_X.loc[:,reg_features])

calculate_RMSE(pred_train, pred_valid)


from sklearn.ensemble import RandomForestRegressor

fm_features = ['Episode_Length_minutes',
              'Host_Popularity_percentage','Guest_Popularity_percentage',
              'ep_len_missing','Guest_Popularity_percentage_missing',
              'ep_num'] + OH_cols

forest_model = RandomForestRegressor(
    random_state = 1,
    min_samples_split = 10,
    max_depth = 30,
    n_estimators =50
)

forest_model.fit(train_X.loc[:, fm_features], train_y)

pred_train_fm = forest_model.predict(train_X.loc[:,fm_features])
pred_valid_fm = forest_model.predict(valid_X.loc[:,fm_features])

calculate_RMSE(pred_train_fm, pred_valid_fm)


calculate_RMSE(pred_train_fm, pred_valid_fm,
               evaluate_subcat=True,
               OH_features=['ep_len_missing',
                            'Genre_Business','Genre_Comedy','Genre_Education','Genre_Health',
                            'Genre_Lifestyle','Genre_Music','Genre_News','Genre_Sports',
                            'Genre_Technology','Genre_True Crime',
                            'Publication_Day_Friday', 'Publication_Day_Monday',
                            'Publication_Day_Saturday', 'Publication_Day_Sunday',
                            'Publication_Day_Thursday', 'Publication_Day_Tuesday',
                            'Publication_Day_Wednesday',
                            'Publication_Time_Afternoon','Publication_Time_Evening',
                            'Publication_Time_Morning','Publication_Time_Night',
                            'Episode_Sentiment_Negative','Episode_Sentiment_Neutral',
                            'Episode_Sentiment_Positive'])


# model 1: for non-missing data
full_fm_features = ['Episode_Length_minutes','ep_len_2','log_ep_len',
                    'ad_density','Number_of_Ads',
                    'Host_Popularity_percentage','Guest_Popularity_percentage',
                    'max_popularity','popularity_diff',
                    'Guest_Popularity_percentage_missing',
                    'ep_num'] + OH_cols


full_forest_model = RandomForestRegressor(
    random_state = 1,
    min_samples_split = 5,
    max_depth = 30
)

full_forest_model.fit(
    train_X.loc[train_X.ep_len_missing==0,full_fm_features], 
    train_y.loc[train_X.ep_len_missing==0])


# Prediction - non-missing data

pred_train_full_fm = full_forest_model.predict(
    train_X.loc[train_X.ep_len_missing==0,full_fm_features]
)

pred_valid_full_fm = full_forest_model.predict(
    valid_X.loc[valid_X.ep_len_missing==0, full_fm_features]
)

calculate_RMSE(pred_train_full_fm, 
               pred_valid_full_fm,
               train_y = train_y.loc[train_X.ep_len_missing==0], 
               valid_y=valid_y.loc[valid_X.ep_len_missing==0])


resid_plot_feature_num(num_features=['Host_Popularity_percentage',
                                     'Guest_Popularity_percentage',
                                     'ep_num',
                                     'Episode_Length_minutes',
                                     'ep_len_2',
                                     'ad_density'],
                       pred_valid_y = pred_valid_full_fm,
                       valid_X=valid_X.loc[valid_X.ep_len_missing==0,:],
                       valid_y=valid_y.loc[valid_X.ep_len_missing==0])


# model 2: for missing data
missing_fm_features = ['Host_Popularity_percentage','Guest_Popularity_percentage',
                       'Guest_Popularity_percentage_missing','ep_num'] + OH_cols

missing_forest_model = RandomForestRegressor(
    random_state = 1
)

missing_forest_model.fit(
    train_X.loc[train_X.ep_len_missing==1, missing_fm_features], 
    train_y.loc[train_X.ep_len_missing==1])


# Prediction - missing
pred_train_missing_fm = missing_forest_model.predict(
    train_X.loc[train_X.ep_len_missing==1,missing_fm_features]
)

pred_valid_missing_fm = missing_forest_model.predict(
    valid_X.loc[valid_X.ep_len_missing==1, missing_fm_features]
)

calculate_RMSE(pred_train_missing_fm, 
               pred_valid_missing_fm,
               train_y = train_y.loc[train_X.ep_len_missing==1], 
               valid_y=valid_y.loc[valid_X.ep_len_missing==1]
              )


resid_plot_feature_num(num_features=['Host_Popularity_percentage',
                                     'Guest_Popularity_percentage',
                                     'ep_num'],
                       pred_valid_y = pred_valid_missing_fm,
                       valid_X=valid_X.loc[valid_X.ep_len_missing==1,:],
                       valid_y=valid_y.loc[valid_X.ep_len_missing==1])


# combine
pred_train_fm3 = pd.concat([
    pd.DataFrame({'pred':pred_train_full_fm},
                index=train_X.loc[train_X.ep_len_missing==0].index),
    pd.DataFrame({'pred':pred_train_missing_fm},
                index=train_X.loc[train_X.ep_len_missing==1].index)
]).merge(train_y, how='right', on='id').drop('Listening_Time_minutes', axis=1)

pred_valid_fm3 = pd.concat([
    pd.DataFrame({'pred':pred_valid_full_fm},
                index=valid_X.loc[valid_X.ep_len_missing==0].index),
    pd.DataFrame({'pred':pred_valid_missing_fm},
                index=valid_X.loc[valid_X.ep_len_missing==1].index)
]).merge(valid_y, how='right',on='id').drop('Listening_Time_minutes', axis=1)

calculate_RMSE(pred_train_fm3, 
               pred_valid_fm3)


# create submission

# 1. prediction for non-missing data
pred_test_full_fm = full_forest_model.predict(
    test_X.loc[test_X.ep_len_missing==0,full_fm_features]
)

# 2. prediction for missing data
pred_test_missing_fm = missing_forest_model.predict(
    test_X.loc[test_X.ep_len_missing==1,missing_fm_features]
)

# 3. combine
pred_test_fm3 = pd.concat([
    pd.DataFrame({'Listening_Time_minutes':pred_test_full_fm},
                index=test_X.loc[test_X.ep_len_missing==0].index),
    pd.DataFrame({'Listening_Time_minutes':pred_test_missing_fm},
                index=test_X.loc[test_X.ep_len_missing==1].index)
]).merge(test_df, how='right', on='id')

# 4. create submission
submission = pd.DataFrame({'id':test_df.index,
                          'Listening_Time_minutes': pred_test_fm3.Listening_Time_minutes})

submission.to_csv('submission.csv', index=False)


