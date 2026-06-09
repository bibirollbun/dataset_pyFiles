# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from wordcloud import WordCloud
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

import warnings
from pandas.errors import SettingWithCopyWarning
warnings.simplefilter(action = 'ignore', category = UserWarning)
warnings.filterwarnings(action = 'ignore', category = FutureWarning)
warnings.filterwarnings(action='ignore', category = RuntimeWarning)
warnings.filterwarnings(action='ignore', category = SettingWithCopyWarning)


#load dataset

train_df = pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv')





train_df.shape


test_df.shape


train_df.head()


test_df.head()


#concatenate train and test data to make it complete data
complete_df = pd.concat((train_df, test_df), axis=0)
complete_df.head()


complete_df.shape


complete_df.columns


#drop target varaibel(column)
complete_df = complete_df.drop(labels='Listening_Time_minutes', axis = 1)



complete_df.head()


print("INfo of Train Data")
train_df.info()


print("INfo of Test Data")
test_df.info()


#check for null values

train_df.isna().sum()


test_df.isna().sum()


#percentage of missing/nulll values
train_null_percentage = train_df.isna().sum()/len(train_df)*100
test_null_percentage = test_df.isna().sum()/len(test_df)*100

print("training data missing Percentage : \n", train_null_percentage)
print("\n----------------------------------\n")
print("testing data missing Percentage : \n", test_null_percentage)


pd.DataFrame({
    'train_null_perc': train_null_percentage,
    'test_null_perc': test_null_percentage
})


# maximum values of Episode Lenght, Number of ads, host popularity, listening Time

train_df['Episode_Length_minutes'].max()


train_df.describe()


train_df['Listening_Time_minutes'].describe()


complete_df.columns


complete_df.head()


complete_df.info()





podcast_name = complete_df.groupby(by='Podcast_Name')['Genre'].count().sort_values(ascending=False)
podcast_name


podcast_dict = {'Podcast Name': podcast_name.index,
'Count': podcast_name.values}

podcast_df = pd.DataFrame(podcast_dict)
podcast_df


#plot the podcast name with barplot
plt.figure(figsize = (14,15))
ax = sns.barplot(y = podcast_df['Podcast Name'],
            x = podcast_df['Count'],
            edgecolor = 'black',
                 palette = 'deep',
            linewidth=1.5,
            orient='h')
for bar in ax.containers:
    plt.bar_label(container = bar, fmt='%d', 
                  label_type='edge', padding=2, fontsize=10, color ='black')
plt.title('Top POdcast name')
plt.show()


plt.figure(figsize = (18, 9))

wordCloud = WordCloud(width=800,
         height=400,
         max_words = 50,
         random_state=2025,
         colormap='rainbow')

wordCloud = wordCloud.generate(text = ' '.join(complete_df['Podcast_Name']))
plt.imshow(wordCloud)
plt.title("WORDCLOUD for PODCAST NAME")


ep_duration = complete_df.groupby(by='Podcast_Name')['Episode_Length_minutes'].mean().sort_values(ascending=False).reset_index(drop=False).head(10)
ep_duration


plt.figure(figsize = (15,7))
ax = sns.barplot(data = ep_duration, x = 'Episode_Length_minutes', y='Podcast_Name',
           palette = 'bright')

for bar in ax.containers:
    plt.bar_label(container = bar, fmt='%.3f', 
                  label_type='edge', padding=2, fontsize=10, color ='black')
plt.title('Top 10 Podcasts as per episode length')
plt.show()


#box plot

plt.figure(figsize=(20,2))
sns.boxplot(data = complete_df,
           x = "Podcast_Name",
           y = "Episode_Length_minutes", 
           palette = 'dark')
plt.xticks(rotation=45)
plt.show()


listen_duration = train_df.groupby(by='Podcast_Name')['Listening_Time_minutes'].mean().sort_values(ascending=False).reset_index(drop=False).head(10)
listen_duration


plt.figure(figsize = (15,7))
ax = sns.barplot(data = listen_duration, x = 'Listening_Time_minutes', y='Podcast_Name',
           palette = 'deep')

for bar in ax.containers:
    plt.bar_label(container = bar, fmt='%.3f', 
                  label_type='center', padding=2, fontsize=10, color ='black')
plt.title('Top 10 Podcasts as maxium listening time')
plt.show()


complete_df.columns


complete_df.head()





complete_df_genre = complete_df['Genre'].sort_values(ascending=False)

plt.figure(figsize=(8,6))
ax = sns.countplot(x = complete_df_genre)

for bar in ax.containers:
    plt.bar_label(container = bar, fmt='%d', 
                  label_type='edge', padding=2, fontsize=8, color ='black')
plt.title('Genre distribution for complete data (train + test')
plt.xlabel('Genre')
plt.ylabel('Count')
plt.show()


# comparsion between train and test
train_genre = train_df['Genre'].sort_values(ascending=False)
test_genre = test_df['Genre'].sort_values(ascending=False)
plt.figure(figsize=(18,5))

plt.subplot(1, 3, 1)
ax1 = sns.countplot(x = train_genre)
for bar in ax1.containers:
    plt.bar_label(container = bar, fmt='%d', 
                  label_type='edge', padding=2, fontsize=8, color ='black')
plt.title('Genre distribution for Train data')
plt.xticks(rotation=45)

plt.subplot(1, 3, 2)
ax2 = sns.countplot(x = test_genre)
for bar in ax2.containers:
    plt.bar_label(container = bar, fmt='%d', 
                  label_type='edge', padding=2, fontsize=8, color ='black')
plt.title('Genre distribution for Test data')
plt.xticks(rotation=45)

plt.subplot(1, 3, 3)
ax3 = sns.countplot(x = complete_df_genre)
for bar in ax3.containers:
    plt.bar_label(container = bar, fmt='%d', 
                  label_type='edge', padding=2, fontsize=8, color ='black')
plt.title('Genre distribution for Complete data')
plt.xlabel('Genre')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()


# comparsion between train and test
train_pub_day = train_df['Publication_Day'].sort_values(ascending=False)
test_pub_day = test_df['Publication_Day'].sort_values(ascending=False)
complete_df_pub_day = complete_df['Publication_Day'].sort_values(ascending=False)

plt.figure(figsize=(18,5))

plt.subplot(1, 3, 1)
ax1 = sns.countplot(x = train_pub_day)
for bar in ax1.containers:
    plt.bar_label(container = bar, fmt='%d', 
                  label_type='edge', padding=2, fontsize=8, color ='black')
plt.title('Publication Day distribution for Train data')
plt.xticks(rotation=45)

plt.subplot(1, 3, 2)
ax2 = sns.countplot(x = test_pub_day)
for bar in ax2.containers:
    plt.bar_label(container = bar, fmt='%d', 
                  label_type='edge', padding=2, fontsize=8, color ='black')
plt.title('Publication Day distribution for Test data')
plt.xticks(rotation=45)

plt.subplot(1, 3, 3)
ax3 = sns.countplot(x = complete_df_pub_day)
for bar in ax3.containers:
    plt.bar_label(container = bar, fmt='%d', 
                  label_type='edge', padding=2, fontsize=8, color ='black')
plt.title('PUblication Day distribution for Complete data')

plt.xticks(rotation=45)
plt.show()


# comparsion between train and test
train_pub_time = train_df['Publication_Time'].sort_values(ascending=False)
test_pub_time = test_df['Publication_Time'].sort_values(ascending=False)

plt.figure(figsize=(18,5))

plt.subplot(1, 2, 1)
ax1 = sns.countplot(x = train_pub_time)
for bar in ax1.containers:
    plt.bar_label(container = bar, fmt='%d', 
                  label_type='edge', padding=2, fontsize=8, color ='black')
plt.title('Publication Time distribution for Train data')
plt.xticks(rotation=45)

plt.subplot(1, 2, 2)
ax2 = sns.countplot(x = test_pub_time)
for bar in ax2.containers:
    plt.bar_label(container = bar, fmt='%d', 
                  label_type='edge', padding=2, fontsize=8, color ='black')
plt.title('Publication Time distribution for Test data')
plt.xticks(rotation=45)

plt.xticks(rotation=45)
plt.show()


train_sentiment = train_df['Episode_Sentiment'].value_counts()
test_sentiment = test_df['Episode_Sentiment'].value_counts()
print(train_sentiment)

plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
#plot
plt.pie(train_sentiment, labels=train_sentiment.index, autopct = '%1.1f%%', shadow=True)
plt.title('Train Data')

plt.subplot(1,2,2)
plt.pie(test_sentiment, labels=test_sentiment.index, autopct = '%1.1f%%', shadow=True)
plt.title('Test Data')

plt.suptitle('1.7 - Sentiment Distribution of Train and Test')
plt.show()


train_df['Episode_Length_minutes'].value_counts()






plt.figure(figsize=(14,5))
plt.subplot(1,2,1)
#plot
sns.histplot(x = train_df['Episode_Length_minutes'], bins=20, color='red')
plt.title('Train Data')

plt.subplot(1,2,2)
sns.histplot(x = test_df['Episode_Length_minutes'], bins=20, color='green')
plt.title('Test Data')

plt.suptitle('1.8 - Episode Length Distribution of Train and Test')
plt.show()



plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
#plot
sns.boxplot(y = train_df['Episode_Length_minutes'])
plt.title('Train Data')

plt.subplot(1,2,2)
sns.boxplot(y = test_df['Episode_Length_minutes'])
plt.title('Test Data')

plt.suptitle('1.8 - OUtliers in Episode Length of Train and Test')
plt.show()


train_df.head()


plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
sns.histplot(data= train_df, x = 'Listening_Time_minutes', bins=100)
plt.title('Histogram of Listening Time')

plt.subplot(1,2,2)
sns.boxplot(data=train_df, y='Listening_Time_minutes')
plt.title('Box Plot of Listening Tiem')

plt.suptitle('Listening Time Distribution')


train_df.head()


#numerical Columns
num_column = train_df.select_dtypes(include='number').drop(columns='id')
num_column.head()


type(num_column)


if hasattr(num_column, 'to_pandas'):
    num_column = num_column.to_pandas()

corr_matrix = num_column.corr(method='spearman')[['Listening_Time_minutes']]
corr_matrix


sns.heatmap(
    corr_matrix.values,
    vmin=-1, 
    vmax = 1,
    annot=True,
    fmt='.2f',
    cmap='rainbow',
    xticklabels=corr_matrix.columns,
    yticklabels=corr_matrix.index,
    annot_kws={'size':10, 'weight':'bold'},
    cbar_kws = {'shrink':0.9, 'aspect':40}

)

plt.title('Correlation Matrix with Dependent Feature',
         fontdict={'size':12, 'weight':'bold'})
plt.show()


num_column = num_column.drop(columns='Listening_Time_minutes')



num_column


fig, axes = plt.subplots(2,2, figsize=(10,8))
for i, feature in enumerate(num_column):
    ax = axes[i//2, i%2]
    sns.scatterplot(
        x = train_df[feature],
        y=train_df['Listening_Time_minutes'],
        ax=ax
    )

    ax.set_title(f'Relation between {feature} and Listening Time',
                     fontsize=8)
    ax.set_xlabel(feature, fontsize=10, color='blue')
    
    ax.set_ylabel('Listening Time(minutes)', fontsize=10, color='blue')
    ax.tick_params(axis='x', rotation=45, labelsize=10, labelcolor='darkgreen')
    ax.tick_params(axis='y', rotation=0, labelsize=10, labelcolor='darkgreen')


plt.suptitle('Correlation of numerical features affecting LIstening Time')
plt.tight_layout()
plt.show()
    


sns.heatmap(data =num_column.corr(method='spearman'),
           annot=True, fmt='.2f',
           cmap = 'Spectral',
           linewidths = 0.5,
           linecolor='black',
           
    annot_kws={'size':10, 'weight':'bold'},
    cbar_kws = {'shrink':0.9, 'aspect':40})


plt.title('Mulitcolinearity')
plt.show()


genre = train_df.groupby(by='Genre')['Listening_Time_minutes'].mean().sort_values(ascending=False).reset_index(drop=False)
genre


type(genre)


ax = sns.barplot(data=genre,
           y= 'Listening_Time_minutes',
           x='Genre',
                 palette ='colorblind',
            edgecolor = 'black',
                linewidth=1.5)
plt.title('Genre with Average Longest Listening Time')


for bar in ax.containers:
    plt.bar_label(container = bar, fmt='%.2f', 
                  label_type='center', padding=3, fontsize=8, color ='black')

plt.xticks(rotation=45)
plt.show()


train_df.info()


cat_columns = train_df.select_dtypes(include='object')   #categorical Columns
cat_columns.columns


for i, feature in enumerate(cat_columns):
    d = train_df.groupby(by=feature)['Listening_Time_minutes'].mean().sort_values(ascending=False).reset_index(drop=False)
    print(d)


#2.3.2 Based on Episode sentiments
fig, axes = plt.subplots(3,2, figsize=(15,12))
for i, feature in enumerate(cat_columns):
    ax = axes[i//2, i%2]
    sns.barplot(
        data = train_df.groupby(by=feature)['Listening_Time_minutes'].mean().sort_values(ascending=False).reset_index(drop=False).head(10),
        y= feature,
        x='Listening_Time_minutes',
        ax=ax
    )
    for bar in ax.containers:
        plt.bar_label(container = bar, fmt='%.2f', 
                  label_type='center', padding=3, fontsize=8, color ='black')

    ax.set_title(f'Relation between {feature} and Listening Time',
                     fontsize=8)
   
    ax.tick_params(axis='x', rotation=45, labelsize=5, labelcolor='darkgreen')
    ax.tick_params(axis='y', rotation=0, labelsize=5, labelcolor='darkgreen')
    


plt.suptitle('Correlation of Categorical features affecting LIstening Time')
plt.show()
    


#2.3.3  Combine Day and Time together
combined_day_time = train_df.groupby(['Publication_Day', 'Publication_Time'])['Listening_Time_minutes'].mean().sort_values(ascending=False).reset_index()
combined_day_time

plt.figure(figsize=(12,6))

ax = sns.barplot(data = combined_day_time,
           x = 'Listening_Time_minutes',
           y='Publication_Day',
           hue='Publication_Time',
           palette='bright', 
           edgecolor='white',
           linewidth=1.5)

plt.title('Listening Duraion by Day and Time of publication Together')
plt.xlabel('Average Listenign Time(in min)')
plt.ylabel('Publication Day')

for bar in ax.containers:
        plt.bar_label(container = bar, fmt='%.2f', 
                  label_type='edge', padding=3, fontsize=8, color ='black')

plt.legend(title='Publication Time',bbox_to_anchor=(1.2,1), loc='upper right')
plt.tight_layout()
plt.show()


train_df.columns


train_df.info()


cat_columns.columns


train_df.isnull().sum()


train_df['Guest_Popularity_percentage'].mean()


train_df['Episode_Length_minutes'].isnull().sum()


#fill the missing values with median
train_df['Episode_Length_minutes'] = train_df['Episode_Length_minutes'].fillna(train_df['Episode_Length_minutes'].median())



train_df['Guest_Popularity_percentage'] = train_df['Guest_Popularity_percentage'].fillna(train_df['Guest_Popularity_percentage'].median())



train_df.isnull().sum()


#drop the null value for Number of Ads
train_df = train_df.dropna()


train_df.isnull().sum()


train_df.info()


(train_df['Episode_Length_minutes']>130).sum()


# drop the episodes whose lenght is greater than 130

train_df = train_df[train_df['Episode_Length_minutes']<=130]


train_df.head()



plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
#plot
sns.boxplot(y = train_df['Episode_Length_minutes'])
plt.title('Train Data')

plt.subplot(1,2,2)
sns.boxplot(y = test_df['Episode_Length_minutes'])
plt.title('Test Data')

plt.suptitle('1.8 - OUtliers in Episode Length of Train and Test')
plt.show()


for col in cat_columns.columns:
    print(f"{col} has unique values: {train_df[col].nunique()}")


categorical_features = [
    'Podcast_Name', 'Genre', 'Publication_Day',
    'Publication_Time', 'Episode_Sentiment'
]  #features with more number of unique values


from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()


le.fit_transform(train_df['Podcast_Name'])


for col in categorical_features:
    train_df[col] = le.fit_transform(train_df[col])
    test_df[col]  = le.transform(test_df[col])


train_df.head()


train_df.info()


test_df.info()


train_df.describe()


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()



num_column.columns


for col in num_column.columns:
    train_df[col] = sc.fit_transform(train_df[[col]])
    test_df[col] = sc.transform(test_df[[col]])
    


train_df.describe()


test_df.isnull().sum(0)


train_df.head()


train_df.columns


X = train_df.drop(['id','Podcast_Name','Episode_Title', 'Listening_Time_minutes'], axis = 1)
y = train_df['Listening_Time_minutes']


X.head()


y.head()


X.shape


y.shape


from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def get_regression_metrics(y_true, y_pred):
    """
    Returns regression metrics as a pandas DataFrame
    """
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    r2 = r2_score(y_true, y_pred)

    metrics = {
        'MAE': [mae],
        'MSE': [mse],
        'RMSE': [rmse],
        'R2 Score': [r2]
    }

    return pd.DataFrame(metrics)


from sklearn.model_selection import train_test_split


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


print(f'X_train shape: {X_train.shape}')
print(f'X_val shape: {X_val.shape}')
print(f'y_train shape: {y_train.shape}')
print(f'y_val shape: {y_val.shape}')



from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error


xgb = XGBRegressor(n_estimators = 100,
                  learning_rate = 0.1,
                  max_depth = 5, 
                  random_state = 42)

xgb.fit(X_train, y_train)


xgb_predict_val = xgb.predict(X_val)

#RMSE 
xgb_rmse = np.sqrt(mean_squared_error(y_val, xgb_predict_val))
xgb_rmse


get_regression_metrics(y_val, xgb_predict_val)


importance = xgb.feature_importances_
pd.DataFrame({'Feature': X_train.columns, 'Importance': importance})


from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor


rf_model = RandomForestRegressor(n_estimators = 100,
                  max_depth = None, 
                  random_state = 42)

rf_model.fit(X_train,y_train)


#Random Forest Prediction
rf_predict_val = rf_model.predict(X_val)

#RMSE 
rf_rmse = np.sqrt(mean_squared_error(y_val, rf_predict_val))
rf_rmse


get_regression_metrics(y_val, rf_predict_val)



gbr_model = GradientBoostingRegressor(n_estimators=100,
                         learning_rate = 0.1,
                         max_depth = 5
                         )

gbr_model.fit(X_train,y_train)


#Random Forest Prediction
gbr_predict_val = gbr_model.predict(X_val)

#RMSE 
gbr_rmse = np.sqrt(mean_squared_error(y_val, gbr_predict_val))
gbr_rmse


get_regression_metrics(y_val, gbr_predict_val)



# Now safely access feature importances
importances = gbr_model.feature_importances_
features = X_train.columns
# Create DataFrame for plotting
feat_imp_df = pd.DataFrame({'Feature': features, 'Importance': importances})
feat_imp_df.sort_values(by='Importance', ascending=False, inplace=True)

# Plot Top 20
plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=feat_imp_df.head(20))
plt.title("Top 20 Feature Importances from XGBoost")
plt.tight_layout()
plt.show()


from lightgbm import LGBMRegressor


lgbm_model = LGBMRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
lgbm_model.fit(X_train, y_train)

lgbm_predict_val = lgbm_model.predict(X_val)
lgbm_rmse = np.sqrt(mean_squared_error(y_val, lgbm_predict_val))

get_regression_metrics(y_val, lgbm_predict_val)



from sklearn.linear_model import LinearRegression, Ridge



# base model
base_model = [
    ('xgb', XGBRegressor(n_estimators = 100,
                  learning_rate = 0.1,
                  max_depth = 5, 
                  random_state = 42)),
                 
    ('lgbm', LGBMRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)),
    ('ridge', Ridge(alpha=1.0)),
    ('gbr',GradientBoostingRegressor(n_estimators=100,
                         learning_rate = 0.1,
                         max_depth = 5
                         ))

]


from sklearn.ensemble import StackingRegressor
#meta model 
meta_model = LinearRegression()

stacking = StackingRegressor(estimators = base_model,
                 final_estimator = meta_model,
                 n_jobs =-1, 
                 passthrough = False)

stacking.fit(X_train, y_train)


# Predict and evaluate using RMSE
# Predict on validation set and evaluate using RMSE
y_val_pred = stacking.predict(X_val)
rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
print(f"Stacking Regressor Validation RMSE: {rmse:.4f}")


test_df.head()



test_df.info()


test_df.isnull().sum()


test_df['Episode_Length_minutes'] = test_df['Episode_Length_minutes'].fillna(train_df['Episode_Length_minutes'].median())
test_df['Guest_Popularity_percentage'] = test_df['Guest_Popularity_percentage'].fillna(train_df['Guest_Popularity_percentage'].median())



test_df.isnull().sum()



num_cols = num_column.columns
num_cols


test_df.describe()


rf_model.predict(test_df)


X_train.columns


test_df.columns


test_df = test_df[X_train.columns]


#prediction
tests_prediciton = rf_model.predict(test_df)


test_df['Episode_Length_minutes'] = tests_prediciton


test_df['Episode_Length_minutes']





test_df2 = pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv')


test_df.shape


test_df['id'] = test_df2['id'].copy()


test_df


test_df[['id', 'Episode_Length_minutes']].to_csv("submission.csv", index =False)




