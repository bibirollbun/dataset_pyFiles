import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os, warnings
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
warnings.filterwarnings('ignore')


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from catboost import CatBoostRegressor, Pool
from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer


train_data=pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv')
test_data=pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv')
submissions=pd.read_csv('/kaggle/input/playground-series-s5e4/sample_submission.csv')


train_data.head()


test_data.shape


train_data.shape


train_data.describe()


train_data[train_data['Number_of_Ads']==103]


train_data[train_data['Listening_Time_minutes']==0].value_counts().sum()


train_data.info()


train_data.isnull().sum()


test_data.isnull().sum()


train_data['Listening_Time_minutes'].hist()
plt.show()


cat_features=[feature for feature in train_data.columns if train_data[feature].dtype=='O'and feature not in ['Episode_Title']]
cat_features


n_cols=2
n_rows=int(np.ceil(len(cat_features)/n_cols))

fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 4))
axes = axes.flatten() 

for idx, feature in enumerate(cat_features):
    data=train_data.copy()
    data.groupby(feature)['Listening_Time_minutes'].median().plot.bar(ax=axes[idx])
    axes[idx].set_title(feature)
    axes[idx].set_xlabel(feature)
    axes[idx].set_ylabel('Listening_Time_minutes')


for j in range(idx + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()


num_features = [feature for feature in train_data.columns if train_data[feature].dtype != 'O' and feature not in ['id']]
num_features


n_cols=2
n_rows=int(np.ceil(len(num_features)/2))

fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, n_rows * 4))
axes = axes.flatten()  # Flatten in case it's a 2D array

for idx,feature in enumerate(num_features):
    train_data[feature].hist(bins=25,ax=axes[idx])
    axes[idx].set_title(feature)
    axes[idx].set_xlabel(feature)
    axes[idx].set_ylabel('count')

for j in range(idx + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()


n_cols=2
n_rows=int(np.ceil(len(num_features)/2))

fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, n_rows * 4))
axes = axes.flatten()  # Flatten in case it's a 2D array

for idx,feature in enumerate(num_features):
    axes[idx].scatter(train_data[feature],train_data['Listening_Time_minutes'])
    axes[idx].set_title(feature)
    axes[idx].set_xlabel(feature)
    axes[idx].set_ylabel('Listening_Time_minutes')

for j in range(idx + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()


sns.boxplot(x='Episode_Length_minutes',y='Genre',data=train_data)
plt.title("BoxPlot of Episode Length Minutes")


#train_data['Guest_Popularity_percentage'].hist()
sns.boxplot(x='Guest_Popularity_percentage', data=train_data)
plt.show()


#filling missing values
#train_data['Episode_Length_minutes']=train_data['Episode_Length_minutes'].fillna(train_data['Episode_Length_minutes'].mean()) instead of mean I am using different stratergy
train_data['Episode_Length_minutes'] = train_data.groupby('Genre')['Episode_Length_minutes'].transform(
    lambda x: x.fillna(x.median()))

train_data['Guest_Popularity_percentage']=train_data['Guest_Popularity_percentage'].fillna(train_data['Guest_Popularity_percentage'].median())
train_data['Number_of_Ads']=train_data['Number_of_Ads'].fillna(0)


#filling missing values in test data 
test_data['Episode_Length_minutes'] = test_data.groupby('Genre')['Episode_Length_minutes'].transform(
    lambda x: x.fillna(x.median()))
test_data['Guest_Popularity_percentage']=test_data['Guest_Popularity_percentage'].fillna(test_data['Guest_Popularity_percentage'].median())


train_data.isnull().sum()


def cap_outliers(df, features):
    df = df.copy()
    for feature in features:
        lower = df[feature].quantile(0.01)
        upper = df[feature].quantile(0.99)
        df[feature] = df[feature].clip(lower=lower, upper=upper)
    return df


outlier_features=['Episode_Length_minutes','Number_of_Ads','Listening_Time_minutes']
df_capped = cap_outliers(train_data, outlier_features)
outlier_features_test=['Episode_Length_minutes','Number_of_Ads']
test_df=cap_outliers(test_data,outlier_features_test)


#making new features on capped and replaced data set to capture the interactions between features
def preprocess_and_engineer(df):
    df = df.copy()
    df.drop(['Episode_Title'],axis=1,inplace=True)
    # Ad rate (ads per minute)
    df['Ad_Rate'] = df['Number_of_Ads'] / (df['Episode_Length_minutes'] + 0.01)
    
    # Host - Guest popularity gap
    df['Host_Guest_Gap'] = df['Host_Popularity_percentage'] - df['Guest_Popularity_percentage']
    
    # Episode length category
    df['Is_Short_Episode'] = (df['Episode_Length_minutes'] < 10).astype(int)
    
    # Encode Sentiment
    sentiment_map = {'Negative': 0, 'Neutral': 1, 'Positive': 2}
    df['Episode_Sentiment_Encoded'] = df['Episode_Sentiment'].map(sentiment_map)

    # --- Interactions ---
    df['Ads_x_Sentiment'] = df['Number_of_Ads'] * df['Episode_Sentiment_Encoded']
    df['Host_x_Guest'] = df['Host_Popularity_percentage'] * df['Guest_Popularity_percentage']
    df['Length_x_Sentiment'] = df['Episode_Length_minutes'] * df['Episode_Sentiment_Encoded']
    df['Length_x_Ads'] = df['Episode_Length_minutes'] * df['Number_of_Ads']

    # --- One-hot encode small categoricals ---
    df = pd.get_dummies(df, columns=['Publication_Time','Publication_Day', 'Genre'], drop_first=True,dtype=int)
    
    return df



df_capped=preprocess_and_engineer(df_capped)

le=LabelEncoder()
df_capped['Podcast_Name']=le.fit_transform(df_capped['Podcast_Name'])
df_capped.drop(['Episode_Sentiment'],axis=1,inplace=True)

df_capped.head()


test_df=preprocess_and_engineer(test_df)

test_df['Podcast_Name']=le.transform(test_df['Podcast_Name'])
test_df.drop(['Episode_Sentiment'],axis=1,inplace=True)
test_df.head()


for feature in cat_features:
    print("Number of unique {} {}".format(feature,len(train_data[feature].unique())))


train_data.drop(['Episode_Title'],axis=1,inplace=True)
test_data.drop(['Episode_Title'],axis=1,inplace=True)


train_encoded=train_data.copy()
test_encoded=test_data.copy()

# using label encoding for podcast name
le=LabelEncoder()
train_encoded['Podcast_Name']=le.fit_transform(train_encoded['Podcast_Name'])
test_encoded['Podcast_Name']=le.transform(test_encoded['Podcast_Name'])

train_encoded['Episode_Sentiment'].unique()

#Mapping sentiment
sentiment_mapping={'Negative':0,'Neutral':1,'Positive':2}
train_encoded['Episode_Sentiment']=train_encoded['Episode_Sentiment'].map(sentiment_mapping)
test_encoded['Episode_Sentiment']=test_encoded['Episode_Sentiment'].map(sentiment_mapping)

#One hot encoding on Genre, publication_day, publication_time
train_encoded = pd.get_dummies(train_encoded, columns=['Genre', 'Publication_Day', 'Publication_Time'],dtype=int, drop_first=True)
test_encoded = pd.get_dummies(test_encoded, columns=['Genre', 'Publication_Day', 'Publication_Time'],dtype=int, drop_first=True)


X_train,X_val,y_train,y_val=train_test_split(df_capped.drop(['Listening_Time_minutes'],axis=1),df_capped['Listening_Time_minutes'],test_size=0.2,random_state=42)


xgb_model = XGBRegressor(
    n_estimators=800, 
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=5,   # L2
    reg_alpha=0.5,  # L1
    random_state=42)
xgb_model.fit(X_train, y_train)
xgb_preds = xgb_model.predict(X_val)
xgb_rmse = np.sqrt(mean_squared_error(y_val, xgb_preds))
print(f"XGBoost RMSE: {xgb_rmse:.4f}")

#CV scores
rmse_scorer = make_scorer(mean_squared_error, squared=False)
scores = cross_val_score(
    xgb_model,
    df_capped.drop(['Listening_Time_minutes'], axis=1),
    df_capped['Listening_Time_minutes'],
    scoring=rmse_scorer,
    cv=5
)

print(f"XGBoost CV RMSE: {np.mean(scores):.4f}")


y_pred=xgb_model.predict(test_df)


submissions['Listening_Time_minutes']=y_pred


submissions.to_csv('submission.csv', index=False)
print("Submission file saved successfully!")

