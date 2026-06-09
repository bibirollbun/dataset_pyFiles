import pandas as pd
import numpy as np
import warnings
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio  
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor,Pool
import optuna
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import LabelEncoder
from plotly.subplots import make_subplots
import plotly.subplots as sp
import plotly.figure_factory as ff  
pio.renderers.default = 'iframe_connected'
warnings.filterwarnings("ignore", category=FutureWarning)


warnings.filterwarnings("ignore", category=DeprecationWarning)
msgs = [
    'invalid value encountered in greater',
    'invalid value encountered in less'
]
for msg in msgs:
    warnings.filterwarnings('ignore', category=RuntimeWarning, message=msg)


train = pd.read_csv("/kaggle/input/playground-series-s5e4/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e4/test.csv")
submission = pd.read_csv("/kaggle/input/playground-series-s5e4/sample_submission.csv")


train.head()


train.info()


train.describe().T


train.isnull().sum()


object_column_names = train.select_dtypes(include=['object']).columns
print("Object Column Names:", object_column_names.tolist())


numerical_column_names = train.select_dtypes(include=['number']).columns
print("Numerical Column Names:", numerical_column_names.tolist())


print("Duplicated Rows:",train.duplicated().sum())

print("Number of Rows:",train.shape[0])

print("Number of Columns:",train.shape[1])


train.nunique()


cat_cols = ['Podcast_Name', 'Episode_Title', 'Genre', 'Publication_Day', 
            'Publication_Time', 'Episode_Sentiment']


for i in cat_cols:
    print(f"Column'{i}' Unique Values")
    print(train[i].unique())
    print("-"*20)


sns.histplot(train['Listening_Time_minutes'], kde=True)
plt.title('Distribution of Listening Time (minutes)')
plt.show()


episode_count = train['Episode_Sentiment'].value_counts().reset_index()
episode_count.columns = ['Sentiment','Count']

ice_palette = ['#d6f0f5', '#a0d9e8', '#72c3dc', '#45add6', '#228cc9']

fig = px.pie(
    episode_count,
    names ='Sentiment',
    values = "Count",
    color = 'Sentiment',
    color_discrete_sequence = ice_palette,
    title = "Episode_Sentiment Distribution"
)
fig.update_traces(textinfo = 'percent+label')
fig.update_layout(width = 600, height=500)

fig.show()


publication_count = train['Publication_Time'].value_counts().reset_index()
publication_count.columns = ['Publication_time','Count']

fig = px.bar(
    publication_count,
    x = 'Publication_time',
    y = 'Count',
    color = 'Publication_time',
    color_discrete_sequence = ice_palette,
    title = "Publication Time Distribution"
)

fig.update_layout(width = 700, height=500)

fig.show()


day_count = train['Publication_Day'].value_counts().reset_index()
day_count.columns = ['Publication_Day','Count']

fig = px.bar(
    day_count,
    x = 'Publication_Day',
    y = 'Count',
    color = 'Publication_Day',
    color_discrete_sequence = ice_palette,
    title = "Publication Day Distribution"
)

fig.update_layout(width = 700, height=500)

fig.show()


genre_count = train['Genre'].value_counts().reset_index()
genre_count.columns = ['Genre','Count']

fig = px.bar(
    genre_count,
    x = 'Genre',
    y = 'Count',
    color = 'Genre',
    color_discrete_sequence = ice_palette,
    title = "Genre Distribution"
)

fig.update_layout(width = 700, height=500)

fig.show()


print(train['Episode_Title'].value_counts())


episode_count = train['Episode_Title'].value_counts().reset_index().head(10)
episode_count.columns = ['Episode_Title', 'Count']

episode_count = episode_count.sort_values('Count', ascending=False)

fig = px.bar(
    episode_count,
    x='Episode_Title',
    y='Count',
    color='Episode_Title',
    color_discrete_sequence=ice_palette,
    title="Top 10 Episode Titles",
    category_orders={'Episode_Title': episode_count['Episode_Title'].tolist()}
)

fig.update_layout(width=700, height=500)
fig.show()



print(train['Podcast_Name'].value_counts())


podcast_count = train['Podcast_Name'].value_counts().reset_index().head(10)
podcast_count.columns = ['Podcast_Name', 'Count']

podcast_count = podcast_count.sort_values('Count', ascending=False)

fig = px.bar(
    podcast_count,
    x='Podcast_Name',
    y='Count',
    color='Podcast_Name',
    color_discrete_sequence=ice_palette,
    title="Top 10 Podcast_Name",
    category_orders={'Podcast_Name': podcast_count['Podcast_Name'].tolist()}
)

fig.update_layout(width=700, height=500)
fig.show()


genre_avg = train.groupby('Genre')['Listening_Time_minutes'].mean().reset_index().round(2)
genre_avg = genre_avg.sort_values(by='Listening_Time_minutes', ascending=False)

fig = px.bar(
    genre_avg,
    x='Genre',
    y='Listening_Time_minutes',
    color='Genre',
    title='Genre-wise Average Listening Time',
    labels={'Listening_Time_minutes': 'Avg Listening Time (min)'},
    color_discrete_sequence=ice_palette,
)

fig.update_layout(
    xaxis_title='Genre',
    yaxis_title='Average Listening Time (minutes)',
    showlegend=False,
    height=500,
    width=700
)

fig.show()


scatter_data = train.dropna(subset=['Episode_Length_minutes', 'Listening_Time_minutes'])

fig = px.scatter(
    scatter_data,
    x='Episode_Length_minutes',
    y='Listening_Time_minutes',
    title='Impact of Episode Length on Listening Time',
    labels={
        'Episode_Length_minutes': 'Episode Length (minutes)',
        'Listening_Time_minutes': 'Listening Time (minutes)'
    },
    trendline='ols',  
    opacity=0.6,
    color_discrete_sequence=['#a1c9f4']  
)

fig.update_layout(
    height=500,
    width=700
)

fig.show()


day_avg = train.groupby('Publication_Day')['Listening_Time_minutes'].mean().reset_index().round(2)

day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
day_avg['Publication_Day'] = pd.Categorical(day_avg['Publication_Day'], categories=day_order, ordered=True)
day_avg = day_avg.sort_values('Publication_Day')

fig = px.bar(
    day_avg,
    x='Publication_Day',
    y='Listening_Time_minutes',
    title='Average Listening Time by Day of the Week',
    labels={
        'Publication_Day': 'Day of the Week',
        'Listening_Time_minutes': 'Average Listening Time (minutes)'
    },
    color='Publication_Day',
    color_discrete_sequence=ice_palette
)

fig.update_layout(
    xaxis_title='Day',
    yaxis_title='Avg Listening Time (minutes)',
    height=500,
    width=800,
    showlegend=False
)

fig.show()


numerical_cols = [
    'Episode_Length_minutes',
    'Host_Popularity_percentage',
    'Guest_Popularity_percentage',
    'Number_of_Ads',
    'Listening_Time_minutes'
]


sns.pairplot(train[numerical_cols])
plt.suptitle("Distribution of Numerical Values" , y=0.02)
plt.show()


genre_mapping = {
    'True Crime': 0, 'Comedy': 1, 'Education': 2, 'Technology': 3, 'Health': 4,
    'News': 5, 'Music': 6, 'Sports': 7, 'Business': 8, 'Lifestyle': 9
}

publication_day_mapping = {
    'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3, 
    'Friday': 4, 'Saturday': 5, 'Sunday': 6
}

publication_time_mapping = {
    'Morning': 0, 'Afternoon': 1, 'Evening': 2, 'Night': 3
}

episode_sentiment_mapping = {
    'Positive': 0, 'Negative': 1, 'Neutral': 2
}


podcast_name_mapping = {
    'Mystery Matters': 0, 'Joke Junction': 1, 'Study Sessions': 2, 'Digital Digest': 3,
    'Mind & Body': 4, 'Fitness First': 5, 'Criminal Minds': 6, 'News Roundup': 7,
    'Daily Digest': 8, 'Music Matters': 9, 'Sports Central': 10, 'Melody Mix': 11, 'Game Day': 12,
    'Gadget Geek': 13, 'Global News': 14, 'Tech Talks': 15, 'Sport Spot': 16, 'Funny Folks': 17,
    'Sports Weekly': 18, 'Business Briefs': 19, 'Tech Trends': 20, 'Innovators': 21,
    'Health Hour': 22, 'Comedy Corner': 23, 'Sound Waves': 24, 'Brain Boost': 25,
    "Athlete's Arena": 26, 'Wellness Wave': 27, 'Style Guide': 28, 'World Watch': 29, 'Humor Hub': 30,
    'Money Matters': 31, 'Healthy Living': 32, 'Home & Living': 33, 'Educational Nuggets': 34,
    'Market Masters': 35, 'Learning Lab': 36, 'Lifestyle Lounge': 37, 'Crime Chronicles': 38,
    'Detective Diaries': 39, 'Life Lessons': 40, 'Current Affairs': 41, 'Finance Focus': 42,
    'Laugh Line': 43, 'True Crime Stories': 44, 'Business Insights': 45, 'Fashion Forward': 46,
    'Tune Time': 47
}


train["Episode_Title"] = train["Episode_Title"].str.replace("Episode ", "", regex=False).astype(int)
test["Episode_Title"] = test["Episode_Title"].str.replace("Episode ", "", regex=False).astype(int)


train['Genre'] = train['Genre'].map(genre_mapping)
test['Genre'] = test['Genre'].map(genre_mapping)

train['Publication_Day'] = train['Publication_Day'].map(publication_day_mapping)
test['Publication_Day'] = test['Publication_Day'].map(publication_day_mapping)

train['Publication_Time'] = train['Publication_Time'].map(publication_time_mapping)
test['Publication_Time'] = test['Publication_Time'].map(publication_time_mapping)

train['Episode_Sentiment'] = train['Episode_Sentiment'].map(episode_sentiment_mapping)
test['Episode_Sentiment'] = test['Episode_Sentiment'].map(episode_sentiment_mapping)


train['Podcast_Name'] = train['Podcast_Name'].map(podcast_name_mapping)
test['Podcast_Name'] = test['Podcast_Name'].map(podcast_name_mapping)


train['Number_of_Ads'].fillna(train['Number_of_Ads'].median(), inplace=True)

train['Episode_Length_minutes'].fillna(train['Episode_Length_minutes'].median(), inplace=True)
test['Episode_Length_minutes'].fillna(test['Episode_Length_minutes'].median(), inplace=True)

train['Guest_Popularity_percentage'].fillna(train['Guest_Popularity_percentage'].median(), inplace=True)
test['Guest_Popularity_percentage'].fillna(test['Guest_Popularity_percentage'].median(), inplace=True)


train['Ads_per_minute'] = train['Number_of_Ads']/train['Episode_Length_minutes']
test['Ads_per_minute'] = test['Number_of_Ads']/test['Episode_Length_minutes']


epsilon = 1e-6  

def add_engineered_features(data):
    data['Host_Popularity_per_Minute'] = data['Host_Popularity_percentage'] / (data['Episode_Length_minutes'] + epsilon)
    data['Guest_Popularity_per_Minute'] = data['Guest_Popularity_percentage'] / (data['Episode_Length_minutes'] + epsilon)

    data['Ads_per_Minute'] = data['Number_of_Ads'] / (data['Episode_Length_minutes'] + epsilon)

    data['Host_Guest_Popularity_Ratio'] = (data['Host_Popularity_percentage'] + epsilon) / (data['Guest_Popularity_percentage'] + epsilon)
    data['Guest_Host_Popularity_Ratio'] = (data['Guest_Popularity_percentage'] + epsilon) / (data['Host_Popularity_percentage'] + epsilon)

    data['Total_Popularity'] = data['Host_Popularity_percentage'] + data['Guest_Popularity_percentage']

    data['Popularity_Difference'] = data['Host_Popularity_percentage'] - data['Guest_Popularity_percentage']

    data['Ads_per_Host_Popularity'] = data['Number_of_Ads'] / (data['Host_Popularity_percentage'] + epsilon)
    data['Ads_per_Guest_Popularity'] = data['Number_of_Ads'] / (data['Guest_Popularity_percentage'] + epsilon)

    return data


train = add_engineered_features(train)
test = add_engineered_features(test)


train.head()


if 'Listening_Time_minutes' not in numerical_cols:
    numerical_cols.append('Listening_Time_minutes')

corr_matrix = train[numerical_cols].corr()

target_corr = corr_matrix['Listening_Time_minutes']

target_corr_sorted = target_corr.sort_values(ascending=False)

print(target_corr_sorted)


numerical_cols = train.select_dtypes(include=['float64', 'int64']).columns
corr_matrix = train[numerical_cols].corr()

plt.figure(figsize=(12, 12))
sns.heatmap(corr_matrix, annot=True, cmap='RdBu', fmt='.2f', linewidths=0.5)
plt.title('Correlation Matrix of Numerical Features', fontsize=14)
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()


X = train.drop(['Listening_Time_minutes'], axis=1)
y = train['Listening_Time_minutes']


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

rmse_scorer = make_scorer(rmse, greater_is_better=False)


def objective(trial):
    # Hyperparameter search space
    params = {
        'iterations': trial.suggest_int('iterations', 500, 3000),
        'depth': trial.suggest_int('depth', 4, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-2, 10.0, log=True),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
        'random_strength': trial.suggest_float('random_strength', 1e-3, 10.0, log=True),
        'border_count': trial.suggest_int('border_count', 32, 255),
        'task_type': 'GPU',
        'devices': '0',
        'verbose': 0,
        'random_state': 42
    }

    # KFold validation inside Optuna
    n_splits = 3
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    rmse_scores = []

    for train_idx, valid_idx in kf.split(X, y):
        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

        model = CatBoostRegressor(**params)
        model.fit(X_train, y_train, eval_set=(X_valid, y_valid), use_best_model=False)

        preds = model.predict(X_valid)
        rmse = mean_squared_error(y_valid, preds, squared=False)
        rmse_scores.append(rmse)

    return np.mean(rmse_scores)



#study = optuna.create_study(direction='minimize')
#study.optimize(objective, n_trials=5) 


#print("âœ… Best trial RMSE:", study.best_value)
#print("ğŸ�† Best hyperparameters:", study.best_trial.params)


catboost_params = {
    'iterations': 2975,
    'depth': 8,
    'learning_rate': 0.11695956589495782,
    'l2_leaf_reg': 0.05083738905484144,
    'bagging_temperature': 0.8059912126709423,
    'random_strength': 0.005287161384835767,
    'border_count': 252,
    'task_type': 'GPU',
    'devices': '0',
    'random_state': 42,
    'verbose': 0
}


n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

# Out-of-fold and test predictions
oof_predictions = np.zeros(len(X))
test_predictions = np.zeros(len(test))
rmse_scores = []

# K-Fold loop
for fold, (train_idx, valid_idx) in enumerate(kf.split(X, y)):
    print(f"ğŸ”� Fold {fold + 1}/{n_splits}")

    X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
    y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

    model = CatBoostRegressor(**catboost_params)
    model.fit(X_train, y_train, eval_set=(X_valid, y_valid), use_best_model=False)

    # Predict validation and test
    oof_pred = model.predict(X_valid)
    oof_predictions[valid_idx] = oof_pred
    test_predictions += model.predict(test) / n_splits

    # Fold RMSE
    rmse = mean_squared_error(y_valid, oof_pred, squared=False)
    rmse_scores.append(rmse)
    print(f"âœ… Fold {fold + 1} RMSE: {rmse:.4f}")

# Final RMSE
mean_rmse = np.mean(rmse_scores)
print(f"\nğŸ“Š Average RMSE across {n_splits} folds: {mean_rmse:.4f}")


rmse_df = pd.DataFrame({
    'Fold': [f'Fold {i+1}' for i in range(n_splits)],
    'RMSE': rmse_scores
})

sns.set(style='whitegrid')

plt.figure(figsize=(8, 6))
sns.lineplot(x='Fold', y='RMSE', data=rmse_df, marker='o', linewidth=2.5, color='mediumvioletred')

plt.title('RMSE Across K-Folds', fontsize=16)
plt.xlabel('Fold')
plt.ylabel('RMSE')

plt.axhline(mean_rmse, color='gray', linestyle='--', label=f'Average RMSE: {mean_rmse:.4f}')
plt.legend()
plt.tight_layout()
plt.show()


submission = pd.DataFrame({'id': test['id'], 'prediction': test_predictions})


submission.head()


submission.to_csv('submission.csv', index=False)
print("File Saved!!")

