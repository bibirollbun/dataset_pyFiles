# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler
import plotly.express as px
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression,Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from joblib import Parallel, delayed
from sklearn.base import clone

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


train = pd.read_csv("/kaggle/input/playground-series-s5e4/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e4/test.csv")
original = pd.read_csv("/kaggle/input/podcast-listening-time-prediction-dataset/podcast_dataset.csv")
submission = pd.read_csv("/kaggle/input/playground-series-s5e4/sample_submission.csv")


train.sample(5)


original.head(2)


test.head(2)


train.isnull().sum()


test.isnull().sum()


original.isnull().sum()


(train.shape), (original.shape), (test.shape)


original.dropna(axis=0, inplace = True, ignore_index=True)


original.shape


datasets = [train, test]

for dataset in datasets:
    dataset = dataset.drop(columns = ['id'], axis = 1, inplace = True)


(train.shape), (original.shape), (test.shape)


train = pd.concat([train, original], ignore_index=True)


train.shape


train.duplicated().sum()


train.drop_duplicates(inplace = True, ignore_index = True)


train.shape


train.describe().T


for i in train.columns:

    BOLD = "\033[1m"
    RESET = "\033[0m"
    print(BOLD + i + RESET)

    print(train[i].nunique(), '\n')


for i in train.columns:

    BOLD = "\033[1m"
    RESET = "\033[0m"
    print(BOLD + i + RESET)

    print(train[i].unique(), '\n')


train.Number_of_Ads.describe()


train.isnull().sum()


datasets = [train, test]


for dataset in datasets:
    dataset["Episode_Length_minutes"] = dataset["Episode_Length_minutes"].fillna(dataset["Episode_Length_minutes"].median())


train["Number_of_Ads"] = train["Number_of_Ads"].fillna(train["Number_of_Ads"].mode()[0])


median_guest_popularity = train["Guest_Popularity_percentage"].median()

for dataset in datasets:
    dataset.loc[(dataset["Guest_Popularity_percentage"].isnull()) & (dataset["Number_of_Ads"] > 0), "Guest_Popularity_percentage"] = median_guest_popularity
    dataset.loc[(dataset["Guest_Popularity_percentage"].isnull()) & (dataset["Number_of_Ads"] == 0), "Guest_Popularity_percentage"] = 0


test.isnull().sum()


num_cols = ['Episode_Length_minutes', 'Host_Popularity_percentage', 
            'Guest_Popularity_percentage', 'Number_of_Ads']


plt.figure(figsize=(10, 6))
for i, col in enumerate(num_cols, 1):
    plt.subplot(2, 2, i)
    sns.boxplot(x=train[col])
    plt.title(f'Boxplot of {col}')
plt.tight_layout()
plt.show()


def detect_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outliers = df[(df[column] < lower) | (df[column] > upper)]
    return outliers, lower, upper

for col in num_cols:
    outliers, lowerlim, upperlim = detect_outliers_iqr(train, col)
    print(f"Nos. of outliers in {col}: {len(outliers)}")


for col in num_cols:
    outliers, lowerlim, upperlim = detect_outliers_iqr(test, col)
    print(f"Nos. of outliers in {col}: {len(outliers)}")


for df_name, df in [("train", train), ("test", test)]:
    for col in ["Host_Popularity_percentage", "Guest_Popularity_percentage"]:
        count = (df[col] > 100).sum()
        print(f"{df_name}.{col} > 100: {count}")
    print("\n")


for dataset in datasets:
    dataset['Host_Popularity_percentage'] = dataset['Host_Popularity_percentage'].clip(upper=100)
    dataset['Guest_Popularity_percentage'] = dataset['Guest_Popularity_percentage'].clip(upper=100)


for col in ['Episode_Length_minutes', 'Number_of_Ads']:
    _, lower, upper = detect_outliers_iqr(train, col)
    train = train[(train[col] >= lower) & (train[col] <= upper)]   # Remove outliers from train dataset
    test[col] = test[col].clip(lower=lower, upper=upper)     # cap outliers of test dataset


train['Episode_Number'] = train['Episode_Title'].str.split().str[-1].astype(int)
test['Episode_Number'] = test['Episode_Title'].str.extract(r'(\d+)').astype(int) #more reliable even extract number from words.
                                                                              # "Episode #98: The Finale" ➜ still gives you 98


for df in [train, test]:
    df['Host_Guest_Interaction'] = df['Host_Popularity_percentage'] * df['Guest_Popularity_percentage']
    df['Length_per_Ad'] = df['Episode_Length_minutes'] / (df['Number_of_Ads'] + 1)


train.head(2)


train_encoded = train.copy()
test_encoded = test.copy()


le = LabelEncoder()
train_encoded['Podcast_Name'] = le.fit_transform(train_encoded['Podcast_Name'])
test_encoded['Podcast_Name'] = le.transform(test_encoded['Podcast_Name'])


one_hot_cols = ['Genre', 'Publication_Time']

train_encoded = pd.get_dummies(train_encoded, columns = one_hot_cols, dtype = int)
test_encoded = pd.get_dummies(test_encoded, columns = one_hot_cols, dtype = int)


train_encoded.head(2)


train_encoded['Publication_Day'] = le.fit_transform(train_encoded['Publication_Day'])
test_encoded['Publication_Day'] = le.transform(test_encoded['Publication_Day'])


sentiment_map = {'Negative': -1, 'Neutral': 0, 'Positive': 1}
train_encoded['Episode_Sentiment'] = train_encoded['Episode_Sentiment'].map(sentiment_map)
test_encoded['Episode_Sentiment'] = test_encoded['Episode_Sentiment'].map(sentiment_map)


train_encoded, test_encoded = train_encoded.align(test_encoded, join='left', axis=1)

test_encoded.fillna(0, inplace=True) # Listening_Time_minutes has all nan values after aligning 


pd.set_option('display.max_columns', None)
train_encoded.sample(2)


train_encoded.info()


numeric_cols = [
    'Episode_Length_minutes',
    'Host_Popularity_percentage',
    'Guest_Popularity_percentage',
    'Number_of_Ads',
    'Episode_Sentiment',
    'Listening_Time_minutes',
    'Episode_Number',
    'Host_Guest_Interaction',
    'Length_per_Ad'
]

corr_matrix = train_encoded[numeric_cols].corr()

# Plot the heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', center=0,
            square=True, linewidths=0.5, cbar_kws={"shrink": 0.75})  # this second line for shrink the output figure

plt.title("Correlation Heatmap of Numeric Features", fontsize=16)
plt.tight_layout()
plt.show()


features_to_scale = [
    'Episode_Length_minutes',
    'Host_Popularity_percentage',
    'Guest_Popularity_percentage',
    'Host_Guest_Interaction',
    'Length_per_Ad'
]
scaler = StandardScaler()

train_scaled = train_encoded.copy()
test_scaled = test_encoded.copy()

train_scaled[features_to_scale] = scaler.fit_transform(train_encoded[features_to_scale])
test_scaled[features_to_scale] = scaler.transform(test_encoded[features_to_scale])


plt.figure(figsize=(8, 5))
sns.histplot(train_encoded['Listening_Time_minutes'], bins=50, kde=True)
plt.title('Distribution of Listening Time (minutes)')
plt.xlabel('Listening Time (minutes)')
plt.ylabel('Frequency')
plt.grid(True)
plt.tight_layout()
plt.show()


num_cols = [
    'Episode_Length_minutes', 'Host_Popularity_percentage',
    'Guest_Popularity_percentage', 'Number_of_Ads', 'Episode_Number',
    'Host_Guest_Interaction', 'Length_per_Ad'
]

for col in num_cols:
    plt.figure(figsize=(6, 4))
    sns.histplot(train[col], kde=True, bins=40)
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.show()


cat_cols = ['Genre', 'Publication_Day', 'Publication_Time', 'Episode_Sentiment']

for col in cat_cols:
    plt.figure(figsize=(6, 4))
    sns.boxplot(data=train, x=col, y='Listening_Time_minutes')
    plt.title(f'Listening Time by {col}')
    plt.xticks(rotation=45)
    plt.show()


avg_ads = train.groupby('Number_of_Ads')['Listening_Time_minutes'].mean().reset_index()

plt.figure(figsize=(6, 4))
sns.barplot(data=avg_ads, x='Number_of_Ads', y='Listening_Time_minutes')
plt.title('Average Listening Time by Number of Ads')
plt.ylabel('Avg Listening Time (minutes)')
plt.xlabel('Number of Ads')
plt.show()


plt.figure(figsize=(6, 4))
genre_means = train.groupby('Genre')['Listening_Time_minutes'].mean().sort_values(ascending=False)
sns.barplot(x=genre_means.index, y=genre_means.values)
plt.title('Average Listening Time by Genre')
plt.ylabel('Listening Time (minutes)')
plt.xticks(rotation=45)
plt.show()


plt.figure(figsize=(6, 4))
day_means = train.groupby('Publication_Day')['Listening_Time_minutes'].mean()
sns.barplot(x=day_means.index, y=day_means.values)
plt.title('Average Listening Time by Day of the Week')
plt.ylabel('Listening Time (minutes)')
plt.xlabel('Day of the Week (0=Monday)')
plt.show()


plt.figure(figsize=(6, 4))
time_means = train.groupby('Publication_Time')['Listening_Time_minutes'].mean().sort_values(ascending=False)
sns.barplot(x=time_means.index, y=time_means.values)
plt.title('Average Listening Time by Time of Day')
plt.ylabel('Listening Time (minutes)')
plt.show()


plt.figure(figsize=(6, 4))
sentiment_means = train.groupby('Episode_Sentiment')['Listening_Time_minutes'].mean()
sns.barplot(x=sentiment_means.index, y=sentiment_means.values)
plt.title('Average Listening Time by Episode Sentiment')
plt.xlabel('Sentiment (-1 = Negative, 0 = Neutral, 1 = Positive)')
plt.ylabel('Listening Time (minutes)')
plt.show()


plt.figure(figsize=(14, 6))
podcast_means = train.groupby('Podcast_Name')['Listening_Time_minutes'].mean().sort_values(ascending=False)
sns.barplot(x=podcast_means.index.astype(str), y=podcast_means.values)
plt.title('Average Listening Time by Podcast Name')
plt.ylabel('Listening Time (minutes)')
plt.xlabel('Podcast Name ID')
plt.xticks(rotation=90)
plt.show()


sns.pairplot(train[numeric_cols].sample(5000), diag_kind='kde')
plt.show()


numeric_cols = [
    'Episode_Length_minutes', 'Host_Popularity_percentage',
    'Guest_Popularity_percentage', 'Number_of_Ads', 'Episode_Number',
    'Host_Guest_Interaction', 'Length_per_Ad', 'Listening_Time_minutes'
]

plt.figure(figsize=(10, 8))
sns.heatmap(train[numeric_cols].corr(), annot=True, cmap='coolwarm', fmt=".2f",
            square=True, linewidths=0.5, cbar_kws={"shrink": 0.75})
plt.title("Correlation Matrix")
plt.show()


pd.crosstab(train['Publication_Day'], train['Episode_Sentiment'], normalize='index') * 100


pd.crosstab(train['Genre'], train['Episode_Sentiment'], normalize='index') * 100


train.groupby(['Genre', 'Episode_Sentiment'])['Listening_Time_minutes'].mean().unstack().plot(kind='bar', figsize=(12,6))
plt.title('Mean Listening Time by Genre and Sentiment')
plt.ylabel('Listening Time (minutes)')
plt.xlabel('Genre')
plt.xticks(rotation=0)
plt.legend(title='Sentiment')
plt.show()


pd.crosstab(train['Publication_Time'], train['Episode_Sentiment'], normalize='index') * 100


train.groupby(['Publication_Day', 'Publication_Time'])['Listening_Time_minutes'].mean().unstack().plot(kind='bar', figsize=(12,6))
plt.title('Avg Listening Time by Day and Time')
plt.ylabel('Listening Time (minutes)')
plt.xlabel('Publication Day')
plt.xticks(rotation=0)
plt.legend(title='Time of Day')
plt.show()


pd.crosstab(train['Podcast_Name'], train['Episode_Sentiment'], normalize='index')*100


train.groupby('Episode_Sentiment')['Listening_Time_minutes'].describe()


agg_dict = {
    'Episode_Length_minutes': 'mean',
    'Host_Popularity_percentage': 'mean',
    'Guest_Popularity_percentage': 'mean',
    'Number_of_Ads': 'mean',
    'Episode_Sentiment': 'first',
    'Listening_Time_minutes': 'mean',
    'Host_Guest_Interaction': 'mean',
    'Length_per_Ad': 'mean',
    'Publication_Day': 'first',
    'Publication_Time': 'first',
    'Episode_Title': 'first',
    'Genre': 'first'
}

episode_grouped = train.groupby(['Podcast_Name', 'Episode_Number']).agg(agg_dict).reset_index()

episode_grouped.head()


top_podcasts = episode_grouped['Podcast_Name'].value_counts().head(2).index.tolist()
filtered_data = episode_grouped[episode_grouped['Podcast_Name'].isin(top_podcasts)]

plt.figure(figsize=(14, 6))
sns.lineplot(data=filtered_data, x='Episode_Number', y='Listening_Time_minutes', hue='Podcast_Name', marker='o')
plt.title('Listening Time per Episode (Top Podcasts)')
plt.xlabel('Episode Number')
plt.ylabel('Average Listening Time (minutes)')
plt.legend(title='Podcast Name')
plt.tight_layout()
plt.show()


# By Name
filtered_data = episode_grouped[episode_grouped['Podcast_Name'] == 'Humor Hub']

plt.figure(figsize=(14, 6))
sns.lineplot(data=filtered_data, x='Episode_Number', y='Listening_Time_minutes', hue='Podcast_Name', marker='o')
plt.title('Listening Time per Episode - Humor Hub')
plt.xlabel('Episode Number')
plt.ylabel('Average Listening Time (minutes)')
plt.legend(title='Podcast Name')
plt.tight_layout()
plt.show()


plt.figure(figsize=(14, 6))
sns.scatterplot(data=filtered_data, x='Episode_Number', y='Guest_Popularity_percentage', hue='Podcast_Name', alpha=0.7)
plt.title('Guest Popularity vs Episode Number')
plt.xlabel('Episode Number')
plt.ylabel('Guest Popularity (%)')
plt.legend(title='Podcast Name')
plt.tight_layout()
plt.show()


X = train_encoded.drop(columns=['Listening_Time_minutes', 'Episode_Title'])
y = train_encoded['Listening_Time_minutes']


test_encoded = test_encoded.drop(columns = ["Episode_Title", "Listening_Time_minutes"])


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


cv = KFold(n_splits = 3, shuffle = True, random_state = 42)





model1 = LinearRegression()


model1.fit(X_train, y_train)
y_pred = model1.predict(X_test)


rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"Baseline Linear Regression RMSE: {rmse:.4f}")





models = {
    "Linear Regression": LinearRegression(n_jobs = -1),
    "Ridge Regression": Ridge(random_state = 42),
    "Lasso Regression": Lasso(random_state = 42),
    "Decision Tree": DecisionTreeRegressor(random_state = 42),
    "Random Forest": RandomForestRegressor(random_state = 42, n_jobs = -1),
    # "Gradient Boosting": GradientBoostingRegressor(random_state = 42),          # Gradient Boosting - RMSE: 13.056252672093557, tested taking so much time
    "XGBoost": XGBRegressor(random_state = 42, verbosity = 0, n_jobs = -1),
    "LightGBM": LGBMRegressor(random_state = 42, verbose = -1, n_jobs = -1),
    "CatBoost": CatBoostRegressor(verbose=0, random_state=42)
}


results = []

for name, model in models.items():
    
    start = time.time()
    model.fit(X_train, y_train)
    end = time.time()
    
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    results.append({"Model": name, "RMSE": rmse})
    
    print(f"{(end - start)/60:.2f} -> {name} - RMSE: {rmse}")


results = pd.DataFrame(results).sort_values(by="RMSE").reset_index(drop=True)
results


plt.figure(figsize=(8, 4))
sns.barplot(x="RMSE", y="Model", data=results, palette="viridis", hue="Model", dodge=False)
plt.title("Model Comparison Based on Cross-Validated RMSE")
plt.xlabel("Root Mean Squared Error")
plt.ylabel("Model")
plt.tight_layout()
plt.show()





top_models = {
    'Random Forest': RandomForestRegressor(random_state=42, n_jobs=-1),
    'XGBoost': XGBRegressor(random_state=42, n_jobs=-1, verbosity=0),
    'CatBoost': CatBoostRegressor(verbose=0, thread_count=-1, random_state=42),
    'LightGBM': LGBMRegressor(random_state=42, verbose=-1, n_jobs=-1)
}


def evaluate_algorithm(name, model, X_train, y_train, cv):
    model_cloned = clone(model)  # Clone to avoid shared state in parallel jobs
    
    start_time = time.time()
    scores = cross_val_score(model_cloned, X_train, y_train, scoring='neg_root_mean_squared_error', cv=cv)
    end_time = time.time() - start_time
    
    rmse = -scores.mean()
    
    return {
        'Model': name,
        'Time': end_time/60,
        'RMSE': rmse
    }

results = Parallel(n_jobs=-1)(
    delayed(evaluate_algorithm)(name, model, X_train, y_train, cv=cv)
    for name, model in top_models.items()
)

results_df = pd.DataFrame(results).sort_values(by='RMSE').reset_index(drop=True)
results_df


plt.figure(figsize=(6, 2))
sns.barplot(x="RMSE", y="Model", data=results_df, palette="viridis", hue="Model", dodge=False)
plt.title("Model Comparison Based on Cross-Validated RMSE")
plt.xlabel("Root Mean Squared Error")
plt.ylabel("Model")
plt.tight_layout()
plt.show()





# param_grid = {
#     'n_estimators': [100, 200],
#     'max_depth': [None, 10, 20],
#     'min_samples_split': [2, 5],
#     'min_samples_leaf': [1, 2]
# }

# rf = RandomForestRegressor(random_state=42, n_jobs = -1)


param_grid = {'max_depth': [None], 'min_samples_leaf': [1], 'min_samples_split': [2], 'n_estimators': [200]}   #tested earlier

rf = RandomForestRegressor(random_state=42, n_jobs = -1)

grid_search_rf = GridSearchCV(rf, param_grid=param_grid, cv = cv,
                              scoring ='neg_root_mean_squared_error', verbose = 2)

grid_search_rf.fit(X_train, y_train)

print("Best Parameters:", grid_search_rf.best_params_)
print("Best CV RMSE Score:", -grid_search_rf.best_score_)


best_rf = grid_search_rf.best_estimator_
y_pred = best_rf.predict(X_test)
mse_test = mean_squared_error(y_test, y_pred)
rmse_test = np.sqrt(mse_test)
print("Test RMSE:", rmse_test)





predict = best_rf.predict(test_encoded)


submission.Listening_Time_minutes = predict
submission.head(2)


submission.to_csv("grid_search_rf_submission.csv", index=False)




