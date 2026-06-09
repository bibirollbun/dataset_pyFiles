import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.sankey import Sankey

from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import VotingRegressor

from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import warnings
warnings.simplefilter(action = "ignore", category = RuntimeWarning)


train = pd.read_csv("/kaggle/input/playground-series-s5e4/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e4/test.csv")


import warnings
warnings.simplefilter(action = "ignore", category = RuntimeWarning)

train.head()


train.shape


import warnings
warnings.simplefilter(action = "ignore", category = RuntimeWarning)

test.head()


test.shape


train.isnull().sum()


test.isnull().sum()





# Eksik verileri model tahminiyle doldurma

features = ["Host_Popularity_percentage", "Listening_Time_minutes"]

train_data_episode_length = train.dropna(subset=["Episode_Length_minutes"])
test_data_episode_length = train[train["Episode_Length_minutes"].isna()]

lgbm_episode_length = LGBMRegressor()
lgbm_episode_length.fit(train_data_episode_length[features], train_data_episode_length["Episode_Length_minutes"])

predictions_episode_length = lgbm_episode_length.predict(test_data_episode_length[features])
train.loc[train["Episode_Length_minutes"].isna(), "Episode_Length_minutes"] = predictions_episode_length

train_data_guest_popularity = train.dropna(subset=["Guest_Popularity_percentage"])
test_data_guest_popularity = train[train["Guest_Popularity_percentage"].isna()]

lgbm_guest_popularity = LGBMRegressor()
lgbm_guest_popularity.fit(train_data_guest_popularity[features], train_data_guest_popularity["Guest_Popularity_percentage"])

predictions_guest_popularity = lgbm_guest_popularity.predict(test_data_guest_popularity[features])
train.loc[train["Guest_Popularity_percentage"].isna(), "Guest_Popularity_percentage"] = predictions_guest_popularity

print("Train Eksik Veriler:")
print(train.isna().sum())


ads_median = train["Number_of_Ads"].median()

train["Number_of_Ads"].fillna(ads_median, inplace=True)

print("Train Eksik Veriler:")
print(train.isna().sum())


features = ["Number_of_Ads", "Host_Popularity_percentage"]

test_data_episode_length = test[test["Episode_Length_minutes"].isna()]
lgbm_episode_length = LGBMRegressor()
lgbm_episode_length.fit(train[features], train["Episode_Length_minutes"])
predictions_episode_length_test = lgbm_episode_length.predict(test_data_episode_length[features])
test.loc[test["Episode_Length_minutes"].isna(), "Episode_Length_minutes"] = predictions_episode_length_test

test_data_guest_popularity = test[test["Guest_Popularity_percentage"].isna()]
lgbm_guest_popularity = LGBMRegressor()
lgbm_guest_popularity.fit(train[features], train["Guest_Popularity_percentage"])
predictions_guest_popularity_test = lgbm_guest_popularity.predict(test_data_guest_popularity[features])
test.loc[test["Guest_Popularity_percentage"].isna(), "Guest_Popularity_percentage"] = predictions_guest_popularity_test

print("Test Eksik Veriler:")
print(test.isna().sum())



day_mapping = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3, 
               'Friday': 4, 'Saturday': 5, 'Sunday': 6}

time_mapping = {'Morning': 9, 'Afternoon': 14, 'Evening': 19, 'Night': 23}

sentiment_mapping = {'Negative': 1, 'Neutral': 2, 'Positive': 3}

train['Publication_Day_num'] = train['Publication_Day'].map(day_mapping)
train['Publication_Time_num'] = train['Publication_Time'].map(time_mapping)
train['Episode_Sentiment_num'] = train['Episode_Sentiment'].map(sentiment_mapping)

test['Publication_Day_num'] = test['Publication_Day'].map(day_mapping)
test['Publication_Time_num'] = test['Publication_Time'].map(time_mapping)
test['Episode_Sentiment_num'] = test['Episode_Sentiment'].map(sentiment_mapping)


train.head()


genre_count = train["Genre"].value_counts()
genre_count


plt.figure(figsize =(10,6))
genre_count.plot(kind = 'bar', color="skyblue")
plt.title("Distribution of Podcast Categories")
plt.xlabel("Podcast Categories")
plt.ylabel("Number of Podcasts")
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.show()


categorical_features = ["Publication_Day", "Publication_Time"]

plt.figure(figsize=(16, 10))
for i, feature in enumerate(categorical_features, 1):
    plt.subplot(2, 2, i)
    sns.countplot(data=train, y=feature, order=train[feature].value_counts().index, palette="viridis")
    plt.title(f"Distribution of {feature}", fontsize=14, fontweight='bold')
    plt.xlabel("Number of Episodes", fontsize=12)
    plt.ylabel(feature, fontsize=12)
plt.tight_layout()
plt.show()


# HaftanÄ±n gÃ¼nlerini sÄ±ralayarak sayma
day_counts = train['Publication_Day'].value_counts().reindex(
    ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'], fill_value=0)

# Verileri kutupsal koordinatlara Ã§evirmek iÃ§in aÃ§Ä±larÄ± belirleme
angles = np.linspace(0, 2 * np.pi, len(day_counts), endpoint=False)
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'projection': 'polar'})
bars = ax.bar(angles, day_counts.values, width=0.6, color=plt.cm.viridis(np.linspace(0.3, 1, len(day_counts))), alpha=0.8)
ax.set_xticks(angles)
ax.set_xticklabels(day_counts.index)
ax.set_title("Podcast Publication Day Distribution (Polar Chart)", pad=20)
plt.show()


#Genre ve Episode_Sentiment iliÅŸkisi
genre_sentiment = train.groupby(['Genre', 'Episode_Sentiment']).size().unstack(fill_value=0)
genre_sentiment.plot(kind='bar', stacked=True, figsize=(12, 7), colormap='coolwarm')
plt.title("Episode Sentiment Distribution by Genre", fontsize=14, fontweight='bold')
plt.xlabel("Genre", fontsize=12)
plt.ylabel("Number of Episodes", fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.legend(title='Sentiment')
plt.tight_layout()
plt.show()


top_10_genel = train[['Podcast_Name', 'Host_Popularity_percentage','Episode_Title', 'Genre', 'Episode_Sentiment']].sort_values(by='Host_Popularity_percentage', ascending=False).head(10)
top_10_genel



top_10_hosts = train[['Podcast_Name', 'Host_Popularity_percentage']].sort_values(by='Host_Popularity_percentage', ascending=False).head(10)

new_colors = sns.color_palette("magma", len(top_10_hosts))

plt.figure(figsize=(10, 6))
bars = plt.barh(top_10_hosts['Podcast_Name'], top_10_hosts['Host_Popularity_percentage'], color=new_colors)

plt.title('Top 10 Podcasts with Highest Host Popularity')
plt.xlabel('Host Popularity Percentage')
plt.ylabel('Podcast Name')
plt.gca().invert_yaxis()
plt.tight_layout() 
plt.show()



crime_podcasts = train[train['Genre'] == 'True Crime']
crime_sentiment_counts = crime_podcasts['Episode_Sentiment'].value_counts()
crime_sentiment_counts.head()



sentiment_counts = train['Episode_Sentiment'].value_counts()

plt.figure(figsize=(8, 8))
plt.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', startangle=90, colors=['lightblue', 'lightgreen', 'lightcoral'])
plt.title('Sentiment Distribution for All Episodes')
plt.axis('equal')  
plt.show()



plt.figure(figsize=(15, 10))

plt.subplot(2, 2, 1) 
sns.histplot(train['Host_Popularity_percentage'], bins=20, kde=True, color='skyblue')
plt.title('Distribution of Host Popularity')
plt.xlabel('Host Popularity (%)')
plt.ylabel('Frequency')

plt.subplot(2, 2, 2) 
sns.histplot(train['Guest_Popularity_percentage'], bins=20, kde=True, color='salmon')
plt.title('Distribution of Guest Popularity')
plt.xlabel('Guest Popularity (%)')
plt.ylabel('Frequency')

plt.subplot(2, 2, 3)
sns.histplot(train['Host_Popularity_percentage'], bins=20, kde=True, color='skyblue', label='Host Popularity')
sns.histplot(train['Guest_Popularity_percentage'], bins=20, kde=True, color='salmon', alpha=0.7, label='Guest Popularity')
plt.title('Comparison of Host and Guest Popularity Distributions')
plt.xlabel('Popularity (%)')
plt.ylabel('Frequency')
plt.legend()

plt.subplot(2, 2, 4) 
sns.scatterplot(x='Number_of_Ads', y='Listening_Time_minutes', data=train)
plt.title('Number of Ads vs. Listening Time')
plt.xlabel('Number of Ads')
plt.ylabel('Listening Time (minutes)')
plt.grid(True)

plt.tight_layout() 
plt.show()


plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.boxplot(train['Episode_Length_minutes'], vert=False, patch_artist=True, 
            boxprops=dict(facecolor='lightcoral', color='red'), 
            medianprops=dict(color='black', linewidth=2))
plt.title('Distribution of Podcast Episode Lengths')
plt.xlabel('Episode Length (minutes)')


plt.subplot(1, 2, 2)
plt.boxplot(train['Listening_Time_minutes'], vert=False, patch_artist=True, 
            boxprops=dict(facecolor='skyblue', color='blue'), 
            medianprops=dict(color='black', linewidth=2))
plt.title('Distribution of Listening Time')
plt.xlabel('Listening Time (minutes)')

plt.tight_layout()  
plt.show()


## Corr 
numeric_columns = ["Episode_Length_minutes", "Host_Popularity_percentage", "Guest_Popularity_percentage", "Number_of_Ads", "Listening_Time_minutes"]
train_numeric = train[numeric_columns]

fig, ax = plt.subplots(figsize=(10,6)) 
sns.heatmap(train_numeric.corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=.5, ax=ax)
plt.title("Korelasyon IsÄ± HaritasÄ±")
plt.show()


## Let's examine the high correlation between episode length and episode listening time.

plt.figure(figsize=(10,6))
sns.scatterplot(data=train, x="Episode_Length_minutes", y="Listening_Time_minutes", alpha=0.5)

sns.regplot(data=train, x="Episode_Length_minutes", y="Listening_Time_minutes", scatter=False, color="red")

plt.xlabel("Episode Length (minutes)")
plt.ylabel("Listening Time (minutes)")
plt.title("Episode Length vs Listening Time")
plt.show()


# Train verisi iÃ§in Cyclinal Encoding
train['Publication_Day_sin'] = np.sin(2 * np.pi * train['Publication_Day_num'] / 7)
train['Publication_Day_cos'] = np.cos(2 * np.pi * train['Publication_Day_num'] / 7)

train['Publication_Time_sin'] = np.sin(2 * np.pi * train['Publication_Time_num'] / 24)
train['Publication_Time_cos'] = np.cos(2 * np.pi * train['Publication_Time_num'] / 24)

# Test verisi iÃ§in Cyclinal Encoding
test['Publication_Day_sin'] = np.sin(2 * np.pi * test['Publication_Day_num'] / 7)
test['Publication_Day_cos'] = np.cos(2 * np.pi * test['Publication_Day_num'] / 7)

test['Publication_Time_sin'] = np.sin(2 * np.pi * test['Publication_Time_num'] / 24)
test['Publication_Time_cos'] = np.cos(2 * np.pi * test['Publication_Time_num'] / 24)




# Ad Density
train['Ad_Density'] = train['Number_of_Ads'] / train['Episode_Length_minutes']
test['Ad_Density'] = test['Number_of_Ads'] / test['Episode_Length_minutes']

# Popularity Interaction
train['Popularity_Interaction'] = train['Host_Popularity_percentage'] * train['Guest_Popularity_percentage']
test['Popularity_Interaction'] = test['Host_Popularity_percentage'] * test['Guest_Popularity_percentage']

# Popularity Difference
train['Popularity_Difference'] = train['Host_Popularity_percentage'] - train['Guest_Popularity_percentage']
test['Popularity_Difference'] = test['Host_Popularity_percentage'] - test['Guest_Popularity_percentage']

# Weekday vs Weekend
train['Is_Weekend'] = train['Publication_Day'].isin(['Saturday', 'Sunday']).astype(int)
test['Is_Weekend'] = test['Publication_Day'].isin(['Saturday', 'Sunday']).astype(int)

# Interaction Features
train['Ad_Length_Interaction'] = train['Number_of_Ads'] * train['Episode_Length_minutes']
test['Ad_Length_Interaction'] = test['Number_of_Ads'] * test['Episode_Length_minutes']

train['Host_Ads_Interaction'] = train['Host_Popularity_percentage'] * train['Number_of_Ads']
test['Host_Ads_Interaction'] = test['Host_Popularity_percentage'] * test['Number_of_Ads']



# Modelde kullanmayacaÄŸÄ±mÄ±z sÃ¼tunlar
drop_columns = ["id", "Podcast_Name", "Episode_Title"]
existing_drop_cols = [col for col in drop_columns if col in train.columns]

# Hedef deÄŸiÅŸken
target = "Listening_Time_minutes"

# SayÄ±sal sÃ¼tunlarÄ± seÃ§ (hedef ve drop sÃ¼tunlarÄ± hariÃ§)
numerical_cols = train.select_dtypes(include=["int64", "float64"]).drop(columns=existing_drop_cols + [target], errors="ignore").columns.tolist()

# SayÄ±sal verileri al ve Ã¶lÃ§ekle
scaler = StandardScaler()
X_numeric = scaler.fit_transform(train[numerical_cols])


# Kategorik sÃ¼tunlarÄ± seÃ§
categorical_cols = train.select_dtypes(include=["object"]).columns.tolist()

# One-Hot Encoding iÅŸlemi
encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
X_categorical = encoder.fit_transform(train[categorical_cols])


# TÃ¼m Ã¶zellikleri birleÅŸtir
X_processed = np.concatenate([X_numeric, X_categorical], axis=1)

# Hedef deÄŸiÅŸken
y = train[target].values

print(f"Final feature shape: {X_processed.shape}")


# SayÄ±sal sÃ¼tun isimleri zaten elimizde
numeric_feature_names = numerical_cols

# One-Hot Encoder'dan Ã§Ä±kan kategorik sÃ¼tun isimlerini al
categorical_feature_names = encoder.get_feature_names_out(categorical_cols).tolist()

# TÃ¼m sÃ¼tun adlarÄ±nÄ± birleÅŸtir
all_feature_names = numeric_feature_names + categorical_feature_names

X_train = pd.DataFrame(X_processed, columns=all_feature_names)

# Gerekirse kontrol et
X_train.head()


# Test verisindeki sayÄ±sal sÃ¼tunlarÄ± aynÄ± scaler ile dÃ¶nÃ¼ÅŸtÃ¼r
X_test_scaled = scaler.transform(test[numerical_cols])


# Test verisindeki kategorik sÃ¼tunlarÄ± dÃ¶nÃ¼ÅŸtÃ¼r
X_test_encoded = encoder.transform(test[categorical_cols])


# Scaled + Encoded birleÅŸimi
X_test_processed = np.hstack([X_test_scaled, X_test_encoded])

# DataFrame oluÅŸtur
X_test = pd.DataFrame(X_test_processed, columns=all_feature_names)

# Kontrol
X_test.head()


X_train = X_train.fillna(X_train.median())
X_test = X_test.fillna(X_test.median())


lreg = LinearRegression()


scores = cross_val_score(lreg, X_train, y, cv = 5, scoring = 'neg_root_mean_squared_error')
rmse_score = np.abs(scores)
print(f"Mean AUC: {rmse_score.mean():.4f}")


lreg.fit(X_train , y)
y_test_pred = lreg.predict(X_test)


submission = pd.DataFrame({
    "id": test["id"],
    "rainfall": y_test_pred
})

submission.to_csv("submission_linear_regresion.csv", index=False)


xgb_model = XGBRegressor(subsample = 0.6,reg_lambda = 1.0,reg_alpha = 0, n_estimators =500 , max_depth = 15, 
                        learning_rate = 0.05,gamma = 0.5,colsample_bytree = 1.0)


scores = cross_val_score(xgb_model, X_train, y, cv = 5, scoring = 'neg_root_mean_squared_error')
rmse_score = np.abs(scores)
print(f"Mean AUC: {rmse_score.mean():.4f}")


xgb_model.fit(X_train,y)
y_pred = xgb_model.predict(X_test)


submission = pd.DataFrame({
    "id": test["id"],
    "rainfall": y_pred
})

submission.to_csv("submission_xgb_regresion.csv", index=False)


# LightGBM modeling
lgbm_model = LGBMRegressor(
    objective='regression',
    learning_rate=0.05,
    n_estimators=1000,
    max_depth=15,
    subsample=0.7,
    colsample_bytree=0.9,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42
)


lgbm_scores = cross_val_score(
    lgbm_model,
    X_train,
    y,
    cv=5,
    scoring='neg_root_mean_squared_error'
)


rmse_scores = np.abs(lgbm_scores)
print(f"LightGBM CV RMSE Scores: {rmse_scores}")
print(f"LightGBM Mean RMSE: {rmse_scores.mean():.4f}")


lgbm_model.fit(X_train, y)


lgbm_pred = lgbm_model.predict(X_test)


submission = pd.DataFrame({
    "id": test["id"],
    "Listening_Time_minutes": lgbm_pred
})
submission.to_csv("submission_lgbm.csv", index=False)


xgb_model = XGBRegressor(
    subsample=1.0,
    reg_lambda=1.5,
    reg_alpha=0,
    n_estimators=500,
    max_depth=15,
    learning_rate=0.05,
    gamma=0.1,
    colsample_bytree=0.8,
    objective='reg:squarederror',
    random_state=42
)

xgb_model.fit(X_train, y)


y_pred = xgb_model.predict(X_test)


submission = pd.DataFrame({
    "id": test["id"],
    "rainfall": y_pred
})

submission.to_csv("submission_xgb_best_params.csv", index=False)



# RidgeCV iÃ§in alpha deÄŸerlerini belirle
alphas = [0.01, 0.1, 1.0, 10.0, 100.0]


ridge_cv = RidgeCV(alphas=alphas, cv=5, scoring='neg_root_mean_squared_error')
ridge_cv.fit(X_train, y)

# En iyi alpha
print("Best Alpha:", ridge_cv.alpha_)

y_pred = ridge_cv.predict(X_test)


submission = pd.DataFrame({
    "id": test["id"],
    "rainfall": y_pred
})
submission.to_csv("submission_ridgecv.csv", index=False)
print("âœ… Submission dosyasÄ± kaydedildi: submission_ridgecv.csv")


xgb_model = XGBRegressor(subsample = 0.6,reg_lambda = 1.0,reg_alpha = 0, n_estimators =500 , max_depth = 15, 
                        learning_rate = 0.05,gamma = 0.5,colsample_bytree = 1.0)
lreg = LinearRegression()


voting = VotingRegressor(
    estimators=[
        ('xgb', xgb_model),
        ('liner', lreg)
    ],
    n_jobs=-1
)

voting.fit(X_train, y)
y_pred = voting.predict(X_test)


submission = pd.DataFrame({
    "id": test["id"],
    "Episode_Length_minutes": y_pred
})

submission.to_csv("submission_voting_liner_xgb.csv", index=False)

