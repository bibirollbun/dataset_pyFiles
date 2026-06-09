import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression


train_df = pd.read_csv("/kaggle/input/playground-series-s5e4/train.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e4/test.csv")


train_df.head(15)


mapping_genre = {"True Crime": 0, "Comedy": 1, "Education": 2, "Technology": 3, "Health": 4, "News": 5, "Music": 6, "Sports": 7, "Business": 8, "Lifestyle": 9}
train_df["Genre"] = train_df["Genre"].map(mapping_genre)
test_df["Genre"] = test_df["Genre"].map(mapping_genre)

mapping_publication_day = {"Thursday": 0, "Saturday": 1, "Tuesday": 2, "Monday": 3, "Sunday": 4, "Wednesday": 5, "Friday": 6}
train_df["Publication_Day"] = train_df["Publication_Day"].map(mapping_publication_day)
test_df["Publication_Day"] = test_df["Publication_Day"].map(mapping_publication_day)

mapping_publication_time = {"Night": 0, "Afternoon": 1, "Evening": 2, "Morning": 3}
train_df["Publication_Time"] = train_df["Publication_Time"].map(mapping_publication_time)
test_df["Publication_Time"] = test_df["Publication_Time"].map(mapping_publication_time)

mapping_episode_sentiment = {"Positive": 0, "Negative": 1, "Neutral": 2}
train_df["Episode_Sentiment"] = train_df["Episode_Sentiment"].map(mapping_episode_sentiment)
test_df["Episode_Sentiment"] = test_df["Episode_Sentiment"].map(mapping_episode_sentiment)


from sklearn.preprocessing import LabelEncoder
import category_encoders as ce
le = LabelEncoder()
train_df["Podcast_Name"] = le.fit_transform(train_df["Podcast_Name"])
test_df["Podcast_Name"] = le.transform(test_df["Podcast_Name"])

train_df["Episode_Title"] = le.fit_transform(train_df["Episode_Title"])
test_df["Episode_Title"] = le.transform(test_df["Episode_Title"])


plt.figure(figsize=(10,10))
sns.histplot(train_df['Listening_Time_minutes'], bins=20, kde=True, color="red")
plt.title("Distribution of Listening Time (minutes)")
plt.xlabel("Listening Time (minutes)")
plt.ylabel("Frequency")
plt.show()


train_df.head()


# Compute the correlation matrix
corr_matrix = train_df[["Episode_Length_minutes", "Host_Popularity_percentage", "Guest_Popularity_percentage", "Number_of_Ads", "Listening_Time_minutes"]].corr()

# Plot the heatmap for all correlation
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
plt.title("Correlation Heatmap")
plt.show()


train_df.describe()


plt.figure(figsize=(15, 6))
sns.boxplot(data=train_df, x=train_df["Number_of_Ads"])
plt.title("Boxplot of Number_of_Ads")
plt.xticks(rotation=45)
plt.show()


Q1 = train_df['Number_of_Ads'].quantile(0.25)
Q3 = train_df['Number_of_Ads'].quantile(0.75)
IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

train_df = train_df[
    (train_df['Number_of_Ads'] >= lower) &
    (train_df['Number_of_Ads'] <= upper)
]


plt.figure(figsize=(15, 6))
sns.boxplot(data=train_df, x=train_df["Number_of_Ads"])
plt.title("Boxplot of Number_of_Ads")
plt.xticks(rotation=45)
plt.show()


# Box Plot
plt.figure(figsize=(15, 6))
sns.boxplot(data=train_df, x="Episode_Length_minutes")
plt.title("Boxplot of Episode_Length_minutes")
plt.xticks(rotation=45)
plt.show()


# Box Plot
plt.figure(figsize=(15, 6))
sns.boxplot(data=train_df, x="Host_Popularity_percentage")
plt.title("Boxplot of Host_Popularity_percentage")
plt.xticks(rotation=45)
plt.show()


# Histogram
plt.figure(figsize=(15, 6))
sns.histplot(data=train_df, x="Host_Popularity_percentage", kde=True, bins=50)
plt.title("Histogram of Host_Popularity_percentage")
plt.xticks(rotation=45)
plt.show()


# Box Plot
plt.figure(figsize=(15, 6))
sns.boxplot(data=train_df, x="Guest_Popularity_percentage")
plt.title("Boxplot of Guest_Popularity_percentage")
plt.xticks(rotation=45)
plt.show()


# Histogram
plt.figure(figsize=(15, 6))
sns.histplot(data=train_df, x="Guest_Popularity_percentage", kde=True, bins=50)
plt.title("HIstogram of Guest_Popularity_percentage")
plt.xticks(rotation=45)
plt.show()


# Compute the correlation matrix
corr_matrix = train_df[["Episode_Length_minutes", "Host_Popularity_percentage", "Guest_Popularity_percentage", "Number_of_Ads", "Listening_Time_minutes"]].corr()

# Plot the heatmap for all correlation
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
plt.title("Correlation Heatmap")
plt.show()


plt.figure(figsize=(20,5))
sns.histplot(train_df['Listening_Time_minutes'], bins=150, kde=True, color="red")
plt.show()


plt.figure(figsize=(20,5))
sns.histplot(train_df['Episode_Length_minutes'], bins=150, kde=True, color="red")
plt.show()


train_df.head()


# Calculate Ads_Per_Minute for both train and test
train_df['Ads_Per_Minute'] = train_df['Number_of_Ads'] / train_df['Episode_Length_minutes']
test_df['Ads_Per_Minute'] = test_df['Number_of_Ads'] / test_df['Episode_Length_minutes']


plt.figure(figsize=(8, 6))
plt.hexbin(
    train_df['Host_Popularity_percentage'],
    train_df['Guest_Popularity_percentage'],
    gridsize=50,
    cmap='plasma'
)
plt.colorbar(label='Counts') 
plt.xlabel('Host Popularity (%)')
plt.ylabel('Guest Popularity (%)')
plt.title('Host vs Guest Popularity Density (Hexbin)')
plt.grid(True)
plt.show()


train_df['Host_Popularity_percentage_log'] = np.log1p(train_df['Host_Popularity_percentage'])
test_df['Host_Popularity_percentage_log'] = np.log1p(test_df['Host_Popularity_percentage'])
train_df['Guest_Popularity_percentage_log'] = np.log1p(train_df['Guest_Popularity_percentage'])
test_df['Guest_Popularity_percentage_log'] = np.log1p(test_df['Guest_Popularity_percentage'])


fig, axes = plt.subplots(2, 2, figsize=(14, 10))

sns.histplot(train_df['Host_Popularity_percentage'], bins=100, ax=axes[0, 0], color='skyblue')
axes[0, 0].set_title('Original Host Popularity')

sns.histplot(train_df['Host_Popularity_percentage_log'], bins=100, ax=axes[0, 1], color='orange')
axes[0, 1].set_title('Log-transformed Host Popularity')

sns.histplot(train_df['Guest_Popularity_percentage'], bins=100, ax=axes[1, 0], color='lightgreen')
axes[1, 0].set_title('Original Guest Popularity')

sns.histplot(train_df['Guest_Popularity_percentage_log'], bins=100, ax=axes[1, 1], color='coral')
axes[1, 1].set_title('Log-transformed Guest Popularity')

plt.tight_layout()
plt.show()


train_df = train_df.drop(['Host_Popularity_percentage', 'Guest_Popularity_percentage'], axis=1)
test_df = test_df.drop(['Host_Popularity_percentage', 'Guest_Popularity_percentage'], axis=1)


train_df.head()


X = train_df.drop(["id", "Listening_Time_minutes"], axis=1)
y = train_df["Listening_Time_minutes"]
X_test = test_df.drop(["id"], axis = 1)

# Split data into training and testing sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


# â‘  CatBoost Model
catboost_model = CatBoostRegressor(
    iterations=1000,
    depth=10,
    learning_rate=0.03,
    loss_function='RMSE',
    cat_features=[],
    random_seed=42,
    verbose=100
)

catboost_model.fit(X_train, y_train)
catboost_pred_val = catboost_model.predict(X_val)

# â‘¡ XGBoost Model
xgb_model = XGBRegressor(
    n_estimators=901,
    max_depth=14,
    learning_rate=0.023634721917800305,
    subsample=0.9422518609857451,
    colsample_bytree=0.801347427407153,
    gamma=0.01725257344830311,
    min_child_weight=7,
    reg_alpha=0.2529313301280318,
    reg_lambda=3.4146535302491694,
    random_state=42,
    n_jobs=-1
)

xgb_model.fit(X_train, y_train)
xgb_pred_val = xgb_model.predict(X_val)

# â‘¢ LightGBM Model
lgb_model = lgb.LGBMRegressor(
    n_estimators=5000,
    learning_rate=0.02,
    max_depth=-1,
    num_leaves=256,
    subsample=0.9,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=2.0,
    min_child_samples=10,
    random_state=42,
    n_jobs=-1
)

lgb_model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    eval_metric="rmse",
    callbacks=[
        lgb.early_stopping(stopping_rounds=50),
        lgb.log_evaluation(period=100)
    ]
)
lgb_pred_val = lgb_model.predict(X_val, num_iteration=lgb_model.best_iteration_)

# â‘£ Combine predictions from XGBoost, LightGBM, and CatBoost
stacked_val = np.vstack([xgb_pred_val, lgb_pred_val, catboost_pred_val]).T

# â‘¤ Meta-model (using Ridge regression)
meta_model = Ridge(alpha=1.0, random_state=42)
meta_model.fit(stacked_val, y_val)

# â‘¥ Final prediction using the meta-model
final_pred = meta_model.predict(stacked_val)

# â‘¦ Evaluation: Calculate RMSE for the stacking model
rmse = np.sqrt(mean_squared_error(y_val, final_pred))
print(f"Stacking Validation RMSE: {rmse:.4f}")


# â‘  Make predictions using XGBoost, LightGBM, and CatBoost models
xgb_pred_test = xgb_model.predict(X_test)
lgb_pred_test = lgb_model.predict(X_test)
catboost_pred_test = catboost_model.predict(X_test)

# â‘¡ Stack the predictions from XGBoost, LightGBM, and CatBoost models
stacked_test = np.vstack([xgb_pred_test, lgb_pred_test, catboost_pred_test]).T

# â‘¢ Make final predictions using the meta-model
test_pred = meta_model.predict(stacked_test)

# â‘£ Create the submission dataframe
submission = pd.DataFrame({"id": test_df["id"], "Listening_Time_minutes": test_pred})

# â‘¤ Save the submission dataframe to a CSV file
submission.to_csv("submission.csv", index=False)

