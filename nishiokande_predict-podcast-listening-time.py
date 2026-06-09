import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


train_df = pd.read_csv("/kaggle/input/playground-series-s5e4/train.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e4/test.csv")


train_df.head(5)


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


train_df = train_df.drop(columns=["Podcast_Name", "Episode_Title"])
test_df = test_df.drop(columns=["Podcast_Name", "Episode_Title"])


train_df.head(5)


test_df.head(5)


# Compute the correlation matrix
corr_matrix = train_df.corr()

# Plot the heatmap for all correlation
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
plt.title("Correlation Heatmap")
plt.show()


plt.figure(figsize=(10,10))
sns.histplot(train_df['Listening_Time_minutes'], bins=20, kde=True, color="red")
plt.title("Distribution of Listening Time (minutes)")
plt.xlabel("Listening Time (minutes)")
plt.ylabel("Frequency")
plt.show()


plt.figure(figsize=(20,5))
sns.histplot(train_df['Listening_Time_minutes'], bins=150, kde=True, color="red")
plt.show()


from sklearn.model_selection import train_test_split

X = train_df.drop(["id", "Listening_Time_minutes"], axis=1)
y = train_df["Listening_Time_minutes"]

# Split data into training and testing sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


import xgboost as xgb
from xgboost import XGBRegressor

from sklearn.metrics import mean_squared_error, r2_score

# Create an XGBoost regression model
model = XGBRegressor(
    n_estimators=400,
    max_depth=14,
    learning_rate=0.03,
    random_state=42
)

# Train the model
model.fit(X_train, y_train)

# Make predictions on the validation set
y_pred = model.predict(X_val)

# Evaluate the model using RMSE
rmse = np.sqrt(mean_squared_error(y_val, y_pred))
print(f"Validation RMSE: {rmse:.4f}")


X_test = test_df.drop(["id"], axis = 1)

test_pred = model.predict(X_test)

submission = pd.DataFrame({"id": test_df["id"], "Listening_Time_minutes": test_pred})

submission.to_csv("submission.csv", index=False)

