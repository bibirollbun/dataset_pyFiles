import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error


train_df = pd.read_csv("/kaggle/input/playground-series-s5e4/train.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e4/test.csv")
submission_data = pd.read_csv("/kaggle/input/playground-series-s5e4/sample_submission.csv")


train_df.shape


train_df.isnull().sum()


categorical_cols = ["Podcast_Name", "Episode_Title", "Genre", "Publication_Day", "Publication_Time"]
numerical_cols = ["Episode_Length_minutes", "Host_Popularity_percentage", "Guest_Popularity_percentage", "Number_of_Ads"]


train_df.fillna(train_df.median(numeric_only=True), inplace=True)
test_df.fillna(test_df.median(numeric_only=True), inplace=True)


# Compute IQR for each numerical column
for col in numerical_cols:
    Q1 = train_df[col].quantile(0.25)
    Q3 = train_df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Count outliers
    outliers = train_df[(train_df[col] < lower_bound) | (train_df[col] > upper_bound)]
    print(f"Outliers in {col}: {len(outliers)}")


import seaborn as sns
import matplotlib.pyplot as plt

# Create boxplots for numerical columns
plt.figure(figsize=(12, 6))
sns.boxplot(data=train_df[numerical_cols])
plt.xticks(rotation=45)
plt.title("Boxplot of Numerical Features")
plt.show()


from scipy.stats import zscore

# Compute Z-scores
z_scores = train_df[numerical_cols].apply(zscore)

# Find outliers where absolute Z-score > 3
outliers = (z_scores.abs() > 3).sum()
print("Outliers detected using Z-score method:\n", outliers)


# Label encode categorical columns
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    train_df[col] = le.fit_transform(train_df[col].astype(str))
    test_df[col] = le.transform(test_df[col].astype(str))
    label_encoders[col] = le 


# Handle outliers using IQR method
def remove_outliers(df, cols):
    for col in cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    return df

train_df = remove_outliers(train_df, numerical_cols + ["Listening_Time_minutes"])


label_encoder = LabelEncoder()
train_df['Episode_Sentiment'] = label_encoder.fit_transform(train_df['Episode_Sentiment'])


scaler = StandardScaler()
train_df[numerical_cols] = scaler.fit_transform(train_df[numerical_cols])
test_df[numerical_cols] = scaler.transform(test_df[numerical_cols])


# Prepare training data
X = train_df.drop(columns=["id", "Listening_Time_minutes"])
y = train_df["Listening_Time_minutes"]
X_test = test_df.drop(columns=["id"])


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


cat_features_indices = [X.columns.get_loc(col) for col in categorical_cols]


model = CatBoostRegressor(
    iterations=1000, 
    depth=6, 
    learning_rate=0.05, 
    loss_function='RMSE', 
    cat_features=cat_features_indices,  # Specify categorical features
    verbose=100
)

model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=50)


y_pred = model.predict(X_val)
rmse = mean_squared_error(y_val, y_pred, squared=False)
print(f"Validation RMSE: {rmse}")


X_test['Episode_Sentiment'] = label_encoder.fit_transform(X_test['Episode_Sentiment'])


X_test.shape


pred = model.predict(X_test)


submission_data


submission = pd.DataFrame({"id": submission_data["id"], "Listening_Time_minutes": pred})
submission.to_csv("submission.csv", index=False)
print("âœ… Submission file saved as submission.csv")


submission




