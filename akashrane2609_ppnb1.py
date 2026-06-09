import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
import xgboost as xgb
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings('ignore')



# Load training and testing data
df_train = pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv')
df_test = pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv')
df_sub = pd.read_csv('/kaggle/input/playground-series-s5e4/sample_submission.csv')


# Quick data overview
print("Train shape:", df_train.shape)
print("Test shape:", df_test.shape)



# Data types
print("\nData types:\n", df_train.dtypes.value_counts())


print("\nMissing values (train):\n", df_train.isnull().sum()[df_train.isnull().sum() > 0])
print("\nMissing values (test):\n", df_test.isnull().sum()[df_test.isnull().sum() > 0])





# Drop 'id' column
df_train.drop(columns=['id'], inplace=True)
df_test.drop(columns=['id'], inplace=True)


# Identify numeric and categorical columns
num_cols = df_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_cols = df_train.select_dtypes(include=['object']).columns.tolist()

# REMOVE target column from num_cols
if 'Listening_Time_minutes' in num_cols:
    num_cols.remove('Listening_Time_minutes')

# Now safely fill missing values
for col in num_cols:
    median_val = df_train[col].median()
    df_train[col] = df_train[col].fillna(median_val)
    df_test[col] = df_test[col].fillna(median_val)

for col in cat_cols:
    mode_val = df_train[col].mode()[0]
    df_train[col] = df_train[col].fillna(mode_val)
    df_test[col] = df_test[col].fillna(mode_val)



# Create useful interaction features
df_train['Total_Popularity'] = df_train['Host_Popularity_percentage'] + df_train['Guest_Popularity_percentage']
df_test['Total_Popularity'] = df_test['Host_Popularity_percentage'] + df_test['Guest_Popularity_percentage']

df_train['Host_vs_Guest'] = df_train['Host_Popularity_percentage'] - df_train['Guest_Popularity_percentage']
df_test['Host_vs_Guest'] = df_test['Host_Popularity_percentage'] - df_test['Guest_Popularity_percentage']

df_train['Length_x_Ads'] = df_train['Episode_Length_minutes'] * df_train['Number_of_Ads']
df_test['Length_x_Ads'] = df_test['Episode_Length_minutes'] * df_test['Number_of_Ads']




# Encode categorical features
for col in cat_cols:
    le = LabelEncoder()
    df_train[col] = le.fit_transform(df_train[col])
    df_test[col] = le.transform(df_test[col])



X = df_train.drop(columns=['Listening_Time_minutes'])
y = df_train['Listening_Time_minutes']

X_test = df_test.copy()

#y = np.log1p(df_train['Listening_Time_minutes'])



xgb_params = {
    'n_estimators': 565,
    'max_depth': 14,
    'learning_rate': 0.04222,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
    'tree_method': 'gpu_hist',
    'n_jobs': -1
}

kf = KFold(n_splits=5, shuffle=True, random_state=42)
test_preds = np.zeros(len(X_test))
scores = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    print(f"Training fold {fold+1}")
    
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    model = XGBRegressor(**xgb_params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric='rmse',
        verbose=False,                         
        callbacks=[
            xgb.callback.EarlyStopping(rounds=50)
        ]
    )
    
    val_preds = model.predict(X_val)
    fold_rmse = np.sqrt(mean_squared_error(y_val, val_preds))
    print(f"Fold {fold+1} RMSE: {fold_rmse:.4f}")
    
    scores.append(fold_rmse)
    test_preds += model.predict(X_test) / 5

print(f"\n✅ Final Average CV RMSE: {np.mean(scores):.5f} ± {np.std(scores):.5f}")




df_sub['Listening_Time_minutes'] = test_preds
df_sub.to_csv('submission_final_1.csv', index=False)

print("Final submission file saved as 'submission_final.csv'")



# Distribution of target
plt.figure(figsize=(8,5))
sns.histplot(df_train['Listening_Time_minutes'], bins=50, kde=True)
plt.title('Distribution of Listening Time (Target Variable)')
plt.xlabel('Listening Time (minutes)')
plt.show()



# Correlation heatmap
plt.figure(figsize=(12,10))
corr = df_train.corr()
sns.heatmap(corr, cmap='coolwarm', annot=False, fmt='.2f', linewidths=0.5)
plt.title('Correlation Heatmap (Numerical Features)')
plt.show()



# Average Listening Time by Categorical Features
cat_features = ['Podcast_Name', 'Genre', 'Publication_Day', 'Publication_Time', 'Episode_Sentiment']

for col in cat_features:
    plt.figure(figsize=(10,5))
    order = df_train.groupby(col)['Listening_Time_minutes'].mean().sort_values().index
    sns.barplot(x=col, y='Listening_Time_minutes', data=df_train, order=order)
    plt.title(f'Mean Listening Time vs {col}')
    plt.xticks(rotation=45)
    plt.show()



# Relationship between episode length, number of ads, and listening time
plt.figure(figsize=(10,6))
sns.scatterplot(data=df_train, x='Episode_Length_minutes', y='Listening_Time_minutes', hue='Number_of_Ads', palette='viridis', alpha=0.7)
plt.title('Listening Time vs Episode Length, colored by Number of Ads')
plt.xlabel('Episode Length (minutes)')
plt.ylabel('Listening Time (minutes)')
plt.show()





