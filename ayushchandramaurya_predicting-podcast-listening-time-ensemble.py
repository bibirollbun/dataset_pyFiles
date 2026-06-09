import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import lightgbm as lgb

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from scipy import stats


train = pd.read_csv("/kaggle/input/playground-series-s5e4/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e4/test.csv")


train.head()


test.head()


train.info()


test.info()


train.describe()


print(train.duplicated().sum())
print(test.duplicated().sum())


target_column = "Listening_Time_minutes"

y_train = train[target_column]
X_train = train.drop(columns=[target_column])
X_test = test


numerical_features = X_train.select_dtypes(include=['number']).columns.tolist()
categorical_features =X_train.select_dtypes(exclude=['number']).columns.tolist()



print("Numerical Features:", numerical_features)  
print("Categorical Features:", categorical_features)


X_train[numerical_features] = X_train[numerical_features].fillna(X_train[numerical_features].median())
X_train[categorical_features] = X_train[categorical_features].fillna(X_train[categorical_features].mode().iloc[0])

X_test[numerical_features] = X_test[numerical_features].fillna(X_test[numerical_features].median())
X_test[categorical_features] = X_test[categorical_features].fillna(X_test[categorical_features].mode().iloc[0])



X_train.isnull().sum()


X_test.isnull().sum()


import seaborn as sns
import matplotlib.pyplot as plt


# Pairplot to visualize relationships
sns.pairplot(train[numerical_features])
plt.suptitle("Numerical Feature Correlations", y=1.02)
plt.show()

# Heatmap of correlations
plt.figure(figsize=(6,4))
sns.heatmap(train[numerical_features].corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap (Numerical Features)")
plt.show()





for col in categorical_features:
    plt.figure(figsize=(8, 4))
    sns.boxplot(x=col, y='Listening_Time_minutes', data=train)
    plt.title(f"Listening Time vs {col}")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()



def engineer_features(X_train, X_test):
    combined = pd.concat([X_train, X_test], axis=0).reset_index(drop=True)

    # 1. Ad Density
    combined['ads_per_minute'] = combined['Number_of_Ads'] / (combined['Episode_Length_minutes'] + 1e-3)

    # 2. Is Weekend
    combined['is_weekend'] = combined['Publication_Day'].isin(['Saturday', 'Sunday']).astype(int)

    # 3. Time of Day Features
    combined['is_morning'] = (combined['Publication_Time'] == 'Morning').astype(int)
    combined['is_night'] = (combined['Publication_Time'] == 'Night').astype(int)

    # 4. Episode Length Buckets
    combined['length_bucket'] = pd.cut(combined['Episode_Length_minutes'], bins=[0, 30, 60, 90, 200],
                                       labels=['short', 'medium', 'long', 'very_long'])

    # 5. Sentiment Ordinal Mapping
    sentiment_map = {'Negative': -1, 'Neutral': 0, 'Positive': 1}
    combined['sentiment_score'] = combined['Episode_Sentiment'].map(sentiment_map)

    # 6. Host-Guest Popularity Ratio
    combined['popularity_ratio'] = combined['Guest_Popularity_percentage'] / (
        combined['Host_Popularity_percentage'] + 1e-3)

    # 7. Episode Number from Title
    combined['episode_number'] = combined['Episode_Title'].str.extract(r'(\d+)').astype(float)

    # 8. Genre + Sentiment Interaction
    combined['genre_sentiment'] = combined['Genre'].astype(str) + "_" + combined['Episode_Sentiment'].astype(str)

    # --- Handle Missing Values ---
    # Fill numeric columns using Genre-wise mean
    for col in ['Episode_Length_minutes', 'Guest_Popularity_percentage']:
        combined[col] = combined.groupby('Genre')[col].transform(lambda x: x.fillna(x.mean()))

    # --- Encode Categorical Features ---
    categorical_cols = ['Podcast_Name', 'Episode_Title', 'Genre', 'Publication_Day',
                        'Publication_Time', 'Episode_Sentiment', 'length_bucket', 'genre_sentiment']

    for col in categorical_cols:
        le = LabelEncoder()
        combined[col] = le.fit_transform(combined[col].astype(str))

    # Split back to train and test
    X_train_fe = combined.iloc[:len(X_train)].reset_index(drop=True)
    X_test_fe = combined.iloc[len(X_train):].reset_index(drop=True)

    return X_train_fe, X_test_fe



X_train_fe, X_test_fe = engineer_features(X_train, X_test)



X_train_fe


X_test_fe


X = X_train_fe.drop(['id'],axis=1)
y = y_train
test_id = X_test_fe['id']
test = X_test_fe.drop(['id'],axis=1)


from xgboost import XGBRegressor, plot_importance
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt

# Best params from Optuna
best_params = {
    'n_estimators': 5000,
    'max_depth': 15,
    'learning_rate': 0.051564535401996674,
    'subsample': 0.6816345671807827,
    'colsample_bytree': 0.9977810444050708,
    'gamma': 1.4032650461122345,
    'reg_alpha': 2.7815627866713517,
    'reg_lambda': 3.780137117381534,
    'random_state': 42,
    'tree_method': 'gpu_hist',  
    'predictor': 'gpu_predictor'
}

# Initialize model
model = XGBRegressor(**best_params)

# K-Fold Cross Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_rmse = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X), start=1):
    X_train_cv, X_val_cv = X.iloc[train_idx], X.iloc[val_idx]
    y_train_cv, y_val_cv = y.iloc[train_idx], y.iloc[val_idx]
    
    model.fit(X_train_cv, y_train_cv, 
              eval_set=[(X_val_cv, y_val_cv)],
              early_stopping_rounds=50,
              verbose=False)
    
    preds = model.predict(X_val_cv)
    rmse_score = np.sqrt(mean_squared_error(y_val_cv, preds))
    cv_rmse.append(rmse_score)

    print(f"Fold {fold} RMSE: {rmse_score:.4f}")

print(f"\nAverage CV RMSE: {np.mean(cv_rmse):.4f}")

# Fit final model on all data
model.fit(X, y)



# Feature importance
plt.figure(figsize=(12, 6))
plot_importance(model, max_num_features=20, importance_type='gain')
plt.title("Top 20 Feature Importances (by Gain)")
plt.show()



test_preds = model.predict(test)




sub1 = pd.read_csv("/kaggle/input/ps-s5e4-listening-time-division-attention/submission.csv")
sub2= pd.read_csv("/kaggle/input/ps-s5-e4-ensemble-of-solutions/submission.csv")
sample_submission = pd.read_csv("/kaggle/input/playground-series-s5e4/sample_submission.csv")

sample_submission['Listening_Time_minutes']= (0.51 * sub1['Listening_Time_minutes']) + (0.48 * sub2['Listening_Time_minutes'])+ (0.01*test_preds)

sample_submission.to_csv('submission.csv', index=False)
sample_submission.head()







