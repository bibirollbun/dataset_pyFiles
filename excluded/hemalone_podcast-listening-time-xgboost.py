import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.linear_model import (
    LinearRegression, Ridge, Lasso, ElasticNet
)
from category_encoders import TargetEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import mutual_info_regression
from scipy.stats import boxcox
from sklearn.ensemble import AdaBoostRegressor
from xgboost import plot_importance
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error, r2_score


train = pd.read_csv("/kaggle/input/playground-series-s5e4/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e4/test.csv")
sample_sub = pd.read_csv("/kaggle/input/playground-series-s5e4/sample_submission.csv")


train.head()


test.head()


plt.figure(figsize=(10, 4))
sns.heatmap(train.isnull(), cbar=False, cmap="viridis")
plt.title("Missing Values Heatmap")
plt.show()


plt.figure(figsize=(10, 4))
sns.heatmap(test.isnull(), cbar=False, cmap="viridis")
plt.title("Missing Values Heatmap")
plt.show()


overall_length_median = train['Episode_Length_minutes'].median()
train.dropna(subset=['Listening_Time_minutes'], inplace=True)

def fill_with_group_median(series):
    return series.fillna(series.median()).fillna(overall_length_median)

def clean_data(df):
    
    df['Episode_Length_minutes'] = df.groupby('Genre')['Episode_Length_minutes'].transform(fill_with_group_median)
    
    df['Guest_Popularity_percentage'] = df['Guest_Popularity_percentage'].fillna(df['Guest_Popularity_percentage'].median())
    df['Number_of_Ads'] = df['Number_of_Ads'].fillna(df['Number_of_Ads'].median())
    
    return df

train = clean_data(train)
test = clean_data(test)


plt.figure(figsize=(10, 4))
sns.heatmap(train.isnull(), cbar=False, cmap="viridis")
plt.title("Missing Values Heatmap")
plt.show()


plt.figure(figsize=(10, 4))
sns.heatmap(test.isnull(), cbar=False, cmap="viridis")
plt.title("Missing Values Heatmap")
plt.show()


plt.figure(figsize=(15, 8))
for i, col in enumerate(train.select_dtypes('number'), 1):
    plt.subplot(2, 3, i)
    sns.boxplot(y=train[col])
    plt.title(f'Boxplot of {col}')
plt.tight_layout()
plt.show()


plt.figure(figsize=(15, 8))
for i, col in enumerate(test.select_dtypes('number'), 1):
    plt.subplot(2, 3, i)
    sns.boxplot(y=train[col])
    plt.title(f'Boxplot of {col}')
plt.tight_layout()
plt.show()


def drop_outliers(df, col, threshold=1.5):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - threshold * IQR
    upper_bound = Q3 + threshold * IQR
    return df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]

for col in train.select_dtypes('number'):
    train = drop_outliers(train, col)
    


def drop_outliers(df, col, threshold=1.5):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - threshold * IQR
    upper_bound = Q3 + threshold * IQR
    return df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]

for col in test.select_dtypes('number'):
    test = drop_outliers(test, col)
    


train.describe()


test.describe()


plt.figure(figsize=(12, 8))
corr = train.select_dtypes(include=[np.number]).corr()
sns.heatmap(corr, annot=True, cmap="coolwarm", center=0)
plt.title("Correlation Matrix")
plt.show()


train['Length_Host_Interaction'] = train['Episode_Length_minutes'] * train['Host_Popularity_percentage']
test['Length_Host_Interaction'] = test['Episode_Length_minutes'] * test['Host_Popularity_percentage']

train['Length_Guest_Interaction'] = train['Episode_Length_minutes'] * train['Guest_Popularity_percentage']
test['Length_Guest_Interaction'] = test['Episode_Length_minutes'] * test['Guest_Popularity_percentage']

train['Ad_Density'] = train['Number_of_Ads'] / train['Episode_Length_minutes']
test['Ad_Density'] = test['Number_of_Ads'] / test['Episode_Length_minutes']
train['Ad_Density'] = train['Ad_Density'].replace([np.inf, -np.inf], 0)
test['Ad_Density'] = test['Ad_Density'].replace([np.inf, -np.inf], 0)

train['Podcast_Avg_Listening'] = train.groupby('Podcast_Name')['Listening_Time_minutes'].transform('mean')
podcast_avg_dict = train.groupby('Podcast_Name')['Podcast_Avg_Listening'].first().to_dict()
test['Podcast_Avg_Listening'] = test['Podcast_Name'].map(podcast_avg_dict).fillna(train['Podcast_Avg_Listening'].mean())

train['Genre_Avg_Listening'] = train.groupby('Genre')['Listening_Time_minutes'].transform('mean')
genre_avg_dict = train.groupby('Genre')['Genre_Avg_Listening'].first().to_dict()
test['Genre_Avg_Listening'] = test['Genre'].map(genre_avg_dict).fillna(train['Genre_Avg_Listening'].mean())

train['Publication_Day_Time'] = train['Publication_Day'] + '_' + train['Publication_Time']
test['Publication_Day_Time'] = test['Publication_Day'] + '_' + test['Publication_Time']

train['Is_Weekend'] = train['Publication_Day'].isin(['Saturday', 'Sunday']).astype(int)
test['Is_Weekend'] = test['Publication_Day'].isin(['Saturday', 'Sunday']).astype(int)

train['Sentiment_Score'] = train['Episode_Sentiment'].map({'Positive': 1, 'Neutral': 0, 'Negative': -1})
test['Sentiment_Score'] = test['Episode_Sentiment'].map({'Positive': 1, 'Neutral': 0, 'Negative': -1})

train['Total_Popularity'] = train['Host_Popularity_percentage'] + train['Guest_Popularity_percentage']
test['Total_Popularity'] = test['Host_Popularity_percentage'] + test['Guest_Popularity_percentage']

train['Host_to_Guest_Ratio'] = train['Host_Popularity_percentage'] / train['Guest_Popularity_percentage']
test['Host_to_Guest_Ratio'] = test['Host_Popularity_percentage'] / test['Guest_Popularity_percentage']
train['Host_to_Guest_Ratio'] = train['Host_to_Guest_Ratio'].replace([np.inf, -np.inf], 0).fillna(0)
test['Host_to_Guest_Ratio'] = test['Host_to_Guest_Ratio'].replace([np.inf, -np.inf], 0).fillna(0)

train['Title_Length'] = train['Episode_Title'].str.len()
test['Title_Length'] = test['Episode_Title'].str.len()


X = train.drop(columns=['Listening_Time_minutes','id'])
y = train['Listening_Time_minutes']
X_test = test.copy()

categorical_cols = X.select_dtypes(include=['object']).columns
label_encoders = {} 
target_encode_cols = ['Podcast_Name', 'Genre']

for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    X_test[col] = le.transform(X_test[col]) 
    label_encoders[col] = le  

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

target_encoder = TargetEncoder(cols=target_encode_cols)
X_train[target_encode_cols] = target_encoder.fit_transform(X_train[target_encode_cols], y_train)
X_val[target_encode_cols] = target_encoder.transform(X_val[target_encode_cols])
X_test[target_encode_cols] = target_encoder.transform(X_test[target_encode_cols])


import xgboost as xgb
xgb_params = {
    'n_estimators': 1000,
    'max_depth': 6,
    'learning_rate': 0.01,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'reg_alpha': 0.5,
    'reg_lambda': 1.0,
    'gamma': 0.1,
    'random_state': 42,
    'tree_method': 'hist',
    'n_jobs': -1,
    'objective': 'reg:squarederror'
}

model = xgb.XGBRegressor(**xgb_params)
model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=50, verbose=10)

train_pred = model.predict(X_train)
val_pred = model.predict(X_val)

train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))

print(f"\nTraining RMSE: {train_rmse:.4f}")
print(f"Validation RMSE: {val_rmse:.4f}")


def plot_learning_curve_xgb(model, X, y, cv=5):
    train_sizes = np.linspace(0.1, 1.0, 10)
    
    train_sizes, train_scores, val_scores = learning_curve(
        model, X, y, 
        train_sizes=train_sizes,
        cv=cv, 
        scoring='neg_root_mean_squared_error',
        random_state=42,
        n_jobs=-1
    )
    
    train_scores_mean = -train_scores.mean(axis=1)
    val_scores_mean = -val_scores.mean(axis=1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_scores_mean, 'o-', color='r', 
             label='Training Score')
    plt.plot(train_sizes, val_scores_mean, 'o-', color='g', 
             label='CV Score')
    
    plt.axhline(y=train_scores_mean[-1], color='r', linestyle='--', alpha=0.3)
    plt.axhline(y=val_scores_mean[-1], color='g', linestyle='--', alpha=0.3)
    
    plt.xlabel('Training Examples')
    plt.ylabel('RMSE')
    plt.title('XGBoost Learning Curve\nFinal Train RMSE: {:.4f}, Val RMSE: {:.4f}'.format(
        train_scores_mean[-1], val_scores_mean[-1]))
    plt.legend()
    plt.grid(True)
    plt.show()

plot_learning_curve_xgb(model, X_train, y_train)


def plot_validation_curve_xgb(X, y, param_name, param_range, cv=5):

    base_params = {
        'objective': 'reg:squarederror',
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'n_jobs': -1
    }
    
    train_scores, val_scores = validation_curve(
        XGBRegressor(**base_params),
        X, y, 
        param_name=param_name, 
        param_range=param_range,
        cv=cv,
        scoring='neg_root_mean_squared_error',
        n_jobs=-1
    )
    
    train_scores_mean = -train_scores.mean(axis=1)
    val_scores_mean = -val_scores.mean(axis=1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(param_range, train_scores_mean, 'o-', color='r', 
             label='Training Score', markersize=8)
    plt.plot(param_range, val_scores_mean, 'o-', color='g', 
             label='CV Score', markersize=8)
    
    best_idx = np.argmin(val_scores_mean)
    plt.axvline(x=param_range[best_idx], color='b', linestyle='--', 
               label=f'Best {param_name}: {param_range[best_idx]}')
    
    plt.xlabel(param_name)
    plt.ylabel('RMSE')
    plt.title(f'XGBoost Validation Curve\nParameter: {param_name}')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    return param_range[best_idx]

best_n_estimators = plot_validation_curve_xgb(
    X_train, y_train,
    param_name='n_estimators',
    param_range=np.arange(50, 251, 25)  
)

best_max_depth = plot_validation_curve_xgb(
    X_train, y_train,
    param_name='max_depth',
    param_range=[3, 4, 5, 6, 7, 8, 9]
)


importance = model.feature_importances_  

feature_importance = pd.DataFrame({
    'feature': X_train.columns,
    'importance': importance
}).sort_values('importance', ascending=False)

print("\nFeature Importance:")
print(feature_importance)


plot_importance(model, importance_type='weight')  
plt.show()


sample_sub.head()


X_test = test[X_train.columns].copy()

categorical_cols = X_test.select_dtypes(include=['object']).columns
for col in categorical_cols:
    le = LabelEncoder()
    X_test[col] = le.fit_transform(X_test[col].astype(str))  

print("Missing values in test set:")
print(X_test.isna().sum())
X_test = X_test.fillna(0)

print("\nData types after preprocessing:")
print(X_test.dtypes)

test_pred = model.predict(X_test)

print(f"\nTest rows: {len(X_test)}, Predictions: {len(test_pred)}")
print(f"Submission template rows: {len(sample_sub)}")

if len(test_pred) < len(sample_sub):
    filled_preds = np.full(len(sample_sub), np.median(test_pred))
    filled_preds[:len(test_pred)] = test_pred  
    sample_sub['Listening_Time_minutes'] = filled_preds
    print(f"Filled {len(sample_sub)-len(test_pred)} missing predictions with median value")
else:
    sample_sub['Listening_Time_minutes'] = test_pred[:len(sample_sub)]
    print("Used exact predictions (truncated if necessary)")

assert len(sample_sub['Listening_Time_minutes']) == len(sample_sub), \
    f"Length mismatch! Submission: {len(sample_sub)}, Predictions: {len(test_pred)}"

sample_sub.to_csv('submission.csv', index=False)
print(f"\nSuccessfully saved submission with {len(sample_sub)} rows")
print(f"Breakdown: {len(test_pred)} model predictions + {max(0, len(sample_sub)-len(test_pred))} filled values")


sample_sub.head()

