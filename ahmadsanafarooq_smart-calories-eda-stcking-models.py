# Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
import xgboost as xgb
import lightgbm as lgb

from sklearn.metrics import mean_squared_error, make_scorer
import warnings
warnings.filterwarnings('ignore')


# Load Data
df_train = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv')
df_test = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')
df_sub = pd.read_csv('/kaggle/input/playground-series-s5e5/sample_submission.csv')


# Encode categorical
df_train['Sex'] = df_train['Sex'].map({'female': 0, 'male': 1})
df_test['Sex'] = df_test['Sex'].map({'female': 0, 'male': 1})


# Feature Engineering
def feature_engineering(df):
    df['BMI'] = df['Weight'] / ((df['Height'] / 100) ** 2)
    df['Duration_HR'] = df['Duration'] * df['Heart_Rate']
    df['Temp_log'] = np.log1p(df['Body_Temp'])
    df['BMI2'] = df['BMI'] ** 2
    return df

df_train = feature_engineering(df_train)
df_test = feature_engineering(df_test)


features_columns = ["Age",
    "Height",
    "Weight",
    "Duration",
    "Heart_Rate",
    "Body_Temp",
    "Calories"
]
for i in features_columns:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Histogram 
    sns.histplot(df_train[i], bins=30, kde=True, color='#FFB000', ax=axes[0])
    axes[0].set_title(f"Histogram of {i}", fontsize=14, fontweight='bold')
    axes[0].set_xlabel(i, fontsize=12)
    axes[0].set_ylabel("Frequency", fontsize=12)
    axes[0].grid(True, linestyle='--', alpha=0.5)

    # Boxplot 
    sns.boxplot(y=df_train[i], color='#FFB000', ax=axes[1], width=0.3, linewidth=1.5)
    axes[1].set_title(f"Boxplot of {i}", fontsize=14, fontweight='bold')
    axes[1].set_ylabel(i, fontsize=12)
    axes[1].grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()


# Data Visualization
plt.figure(figsize=(12, 6))
sns.histplot(df_train['Calories'], bins=40, kde=True, color='skyblue')
plt.title('Distribution of Calories', fontsize=16)
plt.xlabel('Calories')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()


gender_counts = df_train['Sex'].value_counts().sort_index() 

labels = ['Female (0)', 'Male (1)']
colors = ['#FF69B4', '#87CEEB'] 

plt.figure(figsize=(6, 6))
plt.pie(gender_counts, labels=labels, autopct='%1.1f%%', startangle=140, colors=colors)
plt.title("Gender Distribution in Dataset", fontsize=14, fontweight='bold')
plt.legend(labels, title="Legend", loc="upper right")
plt.tight_layout()
plt.show()


gender_counts = df_train['Sex'].value_counts().sort_index() 

labels = ['Female (0)', 'Male (1)']
colors = ['#FF69B4', '#87CEEB'] 

plt.figure(figsize=(6, 6))
plt.pie(gender_counts, labels=labels, autopct='%1.1f%%', startangle=140, colors=colors)
plt.title("Gender Distribution in Dataset", fontsize=14, fontweight='bold')
plt.legend(labels, title="Legend", loc="upper right")
plt.tight_layout()
plt.show()



# Features and target
features = ['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp', 'Sex',
            'BMI', 'Duration_HR', 'Temp_log', 'BMI2']
X = df_train[features]
y = df_train['Calories']
X_test = df_test[features]


# Ensure no negative targets for safety (not required for RMSE, but avoids MSLE issues if ever used)
if (y < 0).any():
    print("âš ï¸� Found negative targets. Filtering them out...")
    valid_idx = y >= 0
    X = X[valid_idx]
    y = y[valid_idx]


# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)


# Models
xgb_model = xgb.XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=4, random_state=42)
lgb_model = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, max_depth=4, random_state=42, verbose=-1)
rf_model = RandomForestRegressor(n_estimators=150, max_depth=6, random_state=42)
ridge_model = Ridge(alpha=0.5)


# Define RMSE scorer explicitly
rmse_scorer = make_scorer(mean_squared_error, greater_is_better=False, squared=False)


# Stacking
stack = StackingRegressor(
    estimators=[
        ('xgb', xgb_model),
        ('lgb', lgb_model),
        ('rf', rf_model),
    ],
    final_estimator=ridge_model,
    cv=KFold(n_splits=5, shuffle=True, random_state=42),
    passthrough=True
)


# Train
stack.fit(X_scaled, y)


# Predict
preds = stack.predict(X_test_scaled)

# Clip predictions to ensure non-negative values
preds = np.clip(preds, 0, None)

if np.any(np.isnan(preds)) or np.any(np.isinf(preds)):
    print("âš ï¸� Prediction contains NaN or Inf. Check your input features or model.")
else:
    df_sub['Calories'] = preds
    df_sub.to_csv('submission.csv', index=False)
    print("âœ… Submission file saved successfully as 'submission.csv'")
    print(df_sub.head())

