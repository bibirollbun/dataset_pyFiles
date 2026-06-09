import pandas as pd
import numpy as np

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Modeling
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_log_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import lightgbm as lgb

import warnings
warnings.filterwarnings("ignore")


df = pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv")


df.head(10)


df.info()


df.drop('id',axis=1).describe().T


plt.figure(figsize=(8, 5))
sns.histplot(df["Calories"], bins=50, kde=True, color='blue')
plt.title("Calories Distribution")
plt.show()


grouped_df = df.groupby(['Age', 'Sex'])['Calories'].mean().reset_index()

custom_palette = {'male': 'blue', 'female': 'pink'}

plt.figure(figsize=(12, 6))
sns.barplot(data=grouped_df, x='Age', y='Calories', hue='Sex', palette=custom_palette)

plt.title('Average Calories by Age and Sex', fontsize=16, fontweight='bold')
plt.xlabel('Age', fontsize=12)
plt.ylabel('Average Calories', fontsize=12)
plt.xticks(rotation=45)
plt.legend(title='Sex')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()


plt.figure(figsize=(10, 6))
sns.heatmap(df.drop(['id','Sex'],axis=1).corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Feature Correlation Matrix")
plt.show()


df[df['Duration'] < 0]


df[df['Heart_Rate'] > 200]


df[df['Heart_Rate'] < 50]


df[df['Body_Temp'] > 42]


df[df['Body_Temp'] <  35]


plt.figure(figsize=(10, 4))
sns.boxplot(x=df['Calories'])
plt.title("Calories Outlier Detection")
plt.show()


high_cal = df[df['Calories'] > df['Calories'].quantile(0.99)]
print(f"Number of high-calorie outliers: {len(high_cal)}")


sex_encoder = LabelEncoder()
df['Sex'] = sex_encoder.fit_transform(df['Sex'])

df['BMI'] = df['Weight'] / ((df['Height']/100) ** 2)
df['MET_proxy'] = df['Duration'] * df['Heart_Rate'] * df['Body_Temp']
df['HR_x_Duration'] = df['Heart_Rate'] * df['Duration']
df['Weight_x_Duration'] = df['Weight'] * df['Duration']
df['Age_x_HR'] = df['Age'] * df['Heart_Rate']


df['Age_Group'] = pd.cut(df['Age'], bins=[19, 30, 50, 80], labels=['20-30', '31-50', '51-79'])

# Average Calories burned by Age Group and Sex
plt.figure(figsize=(10, 6))
sns.barplot(data=df, x='Age_Group', y='Calories', hue='Sex')
plt.title("Average Calories Burned by Age Group and Sex")
plt.ylabel("Average Calories")
plt.xlabel("Age Group")
plt.show()


df['AgeSex_Group'] = df['Age_Group'].astype(str) + '_' + df['Sex'].map({0: 'Female', 1: 'Male'})
df['AgeSex_Group']


age_sex_encoder = LabelEncoder()
df['AgeSex_Group_Code'] = age_sex_encoder.fit_transform(df['AgeSex_Group'])


df['Calories_log'] = np.log1p(df['Calories'])


X = df.drop(columns=["id","Calories", "Calories_log","Age_Group","AgeSex_Group"])
y = df["Calories_log"]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


def rmsle(y_true, y_pred):
    return np.sqrt(mean_squared_log_error(np.expm1(y_true), np.expm1(y_pred)))


# Linear Regression
lr = LinearRegression()
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
lr.fit(X_train_scaled, y_train)
lr_pred = lr.predict(X_test_scaled)
print("Linear Regression RMSLE:", rmsle(y_test, lr_pred))

# Random Forest
rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
print("Random Forest RMSLE:", rmsle(y_test, rf_pred))

# XGBoost
xgbr = xgb.XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=6, random_state=42)
xgbr.fit(X_train, y_train)
xgb_pred = xgbr.predict(X_test)
print("XGBoost RMSLE:", rmsle(y_test, xgb_pred))

# LightGBM
lgbm = lgb.LGBMRegressor(n_estimators=200, learning_rate=0.05, max_depth=6, random_state=42, verbose=-1)
lgbm.fit(X_train, y_train)
lgb_pred = lgbm.predict(X_test)
print("LightGBM RMSLE:", rmsle(y_test, lgb_pred))


lgb.plot_importance(lgbm, max_num_features=10, importance_type='gain')
plt.title("Top 10 Important Features - LightGBM")
plt.show()


ensemble_pred = (xgb_pred + lgb_pred) / 2
print("Ensemble (XGBoost + LightGBM) RMSLE:", rmsle(y_test, ensemble_pred))


new_data = pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv")

new_data['Sex'] = sex_encoder.transform(new_data['Sex'])
new_data['BMI'] = new_data['Weight'] / ((new_data['Height'] / 100) ** 2)
new_data['MET_proxy'] = new_data['Duration'] * new_data['Heart_Rate'] * new_data['Body_Temp']
new_data['HR_x_Duration'] = new_data['Heart_Rate'] * new_data['Duration']
new_data['Weight_x_Duration'] = new_data['Weight'] * new_data['Duration']
new_data['Age_x_HR'] = new_data['Age'] * new_data['Heart_Rate']
new_data['Age_Group'] = pd.cut(new_data['Age'], bins=[19, 30, 50, 80], labels=['20-30', '31-50', '51-79'])
new_data['AgeSex_Group'] = new_data['Age_Group'].astype(str) + '_' + new_data['Sex'].map({0: 'Female', 1: 'Male'})
new_data['AgeSex_Group_Code'] = age_sex_encoder.transform(new_data['AgeSex_Group'])

# Select features
X_new = new_data[X_train.columns]

# Predict
xgb_new_pred = xgbr.predict(X_new)
lgb_new_pred = lgbm.predict(X_new)
ensemble_new_pred = (xgb_new_pred + lgb_new_pred) / 2


final_predictions = np.expm1(ensemble_new_pred)
print(len(final_predictions))
print(final_predictions)


submission = pd.read_csv('/kaggle/input/playground-series-s5e5/sample_submission.csv')
submission["Calories"] = final_predictions
submission.to_csv("submission.csv", index=False)
print('submission saved')
submission.head()




