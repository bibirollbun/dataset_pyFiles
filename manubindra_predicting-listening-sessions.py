# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input/playground-series-s5e4'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


train_df = pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv')
train_df.head()


train_df.info()


train_df.describe().round(2)


train_df[train_df['Episode_Length_minutes'].isna()]['Genre'].value_counts(normalize = True).round(2)


g = sns.FacetGrid(train_df, col = 'Genre',col_wrap= 3, height= 4)
g.map(plt.hist,"Episode_Length_minutes",bins = 20)


from scipy.stats import skew

print(skew(train_df['Episode_Length_minutes'].dropna()))


mean_per_genre = train_df.groupby('Genre')['Episode_Length_minutes'].mean()


train_df['Episode_Length_minutes'] = train_df.apply(
    lambda row: mean_per_genre[row['Genre']] if pd.isna(row['Episode_Length_minutes']) else row['Episode_Length_minutes'],
    axis=1
)
train_df.isna().sum()


mean_per_genre


train_df['Episode_Length_minutes'].hist()


g = sns.FacetGrid(train_df, col = 'Genre',col_wrap= 3, height= 4)
g.map(plt.hist,"Guest_Popularity_percentage",bins = 20)


print(skew(train_df['Guest_Popularity_percentage'].dropna()))


mean_per_genre = train_df.groupby('Genre')['Guest_Popularity_percentage'].median()


train_df['Guest_Popularity_percentage'] = train_df.apply(
    lambda row: mean_per_genre[row['Genre']] if pd.isna(row['Guest_Popularity_percentage']) else row['Guest_Popularity_percentage'],
    axis=1
)
train_df.isna().sum()


train_df = train_df.dropna()
train_df = train_df.drop(columns='id')
train_df


for x in train_df.select_dtypes(include='object').columns:
    print(f'{x} unique values: {train_df[x].unique()}')
    


fig,(ax,ax1) = plt.subplots(ncols=2,figsize=(16,6))
sns.histplot(train_df['Publication_Day'],ax=ax)
sns.histplot(train_df['Publication_Time'],ax=ax1)
sns.despine()


sns.histplot(train_df['Episode_Length_minutes'],bins=60)
sns.despine()


train_df[train_df['Episode_Length_minutes'] == 0]


train_df.groupby('Publication_Day')['Listening_Time_minutes'].sum().sort_values(ascending=False).plot(kind='bar')


train_df.groupby('Genre')['Listening_Time_minutes'].sum().sort_values(ascending=False).plot(kind='bar')


sns.heatmap(train_df.select_dtypes(exclude='object').corr(),annot = True)


x = train_df['Episode_Length_minutes'].tolist()
x = sm.add_constant(x)
y = train_df['Listening_Time_minutes'].tolist()

result = sm.OLS(y,x).fit()
result.summary()


sns.lmplot(train_df,x='Episode_Length_minutes',y='Listening_Time_minutes',hue = 'Genre',ci= False)
sns.despine()


from sklearn.feature_selection import mutual_info_regression


X = train_df.drop(['Listening_Time_minutes','Episode_Title','Podcast_Name'], axis=1)
X = pd.get_dummies(X, drop_first=True)
y = train_df['Listening_Time_minutes']

X_sample = X.sample(n=10000, random_state=42)
y_sample = y.loc[X_sample.index]

importances = mutual_info_regression(X_sample, y_sample)
feature_scores = pd.Series(importances, index=X_sample.columns).sort_values(ascending=False)
print(feature_scores)


top_features = [
    'Episode_Length_minutes',
    'Host_Popularity_percentage',
    'Guest_Popularity_percentage',
    'Number_of_Ads',
    'Episode_Sentiment_Neutral',
    'Publication_Time_Evening',
    'Episode_Sentiment_Positive',
    'Publication_Time_Night',
    'Publication_Time_Morning'
]


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
X = train_df.drop(['Listening_Time_minutes','Episode_Title','Podcast_Name'], axis=1)
X= pd.get_dummies(X, drop_first=True)
X = X[top_features]

y = train_df['Listening_Time_minutes']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = LinearRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.2f}")
print(f"R² Score: {r2:.4f}")


# Residuals
residuals = y_test - y_pred

# Plot
plt.figure(figsize=(8, 5))
sns.scatterplot(x=y_pred, y=residuals, alpha=0.5)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel("Predicted Listening Time")
plt.ylabel("Residuals (Actual - Predicted)")
plt.title("Residual Plot - Linear Regression")
plt.show()


from sklearn.ensemble import RandomForestRegressor
X = train_df.drop(['Listening_Time_minutes','Episode_Title','Podcast_Name'], axis=1)
X= pd.get_dummies(X, drop_first=True)
X = X[top_features]
y_log = np.log1p(train_df['Listening_Time_minutes']) 

X_train, X_test, y_train_log, y_test_log = train_test_split(X, y_log, test_size=0.2, random_state=42)
rf = RandomForestRegressor(
    n_estimators=150,
    max_depth=10,
    min_samples_leaf=3,
    criterion='squared_error',
    n_jobs=-1,
    random_state=42
)
rf.fit(X_train, y_train_log)

# ---- STEP 4: Predict and inverse transform ----
y_pred_log = rf.predict(X_test)
y_pred = np.expm1(y_pred_log)        # Predicted values on original scale
y_test = np.expm1(y_test_log)        # Actual values on original scale

# ---- STEP 5: Evaluate ----
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"Random Forest R²: {r2:.4f}")
print(f"Random Forest RMSE: {rmse:.2f}")


# Residuals
residuals = y_test - y_pred

# Plot
plt.figure(figsize=(8, 5))
sns.scatterplot(x=y_pred_rf, y=residuals, alpha=0.5)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel("Predicted Listening Time")
plt.ylabel("Residuals (Actual - Predicted)")
plt.title("Residual Plot - Random Forest")
plt.show()


from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score

# Model
xgb = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42)

# Fit on training data
xgb.fit(X_train, y_train)

# Predict on test set
y_pred_xgb = xgb.predict(X_test)
r2_xgb = r2_score(y_test, y_pred_xgb)
rmse_xgb = np.sqrt(mean_squared_error(y_test, y_pred_xgb))

print(f"XGBoost RMSE (test): {rmse_xgb:.2f}")
print(f"XGBoost R² Score (test): {r2_xgb:.4f}")

# ---- Cross-validation ----
# For RMSE (note: scores are negative, so take -mean to get positive RMSE)
cv_rmse_scores = cross_val_score(xgb, X_train, y_train, cv=5, scoring='neg_root_mean_squared_error')
cv_r2_scores = cross_val_score(xgb, X_train, y_train, cv=5, scoring='r2')

print(f"\nCross-Validated RMSE: {-cv_rmse_scores.mean():.2f} ± {cv_rmse_scores.std():.2f}")
print(f"Cross-Validated R²: {cv_r2_scores.mean():.4f} ± {cv_r2_scores.std():.4f}")


y_pred = xgb.predict(X_test)
residuals = y_test - y_pred
plt.figure(figsize=(8, 5))
plt.scatter(y_pred, residuals, alpha=0.5)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel("Predicted Listening Time")
plt.ylabel("Residuals (Actual - Predicted)")
plt.title("Residual Plot - XGBoost")
plt.show()


test_df = pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv')
test_df.head()


test_df.set_index('id',inplace=True)


test_df = test_df.drop(columns= ['Podcast_Name','Episode_Title'])
test_df.head()
submission_df = test_df


top_features = [
    'Episode_Length_minutes',
    'Host_Popularity_percentage',
    'Guest_Popularity_percentage',
    'Number_of_Ads',
    'Episode_Sentiment_Neutral',
    'Publication_Time_Evening',
    'Episode_Sentiment_Positive',
    'Publication_Time_Night',
    'Publication_Time_Morning'
]


submission_df['Guest_Popularity_percentage'] = submission_df['Guest_Popularity_percentage'].fillna(submission_df['Guest_Popularity_percentage'].mean())
submission_df = submission_df.dropna(subset='Episode_Length_minutes')
submission_processed = pd.get_dummies(submission_df, drop_first=True)
submission_processed = submission_processed[top_features]
submission_processed_aligned = submission_processed.reindex(columns=X_train.columns, fill_value=0)


submission_processed_aligned


submission_preds_log = xgb.predict(submission_processed_aligned)
submission_preds = submission_preds_log


output_df = pd.DataFrame({
    'id': submission_df.index,
    'Listening_Time_minutes': submission_preds
})
output_df.to_csv("submission.csv", index=False)


fig, (ax,ax1) = plt.subplots(ncols = 2, figsize = (12,6))
output_df['Listening_Time_minutes'].hist(ax = ax)
train_df['Listening_Time_minutes'].hist(ax=ax1)
ax.set_title('Predicted for Submission')
ax1.set_title('Actual for Training')
plt.show()




