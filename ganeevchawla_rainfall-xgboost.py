# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import xgboost as xgb
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


train_data = pd.read_csv("/kaggle/input/playground-series-s5e3/train.csv")
train_data.head()



train_data.info()


df = train_data.copy()


def day_to_season(day):
    if day <= 75 or day >= 354:
        return 'winter'
    elif day <= 171:
        return 'spring'
    elif day <= 264:
        return 'summer'
    elif day <= 353:
        return 'autumn'


df['season'] = df['day'].apply(lambda x: day_to_season(x))


total_rainfall_year = df['rainfall'].sum()
seasonal_rainfall = df.groupby('season')['rainfall'].sum().rename("seasonal_rainfall")


plt.bar(seasonal_rainfall.index, seasonal_rainfall.values, color='b', alpha=0.7)


season_ratios = (seasonal_rainfall / total_rainfall_year).rename("season_num")

# Merge this ratio back into the main DataFrame
df = df.merge(season_ratios, on='season')


df


features = [f for f in df.columns if f not in  ['id' ,'rainfall', 'mintemp', 'temparature', 'season', 'day']]
corr = df[features].corr().abs()

plt.figure(figsize=(10, 6))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.show()


upper_triangle = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

# drop highly correlated features
to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > 0.95)]

df_reduced = df.drop(columns=to_drop)

print(f"Removed features: {to_drop}")



features = [f for f in features if f not in to_drop]
features


def detect_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] < lower_bound) | (df[column] > upper_bound)]

# Count outliers for each numerical feature
outlier_counts = {var: len(detect_outliers_iqr(df, var)) for var in features}

# Create barplot
plt.figure(figsize=(10, 5))
sns.barplot(x=list(outlier_counts.keys()), y=list(outlier_counts.values()), color='coral')
plt.xticks(rotation=45)
plt.title("Number of Outliers per Numerical Feature (IQR Method)", fontsize=14, weight='bold')
plt.ylabel("Number of Outliers")
plt.xlabel("Feature")
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()



# Remove outliers
def remove_outliers(df, features):
    for f in features:
        
        Q1 = df[f].quantile(0.25)
        Q3 = df[f].quantile(0.75)
        
        # Compute IQR
        IQR = Q3 - Q1
        
        # Define bounds
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Filter out outliers
        df = df[(df[f] >= lower_bound) & (df[f] <= upper_bound)]
    return df


df = remove_outliers(df, features)
df


X = df[features]
y = df['rainfall']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
xgb_model = xgb.XGBClassifier(objective='binary:logistic',eval_metric='logloss', random_state=42)



param_grid = {
    'n_estimators': [50, 100, 150,],
    'max_depth': [ 7, 9, ],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.4, 0.7, 0.8],
    'colsample_bytree': [ 0.5, 0.8]
}

grid_search = GridSearchCV(
    estimator=xgb_model,
    param_grid=param_grid,
    scoring='roc_auc',
    cv=5,
    verbose=1,
    n_jobs=-1
)

grid_search.fit(X_train, y_train)
print(grid_search.best_params_)


best_params = grid_search.best_params_
print(grid_search.best_params_)


model = xgb.XGBClassifier(**best_params, objective='binary:logistic',eval_metric='logloss', random_state=42)
model.fit(X_train, y_train)
model


from sklearn.metrics import accuracy_score, roc_auc_score

y_pred = model.predict_proba(X_test)[:,1]

m = roc_auc_score(y_test, y_pred)
print(f"XGBoost CV Score AUC = {m:.3f}")



feature_importance = model.feature_importances_

importance_df = pd.DataFrame({'Feature': X_train.columns, 'Importance': feature_importance})
importance_df = importance_df.sort_values(by="Importance", ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=importance_df, palette="viridis")
plt.title("Feature Importance")
plt.show()


threshold = 0.05
selected_features = importance_df[importance_df["Importance"] > threshold]["Feature"].tolist()
print(f"Features above threshold: {selected_features}")
# Keep only selected features
X_thresh = df[features]
y_thresh = df['rainfall']


test_data = pd.read_csv("/kaggle/input/playground-series-s5e3/test.csv")
test_data.head()


df_test = test_data.copy()


df_test['season'] = df_test['day'].apply(lambda x: day_to_season(x))
df_test['season_num'] = df_test['season'].map(season_ratios)


df_test


df_test.info()


df_test[df_test['winddirection'].isna()]


#df_test['winddirection'] = df_test['winddirection'].fillna(df_test['winddirection'].mean())
df_test.fillna({'winddirection': df_test['winddirection'].mean()}, inplace=True)


df_test[df_test['id'] ==2707]


df_test.info()


X_test = (df_test[features])

predictions = model.predict_proba(X_test)[:,1]


output = pd.DataFrame({'id': df_test.id, 'rainfall': predictions})
output.to_csv('/kaggle/working/submission.csv', index=False)




