!pip install --upgrade scikit-learn
!pip install xgboost


import warnings
warnings.filterwarnings("ignore")


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from xgboost import XGBRegressor
from xgboost import plot_importance

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))



df = pd.read_csv("/kaggle/input/playground-series-s3e14/train.csv", index_col=0)


df.head()


df.describe()


sns.histplot(df['yield'], kde=True)
plt.title("Distribution of Yield")
plt.show()


X = df.drop(columns=['yield'])
y = df['yield']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)


plt.figure(figsize=(8, 6))
sns.histplot(y_train, bins=20, kde=True)
plt.title('Distribution of Blueberry Yield')
plt.xlabel('Yield')
plt.ylabel('Frequency')
plt.show()


train = X_train.copy()
train['yield'] = y_train


numeric_cols = train.select_dtypes(include='number').columns

plt.figure(figsize=(16, 12))
for i, col in enumerate(numeric_cols):
    plt.subplot((len(numeric_cols) - 1) // 3 + 1, 3, i + 1)
    sns.histplot(train[col], kde=True, bins=30, color='skyblue')
    plt.title(f"Histogram: {col}")
    plt.tight_layout()
plt.show()


plt.figure(figsize=(16, 12))
for i, col in enumerate(numeric_cols):
    plt.subplot((len(numeric_cols) - 1) // 3 + 1, 3, i + 1)
    sns.boxplot(x=train[col], color='lightcoral')
    plt.title(f"Boxplot: {col}")
    plt.tight_layout()
plt.show()


selected = ['clonesize', 'honeybee', 'RainingDays', 'fruitset', 'yield']
sns.pairplot(train[selected], corner=True)
plt.suptitle("Pairplots", y=1.02)
plt.show()


numeric_cols = train.select_dtypes(include='number').columns

outlier_counts = pd.Series(dtype=int)

for col in numeric_cols:
    Q1 = train[col].quantile(0.25)
    Q3 = train[col].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = train[(train[col] < lower_bound) | (train[col] > upper_bound)]
    outlier_counts[col] = outliers.shape[0]

outlier_counts = outlier_counts.sort_values(ascending=False)
print(outlier_counts)


def remove_outliers_iqr(df, columns):
    df_cleaned = df.copy()
    for col in columns:
        Q1 = df_cleaned[col].quantile(0.25)
        Q3 = df_cleaned[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df_cleaned = df_cleaned[(df_cleaned[col] >= lower_bound) & (df_cleaned[col] <= upper_bound)]
    return df_cleaned


corr = train.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
plt.figure(figsize=(12,10))
sns.heatmap(corr, mask=mask, annot=True, cmap='mako', fmt=".2f", linewidths=0.5)
plt.title('Macierz korelacji (dolny trójkąt)')
plt.show()


pd.options.display.float_format = '{:.2f}'.format

vif_data = pd.DataFrame()
vif_data["feature"] = X_train.columns
vif_data["VIF"] = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]

print(vif_data)


regr_X_train = X_train.drop(columns=['MaxOfUpperTRange', 'MinOfUpperTRange', 'MaxOfLowerTRange', 'AverageOfUpperTRange', 'MinOfLowerTRange', 'AverageOfLowerTRange', 'AverageRainingDays', 'andrena', 'osmia', 'bumbles', 'fruitmass', 'seeds'])
regr_X_test = X_test.drop(columns=['MaxOfUpperTRange', 'MinOfUpperTRange', 'MaxOfLowerTRange', 'AverageOfUpperTRange', 'MinOfLowerTRange', 'AverageOfLowerTRange', 'AverageRainingDays', 'andrena', 'osmia', 'bumbles', 'fruitmass', 'seeds'])


pd.options.display.float_format = '{:.2f}'.format

vif_data = pd.DataFrame()
vif_data["feature"] = regr_X_train.columns
vif_data["VIF"] = [variance_inflation_factor(regr_X_train.values, i) for i in range(regr_X_train.shape[1])]

print(vif_data)


columns_to_clean = ['fruitmass', 'seeds', 'honeybee']
train_cleaned = remove_outliers_iqr(train, columns_to_clean)
print(train_cleaned.shape)

regr_X_train = train_cleaned.drop(columns=['yield'])
regr_y_train = train_cleaned['yield']

regr_X_train = regr_X_train.drop(columns=['MaxOfUpperTRange', 'MinOfUpperTRange', 'MaxOfLowerTRange', 'AverageOfUpperTRange', 'MinOfLowerTRange', 'AverageOfLowerTRange', 'AverageRainingDays', 'andrena', 'osmia', 'bumbles', 'fruitmass', 'seeds'])


model = LinearRegression()

rfe = RFE(estimator=model, n_features_to_select=3)

rfe.fit(regr_X_train, regr_y_train)

selected_features = regr_X_train.columns[rfe.support_]
print("Selected features (RFE):")
for f in selected_features:
    print("-", f)


X_train_sm = sm.add_constant(regr_X_train)

model_sm = sm.OLS(regr_y_train, X_train_sm).fit()

print(model_sm.summary())

X_test_sm = sm.add_constant(regr_X_test)
y_pred_sm = model_sm.predict(X_test_sm)


mae = mean_absolute_error(y_test, y_pred_sm)
rmse = root_mean_squared_error(y_test, y_pred_sm)
r2 = r2_score(y_test, y_pred_sm)

print(f'MAE: {mae:.4f}')
print(f'RMSE: {rmse:.4f}')
print(f'R^2: {r2:.4f}')


regr_X_train = regr_X_train[['clonesize', 'RainingDays', 'fruitset']]
regr_X_test = regr_X_test[['clonesize', 'RainingDays', 'fruitset']]


X_train_sm = sm.add_constant(regr_X_train)

model_sm = sm.OLS(regr_y_train, X_train_sm).fit()

print(model_sm.summary())

X_test_sm = sm.add_constant(regr_X_test)
y_pred_sm = model_sm.predict(X_test_sm)


X_train_sm = sm.add_constant(regr_X_train)

model_sm = sm.OLS(regr_y_train, X_train_sm).fit()

print(model_sm.summary())

X_test_sm = sm.add_constant(regr_X_test)
y_pred_sm = model_sm.predict(X_test_sm)


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

rf = RandomForestRegressor(n_estimators=100, random_state=42)

rfe = RFE(estimator=rf, n_features_to_select=10)
rfe.fit(X_train, y_train)

selected_features = X_train.columns[rfe.support_]

print("Selected features:")
print(selected_features)

X_train = X_train[selected_features]
X_test = X_test[selected_features]


rf = RandomForestRegressor(n_estimators=300, random_state=42)

rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = root_mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'MAE: {mae:.4f}')
print(f'RMSE: {rmse:.4f}')
print(f'R^2: {r2:.4f}')


importances = rf.feature_importances_
feature_names = X_train.columns

feat_importances = pd.Series(importances, index=feature_names)
feat_importances = feat_importances.sort_values(ascending=False)

plt.figure(figsize=(10, 6))
feat_importances.plot(kind='bar')
plt.title('Feature Importances')
plt.ylabel('Importance')
plt.xlabel('Features')
plt.tight_layout()
plt.show()


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

rf = XGBRegressor(n_estimators=100, random_state=42)

rfe = RFE(estimator=rf, n_features_to_select=10)
rfe.fit(X_train, y_train)

selected_features = X_train.columns[rfe.support_]

print("Selected features:")
print(selected_features)

X_train = X_train[selected_features]
X_test = X_test[selected_features]


xgb_model = XGBRegressor(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    use_label_encoder=False,
    eval_metric='rmse'
)

xgb_model.fit(X_train, y_train)

y_pred = xgb_model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = root_mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'MAE: {mae:.4f}')
print(f'RMSE: {rmse:.4f}')
print(f'R^2: {r2:.4f}')


plt.figure(figsize=(10, 6))
plot_importance(xgb_model, importance_type='gain', max_num_features=20, show_values=False)
plt.title('XGBoost Feature Importances (by gain)')
plt.tight_layout()
plt.show()

