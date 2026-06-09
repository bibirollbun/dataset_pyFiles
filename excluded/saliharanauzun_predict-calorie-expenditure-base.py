import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import seaborn as sns
import warnings; warnings.filterwarnings('ignore')


train = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')
sample_submission = pd.read_csv('/kaggle/input/playground-series-s5e5/sample_submission.csv')


train.head()


test.tail()


sample_submission.sample()


train.info()


test.info()


train.describe().T


test.describe().T


train.corr(numeric_only=True)


train.isnull().sum()


train['Calories'].hist(bins=20)


train['Duration'].value_counts()


train.duplicated().sum()


feat = [c for c in train.columns if c not in ['Calories','id']]; dup = train[train.duplicated(subset=feat, keep=False)].sort_values(feat); 
print(f" pseudoduplicate rows: {len(dup)}"); display(dup.head(10))



train = (
    train.groupby([col for col in train.columns if col not in ['id', 'Calories']])
         .agg({'Calories': 'mean', 'id': 'first'})
         .reset_index()
)


train.head()


train.info()


plt.figure(figsize=(12,8)); sns.heatmap(train.corr(numeric_only=True), annot=True, fmt=".2f", cmap="coolwarm", square=True); plt.show()


# Calculate Q1 (01th percentile) and Q3 (99th percentile) for each numerical column
numerical_columns = train.select_dtypes(include='number')

Q1 = numerical_columns.quantile(0.01)
Q3 = numerical_columns.quantile(0.99)

# IQR 
IQR = Q3 - Q1

# Define lower and upper bounds for outliers
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Find Outliers
outliers = ((numerical_columns < lower_bound) | (numerical_columns > upper_bound))

# Outlier rows
outlier_rows = train[outliers.any(axis=1)]

print(outlier_rows)



df_all = pd.concat([train, test], ignore_index=True)


#    — BMI kg/m²
df_all['BMI'] = df_all['Weight'] / (df_all['Height'] / 100)**2

#    — Age Groups
#df_all['Age_bin'] = pd.cut(df_all['Age'],
#                           bins=[0,20,30,40,50,60,100],
#                           labels=['<20','20–29','30–39','40–49','50–59','60+'])

#    — Temp / HeartRate 
df_all['Temp_HR_ratio'] = df_all['Body_Temp'] / df_all['Heart_Rate']





df_all = pd.get_dummies(df_all, columns=['Sex'], prefix='Sex', drop_first=True, dtype=int)


abs(df_all.corr(numeric_only=True)["Calories"].sort_values(ascending=False))


df_all.info()


del df_all['id']


df_train = df_all[df_all['Calories'].notnull()]  
df_test = df_all[df_all['Calories'].isnull()]    



X = df_train.drop(columns=['Calories'])    
y_log = np.log1p(df_train['Calories'])   


X_train, X_val, y_train_log, y_val_log = train_test_split(X, y_log, test_size=0.2, random_state=42)


y_val = df_train.loc[y_val_log.index, 'Calories']


from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import mean_squared_error
def rmsle(y_true, y_pred):
    y_true = np.maximum(0, y_true)
    y_pred = np.maximum(0, y_pred)
    return np.sqrt(
        mean_squared_error(
            np.log1p(y_true),
            np.log1p(y_pred)
        )
    )


from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb

#LGBM Regressor
lgbm_model = lgb.LGBMRegressor()
lgbm_model.fit(X_train, y_train_log)
y_pred_log_lgb = lgbm_model.predict(X_val)
y_pred_lgb = np.expm1(y_pred_log_lgb)
print(f"LightGBM RMSLE: {rmsle(y_val, y_pred_lgb):.5f}")

# Linear Regression 
lr_model = LinearRegression()
lr_model.fit(X_train, y_train_log)
y_pred_log_lr = lr_model.predict(X_val)
y_pred_lr = np.expm1(y_pred_log_lr)
print(f"Linear Regression RMSLE: {rmsle(y_val, y_pred_lr):.5f}")

# Decision Tree 
dt_model = DecisionTreeRegressor()
dt_model.fit(X_train, y_train_log)
y_pred_log_dt = dt_model.predict(X_val)
y_pred_dt = np.expm1(y_pred_log_dt)
print(f"Decision Tree RMSLE: {rmsle(y_val, y_pred_dt):.5f}")

# Random Forest 
rf_model = RandomForestRegressor()
rf_model.fit(X_train, y_train_log)
y_pred_log_rf = rf_model.predict(X_val)
y_pred_rf = np.expm1(y_pred_log_rf)
print(f"Random Forest RMSLE: {rmsle(y_val, y_pred_rf):.5f}")


plt.figure(figsize=(7, 5))
sns.regplot(
    x=y_val,
    y=y_pred_lgb,
    lowess=True,
    line_kws={"color": "red", "linewidth": 2},       
    scatter_kws={"alpha": 0.3, "color": "gray"}       
)
plt.xlabel("Actual Calories")
plt.ylabel("Predicted Calories (LightGBM)")
plt.title("Actual vs Predicted (LightGBM, with LOWESS smoothing)")
plt.grid(True)
plt.tight_layout()
plt.show()


X_test = df_test.drop(columns=['Calories'], errors='ignore')

X_test = X_test[X_train.columns]

y_pred_log_test = lgbm_model.predict(X_test)

y_pred_test = np.expm1(y_pred_log_test)


submission = pd.DataFrame({
    'id': test['id'],  
    'Calories': y_pred_test  
})

submission.to_csv('submission.csv', index=False)

