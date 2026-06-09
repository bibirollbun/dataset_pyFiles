import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from scipy.stats import skew

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings("ignore")





# data
df_train = pd.read_csv("/kaggle/input/playground-series-s5e10/train.csv")
df_test = pd.read_csv("/kaggle/input/playground-series-s5e10/test.csv")
sample_submission = pd.read_csv("/kaggle/input/playground-series-s5e10/sample_submission.csv")

df_train.shape, df_test.shape



df_train.head()




df_train.info()


df_train.isnull().sum().sort_values(ascending=False).head(10)



for col in df_train.columns:
    unique_vals = df_train[col].unique()
    if len(unique_vals) < 20:  
        print(f"{col}: {unique_vals}")



df_train=df_train.drop('id',axis=1)
df_test=df_test.drop('id',axis=1)


#  train
numerical_cols_train = df_train.select_dtypes(include=['float64', 'int64']).columns.tolist()
categorical_cols_train = df_train.select_dtypes(include=['object', 'bool']).columns.tolist()

print("Train - Numerical Columns:", numerical_cols_train)
print("Train - Categorical Columns:", categorical_cols_train)

#  test
numerical_cols_test = df_test.select_dtypes(include=['float64', 'int64']).columns.tolist()
categorical_cols_test = df_test.select_dtypes(include=['object', 'bool']).columns.tolist()

print("Test - Numerical Columns:", numerical_cols_test)
print("Test - Categorical Columns:", categorical_cols_test)





fig, axs = plt.subplots(ncols=4, nrows=(len(numerical_cols_train) // 4) + 1, figsize=(20, 10))
axs = axs.flatten()
 

for i, col in enumerate(numerical_cols_train):
    sns.boxplot(y=df_train[col], ax=axs[i], color="orange")
    axs[i].set_title(col, fontsize=10)


for j in range(i+1, len(axs)):
   fig.delaxes(axs[j])

plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=5.0)
plt.show()



fig, axs = plt.subplots(ncols=7, nrows=(len(numerical_cols_train)//4)+1, figsize=(20, 10))
axs = axs.flatten()

for i, col in enumerate(numerical_cols_train):
    sns.histplot(df_train[col], ax=axs[i], kde=True, )
    axs[i].set_title(col, fontsize=10)


for j in range(i+1, len(axs)):
    fig.delaxes(axs[j])

plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=5.0)
plt.show()




# correlation
corr_matrix = df_train[numerical_cols_train ].corr()

#  Heatmap
plt.figure(figsize=(10,8))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show()

# correlation with target
corr_target = corr_matrix['accident_risk'].sort_values(ascending=False)
print(corr_target)



df_train['curvature_per_lane'] = df_train['curvature'] / (df_train['num_lanes'] + 1)
df_train['speed_curvature_ratio'] = df_train['speed_limit'] / (df_train['curvature'] + 0.01)


df_test['curvature_per_lane'] = df_test['curvature'] / (df_test['num_lanes'] + 1)
df_test['speed_curvature_ratio'] = df_test['speed_limit'] / (df_test['curvature'] + 0.01)




#time 

df_train['is_night'] = df_train['time_of_day'].isin(['night', 'evening']).astype(int)
df_train['is_peak_hour'] = df_train['time_of_day'].isin(['morning', 'afternoon']).astype(int)


df_test['is_night'] = df_test['time_of_day'].isin(['night', 'evening']).astype(int)
df_test['is_peak_hour'] = df_test['time_of_day'].isin(['morning', 'afternoon']).astype(int)



df_train = df_train.drop(columns=["time_of_day"])
df_test = df_test.drop(columns=["time_of_day"])


 #  train
numerical_cols_train = df_train.select_dtypes(include=['float64', 'int64']).columns.tolist()
categorical_cols_train = df_train.select_dtypes(include=['object', 'bool']).columns.tolist()

print("Train - Numerical Columns:", numerical_cols_train)
print("Train - Categorical Columns:", categorical_cols_train)

#  test
numerical_cols_test = df_test.select_dtypes(include=['float64', 'int64']).columns.tolist()
categorical_cols_test = df_test.select_dtypes(include=['object', 'bool']).columns.tolist()



print(numerical_cols_train)


df_train.head()



from sklearn.model_selection import train_test_split

X = df_train.drop("accident_risk", axis=1)
y = df_train["accident_risk"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)




# bool
bool_cols = X_train.select_dtypes(include=['bool']).columns.tolist()

# bool into int
for df in [X_train, X_test, df_test]:
    df[bool_cols] = df[bool_cols].astype(int)

# 
cat_cols = X_train.select_dtypes(include=['object']).columns.tolist()

# One-hot encoding 
X_train = pd.get_dummies(X_train, columns=cat_cols, drop_first=True)
X_test = pd.get_dummies(X_test, columns=cat_cols, drop_first=True)
df_test = pd.get_dummies(df_test, columns=cat_cols, drop_first=True)

# 
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)
df_test = df_test.reindex(columns=X_train.columns, fill_value=0)







X_train.info()


X_train.head()


X_train.skew()


X_train['speed_curvature_ratio'] = np.log1p(X_train['speed_curvature_ratio'])
X_test['speed_curvature_ratio'] = np.log1p(X_test['speed_curvature_ratio'])

df_test['speed_curvature_ratio'] = np.log1p(X_test['speed_curvature_ratio'])



print("Skewness after log1p:", X_train['speed_curvature_ratio'].skew())


features=['speed_limit','speed_curvature_ratio','num_lanes','num_reported_accidents']

scaler = MinMaxScaler()
X_train[features] = scaler.fit_transform(X_train[features])
X_test[features] = scaler.transform(X_test[features])
df_test[features]= scaler.transform(df_test[features])



model = LGBMRegressor(random_state=42, n_estimators=500)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)



rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"Validation RMSE: {rmse:.4f}")



y_final_pred = model.predict(df_test)



df_test["id"] = pd.read_csv("/kaggle/input/playground-series-s5e10/test.csv")["id"]
submission = pd.DataFrame({
    "id": df_test["id"],
    "accident_risk": y_final_pred
})
submission.to_csv("submission.csv", index=False)
print("✅ submission.csv آماده شد!")


