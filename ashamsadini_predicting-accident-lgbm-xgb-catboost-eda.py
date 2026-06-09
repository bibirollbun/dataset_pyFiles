import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.linear_model import Ridge
import optuna


train_data = pd.read_csv("/kaggle/input/playground-series-s5e10/train.csv")
test_data = pd.read_csv("/kaggle/input/playground-series-s5e10/test.csv")
sample_data = pd.read_csv("/kaggle/input/playground-series-s5e10/sample_submission.csv")


print(f'Train shape: {train_data.shape}')
print(f'Test shape: {test_data.shape}')


train_data = train_data.drop('id', axis=1)
train_data.head()


test_data = test_data.drop('id', axis=1)
test_data.head()


train_data.info()


train_data.describe()


train_data.isnull().sum()


train_data.duplicated().sum()



train_data = train_data.drop_duplicates()




# Detect columns with mixed data types

# This function identifies columns that contain values of more than one data type.
# Mixed data types can cause errors during analysis or visualization, so detecting them is important.
def mixed_types(df):
    mixed_types ={}
    for col in df.columns:
        types = df[col].apply(lambda x: type(x).__name__).value_counts()
        if len(types) > 1:
             mixed_types[col] = types.to_dict()
    return mixed_types


mixed_columns = mixed_types(train_data)
print(f'Columns with mixed types: {mixed_columns}')


binary_cols = ['road_signs_present', 'public_road', 'holiday', 'school_season']
for col in binary_cols:
    sns.barplot(x=col, y='accident_risk', data=train_data)
    plt.title(f'Accident Risk vs {col}')
    plt.show()


plt.figure(figsize=(12,6))
plt.plot(train_data.groupby('time_of_day')['accident_risk'].mean(),
         marker='o', linestyle='-', color='crimson', linewidth=2)
plt.title('Average Accident Risk Over Time of Day', fontsize=16, fontweight='bold')
plt.xlabel('Time of Day', fontsize=12)
plt.ylabel('Accident Risk', fontsize=12)
plt.grid(alpha=0.3)
plt.xticks(rotation=45)
plt.show()


plt.figure(figsize=(15,6))
sns.boxplot(x='weather', y='accident_risk', hue='road_type', data=train_data)
plt.title('Accident Risk by Weather and Road Type')
plt.show()


plt.figure(figsize=(12,6))
sns.histplot(train_data['accident_risk'], bins=50, kde=True, color='teal', alpha=0.7)
plt.title('Accident Risk Distribution', fontsize=16, fontweight='bold')
plt.xlabel('Accident Risk', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.grid(alpha=0.3)
plt.show()


plt.figure(figsize=(10,6))
plt.boxplot(train_data['accident_risk'], patch_artist=True, 
            boxprops=dict(facecolor='lightblue', color='blue'),
            medianprops=dict(color='red', linewidth=2),
            whiskerprops=dict(color='blue'),
            capprops=dict(color='blue'),
            flierprops=dict(marker='o', markerfacecolor='orange', markersize=6, linestyle='none')
           )
plt.title('Accident Risk Distribution', fontsize=16, fontweight='bold')
plt.ylabel('Accident Risk', fontsize=12)
plt.grid(alpha=0.3)
plt.show()



plt.figure(figsize=(15,10))
sns.barplot(
    x='weather',
    y='accident_risk',
    data=train_data,
    order=train_data.groupby('weather')['accident_risk'].mean().nlargest(12).index,
    palette='viridis'
)
plt.title('Distribution of Accident Risk by Weather', fontsize=18, fontweight='bold')
plt.xlabel('Weather', fontsize=14)
plt.ylabel('Accident Risk', fontsize=14)
plt.xticks(rotation=45)
plt.grid(axis='y', alpha=0.3)
plt.show()


plt.figure(figsize=(15,10))
sns.barplot(
    x='road_type',
    y='accident_risk',
    data=train_data,
    order=train_data.groupby('road_type')['accident_risk'].mean().nlargest(12).index,
    palette='viridis'
)
plt.title('Distribution of Accident Risk by Road Type', fontsize=18, fontweight='bold')
plt.xlabel('Road Type', fontsize=14)
plt.ylabel('Accident Risk', fontsize=14)
plt.xticks(rotation=45)
plt.grid(axis='y', alpha=0.3)
plt.show()


plt.figure(figsize=(15, 10))
sns.barplot(x='time_of_day',
            y='accident_risk',
            data=train_data,
            order=train_data.groupby('time_of_day')['accident_risk'].mean().nlargest(12).index,
            palette='viridis',
            )
plt.title('Distribution of Accident Risk')
plt.xlabel('Time of Day')
plt.ylabel('Accident Risk')
plt.show()


corr = train_data.corr(numeric_only=True)
plt.figure(figsize=(12,10))
sns.heatmap(
    corr, 
    annot=True, 
    fmt='.2f', 
    cmap='coolwarm', 
    linewidths=0.5,  
    linecolor='white',
    cbar_kws={"shrink": 0.8, "label": "Correlation"}  
)
plt.title('Correlation Matrix of Numeric Features', fontsize=16, fontweight='bold')
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.show()



num_columns = train_data.select_dtypes(include=np.number).columns

fig, axs = plt.subplots(1, len(num_columns), figsize=(20,6))
fig.suptitle("Outliers in Numeric Features", fontsize=16, fontweight='bold')

for ax, column in zip(axs.flat, num_columns):
    ax.boxplot(
        train_data[column],
        patch_artist=True,
        boxprops=dict(facecolor='lightblue', color='blue'),
        medianprops=dict(color='red', linewidth=2),
        whiskerprops=dict(color='blue', linewidth=1.5),
        capprops=dict(color='blue', linewidth=1.5),
        flierprops=dict(marker='o', markerfacecolor='orange', markersize=6, linestyle='none')
    )
    ax.set_title(column, fontsize=12)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()



 # Skewness (histogram)

num_columns = train_data.select_dtypes(include=np.number).columns

fig, axs = plt.subplots(1, len(num_columns), figsize=(20,5))
fig.suptitle("Skewness of Numeric Features", fontsize=16, fontweight='bold')

for ax, column in zip(axs.flat, num_columns):
    ax.hist(train_data[column], bins=30, color='teal', alpha=0.7, edgecolor='black')
    ax.set_title(column, fontsize=12)
    ax.grid(alpha=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()




num_cols_raw = train_data.select_dtypes(include=np.number).columns.tolist()

#  Outliers (IQR clip) on numeric
for col in num_cols_raw:
  q1, q3 = train_data[col].quantile([0.25, 0.75])
  iqr = q3 - q1
  lower_band, upper_band = q1 - 1.5*iqr, q1 + 1.5*iqr
  train_data[col] = train_data[col].clip(lower_band, upper_band)

# Skewness (log1p if heavy)
for col in num_cols_raw:
  if abs(train_data[col].skew()) > 1:
    train_data[col] = np.log1p(train_data[col])


 # Skewness (histogram)

num_columns = train_data.select_dtypes(include=np.number).columns

fig, axs = plt.subplots(1, len(num_columns), figsize=(20,5))
fig.suptitle("Skewness of Numeric Features", fontsize=16, fontweight='bold')

for ax, column in zip(axs.flat, num_columns):
    ax.hist(train_data[column], bins=30, color='teal', alpha=0.7, edgecolor='black')
    ax.set_title(column, fontsize=12)
    ax.grid(alpha=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()



num_columns = train_data.select_dtypes(include=np.number).columns

fig, axs = plt.subplots(1, len(num_columns), figsize=(20,6))
fig.suptitle("Outliers in Numeric Features", fontsize=16, fontweight='bold')

for ax, column in zip(axs.flat, num_columns):
    ax.boxplot(
        train_data[column],
        patch_artist=True,
        boxprops=dict(facecolor='lightblue', color='blue'),
        medianprops=dict(color='red', linewidth=2),
        whiskerprops=dict(color='blue', linewidth=1.5),
        capprops=dict(color='blue', linewidth=1.5),
        flierprops=dict(marker='o', markerfacecolor='orange', markersize=6, linestyle='none')
    )
    ax.set_title(column, fontsize=12)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()


bool_cols = ['road_signs_present', 'public_road', 'holiday', 'school_season']
train_data[bool_cols] = train_data[bool_cols].astype(int)
test_data[bool_cols] = test_data[bool_cols].astype(int)

cat_cols = ['road_type', 'lighting', 'weather', 'time_of_day']
train_data = pd.get_dummies(train_data, columns =cat_cols, drop_first=True )
test_data = pd.get_dummies(test_data, columns =cat_cols, drop_first=True )


X_train = train_data.drop('accident_risk', axis=1)
y_train = train_data['accident_risk']
X_test = test_data


train_data.head()


X_test = X_test.reindex(columns=X_train.columns, fill_value=0)


X_tr, X_val, y_tr, y_val = train_test_split(
    X_train,
    y_train,
    test_size = 0.2,
    random_state=42
)


def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 400, 1200),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "max_depth": trial.suggest_int("max_depth", 4, 12),
        "num_leaves": trial.suggest_int("num_leaves", 20, 150),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 2.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 2.0),
        "random_state": 42,
        "n_jobs": -1,
        "verbosity": -1
    }
    model = LGBMRegressor(**params)
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    rmse = mean_squared_error(y_val, preds, squared=False)
    return rmse

print(" Running Optuna tuning for LightGBM...")
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=50, show_progress_bar=True)

best_params = study.best_params
print(" Best LightGBM RMSE:", study.best_value)
print(" Best Params:", best_params)


lgb_model = LGBMRegressor(**best_params)
xgb_model = XGBRegressor(
    n_estimators=900,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    loss_function='RMSE',
    tree_method='hist'
)
cat_model = CatBoostRegressor(
    iterations=1995,
    learning_rate=0.03,
    depth=8,
    l2_leaf_reg=6,
    bagging_temperature=0.3,
    loss_function='RMSE',
    random_seed=42,
    verbose=200
)


print("ğŸš€ Training Base Models...")
lgb_model.fit(X_train, y_train)
xgb_model.fit(X_train, y_train)
cat_model.fit(X_train, y_train)


pred_val_lgb = lgb_model.predict(X_val)
pred_val_xgb = xgb_model.predict(X_val)
pred_val_cat = cat_model.predict(X_val)

# Stack features
stack_val = np.vstack([pred_val_lgb, pred_val_xgb, pred_val_cat]).T


ridge = Ridge(alpha=1.0)
ridge.fit(stack_val, y_val)

final_val_pred = ridge.predict(stack_val)
final_rmse = mean_squared_error(y_val, final_val_pred, squared=False)
print(f"\n Final Validation RMSE (Stacking): {final_rmse:.5f}")


lgb_model.fit(X_train, y_train)
xgb_model.fit(X_train, y_train)
cat_model.fit(X_train, y_train)

pred_lgb = lgb_model.predict(X_test)
pred_xgb = xgb_model.predict(X_test)
pred_cat = cat_model.predict(X_test)

stack_test = np.vstack([pred_lgb, pred_xgb, pred_cat]).T
pred_final = ridge.predict(stack_test)

pred_final = np.clip(pred_final, 0, 1)


sample_data["accident_risk"] = pred_final
sample_data.to_csv("submission.csv", index=False)
print("\n submission.csv created successfully and ready for Kaggle upload!")




