import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

import warnings
warnings.filterwarnings('ignore')


df = pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv')
df.head(10)


df.info()


df.isnull().sum()


df.describe()


columns = df.columns
columns


columns[0]
df[columns[0]].value_counts()


for x in columns:
    print(f'{df[x].value_counts()}\n')
    


plt.figure(figsize=(4, 4))
datalabel = sns.countplot(x='road_type', data=df, palette='viridis')

for i in datalabel.containers:
    datalabel.bar_label(i)


plt.title('Distribution of Road Type')
plt.xlabel('Road Type')
plt.ylabel('Count')
plt.savefig('Distribution of Road Type.png')
plt.show()


# Correlation matrix
corr_matrix = df.corr(numeric_only=True)

# Plot heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm")
plt.title("Correlation Matrix for Dataset", fontsize=15)
plt.savefig('Correlation Matrix df.png')
plt.show()


# Continuous variables to visualize
continuous_vars = ['num_lanes', 'speed_limit', 'lighting','weather']

plt.figure(figsize=(12, 10))
for i, var in enumerate(continuous_vars):
    plt.subplot(2, 2, i+1)
    sns.histplot(df[var], kde=True, color="hotpink")
    plt.title(f'{var} Distribution')
    plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('line_speed.png')
plt.show()


bool_cols = df.select_dtypes(include='bool').columns

plt.figure(figsize=(15, 8))  # adjust as needed

for i, var in enumerate(bool_cols):
    plt.subplot( (len(bool_cols)+3)//4 , 4, i+1)  # 4 columns per row
    ax = sns.countplot(x=var, data=df, palette='Greens')
    ax.set_title(var)
    ax.set_xlabel('')
    ax.set_ylabel('Count')

    # Add labels on top of bars
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}', 
                    (p.get_x()+p.get_width()/2., p.get_height()), 
                    ha='center', va='bottom', fontsize=9, xytext=(0, 3),
                    textcoords='offset points')

plt.tight_layout()
plt.show()


from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import mean_squared_error

# df1 = df.drop(columns=['id'],axis=1)

# # Create bins for continuous target
# df1["risk_bin"] = pd.qcut(df1["accident_risk"], q=20, labels=False, duplicates='drop')

# # Stratified sampling: keep full distribution
# sample_df, _ = train_test_split(df1, 
#                                 train_size=0.2, # use 20% of data
#                                 stratify=df1["risk_bin"], 
#                                 random_state=42)

# sample_df = sample_df.drop(columns=["risk_bin"])
# sample_df


# Convert categorical columns
cat_cols = ['road_type', 'lighting', 'weather', 'time_of_day']
for c in cat_cols:
    df[c] = df[c].astype('category')

# Features & target
X = df.drop(columns=['accident_risk', 'id'])
y = df['accident_risk']

# Split
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.15, random_state=42)


df.info()


# from sklearn.preprocessing import StandardScaler, OneHotEncoder
# from sklearn.compose import ColumnTransformer
# from sklearn.pipeline import Pipeline


# num_cols = sample_df.select_dtypes(include=['int64','float64']).columns.drop('accident_risk')
# cat_cols = sample_df.select_dtypes(include=['category','bool']).columns

# preprocessor = ColumnTransformer([
#     ('num', StandardScaler(), num_cols),
#     ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
# ], remainder='drop')


from lightgbm import LGBMRegressor

lgbm = LGBMRegressor(
    n_estimators=2000,
    learning_rate=0.1,
    #max_depth=-1,
    num_leaves=64,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_samples=50,
    reg_alpha=0.1,
    reg_lambda=0.3,
    random_state=42,
    n_jobs=-1,
    verbose=-1
)

lgbm.fit(
    X_train, y_train,
    eval_set=[(X_valid, y_valid)],
    eval_metric='mse',
    categorical_feature=cat_cols,
)

lgbm_pred = lgbm.predict(X_valid)
print("LightGBM MSE:", mean_squared_error(y_valid, lgbm_pred))


from catboost import CatBoostRegressor

catb = CatBoostRegressor(
    iterations=2000,
    learning_rate=0.1,
    depth=8,
    l2_leaf_reg=3,
    random_strength=0.8,
    subsample=0.8,
    loss_function='RMSE',
    cat_features=cat_cols,
    eval_metric='RMSE',
    early_stopping_rounds=200,
    random_seed=42,
    verbose=0
)

catb.fit(X_train, y_train, eval_set=(X_valid, y_valid))
catb_pred = catb.predict(X_valid)
print("CatBoost MSE:", mean_squared_error(y_valid, catb_pred))


# features = ['road_type','num_lanes','curvature','speed_limit','lighting',
#             'weather','road_signs_present','public_road','time_of_day',
#             'holiday','school_season','num_reported_accidents']
       


# standard_scaler = StandardScaler()

# X_train = standard_scaler.fit_transform(X_train)
# X_test = standard_scaler.transform(X_test)


from xgboost import XGBRegressor

xgb = XGBRegressor(
    n_estimators=1000,
    learning_rate=0.1,
    max_depth=12,
    tree_method='hist',       # faster and memory friendly
    verbosity=0, 
    n_jobs=-1,
    random_state=42,
    enable_categorical=True
)

print("Training XGB...")
xgb.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)

preds = xgb.predict(X_valid)
mse = mean_squared_error(y_valid,preds)
rmse = mse ** 0.5

print(f"MSE: {mse:.6f}")
print(f"RMSE: {rmse:.6f}")




# rf = RandomForestRegressor(
#     n_estimators=500,
#     n_jobs=-1,
#     random_state=42,
#     verbose=0,
# )
# print("Training RandomForest...")

# rf.fit(X_train, y_train)

# preds = rf.predict(X_test)
# mse = mean_squared_error(y_test,preds)
# rmse = mse ** 0.5

# print(f"MSE: {mse:.6f}")
# print(f"RMSE: {rmse:.6f}")



# dt = DecisionTreeRegressor(
#     max_depth=12,
#     random_state=42,
# )
# print("Training DecisionTree...")
# dt.fit(X_train, y_train)

# pred = dt.predict(X_test)
# mse = mean_squared_error(y_test,pred)
# rmse = mse ** 0.5

# print(f"MSE: {mse:.6f}")
# print(f"RMSE: {rmse:.6f}")


# # -------------------------
# # Optional: Voting (averaging) ensemble
# # -------------------------
# voting = VotingRegressor(
#     estimators=[('xgb', xgb), 
#                 ('rf', rf), 
#                 ('dt', dt)], 
#             n_jobs=-1, 
#             verbose=0,
#         )
# print("Training VotingRegressor (averaging)...")
# voting.fit(X_train, y_train)

# # -------------------------
# # Predict and evaluate on validation set
# # -------------------------
# models = {'XGB': xgb, 'RandomForest': rf, 'DecisionTree': dt, 'Voting': voting}
# results = {}

# for name, model in models.items():
#     preds = model.predict(X_val)
#     # If you want to enforce risk range [0,1]:
#     #preds = np.clip(preds, 0.0, 1.0)
#     mse = mean_squared_error(y_val, preds)
#     rmse = np.sqrt(mse)
#     results[name] = {'mse': mse, 'rmse': rmse}
#     print(f"{name} -> MSE: {mse:.6f}, RMSE: {rmse:.6f}")


test_df = pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv')
test_df


sample = pd.read_csv("/kaggle/input/playground-series-s5e10/sample_submission.csv")
sample.shape


ids = sample['id']
sample.head(5)


# label_encoder = LabelEncoder()
# test = test_df.drop(columns=['id'],axis=1)

# test['road_type'] = label_encoder.fit_transform(test['road_type'])
# test['lighting'] = label_encoder.fit_transform(test['lighting'])
# test['weather'] = label_encoder.fit_transform(test['weather'])
# test['road_signs_present'] = label_encoder.fit_transform(test['road_signs_present'])
# test['public_road'] = label_encoder.fit_transform(test['public_road'])
# test['time_of_day'] = label_encoder.fit_transform(test['time_of_day'])
# test['holiday'] = label_encoder.fit_transform(test['holiday'])
# test['school_season'] = label_encoder.fit_transform(test['school_season']) #Categorical Data?


# test.head()


test_df = test_df.drop(columns='id',axis=1)
test_df.info()


# test_scaled = standard_scaler.transform(test)
cat_cols = ['road_type', 'lighting', 'weather', 'time_of_day']

for c in cat_cols:
    test_df[c] = test_df[c].astype('category')
    
predict = lgbm.predict(test_df)


test_data = pd.DataFrame({
    'id' : sample['id'],
    'accident_risk' : predict
}) 

test_data


test_data.to_csv('submission.csv',index=False)


sb = pd.read_csv('/kaggle/working/submission.csv')
sb




