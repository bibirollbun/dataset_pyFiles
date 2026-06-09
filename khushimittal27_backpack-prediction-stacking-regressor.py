import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
import lightgbm as lgb 
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.simplefilter("ignore")


train_df = pd.read_csv("/kaggle/input/playground-series-s5e2/train.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e2/test.csv")
# extra_train_df = pd.read_csv("/kaggle/input/playground-series-s5e2/training_extra.csv")
# train_df = pd.concat([train_df, extra_train_df],ignore_index=True)


train_df.head()


train_df.info()


train_df.isnull().sum()


missing_values = train_df.isnull().sum().sort_values(ascending=False)
missing_values_percentage = (missing_values / len(train_df)) * 100

missing_data = pd.DataFrame({"Missing Values": missing_values, "Percentage": missing_values_percentage})
missing_data = missing_data[missing_data["Missing Values"] > 0]  

missing_data


def fill_missing_values(data):
    categorical_cols = data.select_dtypes(include=['object']).columns.tolist()
    for col in categorical_cols:
        data[col].fillna(data[col].mode()[0], inplace=True)
    data['Weight Capacity (kg)'].fillna(data['Weight Capacity (kg)'].median(), inplace=True)


fill_missing_values(train_df)
train_df.isnull().sum()


#plot price distribution
plt.figure(figsize=(10, 5))
sns.histplot(train_df['Price'], bins=50, kde=True, color='blue')
plt.title('Price Distribution')
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.show()


top_brands = train_df['Brand'].value_counts().index[:10]
plt.figure(figsize=(12, 6))
sns.boxplot(x='Brand', y='Price', data=train_df[train_df['Brand'].isin(top_brands)])
plt.xticks(rotation=45)
plt.title('Price Distribution by Brand (Top 10)')
plt.show()


plt.figure(figsize=(8, 5))
sns.heatmap(train_df[['Compartments', 'Weight Capacity (kg)', 'Price']].corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap (Numerical Features)")
plt.show()


X = train_df.drop(columns=['id', 'Price'])
y = train_df['Price']


categorical_features = ['Brand', 'Material', 'Size', 'Laptop Compartment', 'Waterproof', 'Style', 'Color']
preprocessor = ColumnTransformer(transformers=[
    ('cat', OneHotEncoder(handle_unknown='ignore', sparse=False), categorical_features)], remainder='passthrough')
X_encoded = preprocessor.fit_transform(X)
X_encoded = pd.DataFrame(X_encoded, columns=preprocessor.get_feature_names_out())


X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


print("\nShapes after splitting and scaling:")
print("X_train:", X_train_scaled.shape)
print("X_test:", X_test_scaled.shape)
print("y_train:", y_train.shape)
print("y_test:", y_test.shape)


# lr_model = LinearRegression()
# lr_model.fit(X_train_scaled, y_train)
# y_pred = lr_model.predict(X_test_scaled)
# rmse = mean_squared_error(y_test, y_pred, squared=False)
# rmse


# xgb_model = xgb.XGBRegressor(objective='reg:squarederror',
#                              n_estimators=200,
#                              learning_rate=0.05,
#                              max_depth=7,
#                              random_state=42)
# xgb_model.fit(X_train_scaled, y_train,
#               early_stopping_rounds=10,
#               eval_set=[(X_test_scaled, y_test)],
#               verbose=True)


# param_grid = {
#     'n_estimators': [100, 200],
#     'max_depth': [5, 7, 10],
#     'learning_rate': [0.01, 0.05, 0.1],
#     'subsample': [0.8, 1.0],
#     'colsample_bytree': [0.8, 1.0]
# }
# xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
# grid_search = GridSearchCV(estimator=xgb_model, 
#                            param_grid=param_grid, 
#                            cv=3, 
#                            scoring='neg_root_mean_squared_error', 
#                            verbose=2, 
#                            n_jobs=-1)

# grid_search.fit(X_train_scaled, y_train)

# print("Best parameters found: ", grid_search.best_params_)
# print("Best CV RMSE: ", -grid_search.best_score_)

# best_model = grid_search.best_estimator_
# y_pred = best_model.predict(X_test_scaled)
# rmse = mean_squared_error(y_test, y_pred, squared=False)
# r2 = r2_score(y_test, y_pred)
# print(f"RMSE: {rmse}")
# print(f"R2 Score: {r2}")


# param_grid_rf = {
#     'n_estimators': [100, 200],
#     'max_depth': [None, 10, 20],
#     'min_samples_split': [2, 5],
#     'min_samples_leaf': [1, 2]
# }
# rf_model = RandomForestRegressor(random_state=42)
# grid_search_rf = GridSearchCV(estimator=rf_model, 
#                            param_grid=param_grid_rf, 
#                            cv=3, 
#                            scoring='neg_root_mean_squared_error', 
#                            verbose=2, 
#                            n_jobs=-1)

# grid_search_rf.fit(X_train_scaled, y_train)

# print("Best parameters found: ", grid_search_rf.best_params_)
# print("Best CV RMSE: ", -grid_search_rf.best_score_)

# best_model = grid_search_rf.best_estimator_
# y_pred = best_model.predict(X_test_scaled)
# rmse = mean_squared_error(y_test, y_pred, squared=False)
# r2 = r2_score(y_test, y_pred)
# print(f"RMSE: {rmse}")
# print(f"R2 Score: {r2}")


# param_grid_lgb = {
#     'n_estimators': [100, 200],
#     'max_depth': [10, 20, -1],  # -1 implies no limit in LightGBM
#     'learning_rate': [0.01, 0.05, 0.1],
#     'num_leaves': [31, 50, 100],
#     'min_child_samples': [20, 50]
# }
# lgb_model = lgb.LGBMRegressor(random_state=42)
# grid_search_lgb = GridSearchCV(estimator=lgb_model, 
#                            param_grid=param_grid_lgb, 
#                            cv=3, 
#                            scoring='neg_root_mean_squared_error', 
#                            verbose=2, 
#                            n_jobs=-1)

# grid_search_lgb.fit(X_train_scaled, y_train)

# print("Best parameters found: ", grid_search_lgb.best_params_)
# print("Best CV RMSE: ", -grid_search_lgb.best_score_)

# best_model = grid_search_lgb.best_estimator_
# y_pred = best_model.predict(X_test_scaled)
# rmse = mean_squared_error(y_test, y_pred, squared=False)
# r2 = r2_score(y_test, y_pred)
# print(f"RMSE: {rmse}")
# print(f"R2 Score: {r2}")


# param_grid_cat = {
#     'iterations': [200, 500],
#     'depth': [6, 8, 10],
#     'learning_rate': [0.01, 0.05, 0.1],
#     'l2_leaf_reg': [1, 3, 5]
# }
# cat_model = CatBoostRegressor(verbose=0, random_state=42)
# grid_search_cat = GridSearchCV(estimator=cat_model, 
#                            param_grid=param_grid_cat, 
#                            cv=3, 
#                            scoring='neg_root_mean_squared_error', 
#                            verbose=2, 
#                            n_jobs=-1)

# grid_search_cat.fit(X_train_scaled, y_train)

# print("Best parameters found: ", grid_search_cat.best_params_)
# print("Best CV RMSE: ", -grid_search_cat.best_score_)

# best_model = grid_search_cat.best_estimator_
# y_pred = best_model.predict(X_test_scaled)
# rmse = mean_squared_error(y_test, y_pred, squared=False)
# r2 = r2_score(y_test, y_pred)
# print(f"RMSE: {rmse}")
# print(f"R2 Score: {r2}")


# y_pred = xgb_model.predict(X_test_scaled)
# rmse = mean_squared_error(y_test, y_pred, squared=False)
# r2 = r2_score(y_test, y_pred)
# print(f"RMSE: {rmse}")
# print(f"R2 Score: {r2}")


#using stacking Regressor
xgb_model = xgb.XGBRegressor(
    n_estimators=200,
    colsample_bytree=0.8,
    max_depth=5,           
    learning_rate=0.01,    
    subsample=0.8,
    random_state=42
)

rf_model = RandomForestRegressor(
    n_estimators=200,      
    max_depth=10,
    min_samples_leaf=1,
    min_samples_split=2,
    random_state=42
)

cat_model = CatBoostRegressor(
    iterations=500,        
    depth=6,               
    learning_rate=0.01,    
    l2_leaf_reg=5,         
    random_state=42,
    verbose=0
)

estimators = [
    ('xgb', xgb_model),
    ('rf', rf_model),
    ('cat', cat_model)
]

stacking_reg = StackingRegressor(
    estimators=estimators,
    final_estimator=LinearRegression(),
    cv=5,
    n_jobs=-1
)

stacking_reg.fit(X_train_scaled, y_train)
y_pred = stacking_reg.predict(X_test_scaled)
rmse_stack = mean_squared_error(y_test, y_pred, squared=False)
r2_stack = r2_score(y_test, y_pred)

print("Stacking Regressor Validation RMSE: {:.2f}".format(rmse_stack))
print("Stacking Regressor Validation R2 Score: {:.4f}".format(r2_stack))


test_df.isnull().sum()


fill_missing_values(test_df)
test_df.isnull().sum()


test_df.info()


# X_test_new = test_df.drop(columns=['id'])
# X_test_encoded = preprocessor.transform(X_test_new)
# X_test_encoded = pd.DataFrame(X_test_encoded, columns=preprocessor.get_feature_names_out())
# X_test_scaled = scaler.transform(X_test_encoded)
# # X_test_scaled = X_test_scaled.astype(np.float32)
# predictions = lr_model.predict(X_test_scaled)



X_test_new = test_df.drop(columns=['id'])
X_test_encoded = preprocessor.transform(X_test_new)
X_test_encoded = pd.DataFrame(X_test_encoded, columns=preprocessor.get_feature_names_out())
X_test_scaled = scaler.transform(X_test_encoded)
predictions = stacking_reg.predict(X_test_scaled)


submission_df = pd.DataFrame({
    "id": test_df["id"],
    "Price": predictions
})
submission_df.to_csv("submission.csv", index=False)
print("Submission file saved as 'submission.csv'.")


submission_df.head()


# from IPython.display import FileLink
# display(FileLink("submission.csv"))

