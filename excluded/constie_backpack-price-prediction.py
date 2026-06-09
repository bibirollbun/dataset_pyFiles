import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score



train_data = pd.read_csv("../dataset/train.csv")
test_data = pd.read_csv("../dataset/test.csv")
train_extra = pd.read_csv("../dataset/training_extra.csv")
train_data = pd.concat([train_data, train_extra] , ignore_index=True ,axis=0)


train_data.head()


test_data.head()    


train_data.shape


test_data.shape


train_missing_values = pd.DataFrame({"feature" : train_data.columns, 
                                     "missing values": train_data.isnull().sum().values,
                                     "percentage of missing values (%)": train_data.isnull().sum().values/ len(train_data)*100 })
train_missing_values


train_unique_values = pd.DataFrame({'feature': train_data.columns,
                              'no. of unique values': train_data.nunique().values})
train_unique_values


train_feature_types = pd.DataFrame({'feature': train_data.columns,
                              'dataType': train_data.dtypes})

train_feature_types


test_duplicates = test_data.duplicated().sum()
train_duplicates = train_data.duplicated().sum()
print("Number of duplicates in test data: ", test_duplicates)
print("Number of duplicates in train data: ", train_duplicates)


train_data.describe().T


# Add 'Dataset' column to distinguish between train and test data
train_data['Dataset'] = 'Train'
test_data['Dataset'] = 'Test'

# Weight Capacity is the only numercial feature we have
column = "Weight Capacity (kg)"
data_concat = pd.concat([train_data, test_data],ignore_index=True)


sns.boxplot(data=data_concat, x=column, y='Dataset')


custom_palette = ['#3498db', '#e74c3c','#2ecc71']
sns.histplot(data=train_data, x=column, color=custom_palette[0], kde=True, bins=30, label="Train")
sns.histplot(data=test_data, x=column, color=custom_palette[1], kde=True, bins=30, label="Test")
plt.xlabel(column)
plt.ylabel("Frequency")
plt.title(f"Histogram for {column} [TRAIN, TEST & ORIGINAL]")
plt.legend()
plt.tight_layout()


train_data.drop('Dataset', axis=1, inplace=True)
test_data.drop('Dataset', axis=1, inplace=True)


categorical_variables = ['Brand', 'Material', 'Size', 'Compartments', 'Laptop Compartment','Waterproof', 'Style', 'Color']
pie_chart_palette = ['#33638d', '#28ae80', '#d3eb0c', '#ff9a0b', '#7e03a8', '#35b779', '#fde725', '#440154', '#90d743', '#482173', '#22a884', '#f8961e']

countplot_color = '#5C67A3'


def create_categorical_plots(variable):
    sns.set_style('whitegrid')
    
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # Pie Chart TRAIN
    plt.subplot(1, 3, 1)
    train_data[variable].value_counts().plot.pie(
        autopct='%1.1f%%', colors=pie_chart_palette, wedgeprops=dict(width=0.3), startangle=140
    )
    plt.title(f"Pie Chart for {variable} [TRAIN]")

    # Pie Chart TEST
    plt.subplot(1, 3, 2)
    test_data[variable].value_counts().plot.pie(
        autopct='%1.1f%%', colors=pie_chart_palette, wedgeprops=dict(width=0.3), startangle=140
    )
    plt.title(f"Pie Chart for {variable} [TEST]")

    # Bar Graph
    plt.subplot(1, 3, 3)
    sns.countplot(
        data=pd.concat([train_data, test_data],ignore_index=True), 
        x=variable, 
        color=countplot_color,  # Using a single color for the countplot
        alpha=0.8  # Setting 80% opacity
    )
    plt.xlabel(variable)
    plt.ylabel("Count")
    plt.title(f"Bar Graph for {variable} [TRAIN, TEST Combined]")

    # Adjust spacing between subplots
    plt.tight_layout()
    
    # Show the plots
    plt.show()


for variable in categorical_variables:
    create_categorical_plots(variable)


target_palette = ['#3498db', '#e74c3c']
train_data['Dataset'] = 'Train'


sns.set_style('whitegrid')
    
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
target_variable = "Price"

# Box Plot
plt.subplot(1, 2, 1)
sns.boxplot(data=train_data, x=target_variable, y="Dataset", palette=target_palette)
plt.xlabel(target_variable)
plt.title(f"Box Plot for Target Feature '{target_variable}'")

# Histogram
plt.subplot(1, 2, 2)
sns.histplot(data=train_data, x=target_variable, color=target_palette[0], kde=True, bins=30, label="Train")
plt.xlabel(target_variable)
plt.ylabel("Frequency")
plt.title(f"Histogram for Target Feature '{target_variable}' [TRAIN]")
plt.legend()

# Adjust spacing between subplots
plt.tight_layout()

# Show the plots
plt.show()
train_data.drop('Dataset', axis=1, inplace=True)



numerical_variables = ['Compartments','Weight Capacity (kg)', 'Price']
sns.heatmap(train_data[numerical_variables].corr(), annot=True)


categorical_features = ["Brand", "Material", "Size", "Laptop Compartment", "Waterproof", "Style", "Color"]
numerical_features = ["Weight Capacity (kg)"]

for col in categorical_features:
    train_data.fillna({col : train_data[col].mode()[0]}, inplace=True)
    test_data.fillna({col : test_data[col].mode()[0]}, inplace=True)

for col in numerical_features:
    train_data.fillna({col : train_data[col].median()}, inplace=True)
    test_data.fillna({col : test_data[col].median()}, inplace=True)


train_data = train_data.drop("id", axis=1)
test_data_ids = test_data["id"]
test_data = test_data.drop("id", axis=1)  


column = "Weight Capacity (kg)"
Q1 = train_data[column].quantile(0.1)
Q3 = train_data[column].quantile(0.9)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

filtered_data = train_data[(train_data[column] >= lower_bound) & (train_data[column] <= upper_bound)]
no_of_outliers = len(train_data) - len(filtered_data)
print(f"Number of outliers in {column}: {no_of_outliers}")




numerical_variables = ['Weight Capacity (kg)']

skewed_features = test_data[numerical_variables].skew()[test_data[numerical_variables].skew() > 0.75].index.values
print("Features with skewness > 0.75 in test_data:")
display(len(skewed_features))

skewed_features = train_data[numerical_variables].skew()[train_data[numerical_variables].skew() > 0.75].index.values
print("Features with skewness > 0.75 in train_data:")
display(len(skewed_features))


train_data.head()


columns_to_encode = ['Brand', 'Material', 'Size', 'Laptop Compartment','Waterproof', 'Style', 'Color']
train_data_to_encode = train_data[columns_to_encode]
test_data_to_encode = test_data[columns_to_encode]
train_data_encoded = pd.get_dummies(train_data_to_encode, columns=columns_to_encode, drop_first=True)
test_data_encoded = pd.get_dummies(test_data_to_encode, columns=columns_to_encode, drop_first=True)


train_data_encoded.head()


from sklearn.preprocessing import MinMaxScaler

train_data_to_scale = train_data.drop(columns_to_encode, axis=1)
test_data_to_scale = test_data.drop(columns_to_encode, axis=1)

minmax_scaler = MinMaxScaler()
minmax_scaler.fit(train_data_to_scale.drop(["Price"],axis=1))


train_data_scaled = minmax_scaler.transform(train_data_to_scale.drop(["Price"], axis=1))
scaled_train_df = pd.DataFrame(train_data_scaled, columns=train_data_to_scale.drop(['Price'], axis=1).columns)
test_data_scaled = minmax_scaler.transform(test_data_to_scale)
scaled_test_df = pd.DataFrame(test_data_scaled, columns=test_data_to_scale.columns)

# Concatenate train datasets
train_data_combined = pd.concat([train_data_encoded.reset_index(drop=True), scaled_train_df.reset_index(drop=True)], axis=1)
# Concatenate test datasets
test_data_combined = pd.concat([test_data_encoded.reset_index(drop=True), scaled_test_df.reset_index(drop=True)], axis=1)


X = train_data_combined
y = train_data['Price']


X.head()


y.head()


train_mean = train_data.Price.mean()
train_data['pred'] = train_mean
s = np.sqrt(np.mean((train_data.Price-train_data.pred)**2.0 ) )
print(f"RMSE using Train Mean = {s}")


sub = pd.read_csv("../dataset/sample_submission.csv")
sub['Price'] = train_mean
sub.to_csv("submission_mean.csv", index=False)


# split into training and validation set
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

xgb_model = xgb.XGBRegressor(n_estimators=300, learning_rate=0.1, objective='reg:squarederror', random_state=42)
lgb_model = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.1, random_state=42)
rf_model = RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=42, max_depth=10)

print("Training XGBoost...")
xgb_model.fit(X_train, y_train)
print("Training LightGBM...")
lgb_model.fit(X_train, y_train)
print("Training RandomForest...")
rf_model.fit(X_train, y_train)

# Predict on validation sets
val_preds_xgb = xgb_model.predict(X_val)
val_preds_lgb = lgb_model.predict(X_val)
val_preds_rf = rf_model.predict(X_val)


def calculateMetrics(Y_val, val_preds_lgb, val_preds_xgb,val_preds_rf):
    # Calculate metrics for each model
    mse_xgb = mean_squared_error(Y_val, val_preds_xgb)
    mse_lgb = mean_squared_error(Y_val, val_preds_lgb)
    mse_rf = mean_squared_error(Y_val, val_preds_rf)

    mae_xgb = mean_absolute_error(Y_val, val_preds_xgb)
    mae_lgb = mean_absolute_error(Y_val, val_preds_lgb)
    mae_rf = mean_absolute_error(Y_val, val_preds_rf)

    rmse_xgb = mse_xgb ** 0.5
    rmse_lgb = mse_lgb ** 0.5
    rmse_rf = mse_rf ** 0.5

    r2_xgb = r2_score(Y_val, val_preds_xgb)
    r2_lgb = r2_score(Y_val, val_preds_lgb)
    r2_rf = r2_score(Y_val, val_preds_rf)

    # Print the metrics
    print("XGBoost Metrics:")
    print(f"MSE: {mse_xgb:.4f}, MAE: {mae_xgb:.4f}, RMSE: {rmse_xgb:.4f}, R2: {r2_xgb:.4f}")

    print("LightGBM Metrics:")
    print(f"MSE: {mse_lgb:.4f}, MAE: {mae_lgb:.4f}, RMSE: {rmse_lgb:.4f}, R2: {r2_lgb:.4f}")

    print("Random Forest Metrics:")
    print(f"MSE: {mse_rf:.4f}, MAE: {mae_rf:.4f}, RMSE: {rmse_rf:.4f}, R2: {r2_rf:.4f}")

calculateMetrics(y_val,val_preds_lgb, val_preds_xgb,val_preds_rf)


feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'XGBoost': xgb_model.feature_importances_,
    'LightGBM': lgb_model.feature_importances_,
    'RandomForest': rf_model.feature_importances_
})

## normalize the feature importance table
minmax_scaler = MinMaxScaler()
minmax_scaler.fit(feature_importance.drop(["Feature"],axis=1))
normalized_feature_importance = minmax_scaler.transform(feature_importance.drop(["Feature"],axis=1))

normalized_feature_importance = pd.DataFrame(normalized_feature_importance, columns=feature_importance.drop(['Feature'], axis=1).columns)
normalized_feature_importance = pd.concat([feature_importance["Feature"].reset_index(drop=True), normalized_feature_importance.reset_index(drop=True)], axis=1)
normalized_feature_importance['Average_Importance'] = normalized_feature_importance.iloc[:, 1:].mean(axis=1)
normalized_feature_importance = normalized_feature_importance.sort_values(by='Average_Importance', ascending=False)


normalized_feature_importance


# Plot feature importance
plt.figure(figsize=(10, 6))
plt.barh(normalized_feature_importance['Feature'][:20], normalized_feature_importance['Average_Importance'][:20], color='blue')
plt.xlabel("Average Importance Score")
plt.ylabel("Feature")
plt.title("Top 20 Important Features")
plt.gca().invert_yaxis()
plt.show()


threshold = normalized_feature_importance['Average_Importance'].quantile(0.20)
selected_features = normalized_feature_importance[normalized_feature_importance['Average_Importance'] > threshold]['Feature'].tolist()


X_train_selected = X_train[selected_features]
X_val_selected = X_val[selected_features]
test_data_selected = test_data_combined[selected_features]

xgb_model.fit(X_train_selected, y_train)
lgb_model.fit(X_train_selected, y_train)
rf_model.fit(X_train_selected, y_train)

val_preds_xgb = xgb_model.predict(X_val_selected)
val_preds_lgb = lgb_model.predict(X_val_selected)
val_preds_rf = rf_model.predict(X_val_selected)

calculateMetrics(y_val, val_preds_xgb,val_preds_lgb,val_preds_rf)


from scipy.optimize import minimize

def loss(weights):
    weighted_preds = (weights[0] * val_preds_xgb +
                      weights[1] * val_preds_lgb +
                      weights[2] * val_preds_rf)
    return mean_absolute_error(y_val, weighted_preds)

initial_weights = [0.33333, 0.33333, 0.33333]
constraints = ({'type': 'eq', 'fun': lambda w: 1 - sum(w)})
bounds = [(0, 1)] * 3

result = minimize(loss, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)
optimal_weights = result.x



val_preds = (optimal_weights[0] * val_preds_xgb +
             optimal_weights[1] * val_preds_lgb +
             optimal_weights[2] * val_preds_rf)

rmse = np.sqrt(np.mean((y_val-val_preds)**2.0 ) )

print(f"Optimized Validation RMSE: {rmse}")
print(f"Optimal Weights: {optimal_weights}")


test_preds_xgb = xgb_model.predict(test_data_selected)
test_preds_lgb = lgb_model.predict(test_data_selected)
test_preds_rf = rf_model.predict(test_data_selected)

test_preds = (optimal_weights[0] * test_preds_xgb +
              optimal_weights[1] * test_preds_lgb +
              optimal_weights[2] * test_preds_rf)

ensemble_submission_df = pd.DataFrame({
    'id': test_data_ids,
    'Price': test_preds
})

ensemble_submission_df.to_csv('submission_ensemble_with_feature_importance.csv', index=False)
ensemble_submission_df.head(10)


train = pd.read_csv("../dataset/train.csv")
train_extra = pd.read_csv("../dataset/training_extra.csv")
train = pd.concat([train,train_extra],axis=0,ignore_index=True)
train.head()


import pandas as pd
import numpy as np

# Create bins for the numerical variable
train['Weight Capacity Binned'] = pd.qcut(train['Weight Capacity (kg)'], q=65, labels=False)

# Calculate the mean price for each bin
mean_price = train.groupby('Weight Capacity Binned')['Price'].mean()
train['pred'] = train['Weight Capacity Binned'].map(mean_price)

# Calculate RMSE
s = np.sqrt(np.mean((train['Price'] - train['pred']) ** 2.0))
print(f"Validation RMSE using Target Encode Weight Capacity = {s}")


test = pd.read_csv("../dataset/test.csv")
test['Weight Capacity Binned'] = pd.qcut(test['Weight Capacity (kg)'], q=65, labels=False)



test['Price'] = test['Weight Capacity Binned'].map(mean_price)
sub = pd.read_csv("../dataset/sample_submission.csv")
sub['Price'] = test['Price']
sub.fillna({"Price" : sub["Price"].median()}, inplace=True)
sub.to_csv("submission_TE_weight_capacity.csv", index=False)
sub.head()

