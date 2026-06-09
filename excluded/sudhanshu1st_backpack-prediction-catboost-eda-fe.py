import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 


train_data = pd.read_csv("/kaggle/input/playground-series-s5e2/train.csv")
test_data = pd.read_csv("/kaggle/input/playground-series-s5e2/test.csv")


train_data.head(5) 


train_data.shape
test_data.shape 


train_data.info()  



train_data.describe().T


#checking for missing values

train_data.isnull().sum()


numerical_variables = ["Weight Capacity (kg)" ]
target_variables = ["id", "Price"]
categorical_variables = ['Brand', 'Material', 'Size', 'Compartments', 'Laptop Compartment','Waterproof', 'Style', 'Color']


for feature in numerical_variables:  

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Box Plot
    plt.subplot(1, 2, 1)
    sns.boxplot(data=pd.concat([train_data.dropna()]), x=feature)
    plt.xlabel(feature)
    plt.title(f"Box Plot for Target Feature '{feature}'")

    # Histogram
    plt.subplot(1, 2, 2)
    sns.histplot(data=train_data, x=feature, kde=True, bins=30)
    plt.xlabel(feature)
    plt.ylabel("Frequency")
    plt.title(f"histogram for Target Feature '{feature}'")
    plt.legend()

    # Adjust spacing between subplots
    plt.tight_layout()

    # Show the plots
    plt.show()


for feature in categorical_variables:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    #pie chart
    plt.subplot(1, 2, 1)
    train_data[feature].value_counts().plot.pie(
        autopct='%1.1f%%', wedgeprops=dict(width=0.3), startangle=140
    )
    plt.title(f"Pie Chart for {feature}")

    #barchart 
    plt.subplot(1, 2, 2)
    sns.countplot(
        data=pd.concat([train_data.dropna()]), 
        x=feature, 
        alpha=0.8  # Setting 80% opacity
    )
    plt.xlabel(feature)
    plt.ylabel("Count") 

    plt.tight_layout()
    plt.show()


for feature in target_variables:  

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Box Plot
    plt.subplot(1, 2, 1)
    sns.boxplot(data=pd.concat([train_data.dropna()]), x=feature)
    plt.xlabel(feature)
    plt.title(f"Box Plot for Target Feature '{feature}'")

    # Histogram
    plt.subplot(1, 2, 2)
    sns.histplot(data=train_data, x=feature, kde=True, bins=30)
    plt.xlabel(feature)
    plt.ylabel("Frequency")
    plt.title(f"histogram for Target Feature '{feature}'")
    plt.legend()

    # Adjust spacing between subplots
    plt.tight_layout()

    # Show the plots
    plt.show()


#dropping null value

train_data = train_data.dropna()


print(train_data.isnull().sum())


#handling missing values 
categorical_features = ["Brand", "Material", "Size", "Laptop Compartment", "Waterproof", "Style", "Color"]
numerical_features = ["Weight Capacity (kg)"]

for col in categorical_features:
    train_data[col].fillna(train_data[col].mode()[0], inplace=True)
    test_data[col].fillna(test_data[col].mode()[0], inplace=True)

for col in numerical_features:
     train_data[col].fillna(train_data[col].median(),inplace=True)
     test_data[col].fillna(test_data[col].median(), inplace=True)                   
                           


from sklearn.preprocessing import LabelEncoder

def perform_feature_engineering(df):
    df['Brand_Material'] = df['Brand'] + '_' + df['Material']

    # Brand & Size Interaction - Some brands may produce only specific sizes
    df['Brand_Size'] = df['Brand'] + '_' + df['Size']

    # Has Laptop Compartment - Convert Yes/No to 1/0 for easier analysis
    df['Has_Laptop_Compartment'] = df['Laptop Compartment'].map({'Yes': 1, 'No': 0})

    # Is Waterproof - Convert Yes/No to 1/0 for easier analysis
    df['Is_Waterproof'] = df['Waterproof'].map({'Yes': 1, 'No': 0})

    # Compartments Binning - Group compartments into categories
    df['Compartments_Category'] = pd.cut(df['Compartments'], bins=[0, 2, 5, 10, np.inf], labels=['Few', 'Moderate', 'Many', 'Very Many'])

    # Weight Capacity Ratio - Normalize weight capacity using the max value
    df['Weight_Capacity_Ratio'] = df['Weight Capacity (kg)'] / df['Weight Capacity (kg)'].max()

    # Interaction Feature: Weight vs. Compartments - Some bags may hold more with less compartments
    df['Weight_to_Compartments'] = df['Weight Capacity (kg)'] / (df['Compartments'] + 1)  # Avoid division by zero

    # Style and Size Interaction - Certain styles may correlate with sizes
    df['Style_Size'] = df['Style'] + '_' + df['Size']

    return df



# Apply the function to the training data
train_data = perform_feature_engineering(train_data)

# Apply the function to the test data
test_data = perform_feature_engineering(test_data)


id_test = test_data['id']

columns_to_drop = ['id']
train_data.drop(columns_to_drop, axis=1, inplace=True)
test_data.drop(columns_to_drop, axis=1, inplace=True)


columns_to_check = ['Weight Capacity (kg)','Weight_Capacity_Ratio','Weight_to_Compartments']


def outliers_remove(data, column):
    Q1 = data[column].quantile(0.15)
    Q3 = data[column].quantile(0.85)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5*IQR
    upper_bound = Q3 + 1.5*IQR

    # Filter the data
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    
    # Calculate the number of rows deleted
    rows_deleted = len(data) - len(filtered_data)

    # Plot the distribution with outliers
    plt.figure(figsize=(8, 4))
    sns.boxplot(x=data[column], color='lightblue', flierprops={'marker': 'o', 'markersize': 5, 'markerfacecolor': 'red'})
    
    # Highlight Q1 and Q3
    plt.axvline(Q1, color='green', linestyle='--', label='Q1 (10th Percentile)')
    plt.axvline(Q3, color='blue', linestyle='--', label='Q3 (90th Percentile)')
    
    # Highlight lower and upper bounds
    plt.axvline(lower_bound, color='red', linestyle='-', label='Lower Bound')
    plt.axvline(upper_bound, color='red', linestyle='-', label='Upper Bound')

    plt.title(f'Outlier Detection for {column}')
    plt.legend()
    plt.xlabel(column)
    plt.show()
    
    return filtered_data, rows_deleted

# Apply function to each numerical column and visualize
rows_deleted_total = 0

for column in columns_to_check:
    train_data, rows_deleted = outliers_remove(train_data, column)
    rows_deleted_total += rows_deleted
    print(f"Rows deleted for {column}: {rows_deleted}")

print(f"Total rows deleted: {rows_deleted_total}")


y = train_data['Price']


train_data.head(5)


columns_to_encode = ['Brand', 'Material', 'Size', 'Laptop Compartment','Waterproof', 'Style', 'Color','Brand_Material', 'Brand_Size', 'Has_Laptop_Compartment','Is_Waterproof', 'Compartments_Category', 'Style_Size']
train_data_to_encode = train_data[columns_to_encode]
test_data_to_encode = test_data[columns_to_encode]


# Dropping selected columns for scaling
train_data_to_scale = train_data.drop(columns_to_encode, axis=1)
test_data_to_scale = test_data.drop(columns_to_encode, axis=1)


train_data_encoded = pd.get_dummies(train_data_to_encode, columns=columns_to_encode, drop_first=True)
test_data_encoded = pd.get_dummies(test_data_to_encode, columns=columns_to_encode, drop_first=True)


train_data_encoded.head()


from sklearn.preprocessing import MinMaxScaler

# Initialize MinMaxScaler
minmax_scaler = MinMaxScaler()

# Fit the scaler on the training data
minmax_scaler.fit(train_data_to_scale.drop(['Price'], axis=1))

# Scale the training data
scaled_data_train = minmax_scaler.transform(train_data_to_scale.drop(['Price'], axis=1))
scaled_train_df = pd.DataFrame(scaled_data_train, columns=train_data_to_scale.drop(['Price'], axis=1).columns)

# Scale the test data using the parameters from the training data
scaled_data_test = minmax_scaler.transform(test_data_to_scale)
scaled_test_df = pd.DataFrame(scaled_data_test, columns=test_data_to_scale.columns)



test_data_encoded.head()


# Concatenate train datasets
train_data_combined = pd.concat([train_data_encoded.reset_index(drop=True), scaled_train_df.reset_index(drop=True)], axis=1)

# Concatenate test datasets
test_data_combined = pd.concat([test_data_encoded.reset_index(drop=True), scaled_test_df.reset_index(drop=True)], axis=1)


train_data_combined.head()


test_data_combined.head()



from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error
import catboost as cb  

# Define Cross-Validation strategy
kf = KFold(n_splits=10, shuffle=True, random_state=42)

# CatBoost parameters (optimized)
catboost_params = {
    "iterations": 300,
    "learning_rate": 0.1,
    "depth": 6,
    "verbose": 0,
    "random_seed": 42
}

# Lists to store results
rmse_scores = []
mae_scores = []
oof_preds = np.zeros(len(train_data_combined))
test_preds_cb = np.zeros(len(test_data_combined))

# Store feature importances
feature_importance_list = np.zeros(train_data_combined.shape[1])

# Perform K-Fold Cross Validation
print("Training using Cross-Validation...")
for fold, (train_idx, val_idx) in enumerate(kf.split(train_data_combined)):
    print(f"\nTraining Fold {fold+1}...")

    X_train, X_val = train_data_combined.iloc[train_idx], train_data_combined.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    # Define model
    cb_model = cb.CatBoostRegressor(**catboost_params)

    # Train model
    cb_model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=50, verbose=0)

    # Predict on validation set
    val_preds_cb = cb_model.predict(X_val)
    oof_preds[val_idx] = val_preds_cb

    # Calculate and store scores
    rmse = np.sqrt(mean_squared_error(y_val, val_preds_cb))
    mae = mean_absolute_error(y_val, val_preds_cb)
    rmse_scores.append(rmse)
    mae_scores.append(mae)

    print(f"Fold {fold+1} RMSE: {rmse:.4f}, MAE: {mae:.4f}")

    # Accumulate feature importances
    feature_importance_list += cb_model.get_feature_importance() / kf.get_n_splits()

    # Predict on test data and average across folds
    test_preds_cb += cb_model.predict(test_data_combined) / kf.get_n_splits()


print(oof_preds[:10])  # Show first 10 OOF predictions



print(y[:10].values)  # Show first 10 actual values



print(test_preds_cb[:10])  # Show first 10 test predictions



# Create submission file
submission_df = pd.DataFrame({
    'id': id_test,  # Use the stored id_test values
    'Price': test_preds_cb  # Your model's predictions
})

# Save to CSV
submission_df.to_csv("submission.csv", index=False)

# Display first few rows
submission_df.head()





