import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_absolute_error
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import Ridge

# Set random seeds for reproducibility
np.random.seed(42)

# Load data with optimized dtypes
dtypes = {
    'Brand': 'category',
    'Material': 'category',
    'Color': 'category',
    'ClosureType': 'category',
    'Height': 'float32',
    'Width': 'float32',
    'Depth': 'float32',
    'Weight': 'float32',
    'Price': 'float32'
}

train = pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv', dtype=dtypes)
train_extra = pd.read_csv('/kaggle/input/playground-series-s5e2/training_extra.csv', dtype=dtypes)
test = pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv', dtype=dtypes)


print(train.columns)
print(test.columns)
print(train_extra.columns)


# Feature Engineering
def create_features(df):
    # Handle missing values (impute if necessary)
    df = df.fillna(df.mode().iloc[0])  # For categorical, fill with mode
    df['Weight Capacity (kg)'] = df['Weight Capacity (kg)'].fillna(df['Weight Capacity (kg)'].mean())  # For numerical, fill with mean
    
    # Interaction features: Combine categorical features
    df['Brand_Material'] = df['Brand'].astype(str) + "_" + df['Material'].astype(str)
    df['Color_Style'] = df['Color'].astype(str) + "_" + df['Style'].astype(str)
    
    # Binning the 'Weight Capacity (kg)' to create a categorical feature
    df['Weight_Bin'] = pd.qcut(df['Weight Capacity (kg)'], q=5, labels=False)

    # You can add more advanced features here as needed
    
    # Example of feature encoding: Convert categorical variables into dummy variables
    categorical_cols = ['Brand', 'Material', 'Size', 'Style', 'Color']
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)  # One-hot encoding

    return df

# Apply to all datasets
train = create_features(train)
train_extra = create_features(train_extra)
test = create_features(test)

# Check the updated datasets
print(train.head())
print(train_extra.head())
print(test.head())


# Target Transformation & Encoding
# Log-transform target for better performance
train['Price'] = np.log1p(train['Price'])
train_extra['Price'] = np.log1p(train_extra['Price'])

# Check the column names in each dataset
print("Train columns:", train.columns)
print("Train extra columns:", train_extra.columns)
print("Test columns:", test.columns)

# Adjust categorical columns to match the ones in your dataset
cat_cols = ['Brand_Material', 'Color_Style', 'Weight_Bin', 
            'Brand_Jansport', 'Brand_Nike', 'Brand_Puma', 'Brand_Under Armour',
            'Material_Leather', 'Material_Nylon', 'Material_Polyester',
            'Size_Medium', 'Size_Small', 'Style_Messenger', 'Style_Tote',
            'Color_Blue', 'Color_Gray', 'Color_Green', 'Color_Pink', 'Color_Red']

# Label Encoding for categorical features
for col in cat_cols:
    le = LabelEncoder()

    # Concatenate the train, train_extra, and test data to ensure the encoder sees all possible categories
    combined = pd.concat([train[col], train_extra[col], test[col]], axis=0)
    
    # Fit the encoder on all combined data
    le.fit(combined)
    
    # Transform the columns for train, train_extra, and test datasets
    train[col] = le.transform(train[col])
    train_extra[col] = le.transform(train_extra[col])
    test[col] = le.transform(test[col])

# Check the transformed data
print(train.head())
print(train_extra.head())
print(test.head())


# Data Combination Strategy
# Create weighted dataset
train['source_weight'] = 1.5  # Higher weight for competition data
train_extra['source_weight'] = 1.0  # Standard weight for extra data

# Combine datasets
full_train = pd.concat([train, train_extra])

# Features and target
X = full_train.drop(['id', 'Price', 'source_weight'], axis=1)
y = full_train['Price']
weights = full_train['source_weight']

# Make sure 'test' dataset has the same columns as 'full_train', excluding 'id' and 'Price'
# Remove 'id' column from test dataset, but keep other features consistent
X_test = test.drop(['id'], axis=1)

# Check if the features are consistent between training and test sets
assert X.columns.equals(X_test.columns), "Features do not match between train and test datasets."


# Cross-Validation Setup
# 5-fold cross-validation
folds = KFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(len(X))
test_preds = np.zeros(len(X_test))

# Feature importance storage
feature_importance = pd.DataFrame(index=X.columns)


# Encoding categorical features if necessary (handle OneHotEncoding warning)
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor

encoder = OneHotEncoder(sparse_output=False)  # Change sparse to sparse_output
X_encoded = encoder.fit_transform(X.select_dtypes(include=['object']))

# Combine encoded features back into the main dataset (if necessary)

X = pd.concat([
    X.select_dtypes(exclude=['object']),
    pd.DataFrame(X_encoded, index=X.index)
], axis=1)


# Split the data into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Define XGBoost parameters (updated to use early_stopping_rounds in constructor)
xgb_params = {
    'n_estimators': 10000,
    'learning_rate': 0.01,
    'max_depth': 6,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'gamma': 0.1,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'random_state': 42,
    'objective': 'reg:squarederror',
    'eval_metric': 'mae',
    'early_stopping_rounds': 100  # Set early_stopping_rounds in the constructor
}

# Initialize the XGBRegressor model
xgb_model = XGBRegressor(**xgb_params)

# Fit the model on training data
xgb_model.fit(
    X_train, y_train,
    sample_weight=None,  # If you have weights, set this to w_train
    eval_set=[(X_val, y_val)],  # Validation set
    verbose=50  # Print training progress every 50 iterations
)

# Make predictions
y_pred = xgb_model.predict(X_val)

# Evaluate the model
mae = mean_absolute_error(y_val, y_pred)
print(f'Mean Absolute Error: {mae}')

# Optional: Feature importance
import matplotlib.pyplot as plt
xgb.plot_importance(xgb_model)
plt.show()


# Feature Importance Analysis

# Check if model is trained and has booster
if hasattr(xgb_model, 'get_booster'):
    booster = xgb_model.get_booster()
    # Use 'gain' importance for better quality than 'weight'
    feature_scores = booster.get_score(importance_type='gain')

    print("ğŸ”� Raw Feature Importance:")
    print(feature_scores)

    if feature_scores:
        # Get feature names (XGBoost converts them to f0, f1, ..., so map back)
        feature_map = {f'f{idx}': col for idx, col in enumerate(X_train.columns)}

        # Map back feature names
        feature_importance_df = pd.DataFrame([
            {'Feature': feature_map.get(k, k), 'Importance': v} for k, v in feature_scores.items()
        ])

        # Sort and plot
        feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

        print("ğŸ“Š Top Feature Importances:")
        print(feature_importance_df.head())

        # Plot
        plt.figure(figsize=(12, 8))
        sns.barplot(x='Importance', y='Feature', data=feature_importance_df.head(20))
        plt.title('Top 20 Feature Importances (XGBoost - Gain)')
        plt.xlabel('Importance (Gain)')
        plt.ylabel('Feature')
        plt.tight_layout()
        plt.show()
    else:
        print("âš ï¸� No feature importance scores found.")
else:
    print("â�Œ Model is not trained or incompatible for feature importance extraction.")


# Final Submission
# Inverse log transform
final_pred = np.expm1(test_preds)

# Create submission
submission = pd.DataFrame({
    'id': test['id'],
    'Price': final_pred
})

# Save submission
submission.to_csv('submission.csv', index=False)
print("Submission file created!")

