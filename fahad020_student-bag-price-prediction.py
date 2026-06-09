import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error


# Load data
train = pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv')
train_extra = pd.read_csv('/kaggle/input/playground-series-s5e2/training_extra.csv')

# Combine training data
train = pd.concat([train, train_extra], axis=0, ignore_index=True)
print(f'Training data shape: {train.shape}')


def process_data(df):
    df = df.copy()
    
    # Fill missing values
    for col in ['Brand', 'Material', 'Size', 'Style', 'Color']:
        df[col] = df[col].fillna('Unknown')
    
    # Binary features
    df['Laptop Compartment'] = df['Laptop Compartment'].map({'Yes': 1, 'No': 0, 'Unknown': -1})
    df['Waterproof'] = df['Waterproof'].map({'Yes': 1, 'No': 0, 'Unknown': -1})
    
    # Size encoding
    df['size_numeric'] = df['Size'].map({'Small': 1, 'Medium': 2, 'Large': 3, 'Unknown': 0})
    
    # Feature engineering
    df['premium_features'] = df['Laptop Compartment'] + df['Waterproof']
    df['compartment_density'] = df['Compartments'] / df['size_numeric'].replace(0, 1)
    df['brand_material'] = df['Brand'] + '_' + df['Material']
    df['style_size'] = df['Style'] + '_' + df['Size']
    
    # Handle outliers
    df['Weight Capacity (kg)'] = df['Weight Capacity (kg)'].clip(
        df['Weight Capacity (kg)'].quantile(0.01),
        df['Weight Capacity (kg)'].quantile(0.99)
    )
    
    return df

# Process data
train = process_data(train)
test = process_data(test)


# Prepare categorical features
cat_features = ['Brand', 'Material', 'Size', 'Style', 'Color', 'brand_material', 'style_size']
encoders = {}

for col in cat_features:
    encoders[col] = LabelEncoder()
    train[col] = encoders[col].fit_transform(train[col])
    test[col] = test[col].map(lambda x: 'Unknown' if x not in encoders[col].classes_ else x)
    test[col] = encoders[col].transform(test[col])


# Features for model
features = ['Brand', 'Material', 'size_numeric', 'Compartments',
           'Laptop Compartment', 'Waterproof', 'Style', 'Color',
           'Weight Capacity (kg)', 'premium_features', 'compartment_density',
           'brand_material', 'style_size']

X = train[features]
y = train['Price']
X_test = test[features]


# LightGBM parameters
params = {
    'objective': 'regression',
    'metric': 'rmse',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'colsample_bytree': 0.9,    # Changed from feature_fraction
    'subsample': 0.8,           # Changed from bagging_fraction
    'subsample_freq': 5,        # Changed from bagging_freq
    'n_estimators': 1000,
    'force_row_wise': True,     # Added to remove overhead
    'feature_pre_filter': False # Added to handle the feature names warning
}

# K-fold settings
n_folds = 5
kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

# Arrays for storing results
oof_predictions = np.zeros(len(X))
test_predictions = np.zeros(len(X_test))
scores = []

# Clean feature names (remove whitespace)
X.columns = [col.replace(' ', '_') for col in X.columns]
X_test.columns = [col.replace(' ', '_') for col in X_test.columns]

# K-fold cross validation
for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    print(f'Fold {fold + 1}')
    
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    model = lgb.LGBMRegressor(**params)
    model.fit(X_train, y_train)
    
    # Validation predictions
    val_pred = model.predict(X_val)
    oof_predictions[val_idx] = val_pred
    
    # Test predictions
    test_predictions += model.predict(X_test) / n_folds
    
    # Calculate score
    score = np.sqrt(mean_squared_error(y_val, val_pred))
    scores.append(score)
    print(f'RMSE: {score}')

print(f'\nAverage RMSE: {np.mean(scores):.4f} ± {np.std(scores):.4f}')


# Create submission
submission = pd.DataFrame({
    'id': test['id'],
    'Price': test_predictions
})

submission.to_csv('submission.csv', index=False)
print('Submission file created!')

