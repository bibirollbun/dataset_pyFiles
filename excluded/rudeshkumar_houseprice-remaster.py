import numpy as np
import pandas as pd
import category_encoders as ce
from sklearn.model_selection import cross_val_score
from sklearn.impute import KNNImputer
from sklearn.feature_selection import SelectKBest, f_regression
import lightgbm as lgb
from bayes_opt import BayesianOptimization
import pickle

# Load data
df_train = pd.read_csv('/kaggle/input/local-hack-day-data-whats-that-house-worth/train.csv')
df_test = pd.read_csv('/kaggle/input/local-hack-day-data-whats-that-house-worth/test.csv')

def calculate_total_area(df):
    df['lotUnit'] = df['lotUnit'].replace({'acres': 43560.0, 'sqft': 1.0})
    df['lotArea'] = pd.to_numeric(df['lotArea'], errors='coerce')
    df['lotUnit'] = pd.to_numeric(df['lotUnit'], errors='coerce')
    df['totalArea'] = df['lotArea'] * df['lotUnit']
    return df

def create_features(df, is_train=True):
    if is_train:
        df['price_per_sqft'] = df['price'] / df['livingArea']
    df['total_rooms'] = df['bedrooms'] + df['bathrooms']
    df['dateSold'] = pd.to_datetime(df['dateSold'])
    df['sale_month'] = df['dateSold'].dt.month
    df['sale_quarter'] = df['dateSold'].dt.quarter
    return df

def handle_outliers(df, columns):
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df[col] = df[col].clip(lower_bound, upper_bound)
    return df

def select_features(X, y, k=20):
    n_features = X.shape[1]
    k = min(k, n_features)
    selector = SelectKBest(f_regression, k=k)
    X_new = selector.fit_transform(X, y)
    selected_features = X.columns[selector.get_support()].tolist()
    return X_new, selected_features

def encode_features(df, categorical_columns, encoder=None, target_column=None):
    if encoder is None:
        encoder = ce.CatBoostEncoder(cols=categorical_columns)
        if target_column is not None:
            y = df[target_column]
            df_encoded = encoder.fit_transform(df[categorical_columns], y)
        else:
            raise ValueError("Target column is required to fit the encoder")
    else:
        df_encoded = encoder.transform(df[categorical_columns])
    
    df = df.drop(columns=categorical_columns).join(df_encoded)
    return df, encoder

def preprocess_data(df, is_train=True, encoder=None):
    df = calculate_total_area(df)
    df = create_features(df, is_train)
    
    if is_train:
        df = handle_outliers(df, ['price', 'livingArea', 'totalArea'])
    else:
        df = handle_outliers(df, ['livingArea', 'totalArea'])
    
    categorical_columns = ['state', 'zipcode', 'city', 'homeType', 'homeStatus']
    if is_train:
        df, encoder = encode_features(df, categorical_columns, target_column='price')
    else:
        if encoder is None:
            raise ValueError("Encoder is required for test data preprocessing")
        df, _ = encode_features(df, categorical_columns, encoder=encoder)
    
    df = df.drop(['id', 'address', 'dateSold'], axis=1)
    
    if not is_train:
        df = df.drop('price_per_sqft', axis=1, errors='ignore')
    
    imputer = KNNImputer(n_neighbors=5)
    df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
    
    return df, encoder

# Preprocess training data
df_train_processed, encoder = preprocess_data(df_train, is_train=True)

# Split data
X = df_train_processed.drop(['price', 'price_per_sqft'], axis=1)
y = df_train_processed['price']

print(f"Number of features before selection: {X.shape[1]}")
print("Features before selection:", X.columns.tolist())

# Select best features
X_new, selected_features = select_features(X, y)
print(f"Number of features after selection: {len(selected_features)}")
print("Selected features:", selected_features)
X = X[selected_features]  # Use only the selected features

# Define optimization function for LightGBM
def optimize_lgb(max_depth, learning_rate, n_estimators, min_child_samples, subsample, colsample_bytree):
    params = {
        'max_depth': int(max_depth),
        'learning_rate': learning_rate,
        'n_estimators': int(n_estimators),
        'min_child_samples': int(min_child_samples),
        'subsample': subsample,
        'colsample_bytree': colsample_bytree,
        'objective': 'regression',
        'random_state': 42,
        'n_jobs': -1,
    }
    model = lgb.LGBMRegressor(**params)
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
    return np.mean(np.sqrt(-cv_scores))

# Optimize LightGBM hyperparameters using Bayesian Optimization
pbounds = {
    'max_depth': (3, 10),
    'learning_rate': (0.01, 0.3),
    'n_estimators': (100, 1000),
    'min_child_samples': (1, 10),
    'subsample': (0.5, 1.0),
    'colsample_bytree': (0.5, 1.0)
}

optimizer = BayesianOptimization(f=optimize_lgb, pbounds=pbounds, random_state=42)
optimizer.maximize(init_points=5, n_iter=25)

# Retrieve best parameters
best_params = optimizer.max['params']
best_params['max_depth'] = int(best_params['max_depth'])
best_params['n_estimators'] = int(best_params['n_estimators'])
best_params['min_child_samples'] = int(best_params['min_child_samples'])
print("Best parameters:", best_params)
print("Best cross-validation RMSE:", optimizer.max['target'])

# Train final LightGBM model
final_model = lgb.LGBMRegressor(objective='regression', random_state=42, n_jobs=-1, **best_params)
final_model.fit(X, y)

# Process test data using the same preprocessing pipeline
df_test_processed, _ = preprocess_data(df_test, is_train=False, encoder=encoder)
X_test = df_test_processed[selected_features]

# Make predictions on test data
test_predictions = final_model.predict(X_test)

# Create submission file
submission = pd.DataFrame({'Id': df_test['id'], 'PredictedPrice': test_predictions})
submission.to_csv('submission.csv', index=False)
print("Predictions saved to submission.csv")

# Save the encoder and final model for deployment
with open('encoder.pkl', 'wb') as f:
    pickle.dump(encoder, f)
with open('house_price_model.pkl', 'wb') as f:
    pickle.dump(final_model, f)


