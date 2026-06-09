# Essential libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error
import seaborn as sns
import matplotlib.pyplot as plt


# Load datasets
train = pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv')
train_extra = pd.read_csv('/kaggle/input/playground-series-s5e2/training_extra.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv')


# Basic information
print("Train shape:", train.shape)
print("\nTrain Extra shape:", train_extra.shape)
print("\nTest shape:", test.shape)

# Summary statistics of Price
print("\nPrice Statistics:")
print(train['Price'].describe())

# Missing values check
print("\nMissing Values in Train:")
print(train.isnull().sum())


# 1. Data Preprocessing
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor

def preprocess_data(df, le_dict=None, is_train=True):
    # Create a copy
    data = df.copy()
    
    # Handle missing values
    # For numerical: Weight Capacity
    num_imputer = SimpleImputer(strategy='median')
    data['Weight_Capacity_(kg)'] = num_imputer.fit_transform(data[['Weight Capacity (kg)']])
    # Drop the original column to avoid duplicates
    data = data.drop('Weight Capacity (kg)', axis=1)
    
    # For categorical: use 'missing' as a category
    cat_columns = ['Brand', 'Material', 'Size', 'Laptop Compartment', 
                  'Waterproof', 'Style', 'Color']
    for col in cat_columns:
        data[col] = data[col].fillna('missing')
    
    # Label encoding for categorical variables
    if is_train:
        le_dict = {}
        for col in cat_columns:
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col].astype(str))
            le_dict[col] = le
    else:
        # Use provided label encoders for test data
        for col in cat_columns:
            le = le_dict[col]
            # Handle unseen categories
            data[col] = data[col].astype(str)
            data[col] = data[col].map(lambda x: 'missing' if x not in le.classes_ else x)
            data[col] = le.transform(data[col])
    
    # Feature engineering
    # Create interaction features
    data['Size_Weight'] = data['Size'].astype(float) * data['Weight_Capacity_(kg)']
    data['Compartments_Size'] = data['Compartments'].astype(float) * data['Size'].astype(float)
    
    # Binary features
    data['Has_Laptop_Compartment'] = (data['Laptop Compartment'] != 'missing').astype(int)
    data['Is_Waterproof'] = (data['Waterproof'] != 'missing').astype(int)
    
    if is_train:
        return data, le_dict
    return data

# Rest of the code remains the same...

# 2. Model Training
def train_models(train_data, y):
    models = {
        'lgb': lgb.LGBMRegressor(
            n_estimators=1000,
            learning_rate=0.05,
            num_leaves=31,
            random_state=42
        ),
        'xgb': xgb.XGBRegressor(
            n_estimators=1000,
            learning_rate=0.05,
            max_depth=7,
            random_state=42
        ),
        'rf': RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            random_state=42
        )
    }
    
    trained_models = {}
    scores = {}
    
    feature_cols = [col for col in train_data.columns if col not in ['id', 'Price']]
    
    for name, model in models.items():
        # K-fold cross-validation
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        fold_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(train_data)):
            X_train = train_data.iloc[train_idx][feature_cols]
            y_train = y.iloc[train_idx]
            X_val = train_data.iloc[val_idx][feature_cols]
            y_val = y.iloc[val_idx]
            
            model.fit(X_train, y_train)
            pred = model.predict(X_val)
            rmse = np.sqrt(mean_squared_error(y_val, pred))
            fold_scores.append(rmse)
        
        scores[name] = np.mean(fold_scores)
        trained_models[name] = model
        print(f"{name} CV RMSE: {np.mean(fold_scores):.4f}")
    
    return trained_models, scores

# 3. Prediction and Ensemble
def make_predictions(models, test_data):
    feature_cols = [col for col in test_data.columns if col != 'id']
    predictions = {}
    
    for name, model in models.items():
        pred = model.predict(test_data[feature_cols])
        predictions[name] = pred
    
    # Weighted average ensemble
    weights = {
        'lgb': 0.4,
        'xgb': 0.4,
        'rf': 0.2
    }
    
    final_pred = np.zeros(len(test_data))
    for name, pred in predictions.items():
        final_pred += weights[name] * pred
    
    return final_pred

# 4. Main execution
def main():
    # Load data
# Load datasets
    train = pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv')
    train_extra = pd.read_csv('/kaggle/input/playground-series-s5e2/training_extra.csv')
    test = pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv')
        # Preprocess training data
    train_processed, le_dict = preprocess_data(train, is_train=True)
    train_extra_processed, _ = preprocess_data(train_extra, le_dict=le_dict, is_train=True)
    
    # Combine training data
    combined_train = pd.concat([train_processed, train_extra_processed], axis=0)
    
    # Prepare target variable
    y = combined_train['Price']
    
    # Train models
    trained_models, scores = train_models(combined_train, y)
    
    # Preprocess test data
    test_processed = preprocess_data(test, le_dict=le_dict, is_train=False)
    
    # Make predictions
    final_predictions = make_predictions(trained_models, test_processed)
    
    # Create submission file
    submission = pd.DataFrame({
        'id': test['id'],
        'Price': final_predictions
    })
    submission.to_csv('submission.csv', index=False)

if __name__ == "__main__":
    main()

