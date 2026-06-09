import h5py
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor

# Load training data
def load_train_data(h5_file_path):
    train_spot_tables = {}
    with h5py.File(h5_file_path, "r") as f:
        train_spots = f["spots/Train"]
        for slide_name in train_spots.keys():
            spot_array = np.array(train_spots[slide_name])
            df = pd.DataFrame(spot_array)
            train_spot_tables[slide_name] = df
    print("Training data loaded successfully.")
    return train_spot_tables

# Prepare training set
def prepare_training_set(train_spot_tables, slides=None):
    if slides is None:
        slides = list(train_spot_tables.keys())
    
    X_list, y_list = [], []
    for slide_id in slides:
        df = train_spot_tables[slide_id]
        feature_cols = ['x', 'y']
        target_cols = [col for col in df.columns if col not in feature_cols]
        
        X_list.append(df[feature_cols].values.astype(float))
        y_list.append(df[target_cols].values.astype(float))
    
    return np.vstack(X_list), np.vstack(y_list), target_cols

# Load test data
def load_test_data(h5_file_path, slide_id):
    with h5py.File(h5_file_path, "r") as f:
        test_spots = f["spots/Test"]
        if slide_id not in test_spots:
            raise ValueError(f"Slide {slide_id} not found in test data.")
        spot_array = np.array(test_spots[slide_id])
        test_df = pd.DataFrame(spot_array)
    print(f"Test data for slide {slide_id} loaded successfully.")
    return test_df

# Train MultiOutputRegressor with XGBoost using K-Fold
def train_multioutput_xgb(X, y, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    models = []
    fold = 1
    for train_index, val_index in kf.split(X):
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]
        
        model = MultiOutputRegressor(xgb.XGBRegressor(objective='reg:squarederror', n_estimators=1000, max_depth=6, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8, random_state=42))
        model.fit(X_train, y_train)
        models.append(model)
        
        y_pred = model.predict(X_val)
        mse = mean_squared_error(y_val, y_pred)
        r2 = r2_score(y_val, y_pred)
        print(f"Fold {fold}: MSE={mse:.4f}, R²={r2:.4f}")
        fold += 1
    
    return models

# Make predictions
def predict(models, X_test):
    predictions = np.mean([model.predict(X_test) for model in models], axis=0)
    return predictions

# Create submission file
def create_submission(test_df, predictions, target_cols, submission_filename="submission.csv"):
    pred_df = pd.DataFrame(predictions, columns=target_cols, index=test_df.index)
    pred_df.insert(0, 'ID', pred_df.index)
    pred_df.to_csv(submission_filename, index=False)
    print(f"Submission file '{submission_filename}' created!")

# Main execution
if __name__ == "__main__":
    h5_file_path = "/kaggle/input/el-hackathon-2025/elucidata_ai_challenge_data.h5"
    
    # Load training data
    train_spot_tables = load_train_data(h5_file_path)
    X, y, target_cols = prepare_training_set(train_spot_tables)
    
    # Train MultiOutputRegressor with XGBoost
    models = train_multioutput_xgb(X, y, n_splits=5)
    
    # Load test data
    test_df = load_test_data(h5_file_path, slide_id='S_7')
    X_test = test_df[['x', 'y']].values.astype(float)
    
    # Make predictions
    predictions = predict(models, X_test)
    
    # Save predictions
    create_submission(test_df, predictions, target_cols)


