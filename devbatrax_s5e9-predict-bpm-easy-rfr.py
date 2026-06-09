import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import warnings

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)


TARGET = 'BeatsPerMinute'
ID_COL = 'id'
N_SPLITS = 5         
RANDOM_STATE = 42



train_data = pd.read_csv("/kaggle/input/playground-series-s5e9/train.csv") 
test_data = pd.read_csv("/kaggle/input/playground-series-s5e9/test.csv")    

test_ids = test_data[ID_COL] 
X_train = train_data.drop(columns=[ID_COL, TARGET])
y_train = train_data[TARGET]
X_test_orig = test_data.drop(columns=[ID_COL])

print(f"Data Loaded. Training samples: {X_train.shape[0]}, Features: {X_train.shape[1]}")
    
#Impute Missing Values with Median
for col in X_train.columns:
        if X_train[col].isnull().any():
            median_val = X_train[col].median()
            X_train[col] = X_train[col].fillna(median_val)
            X_test_orig[col] = X_test_orig[col].fillna(median_val)




if 'TrackDurationMs' in X_train.columns:
    X_train['Log_Duration'] = np.log1p(X_train['TrackDurationMs'])
    X_test_orig['Log_Duration'] = np.log1p(X_test_orig['TrackDurationMs'])
    X_train = X_train.drop(columns=['TrackDurationMs'])
    X_test_orig = X_test_orig.drop(columns=['TrackDurationMs'])

if 'AudioLoudness' in X_train.columns and 'Energy' in X_train.columns:
    X_train['Loudness_x_Energy'] = X_train['AudioLoudness'] * X_train['Energy']
    X_test_orig['Loudness_x_Energy'] = X_test_orig['AudioLoudness'] * X_test_orig['Energy']


#random forest model

rfr_model = RandomForestRegressor(
    n_estimators=500,
    max_depth=10, 
    min_samples_split=5,
    min_samples_leaf=3,
    n_jobs=-1,
    random_state=RANDOM_STATE,
    verbose=0
)

print("\n--- Starting 5-Fold Cross-Validation for Random Forest ---")

kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
oof_predictions = np.zeros(X_train.shape[0])
test_predictions_sum = np.zeros(X_test_orig.shape[0])
rmse_scores = []


# k-fold training and prediction

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train, y_train)):
    
    X_train_fold, y_train_fold = X_train.iloc[train_idx], y_train.iloc[train_idx]
    X_val_fold = X_train.iloc[val_idx]
    y_val_fold = y_train.iloc[val_idx]

    rfr_model.fit(X_train_fold, y_train_fold)

    val_preds = rfr_model.predict(X_val_fold)
    oof_predictions[val_idx] = val_preds
    
    fold_rmse = np.sqrt(mean_squared_error(y_val_fold, val_preds))
    rmse_scores.append(fold_rmse)
    print(f"  -> Fold {fold+1} RMSE: {fold_rmse:.4f}")
    
    test_predictions_sum += rfr_model.predict(X_test_orig)

avg_rmse = np.mean(rmse_scores)
final_test_predictions = test_predictions_sum / N_SPLITS

print("\n" + "="*50)
print(f"âœ… FINAL RANDOM FOREST CV RMSE: {avg_rmse:.4f}")
print("="*50)


#submission

submission_df = pd.DataFrame({
    ID_COL: test_ids,
    TARGET: final_test_predictions
})

submission_df.to_csv('rfr_submission.csv', index=False)
print("Submission file 'rfr_submission.csv' created.")




