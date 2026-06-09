import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
import xgboost as xgb


train = pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv')


# Các cột cat cần encode
categorical_cols = ['road_type', 'lighting', 'weather', 'road_signs_present', 'public_road', 'time_of_day', 'holiday', 'school_season']

# Sử dụng LabelEncoder cho từng cột cat
le_dict = {}
for col in categorical_cols:
    le = LabelEncoder()
    # Fit trên train + test để tránh missing labels
    combined = pd.concat([train[col], test[col]])
    le.fit(combined)
    train[col] = le.transform(train[col])
    test[col] = le.transform(test[col])
    le_dict[col] = le

# Các cột features
features = ['road_type', 'num_lanes', 'curvature', 'speed_limit', 'lighting', 'weather', 
            'road_signs_present', 'public_road', 'time_of_day', 'holiday', 'school_season', 
            'num_reported_accidents']

# Target
target = 'accident_risk'

# Chia train thành train/val để evaluate
X = train[features]
y = train[target]
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


model = xgb.XGBRegressor(objective='reg:squarederror', 
                         n_estimators=1000, 
                         learning_rate=0.05, 
                         max_depth=6, 
                         subsample=0.8, 
                         colsample_bytree=0.8, 
                         random_state=42,
                         eval_metric='rmse')

# Fit model với early stopping
model.fit(X_train, y_train, 
          eval_set=[(X_val, y_val)], 
          early_stopping_rounds=50, 
          verbose=100)

# Dự đoán trên val để check MSE
val_preds = model.predict(X_val)
mse = mean_squared_error(y_val, val_preds)
print(f'Validation MSE: {mse}')

# Dự đoán trên test
X_test = test[features]
test_preds = model.predict(X_test)


# Create submission
submission = pd.DataFrame({'id': test['id'], 'accident_risk': test_preds})
submission.to_csv('submission.csv', index=False)
print('Submission file created!')




