import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from xgboost import XGBClassifier
from collections import Counter
import uuid


np.random.seed(42)


def map_at_3(y_true, y_pred, k=3):
    score = 0.0
    num_hits = 0.0
    for i, pred in enumerate(y_pred):
        pred_k = pred[:k]
        if y_true[i] in pred_k:
            idx = np.where(pred_k == y_true[i])[0][0]
            num_hits += 1.0 / (idx + 1)
    return num_hits / len(y_true) if len(y_true) > 0 else 0.0


train_file = '/kaggle/input/playground-series-s5e6/train.csv'
test_file = '/kaggle/input/playground-series-s5e6/test.csv'

train_df = pd.read_csv(train_file)
test_df = pd.read_csv(test_file)


# Encode categorical variables
soil_encoder = LabelEncoder()
crop_encoder = LabelEncoder()
fertilizer_encoder = LabelEncoder()
    
# Fit encoders on combined data to ensure consistency
combined_soil = pd.concat([train_df['Soil Type'], test_df['Soil Type']])
combined_crop = pd.concat([train_df['Crop Type'], test_df['Crop Type']])
soil_encoder.fit(combined_soil)
crop_encoder.fit(combined_crop)
fertilizer_encoder.fit(train_df['Fertilizer Name'])
    
# Transform categorical features
train_df['Soil Type'] = soil_encoder.transform(train_df['Soil Type'])
train_df['Crop Type'] = crop_encoder.transform(train_df['Crop Type'])
train_df['Fertilizer Name'] = fertilizer_encoder.transform(train_df['Fertilizer Name'])
test_df['Soil Type'] = soil_encoder.transform(test_df['Soil Type'])
test_df['Crop Type'] = crop_encoder.transform(test_df['Crop Type'])
    
# Define features
feature_cols = ['Temparature', 'Humidity', 'Moisture', 'Soil Type', 'Crop Type', 
                    'Nitrogen', 'Potassium', 'Phosphorous']
    
# Scale numerical features
scaler = StandardScaler()
numerical_cols = ['Temparature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous']
train_df[numerical_cols] = scaler.fit_transform(train_df[numerical_cols])
test_df[numerical_cols] = scaler.transform(test_df[numerical_cols])
    
# Prepare features and labels
X = train_df[feature_cols].values
y = train_df['Fertilizer Name'].values
X_test = test_df[feature_cols].values
test_ids = test_df['id'].values


# Split data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
# Initialize and train XGBoost model
model = XGBClassifier(
    objective='multi:softprob',
    num_class=len(fertilizer_encoder.classes_),
    eval_metric='mlogloss',
    max_depth=8,
    learning_rate=0.1,
    n_estimators=500,
    random_state=42
)

model.fit(X_train, y_train)
    
# Validate model
val_probs = model.predict_proba(X_val)
val_preds = np.argsort(-val_probs, axis=1)[:, :3]  # Top-3 predictions
map_score = map_at_3(y_val, val_preds)
print(f'Validation MAP@3: {map_score:.4f}')
    
# Predict on test set
test_probs = model.predict_proba(X_test)
test_preds = np.argsort(-test_probs, axis=1)[:, :3]  # Top-3 predictions
    
# Convert predictions to fertilizer names
predictions = []
for pred in test_preds:
    pred_fertilizers = fertilizer_encoder.inverse_transform(pred)
    predictions.append(' '.join(pred_fertilizers))


# Create submission dataframe
submission = pd.DataFrame({
    'id': test_ids,
    'Fertilizer Name': predictions
})


submission.head()


# Save submission
submission.to_csv('submission.csv', index=False)
print("Submission file created: submission.csv")

