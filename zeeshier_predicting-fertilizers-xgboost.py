import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
import lightgbm as lgb
from sklearn.metrics import average_precision_score
import warnings
warnings.filterwarnings('ignore')


# Load data
train = pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv')


# Encode Fertilizer Name early to use numeric values for target encoding
le_fertilizer = LabelEncoder()
train['Fertilizer Name'] = le_fertilizer.fit_transform(train['Fertilizer Name'])


# Feature engineering
train['N_P_ratio'] = train['Nitrogen'] / (train['Phosphorous'] + 1e-6)
test['N_P_ratio'] = test['Nitrogen'] / (test['Phosphorous'] + 1e-6)
train['N_K_ratio'] = train['Nitrogen'] / (train['Potassium'] + 1e-6)
test['N_K_ratio'] = test['Nitrogen'] / (test['Potassium'] + 1e-6)
train['P_K_ratio'] = train['Phosphorous'] / (train['Potassium'] + 1e-6)
test['P_K_ratio'] = test['Phosphorous'] / (test['Potassium'] + 1e-6)


# Bin continuous features
train['Temp_bin'] = pd.cut(train['Temparature'], bins=3, labels=['Low', 'Medium', 'High'])
test['Temp_bin'] = pd.cut(test['Temparature'], bins=3, labels=['Low', 'Medium', 'High'])
train['Moisture_bin'] = pd.cut(train['Moisture'], bins=3, labels=['Low', 'Medium', 'High'])
test['Moisture_bin'] = pd.cut(test['Moisture'], bins=3, labels=['Low', 'Medium', 'High'])


# Target encoding for Soil Type and Crop Type using encoded Fertilizer Name
default_fertilizer = train['Fertilizer Name'].mode()[0]  # Most frequent encoded fertilizer
for col in ['Soil Type', 'Crop Type']:
    means = train.groupby(col)['Fertilizer Name'].apply(lambda x: x.mode()[0])
    train[f'{col}_target'] = train[col].map(means)
    test[f'{col}_target'] = test[col].map(means).fillna(default_fertilizer)


# Encode categorical variables
for col in ['Soil Type', 'Crop Type', 'Temp_bin', 'Moisture_bin']:
    le = LabelEncoder()
    train[col] = le.fit_transform(train[col])
    test[col] = le.transform(test[col])


# Check for unseen categories
for col in ['Soil Type', 'Crop Type']:
    unseen = set(test[col]) - set(train[col])
    if unseen:
        print(f"Warning: Unseen {col} categories in test set: {unseen}")
        test[col] = test[col].apply(lambda x: train[col].mode()[0] if x in unseen else x)


# Encode categorical variables
for col in ['Soil Type', 'Crop Type', 'Temp_bin', 'Moisture_bin']:
    le = LabelEncoder()
    train[col] = le.fit_transform(train[col])
    test[col] = le.transform(test[col])


# Normalize numerical features
scaler = StandardScaler()
num_features = ['Temparature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 
                'Phosphorous', 'N_P_ratio', 'N_K_ratio', 'P_K_ratio', 
                'Soil Type_target', 'Crop Type_target']
train[num_features] = scaler.fit_transform(train[num_features])
test[num_features] = scaler.transform(test[num_features])


# Define features and target
features = ['Temparature', 'Humidity', 'Moisture', 'Soil Type', 'Crop Type', 
            'Nitrogen', 'Potassium', 'Phosphorous', 'N_P_ratio', 'N_K_ratio', 
            'P_K_ratio', 'Temp_bin', 'Moisture_bin', 'Soil Type_target', 'Crop Type_target']
target = 'Fertilizer Name'


# Prepare data
X = train[features]
y = train[target]
X_test = test[features]


# Custom MAP@3 evaluation function
def map_at_3(y_true, y_pred_proba, n=3):
    map_score = 0.0
    for i in range(len(y_true)):
        true_label = y_true.iloc[i] if isinstance(y_true, pd.Series) else y_true[i]
        pred_probs = y_pred_proba[i]
        top_n = np.argsort(pred_probs)[::-1][:n]
        score = 0.0
        relevant = False
        for k in range(n):
            if top_n[k] == true_label and not relevant:
                score += 1.0 / (k + 1)
                relevant = True
        map_score += score
    return map_score / len(y_true)


# K-fold cross-validation with LightGBM
kf = KFold(n_splits=5, shuffle=True, random_state=42)
map_scores = []
for train_idx, val_idx in kf.split(X):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    # Train LightGBM
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    params = {
        'objective': 'multiclass',
        'num_class': len(le_fertilizer.classes_),
        'metric': 'multi_logloss',
        'learning_rate': 0.05,
        'num_leaves': 15,
        'max_depth': 4,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'random_state': 42
    }
    model = lgb.train(params, train_data, num_boost_round=200, valid_sets=[val_data], 
                      callbacks=[lgb.early_stopping(10)])
    
    # Evaluate
    val_proba = model.predict(X_val)
    map_score = map_at_3(y_val, val_proba)
    map_scores.append(map_score)

print(f"Mean CV MAP@3: {np.mean(map_scores):.4f} ± {np.std(map_scores):.4f}")


# Train final model on full data
train_data = lgb.Dataset(X, label=y)
model = lgb.train(params, train_data, num_boost_round=200)


# Predict top 3 fertilizers for test set
test_proba = model.predict(X_test)
predictions = []
for proba in test_proba:
    top_3_indices = np.argsort(proba)[::-1][:3]
    top_3_fertilizers = le_fertilizer.inverse_transform(top_3_indices)
    predictions.append(' '.join(top_3_fertilizers))


# Create submission
submission = pd.DataFrame({'id': test['id'], 'Fertilizer Name': predictions})
submission.to_csv('submission.csv', index=False)
print("Submission file created: submission.csv")

