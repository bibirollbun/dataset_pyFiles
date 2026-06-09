# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import numpy as np
import warnings
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')


train = pd.read_csv("/kaggle/input/playground-series-s5e6/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e6/test.csv")

columns = train.columns.tolist()
train.head()


print(train.shape)
print(test.shape)


train.describe()


train.dtypes


## Categorical data
cat_feat = train.select_dtypes(include=['object'])

## Numeric Data
num_feat = train.select_dtypes(include=['number'])



num_feat.corr()
cat_feat





from sklearn.preprocessing import LabelEncoder

# Get list of categorical columns
cat_feat = train.select_dtypes(include=['object']).columns.tolist()

# Initialize the encoder
le = LabelEncoder()

# Apply Label Encoding to each categorical feature
for col in cat_feat:
    train[col] = le.fit_transform(train[col])



train.corr()



# Set style
sns.set(style="whitegrid")

# Set figure size
train.hist(figsize=(15, 12), bins=20, color='skyblue', edgecolor='black')
plt.suptitle("Histogram of All Features", fontsize=18)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()


for col in train.columns:
    plt.figure(figsize=(6, 4))
    sns.histplot(train[col], kde=True, bins=20, color='teal', edgecolor='black')
    plt.title(f"Distribution of {col}")
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()




# Optional: Exclude 'id' if it's not meaningful
features_to_plot = [col for col in train.columns if col != 'id']

# Loop through features and plot unique value counts
for col in features_to_plot:
    unique_vals = train[col].nunique()
    
    # Plot only if unique values are reasonable to display (e.g. <= 20)
    if unique_vals <= 20:
        plt.figure(figsize=(8, 4))
        sns.countplot(data=train, x=col, palette='Set2')
        plt.title(f"Count Plot of {col} (Unique: {unique_vals})")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    else:
        print(f"Skipping '{col}' â€” too many unique values ({unique_vals}) to display cleanly.")



import matplotlib.pyplot as plt
import seaborn as sns

# Exclude non-numeric columns like 'id' and 'Fertilizer Name' (if it's the target)
num_features = train.select_dtypes(include='number').drop(columns=['id', 'Fertilizer Name'], errors='ignore')

# Plot box plots for all numerical features
for col in num_features.columns:
    plt.figure(figsize=(6, 4))
    sns.boxplot(y=train[col], color='lightcoral')
    plt.title(f"Box Plot of {col}")
    plt.ylabel(col)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()



import pandas as pd
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Split into X (features) and y (target)
X = train.drop(columns=['Fertilizer Name', 'id'])  # drop target and id
y = train['Fertilizer Name']

# 2. (Optional) If not already numeric, label-encode y
#    â€” but it looks like 'Fertilizer Name' is already integer-encoded in your DataFrame.

# 3. Train-test split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. Fit a LightGBM classifier
model = lgb.LGBMClassifier(
    n_estimators=200,
    learning_rate=0.05,
    random_state=42
)
model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    # early_stopping_rounds=20,
    # verbose=False
)

# 5. Extract feature importances
feat_imp = pd.Series(
    model.feature_importances_, 
    index=X.columns
).sort_values(ascending=False)

# 6. Show the ranked importances
print("Feature importances (highest â†’ lowest):")
print(feat_imp)

# 7. (Optional) Visualize as a bar plot
plt.figure(figsize=(8, 5))
sns.barplot(x=feat_imp.values, y=feat_imp.index, palette='viridis')
plt.title("LightGBM Feature Importances")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()



import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# 1. Select top 4 features and target
top_features = ['Phosphorous', 'Nitrogen', 'Moisture', 'Potassium']
X = train[top_features]
y = train['Fertilizer Name']

# 2. Train-test split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Initialize XGBoost Classifier
model = xgb.XGBClassifier(
    n_estimators=200,
    learning_rate=0.05,
    use_label_encoder=False,
    eval_metric='mlogloss',
    random_state=42
)

# 4. Train the model
model.fit(X_train, y_train)

# 5. Make predictions
y_pred = model.predict(X_val)

# 6. Evaluate performance
acc = accuracy_score(y_val, y_pred)
print(f"Validation Accuracy: {acc:.4f}")
print("\nClassification Report:")
print(classification_report(y_val, y_pred))



import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# 1. Select all features except target and ID
exclude_cols = ['Fertilizer Name', 'id']  # exclude target and id
features = [col for col in train.columns if col not in exclude_cols]

X = train[features]
y = train['Fertilizer Name']

print(f"Features used for training: {features}")

# 2. Train-test split
print("Splitting data into train and validation sets...")
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Train set size: {X_train.shape[0]} samples")
print(f"Validation set size: {X_val.shape[0]} samples")

# 3. Initialize XGBoost Classifier
print("Initializing XGBoost classifier...")
model = xgb.XGBClassifier(
    n_estimators=200,
    learning_rate=0.05,
    use_label_encoder=False,
    eval_metric='mlogloss',
    random_state=42
)

# 4. Train the model
print("Training model...")
model.fit(X_train, y_train)
print("Training completed.")

# 5. Make predictions
print("Predicting on validation set...")
y_pred = model.predict(X_val)

# 6. Evaluate performance
acc = accuracy_score(y_val, y_pred)
print(f"\nValidation Accuracy: {acc:.4f}")

print("\nClassification Report:")
print(classification_report(y_val, y_pred))



import numpy as np
from sklearn.preprocessing import LabelEncoder

# Step 1: Encode target labels
le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)
y_val_enc = le.transform(y_val)  # must match training encoder

# Step 2: Predict probabilities
probs = model.predict_proba(X_val)

# Step 3: Get top 3 prediction indices
top_3 = np.argsort(probs, axis=1)[:, ::-1][:, :3]  # shape = (n_samples, 3)

# Step 4: MAP@3 metric definition
def apk(actual, predicted, k=3):
    """Average Precision at k for one sample"""
    if actual in predicted[:k]:
        return 1.0 / (np.where(predicted == actual)[0][0] + 1)
    return 0.0

def mapk(y_true, y_pred, k=3):
    """Mean Average Precision at k"""
    return np.mean([apk(a, p) for a, p in zip(y_true, y_pred)])

# Step 5: Evaluate
map3 = mapk(y_val_enc, top_3, k=3)
print(f"\nðŸ“Š MAP@3 Score: {map3:.4f}")



xgb.XGBClassifier(
    n_estimators=500,
    learning_rate=0.03,
    max_depth=7,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    use_label_encoder=False,
    eval_metric='mlogloss'
)



import pandas as pd
import xgboost as xgb
import numpy as np
from sklearn.preprocessing import LabelEncoder

# 1. Load data
train = pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv')

# 2. Encode categorical features
cat_features = ['Soil Type', 'Crop Type']
for col in cat_features:
    le_cat = LabelEncoder()
    # Fit on combined to handle unseen labels safely
    all_values = pd.concat([train[col], test[col]], axis=0)
    le_cat.fit(all_values)
    train[col] = le_cat.transform(train[col])
    test[col] = le_cat.transform(test[col])

# 3. Prepare features and target
exclude_cols = ['Fertilizer Name', 'id']
features = [col for col in train.columns if col not in exclude_cols]

X_train = train[features]
y_train = train['Fertilizer Name']
X_test = test[features]

# 4. Encode target labels
le_target = LabelEncoder()
y_train_enc = le_target.fit_transform(y_train)

# 5. Train model
model = xgb.XGBClassifier(
    n_estimators=500,
    learning_rate=0.03,
    max_depth=7,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    use_label_encoder=False,
    eval_metric='mlogloss'
)

print("Training model...")
model.fit(X_train, y_train_enc)
print("Training complete.")

# 6. Predict on test set
print("Predicting on test data...")
test_probs = model.predict_proba(X_test)

# 7. Get top 3 predicted classes (as class names)
top_3_indices = np.argsort(test_probs, axis=1)[:, ::-1][:, :3]
top_3_labels = le_target.inverse_transform(top_3_indices.ravel()).reshape(-1, 3)

# 8. Display sample
print("\nðŸ“¦ Sample predictions (top 3 per row):")
for i in range(5):
    print(f"Sample {i+1}: {top_3_labels[i]}")

# 9. Create submission file
submission = pd.DataFrame({
    'id': test['id'],
    'Fertilizer Name': [' '.join(row) for row in top_3_labels]
})
submission.to_csv("submission1.csv", index=False)
print("\nâœ… Submission file saved as submission.csv")



from sklearn.model_selection import train_test_split

# Split training set for local MAP@3 evaluation
X_train_split, X_val, y_train_split, y_val = train_test_split(
    X_train, y_train_enc, test_size=0.2, random_state=42, stratify=y_train_enc
)

# Train on the split data
model_split = xgb.XGBClassifier(
    n_estimators=500,
    learning_rate=0.03,
    max_depth=7,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    use_label_encoder=False,
    eval_metric='mlogloss'
)

print("Training split model for validation...")
model_split.fit(X_train_split, y_train_split)
print("Validation training complete.")

# Predict on validation
val_probs = model_split.predict_proba(X_val)
top_3_val = np.argsort(val_probs, axis=1)[:, ::-1][:, :3]

# Define MAP@3
def apk(actual, predicted, k=3):
    if actual in predicted[:k]:
        return 1.0 / (np.where(predicted == actual)[0][0] + 1)
    return 0.0

def mapk(y_true, y_pred, k=3):
    return np.mean([apk(a, p) for a, p in zip(y_true, y_pred)])

# Evaluate MAP@3 on validation
map3_val_score = mapk(y_val, top_3_val, k=3)
print(f"\nðŸ“Š Local Validation MAP@3 Score: {map3_val_score:.4f}")


submission = pd.DataFrame({
    'id': test['id'],
    'Fertilizer Name': [' '.join(row) for row in top_3_labels]
})

# Save the submission file
submission.to_csv("submission2.csv", index=False)
print("âœ… Submission file saved as 'submission.csv'")



# Avoid division by zero
train['N_by_P'] = train['Nitrogen'] / (train['Phosphorous'] + 1e-6)
train['N_by_K'] = train['Nitrogen'] / (train['Potassium'] + 1e-6)
train['P_by_K'] = train['Phosphorous'] / (train['Potassium'] + 1e-6)

test['N_by_P'] = test['Nitrogen'] / (test['Phosphorous'] + 1e-6)
test['N_by_K'] = test['Nitrogen'] / (test['Potassium'] + 1e-6)
test['P_by_K'] = test['Phosphorous'] / (test['Potassium'] + 1e-6)



import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder

# 1. Load data
train = pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv')

# 2. Encode categorical features 'Soil Type' and 'Crop Type' using LabelEncoder on combined data
cat_features = ['Soil Type', 'Crop Type']
for col in cat_features:
    le_cat = LabelEncoder()
    combined = pd.concat([train[col], test[col]], axis=0)
    le_cat.fit(combined)
    train[col] = le_cat.transform(train[col])
    test[col] = le_cat.transform(test[col])

# 3. Create safe divide function to avoid division by zero
def safe_divide(a, b):
    return np.where(b != 0, a / b, 0)

# 4. Feature engineering on train and test
for df in [train, test]:
    df['N_by_P'] = safe_divide(df['Nitrogen'], df['Phosphorous'])
    df['N_by_K'] = safe_divide(df['Nitrogen'], df['Potassium'])
    df['P_by_K'] = safe_divide(df['Phosphorous'], df['Potassium'])
    df['N_P_ratio'] = df['N_by_P']  # just to keep consistent naming if needed
    df['N_K_ratio'] = df['N_by_K']
    df['P_K_ratio'] = df['P_by_K']
    # Combine Soil Type and Crop Type as a single categorical feature
    df['Soil_Crop_Combo'] = df['Soil Type'].astype(str) + "_" + df['Crop Type'].astype(str)

# 5. Encode the new combined feature 'Soil_Crop_Combo'
le_combo = LabelEncoder()
combined_combo = pd.concat([train['Soil_Crop_Combo'], test['Soil_Crop_Combo']], axis=0)
le_combo.fit(combined_combo)
train['Soil_Crop_Combo'] = le_combo.transform(train['Soil_Crop_Combo'])
test['Soil_Crop_Combo'] = le_combo.transform(test['Soil_Crop_Combo'])

# 6. Prepare feature list (exclude target and id)
exclude_cols = ['Fertilizer Name', 'id']
features = [col for col in train.columns if col not in exclude_cols]

print("Features used:", features)

X_train = train[features]
y_train = train['Fertilizer Name']

# 7. Encode target labels
le_target = LabelEncoder()
y_train_enc = le_target.fit_transform(y_train)

# 8. Initialize and train XGBoost model
model = xgb.XGBClassifier(
    n_estimators=500,
    learning_rate=0.03,
    max_depth=7,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    use_label_encoder=False,
    eval_metric='mlogloss'
)

print("Training model with engineered features...")
model.fit(X_train, y_train_enc)
print("Training complete.")

# 9. Predict on test set
X_test = test[features]
print("Predicting on test data...")
test_probs = model.predict_proba(X_test)

# 10. Get top 3 predicted class indices and convert to class names
top_3_indices = np.argsort(test_probs, axis=1)[:, ::-1][:, :3]
top_3_labels = le_target.inverse_transform(top_3_indices.ravel()).reshape(-1, 3)

# 11. Display sample predictions
print("\nSample top-3 predictions:")
for i in range(5):
    print(f"Sample {i+1}: {top_3_labels[i]}")

# 12. Prepare submission DataFrame (optional)
submission = pd.DataFrame({
    'id': test['id'],
    'Fertilizer Name': [' '.join(row) for row in top_3_labels]
})

# Uncomment below to save submission file
# submission.to_csv("submission.csv", index=False)
# print("Submission saved.")



from sklearn.model_selection import train_test_split

# Split train data
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train, y_train_enc, test_size=0.2, random_state=42, stratify=y_train_enc
)

# Train on train split
model.fit(X_tr, y_tr)

# Predict on validation split
val_probs = model.predict_proba(X_val)
top_3_val_indices = np.argsort(val_probs, axis=1)[:, ::-1][:, :3]
top_3_val_labels = le_target.inverse_transform(top_3_val_indices.ravel()).reshape(-1, 3)
y_val_labels = le_target.inverse_transform(y_val)

# MAP@3 metric
def apk(actual, predicted, k=3):
    for i, p in enumerate(predicted[:k]):
        if p == actual:
            return 1.0 / (i + 1)
    return 0.0

def mapk(actuals, predicted, k=3):
    return np.mean([apk(a, p, k) for a, p in zip(actuals, predicted)])

map3_score = mapk(y_val_labels, top_3_val_labels, k=3)
print(f"Local Validation MAP@3 Score: {map3_score:.4f}")



submission.to_csv("submission.csv", index=False)
print("Submission saved.")


