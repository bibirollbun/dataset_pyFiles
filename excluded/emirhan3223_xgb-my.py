%load_ext cuml.accel


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O 
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# GÃ¶rselleÅŸtirme ayarlarÄ±
sns.set_theme(style="whitegrid") 
sns.set(font_scale=1.1)


train = pd.read_csv("/kaggle/input/playground-series-s5e6/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e6/test.csv")
sample_submission = pd.read_csv("/kaggle/input/playground-series-s5e6/sample_submission.csv")

print("Train shape:", train.shape)
print("Test shape:", test.shape)
print("Sample submission head:")
display(sample_submission.head())


print("Train dataset:")
display(train.head())

print("Missing values:")
display(train.isnull().sum())

print("ğŸ�¯ Target variable (label) examples:")
print(train['Fertilizer Name'].value_counts())


print("Soil Types:", train['Soil Type'].unique())
print("Crop Types:", train['Crop Type'].unique())
print("Fertilizers:", train['Fertilizer Name'].unique())
print("Number of Fertilizers:", train['Fertilizer Name'].nunique())


plt.figure(figsize=(10,6))
sns.countplot(data=train, y='Fertilizer Name', order=train['Fertilizer Name'].value_counts().index)
plt.title("Fertilizer Name â€“ Class Distribution")
plt.tight_layout()
plt.show()


numeric_cols = ['Temparature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous']
train[numeric_cols].describe().T


# Copy it
df = train.copy()


from sklearn.preprocessing import LabelEncoder
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1.
df = train.copy()

# 2. Label encoding
le_fert = LabelEncoder()
df['Fertilizer Code'] = le_fert.fit_transform(df['Fertilizer Name'])

# 3. Categorical encoding (for Correlation)
le_soil = LabelEncoder()
le_crop = LabelEncoder()
df['Soil Code'] = le_soil.fit_transform(df['Soil Type'])
df['Crop Code'] = le_crop.fit_transform(df['Crop Type'])

# 4. Feature engineering based on NPK â€“ from raw data
df['NPK_sum'] = df['Nitrogen'] + df['Phosphorous'] + df['Potassium']
df['PK_sum'] = df['Phosphorous'] + df['Potassium']
df['NP_sum'] = df['Nitrogen'] + df['Phosphorous']
df['SMI'] = df['Humidity'] / (df['Temparature'] + 1e-6)  # bÃ¶lme hatasÄ±na karÅŸÄ± epsilon

# 5. Select numerical columns (from raw data)
num_cols = ['Temparature', 'Humidity', 'Moisture', 'Nitrogen', 'Phosphorous', 'Potassium']
eng_cols = ['NPK_sum', 'PK_sum', 'NP_sum','SMI']
corr_cols = num_cols + eng_cols + ['Fertilizer Code']

# 6. Compute correlation matrix
corr_matrix = df[corr_cols].corr()

# 7. Visualize â€“ correlations with the target
target_corr = corr_matrix['Fertilizer Code'].drop('Fertilizer Code').sort_values(ascending=False)

plt.figure(figsize=(8, 5))
sns.barplot(x=target_corr.values, y=target_corr.index)
plt.title("Fertilizer Code and Pearson Korelasyon (Raw Data)")
plt.xlabel("Correlation Coefficient")
plt.tight_layout()
plt.show()


from sklearn.preprocessing import StandardScaler

scale_cols = num_cols + eng_cols
scaler = StandardScaler()
df[scale_cols] = scaler.fit_transform(df[scale_cols])


df['Crop_Soil_Combo'] = df['Crop Code'].astype(str) + "_" + df['Soil Code'].astype(str)
le_combo = LabelEncoder()
df['Combo Code'] = le_combo.fit_transform(df['Crop_Soil_Combo'])


final_features = scale_cols + ['Soil Code', 'Crop Code', 'Combo Code']


X = df[final_features]
y = df['Fertilizer Code']


print("Feature matrix size:", X.shape)
print("Label vector example:", y.unique())


X.head()


import pandas as pd
import numpy as np
import time
import optuna
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from xgboost import XGBClassifier

# Separate Training/Validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

def evaluate_model(model, X_train, y_train, X_val, y_val):
    start = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start

    preds = model.predict(X_val)
    proba = model.predict_proba(X_val)

    acc = accuracy_score(y_val, preds)
    f1 = f1_score(y_val, preds, average='weighted')
    try:
        auc = roc_auc_score(y_val, proba, multi_class='ovr')
    except:
        auc = np.nan

    return {
        "Accuracy": acc,
        "F1-Score": f1,
        "AUC": auc,
        "Train Time (s)": train_time
    }

# Optuna objective func
def objective(trial):
    params = {
        'objective': 'multi:softprob',
        'num_class': len(np.unique(y)),
        'eval_metric': 'mlogloss',
        'use_label_encoder': False,
        'tree_method': 'hist',
        'device': 'cuda',
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'gamma': trial.suggest_float('gamma', 0, 5),
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'random_state': 42
    }

    model = XGBClassifier(**params)
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    f1 = f1_score(y_val, preds, average='weighted')
    return f1  # Maximize F1-score

# Start Optim
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)

best_params = study.best_params
best_params.update({
    'objective': 'multi:softprob',
    'num_class': len(np.unique(y)),
    'eval_metric': 'mlogloss',
    'use_label_encoder': False,
    'tree_method': 'hist',
    'device': 'cuda',
    'random_state': 42
})
final_model = XGBClassifier(**best_params)

# Evaluate
print("\nğŸ�† Best hyperparams:")
print(best_params)

results = evaluate_model(final_model, X_train, y_train, X_val, y_val)
print("\nğŸ“Š Best results of XGBoost:")
for key, value in results.items():
    print(f"{key}: {value:.4f}")


import numpy as np

# 1. Get prediction probabilities
probs = final_model.predict_proba(X_val)

# 2. Get top 3 predictions in order
top3_preds = np.argsort(probs, axis=1)[:, -3:][:, ::-1]  # shape: (n_samples, 3)

# 3. Get true labels (with string representations)
true_labels = le_fert.inverse_transform(y_val)

# 4. Convert top 3 predicted labels to string representation
top3_labels = le_fert.inverse_transform(top3_preds.ravel()).reshape(top3_preds.shape)

# 5. MAP@3
def mapk(actual, predicted, k=3):
    score = 0.0
    for a, p in zip(actual, predicted):
        try:
            score += 1.0 / (p[:k].index(a) + 1)
        except ValueError:
            continue
    return score / len(actual)

# 6. MAP@3 calculation
map3_score = mapk(true_labels, top3_labels.tolist(), k=3)
print(f"âœ… MAP@3 Score: {map3_score:.4f}")


import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

# 1. Test Data
test = pd.read_csv("/kaggle/input/playground-series-s5e6/test.csv")

# 2. Trained Encoders
test['Soil Code'] = le_soil.transform(test['Soil Type'])
test['Crop Code'] = le_crop.transform(test['Crop Type'])

# 3. NPK and SMI 
test['NPK_sum'] = test['Nitrogen'] + test['Phosphorous'] + test['Potassium']
test['PK_sum'] = test['Phosphorous'] + test['Potassium']
test['NP_sum'] = test['Nitrogen'] + test['Phosphorous']
test['SMI'] = test['Humidity'] / test['Temparature']

# 4. Scaling scalar columns
scale_cols = ['Temparature', 'Humidity', 'Moisture', 'Nitrogen', 'Phosphorous', 'Potassium',
              'NPK_sum', 'PK_sum', 'NP_sum', 'SMI']
test[scale_cols] = scaler.transform(test[scale_cols])

# 5. Prepare X_test according to the input format of the model
X_test = test[scale_cols + ['Soil Code', 'Crop Code']]


X_test = X_test.copy() 
test['Combo Code'] = test['Soil Code'] * 100 + test['Crop Code']
X_test['Combo Code'] = test['Combo Code']


# 6. Predict (top-3)
test_probs = final_model.predict_proba(X_test)
top3_test_preds = np.argsort(test_probs, axis=1)[:, -3:][:, ::-1]
top3_test_labels = le_fert.inverse_transform(top3_test_preds.ravel()).reshape(top3_test_preds.shape)

# 7. Submission format
submission = pd.DataFrame({
    "id": test["id"],
    "Fertilizer Name": [' '.join(row) for row in top3_test_labels]
})

# 8. Save as CSV 
submission.to_csv("submission.csv", index=False)
print("âœ… submission.csv successfully created.")


submission.head()

