import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report


!pip install --upgrade torch torchvision torchaudio
!pip install git+https://github.com/dreamquark-ai/tabnet.git


# Load Kaggle datasets
train_data = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv')
test_data = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv')

# Drop unnecessary columns
train_data.drop(columns=['id'], inplace=True)
test_ids = test_data['id']
test_data.drop(columns=['id'], inplace=True)


train_data.head()


train_data.fillna(train_data.median(), inplace=True)
test_data.fillna(test_data.median(), inplace=True)


categorical_cols = train_data.select_dtypes(include=['object']).columns
label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    train_data[col] = le.fit_transform(train_data[col])
    test_data[col] = le.transform(test_data[col])
    label_encoders[col] = le


# Split features and target
X = train_data.drop(columns=['rainfall'])  
y = train_data['rainfall']  

# Train-test split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)



# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(test_data)

# Model Training - XGBoost and Random Forest
xgb_model = XGBClassifier(n_estimators=300, max_depth=10, learning_rate=0.05, random_state=42)
rf_model = RandomForestClassifier(n_estimators=300, max_depth=15, random_state=42)


xgb_model.fit(X_train, y_train)
rf_model.fit(X_train, y_train)

xgb_pred = xgb_model.predict(X_val)
rf_pred = rf_model.predict(X_val)

# Accuracy Scores
print("XGBoost Accuracy:", accuracy_score(y_val, xgb_pred))
print("Random Forest Accuracy:", accuracy_score(y_val, rf_pred))


# Stacking Classifier
stacking_model = StackingClassifier(
    estimators=[('xgb', xgb_model), ('rf', rf_model)],
    final_estimator=GradientBoostingClassifier(n_estimators=100)
)

stacking_model.fit(X_train, y_train)
stack_pred = stacking_model.predict(X_val)
print("Stacking Model Accuracy:", accuracy_score(y_val, stack_pred))


# Final Predictions
final_pred = stacking_model.predict(X_test)

# Create submission file
submission = pd.DataFrame({'id': test_ids, 'target': final_pred})
submission.to_csv('submission_revised.csv', index=False)

print("Submission file created successfully!")


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score
import lightgbm as lgb
import optuna

# Load Kaggle datasets
train_data = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv')
test_data = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv')

# Drop unnecessary columns
train_data.drop(columns=['id'], inplace=True)
test_ids = test_data['id']
test_data.drop(columns=['id'], inplace=True)

# Handle missing values
train_data.fillna(train_data.median(), inplace=True)
test_data.fillna(test_data.median(), inplace=True)

# Encode categorical variables
categorical_cols = train_data.select_dtypes(include=['object']).columns
label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    train_data[col] = le.fit_transform(train_data[col])
    test_data[col] = le.transform(test_data[col])
    label_encoders[col] = le

# Split features and target
X = train_data.drop(columns=['rainfall'])
y = train_data['rainfall']

# Train-test split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(test_data)

# LightGBM Hyperparameter Optimization with Optuna
def objective(trial):
    params = {
        'objective': 'binary',
        'metric': 'accuracy',
        'boosting_type': 'gbdt',
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'num_leaves': trial.suggest_int('num_leaves', 20, 200),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 50),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0)
    }
    
    model = lgb.LGBMClassifier(**params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], eval_metric='accuracy')
    preds = model.predict(X_val)
    return accuracy_score(y_val, preds)

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=30)

# Train the best LightGBM model
best_params = study.best_params
lgb_model = lgb.LGBMClassifier(**best_params)
lgb_model.fit(X_train, y_train)

# Validate model
lgb_pred = lgb_model.predict(X_val)
print("Optimized LightGBM Accuracy:", accuracy_score(y_val, lgb_pred))

# Make final predictions
final_pred = lgb_model.predict(X_test)

# Create submission file
submission = pd.DataFrame({'id': test_ids, 'target': final_pred})
submission.to_csv('submissionN.csv', index=False)

print("Submission file created successfully!")



import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score
from catboost import CatBoostClassifier
import optuna

# Load Kaggle datasets
train_data = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv')
test_data = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv')

# Drop unnecessary columns
train_data.drop(columns=['id'], inplace=True)
test_ids = test_data['id']
test_data.drop(columns=['id'], inplace=True)

# Handle missing values
train_data.fillna(train_data.median(), inplace=True)
test_data.fillna(test_data.median(), inplace=True)

# Encode categorical variables
categorical_cols = train_data.select_dtypes(include=['object']).columns.tolist()
label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    train_data[col] = le.fit_transform(train_data[col])
    test_data[col] = le.transform(test_data[col])
    label_encoders[col] = le

# Split features and target
X = train_data.drop(columns=['rainfall'])
y = train_data['rainfall']

# Train-test split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling (CatBoost doesn't need scaling but helps if mixed numeric values)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(test_data)

# Hyperparameter tuning with Optuna
def objective(trial):
    params = {
        'iterations': trial.suggest_int('iterations', 500, 2000),
        'depth': trial.suggest_int('depth', 4, 12),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-5, 100),
        'border_count': trial.suggest_int('border_count', 32, 255),
        'loss_function': 'Logloss',
        'eval_metric': 'Accuracy',
        'random_state': 42,
        'verbose': 0
    }

    model = CatBoostClassifier(**params)
    model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=50, verbose=False)
    preds = model.predict(X_val)
    return accuracy_score(y_val, preds)

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=30)

# Train the best CatBoost model
best_params = study.best_params
cat_model = CatBoostClassifier(**best_params)
cat_model.fit(X_train, y_train)

# Validate model
cat_pred = cat_model.predict(X_val)
print("Optimized CatBoost Accuracy:", accuracy_score(y_val, cat_pred))

# Make final predictions
final_pred = cat_model.predict(X_test)

# Create submission file
submission = pd.DataFrame({'id': test_ids, 'target': final_pred})
submission.to_csv('submission_revised2.csv', index=False)

print("Submission file created successfully!")



import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score
from pytorch_tabnet.tab_model import TabNetClassifier

# Load Kaggle datasets
train_data = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv')
test_data = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv')

# Drop unnecessary columns
train_data.drop(columns=['id'], inplace=True)
test_ids = test_data['id']
test_data.drop(columns=['id'], inplace=True)

# Handle missing values
train_data.fillna(train_data.median(), inplace=True)
test_data.fillna(test_data.median(), inplace=True)

# Encode categorical variables
categorical_cols = train_data.select_dtypes(include=['object']).columns.tolist()
label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    train_data[col] = le.fit_transform(train_data[col])
    test_data[col] = le.transform(test_data[col])
    label_encoders[col] = le

# Split features and target
X = train_data.drop(columns=['rainfall'])
y = train_data['rainfall']

# Train-test split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(test_data)

# Train TabNet Model
tabnet_model = TabNetClassifier(optimizer_params=dict(lr=2e-2), verbose=10, seed=42)

tabnet_model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    patience=20,
    max_epochs=200
)

# Validate model
tabnet_pred = tabnet_model.predict(X_val)
print("TabNet Accuracy:", accuracy_score(y_val, tabnet_pred))

# Make final predictions
final_pred = tabnet_model.predict(X_test)

# Create submission file
submission = pd.DataFrame({'id': test_ids, 'target': final_pred})
submission.to_csv('submission_tabnet.csv', index=False)

print("Submission file created successfully!")



pip install pytorch-tabnet



import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, PowerTransformer
from sklearn.metrics import accuracy_score
from pytorch_tabnet.tab_model import TabNetClassifier
import optuna
import torch

# Load Kaggle datasets
train_data = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv')
test_data = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv')

# Drop unnecessary columns
train_data.drop(columns=['id'], inplace=True)
test_ids = test_data['id']
test_data.drop(columns=['id'], inplace=True)

# Handle missing values
for col in train_data.columns:
    if train_data[col].isnull().sum() > 0:
        if train_data[col].dtype == "object":
            train_data[col].fillna(train_data[col].mode()[0], inplace=True)
            test_data[col].fillna(test_data[col].mode()[0], inplace=True)
        else:
            train_data[col].fillna(train_data[col].median(), inplace=True)
            test_data[col].fillna(test_data[col].median(), inplace=True)

# Encode categorical variables
categorical_cols = train_data.select_dtypes(include=['object']).columns.tolist()
label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    train_data[col] = le.fit_transform(train_data[col])
    test_data[col] = le.transform(test_data[col])
    label_encoders[col] = le

# Apply Power Transformation for normalizing features
scaler = PowerTransformer()
num_cols = train_data.select_dtypes(include=['int64', 'float64']).columns
train_data[num_cols] = scaler.fit_transform(train_data[num_cols])
test_data[num_cols] = scaler.transform(test_data[num_cols])

# Split features and target
X = train_data.drop(columns=['rainfall'])
y = train_data['rainfall']

# Train-test split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert data to tensors for TabNet
X_train = np.array(X_train).astype(np.float32)
X_val = np.array(X_val).astype(np.float32)
X_test = np.array(test_data).astype(np.float32)
y_train = np.array(y_train).astype(np.int64)
y_val = np.array(y_val).astype(np.int64)

# Hyperparameter tuning with Optuna
def objective(trial):
    params = {
        "n_d": trial.suggest_int("n_d", 8, 64),
        "n_a": trial.suggest_int("n_a", 8, 64),
        "n_steps": trial.suggest_int("n_steps", 3, 10),
        "gamma": trial.suggest_float("gamma", 1.0, 2.0),
        "lambda_sparse": trial.suggest_float("lambda_sparse", 0, 1e-3),
        "optimizer_params": dict(lr=trial.suggest_loguniform("lr", 1e-4, 2e-2)),
        "mask_type": "entmax",
        "seed": 42,
        "verbose": 0
    }

    model = TabNetClassifier(**params, device_name='cuda' if torch.cuda.is_available() else 'cpu')
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        patience=20,
        max_epochs=200,
        eval_metric=['accuracy'],
        batch_size=512
    )

    preds = model.predict(X_val)
    return accuracy_score(y_val, preds)

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=30)

# Train the best TabNet model
best_params = study.best_params
tabnet_model = TabNetClassifier(**best_params, device_name='cuda' if torch.cuda.is_available() else 'cpu')

tabnet_model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    patience=20,
    max_epochs=200,
    eval_metric=['accuracy'],
    batch_size=512
)

# Validate model
tabnet_pred = tabnet_model.predict(X_val)
print("Optimized TabNet Accuracy:", accuracy_score(y_val, tabnet_pred))

# Make final predictions
final_pred = tabnet_model.predict(X_test)

# Create submission file
submission = pd.DataFrame({'id': test_ids, 'target': final_pred})
submission.to_csv('submission_tabnet_optimized.csv', index=False)

print("Submission file created successfully!")





