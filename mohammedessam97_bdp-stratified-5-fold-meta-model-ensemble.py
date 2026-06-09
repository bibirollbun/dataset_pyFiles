!pip install imbalanced-learn


import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


train = pd.read_csv('/kaggle/input/playground-series-s5e8/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e8/test.csv')


train.head()


# Target distribution
print("\nTarget distribution:")
print(train['y'].value_counts(normalize=True))

# Nulls check
print("\nMissing values:")
print(train.isnull().sum())


# Categorical columns
cat_cols = train.select_dtypes(include='object').columns.tolist()
print("\nCategorical columns:", cat_cols)

# Numerical columns
num_cols = train.select_dtypes(include='number').drop(['id', 'y'], axis=1).columns.tolist()
print("\nNumerical columns:", num_cols)

# Quick look at categorical unique values
for col in cat_cols:
    print(f"\n{col}: {train[col].nunique()} unique values")


from sklearn.preprocessing import LabelEncoder

encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    train[col] = le.fit_transform(train[col])
    test[col] = le.transform(test[col])
    encoders[col] = le


from sklearn.model_selection import StratifiedKFold

X = train.drop(['id', 'y'], axis=1)
y = train['y']
X_test = test.drop('id', axis=1)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


classes = np.array([0,1])
weights = compute_class_weight('balanced',classes=classes,y=train['y'])
class_weight_dict = dict(zip(classes,weights))

# for XGBoost: scale_pos_weight = negative/positive
scale_pos = class_weight_dict[0] / class_weight_dict[1]
print("Class Weights: ",class_weight_dict)
print("XGBoost scale_pos_weight: ",scale_pos)

# Base models with class imbalance handling
model_dict = {
    'catboost': CatBoostClassifier(verbose=0, random_state=42,
                                   class_weights=[class_weight_dict[0], class_weight_dict[1]]),
    'xgboost': XGBClassifier(use_label_encoder=False, eval_metric='logloss',
                             scale_pos_weight=scale_pos, random_state=42),
    'lightgbm': LGBMClassifier(random_state=42, class_weight='balanced'),
    'randomforest': RandomForestClassifier(n_estimators=100,
                                           random_state=42, class_weight='balanced'),
    'logistic': LogisticRegression(max_iter=1000,
                                   random_state=42, class_weight='balanced')
}


# Storage
oof_preds = {name: np.zeros(len(train)) for name in model_dict}
test_preds = {name: np.zeros(len(test)) for name in model_dict}

# Choose whether to apply SMOTE
use_smote = False
sm = SMOTE(random_state=42)

for model_name, model in model_dict.items():
    print(f"\nTraining model: {model_name}")
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        if use_smote:
            X_train,y_train = sm.fit_resample(X_train,y_train)
       
        clf = model.fit(X_train, y_train)
        oof_preds[model_name][val_idx] = clf.predict_proba(X_val)[:, 1]
        test_preds[model_name] += clf.predict_proba(X_test)[:, 1] / skf.n_splits

        print(f"  Fold {fold+1} AUC:", roc_auc_score(y_val, oof_preds[model_name][val_idx]))


# Create meta dataset
oof_stack = pd.DataFrame(oof_preds)
test_stack = pd.DataFrame(test_preds)

# Train meta-model
meta_model = LogisticRegression(max_iter=1000, 
            class_weight='balanced', random_state=42)

meta_model.fit(oof_stack, y)
final_preds = meta_model.predict_proba(test_stack)[:, 1]



final_preds


submission = pd.DataFrame({
    "id": test['id'],
    "y": final_preds
})
submission.to_csv("submission.csv", index=False)




