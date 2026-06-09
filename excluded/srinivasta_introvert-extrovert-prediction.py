import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import KNNImputer
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

# === Load Provided Data Only ===
train = pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv")

# === Label encode target ===
target_le = LabelEncoder()
train['Personality'] = target_le.fit_transform(train['Personality'])

# === Combine for unified preprocessing ===
test['Personality'] = -1  # Placeholder for consistent processing
df_all = pd.concat([train, test], ignore_index=True)

# === Define column types ===
num_cols = ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside',
            'Friends_circle_size', 'Post_frequency']
cat_cols = ['Stage_fear', 'Drained_after_socializing']

# === Encode categorical (binary: Yes/No -> 1/0) ===
for col in cat_cols:
    df_all[col] = df_all[col].map({'Yes': 1, 'No': 0})

# === KNN Imputation ===
imputer = KNNImputer(n_neighbors=5)
df_all[num_cols + cat_cols] = imputer.fit_transform(df_all[num_cols + cat_cols])

# === Feature Engineering ===
df_all['Social_activity_level'] = df_all['Social_event_attendance'] + df_all['Going_outside']
df_all['Alone_ratio'] = df_all['Time_spent_Alone'] / (df_all['Social_event_attendance'] + 1)
df_all['Post_friend_ratio'] = df_all['Post_frequency'] / (df_all['Friends_circle_size'] + 1)
df_all['Is_socially_active'] = (
    (df_all['Social_event_attendance'] > df_all['Social_event_attendance'].median()) &
    (df_all['Going_outside'] > df_all['Going_outside'].median()) &
    (df_all['Post_frequency'] > df_all['Post_frequency'].median())
).astype(int)
df_all['Fear_and_drain'] = df_all['Stage_fear'] * df_all['Drained_after_socializing']

# === Split preprocessed data ===
train_clean = df_all[df_all['Personality'] != -1].copy()
test_clean = df_all[df_all['Personality'] == -1].copy()

X_train = train_clean.drop(columns=['id', 'Personality'])
y_train = train_clean['Personality'].astype(int)
X_test = test_clean.drop(columns=['id', 'Personality'])

# === Feature Scaling ===
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# === Define Base Models ===
lgb = LGBMClassifier(n_estimators=500, learning_rate=0.05, class_weight='balanced', random_state=42)
xgb = XGBClassifier(n_estimators=500, learning_rate=0.05, use_label_encoder=False, eval_metric='logloss', random_state=42)
cat = CatBoostClassifier(n_estimators=500, learning_rate=0.05, verbose=0, random_state=42)

# === Stacking Ensemble ===
stack_model = StackingClassifier(
    estimators=[('lgb', lgb), ('xgb', xgb), ('cat', cat)],
    final_estimator=LogisticRegression(),
    cv=5,
    passthrough=True
)

# === Cross-validation Evaluation ===
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = []
test_preds = np.zeros(len(X_test_scaled))

for train_idx, val_idx in skf.split(X_train_scaled, y_train):
    X_tr, X_val = X_train_scaled[train_idx], X_train_scaled[val_idx]
    y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
    
    stack_model.fit(X_tr, y_tr)
    val_preds = stack_model.predict(X_val)
    acc = accuracy_score(y_val, val_preds)
    cv_scores.append(acc)
    
    test_preds += stack_model.predict_proba(X_test_scaled)[:, 1]

print(f"âœ… Cross-Validated Accuracy: {np.mean(cv_scores):.4f}")

# === Final Prediction ===
final_preds = (test_preds / skf.n_splits > 0.5).astype(int)
final_labels = target_le.inverse_transform(final_preds)

# === Save Submission ===
submission = pd.DataFrame({'id': test_clean['id'].astype(int), 'Personality': final_labels})
submission.to_csv("submission.csv", index=False)
print("ğŸ“� Submission saved as 'submission.csv'")


