# !pip install --update scikit-learn
# !pip install --update imbalanced-learn
# !pip install category_encoders
!pip uninstall scikit-learn imbalanced-learn -y
!pip install scikit-learn==1.3.0 imbalanced-learn==0.9.1


import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
import category_encoders as ce
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score, roc_auc_score


SEED = 42
random.seed(SEED)
np.random.seed(SEED)
warnings.filterwarnings('ignore')
sns.set_style('whitegrid')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


df = pd.read_csv('/kaggle/input/hotel-booking-demand-3/train_final.csv')
df_test = pd.read_csv('/kaggle/input/hotel-booking-demand-3/test_final.csv')
print('Data has been loaded from the Kaggle input directory.')


plt.figure(figsize=(8, 6))
ax = sns.countplot(x='is_canceled', data=df, palette='viridis')
plt.title('Distribution of Booking Cancellation Status', fontsize=16)
plt.xlabel('Cancellation Status (0: Not Canceled, 1: Canceled)', fontsize=12)
plt.ylabel('Number of Bookings', fontsize=12)
plt.xticks([0, 1], ['Not Canceled', 'Canceled'])
total = len(df)
for p in ax.patches:
    percentage = '{:.1f}%'.format(100 * p.get_height() / total)
    x = p.get_x() + p.get_width() / 2
    y = p.get_height()
    ax.annotate(f"{p.get_height()}\n({percentage})", (x, y), ha='center', va='bottom', fontsize=9, xytext=(0, 5), textcoords='offset points')
plt.show()


train_df = df.copy()
test_df = df_test.copy()
all_df = pd.concat([train_df.drop('is_canceled', axis=1), test_df], axis=0, ignore_index=True)

all_df["hotel"] = all_df["hotel"].str.replace(" Hotel", "")
month_mapping = {"January": 1, "February": 2, "March": 3, "April": 4, "May": 5, "June": 6, "July": 7, "August": 8, "September": 9, "October": 10, "November": 11, "December": 12}
all_df['arrival_date'] = pd.to_datetime(all_df['arrival_date_year'].astype(str) + '-' + all_df['arrival_date_month'].map(month_mapping).astype(str) + '-' + all_df['arrival_date_day_of_month'].astype(str))
all_df["arrival_season"] = all_df["arrival_date"].dt.month.map({1: "winter", 2: "winter", 3: "spring", 4: "spring", 5: "spring", 6: "summer", 7: "summer", 8: "summer", 9: "autumn", 10: "autumn", 11: "autumn", 12: "winter"})
all_df["arrival_weekend"] = (all_df["arrival_date"].dt.dayofweek >= 5).astype(int)
all_df["month_num"] = all_df["arrival_date_month"].map(month_mapping)
all_df["month_sin"] = np.sin(2 * np.pi * all_df["month_num"] / 12)
all_df["month_cos"] = np.cos(2 * np.pi * all_df["month_num"] / 12)
all_df["week_sin"] = np.sin(2 * np.pi * all_df["arrival_date_week_number"] / 52)
all_df["week_cos"] = np.cos(2 * np.pi * all_df["arrival_date_week_number"] / 52)
all_df['total_stays'] = all_df['stays_in_weekend_nights'] + all_df['stays_in_week_nights']
all_df['total_guests'] = all_df['adults'] + all_df['children'] + all_df['babies']
all_df['room_changed'] = (all_df['reserved_room_type'] != all_df['assigned_room_type']).astype(int)
all_df["meal"] = all_df["meal"].replace("Undefined", all_df.meal.value_counts().index[0])
all_df.drop(columns=['reservation_status', 'reservation_status_date', 'arrival_date', 'arrival_date_year', 'arrival_date_month', 'arrival_date_day_of_month', 'month_num', 'arrival_date_week_number'], inplace=True, errors='ignore')
print("Feature engineering completed!")


X = all_df[:len(train_df)].copy()
X_test = all_df[len(train_df):].copy()
y = train_df['is_canceled']

cat_cols = [col for col in X.columns if X[col].dtype == 'object']
num_cols = [col for col in X.columns if X[col].dtype != 'object']

high_card_cols = ['country', 'assigned_room_type', 'reserved_room_type']
target_encoder = ce.TargetEncoder(cols=high_card_cols, smoothing=5)
X[high_card_cols] = target_encoder.fit_transform(X[high_card_cols], y)
X_test[high_card_cols] = target_encoder.transform(X_test[high_card_cols])

low_card_cols = [col for col in cat_cols if col not in high_card_cols]
X = pd.get_dummies(X, columns=low_card_cols, drop_first=True)
X_test = pd.get_dummies(X_test, columns=low_card_cols, drop_first=True)

train_cols = X.columns
test_cols = X_test.columns
missing_in_test = set(train_cols) - set(test_cols)
for c in missing_in_test:
    X_test[c] = 0
missing_in_train = set(test_cols) - set(train_cols)
for c in missing_in_train:
    X[c] = 0
X_test = X_test[train_cols]

scaler = RobustScaler()
X[num_cols] = scaler.fit_transform(X[num_cols])
X_test[num_cols] = scaler.transform(X_test[num_cols])

smote = SMOTE(random_state=SEED)
X_resampled, y_resampled = smote.fit_resample(X, y)
X_train, X_val, y_train, y_val = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=SEED, stratify=y_resampled)
print("Encoding, scaling, and SMOTE completed!")


base_estimators = [
    ('RandomForest', RandomForestClassifier(n_estimators=200, max_depth=15, random_state=SEED, n_jobs=-1)),
    ('XGBoost', XGBClassifier(random_state=SEED, n_estimators=1000, max_depth=7, learning_rate=0.05, colsample_bytree=0.8, subsample=0.8, eval_metric='logloss', use_label_encoder=False)),
    ('CatBoost', CatBoostClassifier(random_state=SEED, n_estimators=1500, learning_rate=0.08, max_depth=6, verbose=0)),
    ('LightGBM', LGBMClassifier(random_state=SEED, n_estimators=1200, learning_rate=0.07, num_leaves=31, max_depth=-1))
]

stacking_model = StackingClassifier(
    estimators=base_estimators, 
    final_estimator=LogisticRegression(),
    cv=5,
    n_jobs=-1
)

all_models = dict(base_estimators)
all_models['Stacking'] = stacking_model

results = {}
trained_models = {}

for name, model in all_models.items():
    print(f"--- Training: {name} ---")
    if name == 'XGBoost':
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    else:
        model.fit(X_train, y_train)
    
    trained_models[name] = model
    y_pred = model.predict(X_val)
    y_pred_proba = model.predict_proba(X_val)[:, 1]
    
    accuracy = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)
    roc_auc = roc_auc_score(y_val, y_pred_proba)
    
    results[name] = {'accuracy': accuracy, 'f1_score': f1, 'roc_auc': roc_auc}
    
    print(f"--- {name} Evaluation Report ---")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"AUC-ROC: {roc_auc:.4f}")
    
    cm = confusion_matrix(y_val, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Canceled', 'Canceled'], yticklabels=['Not Canceled', 'Canceled'])
    plt.title(f'{name} Confusion Matrix', fontsize=16)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.show()


results_df = pd.DataFrame(results).T.reset_index().rename(columns={'index': 'Model'})
print("Summary of all model performances:")
display(results_df.sort_values('roc_auc', ascending=False))

plt.figure(figsize=(12, 7))
sns.barplot(data=results_df.sort_values('roc_auc', ascending=False), x='Model', y='roc_auc', palette='viridis')
plt.title('Comparison of AUC-ROC Scores Across Models', fontsize=16)
plt.xlabel('Model', fontsize=12)
plt.ylabel('AUC-ROC Score', fontsize=12)
plt.ylim(0.85, 1.0)
plt.show()

best_model_name = results_df.sort_values('roc_auc', ascending=False)['Model'].iloc[0]
best_model = trained_models[best_model_name]
print(f"The selected best model is: {best_model_name}")


final_predictions = best_model.predict(X_test)
submission = pd.DataFrame({'index': df_test.index, 'is_canceled': final_predictions})
submission.to_csv('submission.csv', index=False)
print("Submission file 'submission.csv' has been generated!")
display(submission.head())

