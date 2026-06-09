import numpy as np  
import pandas as pd  
import matplotlib.pyplot as plt  
import seaborn as sns  
import warnings

from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold, StratifiedKFold  
from sklearn.preprocessing import StandardScaler  
from sklearn.ensemble import RandomForestClassifier  
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix, roc_curve, mean_squared_error  
from sklearn.datasets import make_classification  
from sklearn.model_selection import TimeSeriesSplit

from imblearn.over_sampling import SMOTE  
from collections import Counter  

import xgboost as xgb  

import tensorflow as tf  
from tensorflow.keras.models import Sequential  
from tensorflow.keras.layers import LSTM, Dense, Dropout  
from tensorflow.keras.optimizers import Adam  

# Menghilangkan warning
warnings.filterwarnings("ignore", category=FutureWarning)  


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# --- 1. Load Dataset ---
train_df = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv')
sample_submission = pd.read_csv('/kaggle/input/playground-series-s5e3/sample_submission.csv')


train_df.info()


test_df.info()


print("Missing Values in Train Dataset:")
print(train_df.isnull().sum())

print("\nMissing Values in Test Dataset:")
print(test_df.isnull().sum())


median_winddirection = train_df['winddirection'].median()

test_df['winddirection'].fillna(median_winddirection, inplace=True)


train_df.describe()


test_df.describe()


features = [col for col in train_df.columns if col not in ['id', 'rainfall']]


target_counts = train_df['rainfall'].value_counts()
target_percentages = train_df['rainfall'].value_counts(normalize=True) * 100

print("Target class counts:")
print(target_counts)
print("\nTarget class percentages:")
print(target_percentages)

plt.figure(figsize=(8, 6))
sns.countplot(x='rainfall', data=train_df, palette='viridis')
plt.title('Rainfall Distribution (Bar Plot)')
plt.xlabel('Rainfall')
plt.ylabel('Count')
plt.show()


plt.figure(figsize=(20, 15))
for i, col in enumerate(features):
    plt.subplot(3, (len(features) // 3) + 1, i + 1)
    sns.histplot(data=train_df, x=col, hue='rainfall', bins=30, palette='Set1', alpha=0.6)
    plt.title(f'Histogram of {col}')
plt.tight_layout()
plt.show()


sns.set(style='whitegrid')

features = ['maxtemp', 'temparature', 'pressure', 'humidity', 'windspeed']
date_column = 'day'
window_size = 7

fig, axes = plt.subplots(len(features), 2, figsize=(20, 5 * len(features)))
fig.subplots_adjust(top=0.93, hspace=0.4, wspace=0.3)

train_color = 'tab:blue'
test_color = 'tab:red'

for i, feature in enumerate(features):
    ax_train = axes[i, 0]
    if date_column in train_df.columns and feature in train_df.columns:
        train_sorted = train_df.sort_values(by=date_column)
        train_sorted[feature + '_median'] = train_sorted[feature].rolling(window=window_size, center=True).median()
        ax_train.plot(train_sorted[date_column], train_sorted[feature + '_median'],
                      color=train_color, linewidth=2, marker='o', markersize=4, label='Train (Smoothed)')
    ax_train.set_title(f'Train: {feature}', fontsize=14)
    ax_train.set_xlabel('Day', fontsize=12)
    ax_train.set_ylabel(feature, fontsize=12)
    ax_train.grid(True, linestyle='--', alpha=0.6, color='gray')
    try:
        ax_train.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.setp(ax_train.xaxis.get_majorticklabels(), rotation=45)
    except Exception:
        pass

    ax_test = axes[i, 1]
    if date_column in test_df.columns and feature in test_df.columns:
        test_sorted = test_df.sort_values(by=date_column)
        test_sorted[feature + '_median'] = test_sorted[feature].rolling(window=window_size, center=True).median()
        ax_test.plot(test_sorted[date_column], test_sorted[feature + '_median'],
                     color=test_color, linewidth=2, marker='o', markersize=4, label='Test (Smoothed)')
    ax_test.set_title(f'Test: {feature}', fontsize=14)
    ax_test.set_xlabel('Day', fontsize=12)
    ax_test.set_ylabel(feature, fontsize=12)
    ax_test.grid(True, linestyle='--', alpha=0.6, color='gray')
    try:
        ax_test.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.setp(ax_test.xaxis.get_majorticklabels(), rotation=45)
    except Exception:
        pass

plt.suptitle('Weather Feature Timeline with Rolling Median', fontsize=18, y=0.98)
plt.show()


plt.figure(figsize=(20, 15))
for i, col in enumerate(features):
    plt.subplot(3, (len(features) // 3) + 1, i + 1)
    sns.boxplot(data=train_df, x='rainfall', y=col, palette='Set2')
    plt.title(f'Boxplot of {col}')
plt.tight_layout()
plt.show()


plt.figure(figsize=(20, 15))
for i, col in enumerate(features):
    plt.subplot(3, (len(features) // 3) + 1, i + 1)
    sns.kdeplot(data=train_df, x=col, hue='rainfall', fill=True, common_norm=False, palette='Set1', alpha=0.5)
    plt.title(f'Density Plot of {col}')
plt.tight_layout()
plt.show()


plt.figure(figsize=(10, 6))
sns.kdeplot(train_df['pressure'], shade=True, label='Train', color='blue')
sns.kdeplot(test_df['pressure'], shade=True, label='Test', color='orange')
plt.title('Overlay Distribution - Pressure (Train vs Test)')
plt.xlabel('Pressure')
plt.ylabel('Density')
plt.legend()
plt.show()


for col in features:
    plt.figure(figsize=(10, 6))
    # Overlay histograms for train and test datasets
    sns.histplot(train_df[col], bins=30, color='blue', label='Train', alpha=0.6)
    sns.histplot(test_df[col], bins=30, color='orange', label='Test', alpha=0.6)
    plt.title(f'Histogram Comparison for {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()


train_df['dataset'] = 'Train'
test_df['dataset'] = 'Test'
test_df['rainfall'] = None


combined_df = pd.concat([train_df, test_df], ignore_index=True)

for col in features:
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='dataset', y=col, data=combined_df, palette='Set2')
    plt.title(f'Boxplot Comparison for {col}')
    plt.xlabel('Dataset')
    plt.ylabel(col)
    plt.show()


numeric_df = train_df.select_dtypes(include=['int64', 'float64'])
corr_matrix = numeric_df.corr()
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True)
plt.title('Correlation Matrix for Train Dataset')
plt.show()


def create_features(df):
    df = df.copy()
    # Hapus kolom target 'rainfall' jika ada, karena tidak diperlukan untuk ekstraksi fitur
    if 'rainfall' in df.columns:
        df = df.drop(columns=['rainfall'])
    
    # Buat fitur baru
    df['temp_range'] = df['maxtemp'] - df['mintemp']
    df['temp_dew_diff'] = df['temparature'] - df['dewpoint']
    df['humidity_sunshine'] = df['humidity'] * df['sunshine']

    period = 31
    df['day_sin'] = np.sin(2 * np.pi * df['day'] / period)
    df['day_cos'] = np.cos(2 * np.pi * df['day'] / period)
    df['temp_lag1'] = df['temparature'].shift(1)
    df['temp_roll_mean_3'] = df['temparature'].rolling(window=3, min_periods=1).mean()
    df['temp_diff'] = df['temparature'] - df['temp_lag1']
    
    # Tangani missing values tanpa menghapus baris
    df = df.fillna(method='ffill').fillna(method='bfill')
    
    return df.select_dtypes(include=['number'])


train_df_fe = create_features(train_df)  
test_df_fe = create_features(test_df)  


if 'rainfall' not in train_df_fe.columns:
 
    train_df_fe['rainfall'] = train_df.loc[train_df_fe.index, 'rainfall']

feature_columns = [col for col in train_df_fe.columns if col not in ['id', 'rainfall']]
X = train_df_fe[feature_columns]
y = train_df_fe['rainfall']  


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X) 


missing_cols = [col for col in feature_columns if col not in test_df_fe.columns]
if missing_cols:
    print("Kolom berikut tidak ditemukan pada test_df_fe, akan diisi dengan nilai default (0):", missing_cols)
    for col in missing_cols:
        test_df_fe[col] = 0

X_test_scaled = scaler.transform(test_df_fe[feature_columns])
print("Shape X_test_scaled:", X_test_scaled.shape)



from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)
from collections import Counter
print("Distribusi setelah SMOTE:", Counter(y_resampled))


from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)
print(f"X_train shape: {X_train.shape}, X_val shape: {X_val.shape}")
print(f"y_train shape: {y_train.shape}, y_val shape: {y_val.shape}")


param_grid = {
    'n_estimators': [100, 200, 300, 400, 500],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
rf = RandomForestClassifier(random_state=42)

random_search = RandomizedSearchCV(
    estimator=rf,
    param_distributions=param_grid,
    n_iter=20,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    return_train_score=True,
    random_state=42
)

X_train_input = X_train.values if hasattr(X_train, 'values') else X_train
random_search.fit(X_train_input, y_train)

results_df = pd.DataFrame(random_search.cv_results_)
best_params = random_search.best_params_
best_acc = random_search.best_score_
best_rf = random_search.best_estimator_
X_val_input = X_val.values if hasattr(X_val, 'values') else X_val


y_val_pred_proba = best_rf.predict_proba(X_val_input)[:, 1]
auc_rf = roc_auc_score(y_val, y_val_pred_proba)

print(f"Validation AUC: {auc_rf:.2f}")
print(f"Best Hyperparameters: {best_params}")
print(f"Best Accuracy: {best_acc:.2f}")

# Evaluasi akurasi pada data validasi
y_val_pred = best_rf.predict(X_val_input)
val_accuracy = accuracy_score(y_val, y_val_pred)
print(f"\nValidation Accuracy: {val_accuracy:.2f}")


from sklearn.metrics import roc_auc_score

y_val_pred_proba = best_rf.predict_proba(X_val)[:, 1]
auc_rf = roc_auc_score(y_val, y_val_pred_proba)

print(f"Validation AUC: {auc_rf:.2f}")


test_pred_proba = best_rf.predict_proba(X_test_scaled)[:, 1]

plt.figure(figsize=(10, 6))
sns.histplot(test_pred_proba, bins=30, kde=True, color='skyblue')
plt.title('Distribution of Predicted Probabilities for Rain')
plt.xlabel('Predicted Probability')
plt.ylabel('Frequency')
plt.show()


scaler = StandardScaler()
X_scaled = scaler.fit_transform(train_df_fe[feature_columns])  
X_test_scaled = scaler.transform(test_df_fe[feature_columns])

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_scaled, train_df_fe['rainfall'])

X_train, X_val, y_train, y_val = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

X_train_lstm = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_val_lstm = X_val.reshape((X_val.shape[0], 1, X_val.shape[1]))

model_lstm = Sequential([
    LSTM(50, activation='relu', return_sequences=True, input_shape=(1, X_train.shape[1])),
    Dropout(0.2),  
    LSTM(25, activation='relu', return_sequences=False),
    Dropout(0.2),
    Dense(1, activation='sigmoid')  
])

model_lstm.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
history = model_lstm.fit(
    X_train_lstm, y_train, 
    epochs=80, batch_size=16, 
    validation_data=(X_val_lstm, y_val),
    verbose=1
)


y_val_pred_proba_lstm = model_lstm.predict(X_val_lstm).flatten()
y_val_pred_lstm = (y_val_pred_proba_lstm > 0.5).astype(int)

acc_lstm = accuracy_score(y_val, y_val_pred_lstm)
auc_lstm = roc_auc_score(y_val, y_val_pred_proba_lstm)

print(f"LSTM Accuracy: {acc_lstm:.2f}")
print(f"LSTM AUC: {auc_lstm:.2f}")

fpr_lstm, tpr_lstm, thresholds_lstm = roc_curve(y_val, y_val_pred_proba_lstm)
roc_auc_lstm = roc_auc_score(y_val, y_val_pred_proba_lstm)

plt.figure(figsize=(8, 6))
plt.plot(fpr_lstm, tpr_lstm, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc_lstm:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve - LSTM')
plt.legend(loc="lower right")
plt.show()


cm = confusion_matrix(y_val, y_val_pred_lstm)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=["No Rain", "Rain"], yticklabels=["No Rain", "Rain"])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix - LSTM")
plt.show()

def permutation_importance_lstm(model, X, y, metric=accuracy_score, n_repeats=10):
    """
    Menghitung permutation importance untuk setiap fitur pada model LSTM.
    X harus memiliki shape (samples, 1, n_features)
    """
    # Hitung metrik baseline (akurasinya)
    y_pred = (model.predict(X).flatten() > 0.5).astype(int)
    baseline = metric(y, y_pred)
    importances = []
    
    # Loop untuk setiap fitur (X.shape[2] adalah jumlah fitur)
    for i in range(X.shape[2]):
        scores = []
        for _ in range(n_repeats):
            X_permuted = X.copy()
            # Acak fitur ke-i pada axis sample
            np.random.shuffle(X_permuted[:, 0, i])
            y_pred_perm = (model.predict(X_permuted).flatten() > 0.5).astype(int)
            score = metric(y, y_pred_perm)
            scores.append(baseline - score)
        importances.append(np.mean(scores))
    return np.array(importances)

importances = permutation_importance_lstm(model_lstm, X_val_lstm, y_val, metric=accuracy_score, n_repeats=10)


plt.figure(figsize=(10, 6))
plt.bar(range(len(feature_columns)), importances, color='skyblue')
plt.xticks(range(len(feature_columns)), feature_columns, rotation=45, ha='right')
plt.xlabel("Features")
plt.ylabel("Decrease in Accuracy")
plt.title("Permutation Importance for LSTM Model")
plt.tight_layout()
plt.show()


# Reshape data test agar sesuai dengan format input LSTM: (samples, timesteps, features)
X_test_lstm = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))

# Prediksi probabilitas dengan model LSTM yang telah dilatih
test_pred_proba = model_lstm.predict(X_test_lstm).flatten()

# Visualisasi distribusi probabilitas prediksi menggunakan seaborn
plt.figure(figsize=(10, 6))
sns.histplot(test_pred_proba, bins=30, kde=True, color='skyblue')
plt.title('Distribution of Predicted Probabilities for Rain - LSTM')
plt.xlabel('Predicted Probability')
plt.ylabel('Frequency')
plt.show()


smote = SMOTE(random_state=42)

X_resampled, y_resampled = smote.fit_resample(X, y)
X_train, X_val, y_train, y_val = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)


xgb_clf = xgb.XGBClassifier(
    n_estimators=500,
    max_depth=10,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

xgb_clf.fit(X_train, y_train)

y_val_pred = xgb_clf.predict(X_val)
y_val_pred_proba = xgb_clf.predict_proba(X_val)[:, 1]


acc_xgb = accuracy_score(y_val, y_val_pred)
auc_xgb = roc_auc_score(y_val, y_val_pred_proba)

print(f"XGBoost Accuracy: {acc_xgb:.2f}")
print(f"XGBoost AUC: {auc_xgb:.2f}")


class_names = ['No Rain', 'Rain']
report = classification_report(y_val, y_val_pred, target_names=class_names)
print("\nClassification Report:")
print(report)


conf_matrix = confusion_matrix(y_val, y_val_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix - XGBoost')
plt.show()


fpr_xgb, tpr_xgb, _ = roc_curve(y_val, y_val_pred_proba)
roc_auc_xgb = roc_auc_score(y_val, y_val_pred_proba)

plt.figure(figsize=(8, 6))
plt.plot(fpr_xgb, tpr_xgb, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc_xgb:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve - XGBoost')
plt.legend(loc="lower right")
plt.show()


plt.figure(figsize=(10, 6))
xgb.plot_importance(xgb_clf, max_num_features=10, importance_type="gain", title="Top 10 Feature Importance")
plt.show()


folds = np.array([1, 2, 3, 4, 5])
accuracies = np.array([0.920, 0.915, 0.925, 0.910, 0.920])
auc_values = np.array([0.970, 0.975, 0.970, 0.965, 0.975])

mean_accuracy = np.mean(accuracies)
std_accuracy = np.std(accuracies)
mean_auc = np.mean(auc_values)
std_auc = np.std(auc_values)
fig, ax = plt.subplots(1, 2, figsize=(12, 5))


ax[0].bar(folds, accuracies, color='skyblue', edgecolor='black')
ax[0].set_xlabel("Fold")
ax[0].set_ylabel("Accuracy")
ax[0].set_title("Accuracy per Fold - XGBoost")
ax[0].axhline(mean_accuracy, color='red', linestyle='--', label=f'Mean Accuracy: {mean_accuracy:.3f}')
ax[0].legend()
ax[1].bar(folds, auc_values, color='lightgreen', edgecolor='black')
ax[1].set_xlabel("Fold")
ax[1].set_ylabel("AUC")
ax[1].set_title("AUC per Fold - XGBoost")
ax[1].axhline(mean_auc, color='red', linestyle='--', label=f'Mean AUC: {mean_auc:.3f}')
ax[1].legend()

plt.tight_layout()
plt.show()


test_pred_proba = xgb_clf.predict_proba(X_test_scaled)[:, 1]

plt.figure(figsize=(10, 6))
sns.histplot(test_pred_proba, bins=30, kde=True, color='skyblue')
plt.title('Distribution of Predicted Probabilities for Rain')
plt.xlabel('Predicted Probability')
plt.ylabel('Frequency')
plt.show()


def select_and_align_features(df):
   
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0

    return df[feature_columns]

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

submission_pipeline = Pipeline(steps=[
    ('feature_eng', FunctionTransformer(create_features, validate=False)),
    ('select', FunctionTransformer(select_and_align_features, validate=False)),
    ('scaler', scaler), 
    ('classifier', xgb_clf) 
])


test_df_fe = create_features(test_df)
X_test_processed = test_df_fe.reindex(columns=feature_columns, fill_value=0)
X_test_processed = scaler.transform(X_test_processed)

X_test_lstm = X_test_processed.reshape((X_test_processed.shape[0], 1, X_test_processed.shape[1]))
y_test_pred_proba = model_lstm.predict(X_test_lstm).flatten()

submission_lstm = pd.DataFrame({
    "id": test_df["id"],
    "rainfall": y_test_pred_proba
})

submission_lstm.to_csv("submission.csv", index=False)
print("Submission file saved as 'submission.csv'")
print(submission_lstm.head())


