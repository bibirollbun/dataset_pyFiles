import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score

# Load and preprocess
df = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv')
df = df.dropna()  # Drop missing rows (or impute)
df = df.drop(columns=['id'])  # Drop irrelevant column




df.head()


# Feature engineering
df['temp_range'] = df['maxtemp'] - df['mintemp']
df['dewpoint_diff'] = df['temparature'] - df['dewpoint']
df['month'] = np.ceil(df['day'] / 30.5).astype(int)

# EDA (example: check rainfall distribution)
print(df['rainfall'].value_counts(normalize=True))




df.head()


import matplotlib.pyplot as plt

# Rainfall distribution
rainfall_counts = df['rainfall'].value_counts(normalize=True) * 100
print("Rainfall Distribution (%):")
print(rainfall_counts)

# Bar plot
plt.bar(['No Rain (0)', 'Rain (1)'], rainfall_counts, color=['blue', 'orange'])
plt.title('Rainfall Distribution')
plt.ylabel('Percentage (%)')
plt.show()


# Histograms
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
df['humidity'].hist(ax=axes[0], bins=20, color='green')
axes[0].set_title('Humidity Distribution')
df['cloud'].hist(ax=axes[1], bins=20, color='gray')
axes[1].set_title('Cloud Cover Distribution')
df['temparature'].hist(ax=axes[2], bins=20, color='red')
axes[2].set_title('Temperature Distribution')
plt.tight_layout()
plt.show()


# Boxplots
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
df.boxplot(column='humidity', by='rainfall', ax=axes[0])
axes[0].set_title('Humidity vs Rainfall')
df.boxplot(column='cloud', by='rainfall', ax=axes[1])
axes[1].set_title('Cloud Cover vs Rainfall')
df.boxplot(column='windspeed', by='rainfall', ax=axes[2])
axes[2].set_title('Windspeed vs Rainfall')
plt.suptitle('')  # Remove default title
plt.tight_layout()
plt.show()


import seaborn as sns

# Correlation matrix
corr = df.corr()

# Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Heatmap')
plt.show()


# Split data
X = df.drop(columns=['rainfall'])
y = df['rainfall']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)




# Base models
rf = RandomForestClassifier(n_estimators=100, max_depth=10, class_weight='balanced', random_state=42)
svm = SVC(kernel='rbf', C=1.0, class_weight='balanced', probability=True, random_state=42)

# Fit base models
rf.fit(X_train_scaled, y_train)
svm.fit(X_train_scaled, y_train)




# Stacking: predictions for meta-model
rf_preds_train = rf.predict_proba(X_train_scaled)[:, 1]
svm_preds_train = svm.predict_proba(X_train_scaled)[:, 1]
stacked_train = np.column_stack((rf_preds_train, svm_preds_train))

# Meta-model
meta_model = LogisticRegression()
meta_model.fit(stacked_train, y_train)




# Test predictions
rf_preds_test = rf.predict_proba(X_test_scaled)[:, 1]
svm_preds_test = svm.predict_proba(X_test_scaled)[:, 1]
stacked_test = np.column_stack((rf_preds_test, svm_preds_test))
final_preds = meta_model.predict(stacked_test)

# Evaluate
print("F1 Score:", f1_score(y_test, final_preds))
print("AUC-ROC:", roc_auc_score(y_test, meta_model.predict_proba(stacked_test)[:, 1]))


# Load and preprocess
df_test = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv')
test_ids = df_test['id']

df_test = df_test.drop(columns=['id'])  # Drop irrelevant column


# Feature engineering
df_test['temp_range'] = df_test['maxtemp'] - df_test['mintemp']
df_test['dewpoint_diff'] = df_test['temparature'] - df_test['dewpoint']
df_test['month'] = np.ceil(df_test['day'] / 30.5).astype(int)








# Prepare test data
X_test_new = df_test
X_test_new = X_test_new.fillna(method='ffill')
X_test_new_scaled = scaler.transform(X_test_new)

# Get predictions from base models
rf_preds_test_new = rf.predict_proba(X_test_new_scaled)[:, 1]
svm_preds_test_new = svm.predict_proba(X_test_new_scaled)[:, 1]

# Stack predictions
stacked_test_new = np.column_stack((rf_preds_test_new, svm_preds_test_new))




# Predict rainfall using meta-model

final_preds_proba_new = meta_model.predict_proba(stacked_test_new)[:, 1]
final_preds_proba_new = np.round(final_preds_proba_new, 4)

# Create a DataFrame with predictions
df_submission = pd.DataFrame({'id': test_ids, 'rainfall': final_preds_proba_new})

# Save to CSV
df_submission.to_csv('submission1.csv', index=False)

