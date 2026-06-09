import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Set random seed
def set_global_seed(seed=42):
    np.random.seed(seed)
    import random
    random.seed(seed)
set_global_seed()



df_train = pd.read_csv('/kaggle/input/child-mind-institute-problematic-internet-use/train.csv')
df_test = pd.read_csv('/kaggle/input/child-mind-institute-problematic-internet-use/test.csv')
df_dict = pd.read_csv('/kaggle/input/child-mind-institute-problematic-internet-use/data_dictionary.csv')

print("Train shape:", df_train.shape)
print("Test shape:", df_test.shape)


# Target Distribution
print(df_train['sii'].value_counts())
sns.countplot(x='sii', data=df_train)
plt.title('Target Variable Distribution (sii)')
plt.show()


# Correlation Heatmap
train_filled = df_train.fillna(df_train.median(numeric_only=True))
plt.figure(figsize=(12, 8))
sns.heatmap(train_filled.corr(numeric_only=True), cmap='coolwarm', center=0)
plt.title('Feature Correlation Heatmap')
plt.show()


# BMI Boxplot by sii
sns.boxplot(x='sii', y='Physical-BMI', data=df_train)
plt.title('BMI by Problematic Internet Use Level')
plt.show()


# Handle Missing Values
train_filled = df_train.fillna(df_train.median(numeric_only=True))
test_filled = df_test.fillna(df_train.median(numeric_only=True))


# Encode Categorical Columns
X = train_filled.drop(['id', 'sii'], axis=1)
y = train_filled['sii']
X_test = test_filled.drop(['id'], axis=1)

# Encode remaining season columns
season_cols = [
    'Basic_Demos-Enroll_Season', 'CGAS-Season', 'Physical-Season',
    'Fitness_Endurance-Season', 'FGC-Season', 'BIA-Season',
    'PAQ_A-Season', 'PAQ_C-Season',
    'SDS-Season', 'PreInt_EduHx-Season'
]


le = LabelEncoder()
for col in season_cols:
    if col in X.columns and col in X_test.columns:
        X[col] = le.fit_transform(X[col].astype(str))
        X_test[col] = le.transform(X_test[col].astype(str))

# Drop PCIAT-Season if it exists
X = X.drop(columns=['PCIAT-Season'], errors='ignore')

# Align and fill X_test
X_test = X_test.reindex(columns=X.columns, fill_value=np.nan)
X_test = X_test.fillna(X.median(numeric_only=True))


 # Train-Test Split & Scaling
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Initialize and train
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train_scaled, y_train)

# Predict and evaluate
y_pred = rf_model.predict(X_val_scaled)
print("Classification Report:")
print(classification_report(y_val, y_pred))

# Confusion matrix
import seaborn as sns
import matplotlib.pyplot as plt

conf_matrix = confusion_matrix(y_val, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()



importances = rf_model.feature_importances_
importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances})
top_feats = importance_df.sort_values(by='Importance', ascending=False).head(15)

sns.barplot(data=top_feats, x='Importance', y='Feature')
plt.title('Top 15 Important Features (Random Forest)')
plt.show()


cm = confusion_matrix(y_val, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Purples')
plt.title("Confusion Matrix (Validation Set)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_train_scaled)

sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y_train, palette='deep')
plt.title("PCA Visualization of Training Set")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()


# Predict on test set
test_preds = rf_model.predict(X_test_scaled)

# Prepare submission
submission = pd.DataFrame({
    'id': df_test['id'],
    'sii': test_preds
})
submission.to_csv('submission.csv', index=False)


