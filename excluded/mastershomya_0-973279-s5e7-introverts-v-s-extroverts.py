import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold,GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder


train_data = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
train_data.head()


train_data.shape


train_data.info()


train_data.describe()


train_data.isna().sum()


sns.countplot(x='Personality', data=train_data)
plt.title("Introvert - Extrovert Count")
plt.show()

print(train_data['Personality'].value_counts())


def missing_data(df):
    overview = pd.DataFrame({
        'Missing': df.isnull().sum(),
        'Dtype': df.dtypes,
        'Unique': df.nunique()
    })
    return overview

missing_data(train_data)


le = LabelEncoder()
train_data['Personality'] = le.fit_transform(train_data['Personality'])


train_data.head() #Extrovert = 0 and introvert = 1


numerical_features_to_be_filled = [
    'Time_spent_Alone',
    'Social_event_attendance',
    'Going_outside',
    'Friends_circle_size',
    'Post_frequency'
]

for col in numerical_features_to_be_filled:
    train_data[col].fillna(train_data[col].median(), inplace=True)


categorical_features_to_be_filled = ['Stage_fear', 'Drained_after_socializing']
for col in categorical_features_to_be_filled:
    train_data[col].fillna(train_data[col].mode()[0], inplace=True)


mapping = {'Yes': 1, 'No': 0}
for col in categorical_features_to_be_filled:
    train_data[col] = train_data[col].map(mapping)


train_data.head()


train_data.isna().sum()


plt.figure(figsize=(10, 6))
sns.heatmap(train_data.corr(), annot=True, cmap='magma', center=0)
plt.title("Feature Correlation")
plt.show()


X = train_data.drop(columns=['id','Personality'])
y = train_data['Personality']


#Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#Apply PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

#Create a DataFrame with PCA results and add target labels
pca_df = pd.DataFrame(X_pca, columns=['PCA1', 'PCA2'])
pca_df['Personality'] = y.values

#Plot
plt.figure(figsize=(8, 6))
sns.scatterplot(data=pca_df, x='PCA1', y='PCA2', hue='Personality', palette='Set2')
plt.title("Personality Clustering (PCA Projection)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.grid(True)
plt.tight_layout()
plt.show()



# Hyperparameter grid
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5],
    'learning_rate': [0.01, 0.05],
    'subsample': [0.7, 1.0],
    'colsample_bytree': [0.7, 1.0],
    'gamma': [0, 1]
}

# Base model
xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)

# Stratified 5-fold CV
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Grid search
grid_search = GridSearchCV(
    estimator=xgb,
    param_grid=param_grid,
    scoring='accuracy',
    cv=cv,
    n_jobs=-1,  # Use all CPU cores
    verbose=1
)

# Fit grid search
grid_search.fit(X, y)

# Best model and parameters
best_model = grid_search.best_estimator_
print("\n✅ Best Parameters:")
print(grid_search.best_params_)

print("\n✅ Best CV Accuracy:", round(grid_search.best_score_, 4))

# Optional: Evaluate best model using StratifiedKFold again (detailed output)
fold_accuracies = []
fold = 1
for train_idx, val_idx in cv.split(X, y):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_val)
    
    acc = accuracy_score(y_val, y_pred)
    fold_accuracies.append(acc)

    print(f"\nFold {fold}")
    print("Accuracy:", round(acc, 4))
    print("Confusion Matrix:\n", confusion_matrix(y_val, y_pred))
    print("Classification Report:\n", classification_report(y_val, y_pred))
    fold += 1

# Summary
print("\n=== Final Cross-Validation Summary ===")
print("Fold Accuracies:", [round(a, 4) for a in fold_accuracies])
print("Mean CV Accuracy:", round(np.mean(fold_accuracies), 4))
print("Standard Deviation:", round(np.std(fold_accuracies), 4))

# Fit final model on full data
best_model.fit(X, y)
print("\n✅ Final best model trained on full dataset.")



test_data = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')
test_data.head()


numerical_features = [
    'Time_spent_Alone',
    'Social_event_attendance',
    'Going_outside',
    'Friends_circle_size',
    'Post_frequency'
]
categorical_features = ['Stage_fear', 'Drained_after_socializing']

for col in categorical_features:
    test_data[col].fillna(test_data[col].mode()[0], inplace=True)
for col in numerical_features:
    test_data[col].fillna(test_data[col].median(), inplace=True)
for col in categorical_features:
    test_data[col] = test_data[col].map(mapping)


test_data.head()


test_data.isna().sum()


test_ids = test_data['id']

X_test = test_data.drop(columns='id')

test_preds = best_model.predict(X_test)

submission_df = pd.DataFrame({
    'id': test_ids,
    'Personality': test_preds
})

submission_df.to_csv('submission.csv', index=False)
print("✅ Submission DataFrame created successfully!")


submission_file = pd.read_csv('/kaggle/working/submission.csv')
submission_file.head()


submission_file['Personality'] = submission_file['Personality'].map({1:'Introvert', 0:'Extrovert'})


submission_file.head()


submission_file.to_csv('submission_final.csv', index=False)

