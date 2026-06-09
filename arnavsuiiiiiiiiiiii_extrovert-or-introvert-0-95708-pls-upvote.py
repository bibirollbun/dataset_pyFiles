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


import warnings
warnings.filterwarnings("ignore")



df=pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv")
dt=pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv")
samp=pd.read_csv("/kaggle/input/playground-series-s5e7/sample_submission.csv")


df.info()


df.head(2)


df=df.drop(columns=['id'])
ids=dt['id']
dt=dt.drop(columns=['id'])


# Fill numeric/categorical -1
for col in ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside', 'Friends_circle_size', 'Post_frequency']:
    df[col] = df[col].fillna(-1)
    dt[col] = dt[col].fillna(-1)

# Fill categorical "Unknown"
for col in ['Stage_fear', 'Drained_after_socializing']:
    df[col] = df[col].fillna("Unknown")
    dt[col] = dt[col].fillna("Unknown")



# # Fill numeric columns with mean (rounded to int)
# for col in ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside', 'Friends_circle_size', 'Post_frequency']:
#     mean_val_df = int(df[col].mean())
#     mean_val_dt = int(dt[col].mean())
    
#     df[col] = df[col].fillna(mean_val_df)
#     dt[col] = dt[col].fillna(mean_val_dt)

# # Fill categorical columns with mode
# for col in ['Stage_fear', 'Drained_after_socializing']:
#     mode_val_df = df[col].mode()[0]
#     mode_val_dt = dt[col].mode()[0]
    
#     df[col] = df[col].fillna(mode_val_df)
#     dt[col] = dt[col].fillna(mode_val_dt)



# df = df.dropna()
# dt = dt.dropna()



df.head()


# Define mapping
mapping = {
    'No': 0,
    'Yes': 1,
    'Unknown': 2
}

# Apply mapping to both DataFrames
for col in ['Stage_fear', 'Drained_after_socializing']:
    df[col] = df[col].map(mapping)
    dt[col] = dt[col].map(mapping)



df['Personality']=df['Personality'].map({
    'Extrovert' : 0,
    'Introvert' : 1
})


df.head()


# def flip_outlier_labels_iqr(data, label_col='Personality'):
#     data = data.copy()

#     for label in data[label_col].unique():
#         subset = data[data[label_col] == label]

#         for col in subset.columns:
#             if col == label_col:
#                 continue

#             Q1 = subset[col].quantile(0.25)
#             Q3 = subset[col].quantile(0.75)
#             IQR = Q3 - Q1
#             lower_bound = Q1 - 1.5 * IQR
#             upper_bound = Q3 + 1.5 * IQR

#             outlier_indices = subset[
#                 (subset[col] < lower_bound) | (subset[col] > upper_bound)
#             ].index

#             # Flip labels for detected outliers
#             data.loc[outlier_indices, label_col] = 1 - label  # Flip 0 to 1, 1 to 0

#     return data

# # Apply to df
# df = flip_outlier_labels_iqr(df)



df = df.astype(int)
dt = dt.astype(int)



df.describe().T


df['Introversion_Score'] = (
    df['Time_spent_Alone'] + 
    (10 - df['Social_event_attendance']) + 
    (10 - df['Going_outside']) + 
    (10 - df['Friends_circle_size']) +
    df['Drained_after_socializing'] * 5
)

dt['Introversion_Score'] = (
    dt['Time_spent_Alone'] + 
    (10 - dt['Social_event_attendance']) + 
    (10 - dt['Going_outside']) + 
    (10 - dt['Friends_circle_size']) +
    df['Drained_after_socializing'] * 5
)






df.isnull().sum()


df.info()


import seaborn as sns
import matplotlib.pyplot as plt

for col in df.columns[:-1]:  # Exclude 'Personality'
    plt.figure(figsize=(6, 4))
    sns.boxplot(x='Personality', y=col, data=df)
    plt.title(f'Boxplot of {col} by Personality')
    plt.show()



# sns.pairplot(df, hue='Personality', diag_kind='kde')
# plt.show()



corr = df.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()



from sklearn.decomposition import PCA

X = df.drop(columns='Personality')
y = df['Personality']

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

plt.figure(figsize=(6, 4))
sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1], hue=y, palette='Set1')
plt.title("PCA Projection by Personality")
plt.show()



import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D

# Features and target
X = df.drop(columns='Personality')
y = df['Personality']

# PCA with 3 components
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X)

# 3D Scatter plot
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

scatter = ax.scatter(
    X_pca[:, 0], X_pca[:, 1], X_pca[:, 2],
    c=y, cmap='Set1', s=50, alpha=0.7
)

ax.set_title('3D PCA Projection by Personality')
ax.set_xlabel('PCA1')
ax.set_ylabel('PCA2')
ax.set_zlabel('PCA3')

# Legend mapping 0 -> Extrovert, 1 -> Introvert
legend_labels = ['Extrovert', 'Introvert']
legend = ax.legend(handles=scatter.legend_elements()[0], labels=legend_labels, loc="upper right")
plt.show()



from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

# Features and target
# X = df.drop(columns=['Personality'])
# y = df['Personality']
X = df[['Stage_fear','Post_frequency']]
y = df['Personality']
# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Models to evaluate
models = {
    'LogisticRegression': LogisticRegression(max_iter=1000),
    'RidgeClassifier': RidgeClassifier(),
    'SGDClassifier': SGDClassifier(),
    'DecisionTree': DecisionTreeClassifier(),
    'RandomForest': RandomForestClassifier(),
    'GradientBoosting': GradientBoostingClassifier(),
    'AdaBoost': AdaBoostClassifier(),
    'Bagging': BaggingClassifier(),
    'ExtraTrees': ExtraTreesClassifier(),
    'GaussianNB': GaussianNB(),
    'KNeighbors': KNeighborsClassifier(),
    'SVC': SVC(probability=True),
    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
    'LightGBM': LGBMClassifier(),
    'CatBoost': CatBoostClassifier(verbose=0)
}

# Store results
results = []

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba) if y_proba is not None else None

    results.append({
        'Model': name,
        'Accuracy': acc,
        'F1 Score': f1,
        'AUC Score': auc
    })

# Convert to DataFrame and sort by Accuracy
results_df = pd.DataFrame(results).sort_values(by='Accuracy', ascending=False).reset_index(drop=True)
results_df



# from catboost import CatBoostClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
# import matplotlib.pyplot as plt

# # Features and target
# # X = df.drop(columns=['Personality'])
# # y = df['Personality']

# # Train-test split
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42, stratify=y
# )

# # Train CatBoost model
# model = CatBoostClassifier(verbose=0, random_state=42)
# model.fit(X_train, y_train)

# # Predictions
# y_pred = model.predict(X_test)
# y_proba = model.predict_proba(X_test)[:, 1]

# # Metrics
# acc = accuracy_score(y_test, y_pred)
# f1 = f1_score(y_test, y_pred)
# auc = roc_auc_score(y_test, y_proba)

# print(f"Accuracy: {acc:.4f}")
# print(f"F1 Score: {f1:.4f}")
# print(f"AUC Score: {auc:.4f}")

# # Confusion matrix
# cm = confusion_matrix(y_test, y_pred)
# disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Extrovert', 'Introvert'])

# # Plot
# plt.figure(figsize=(6, 4))
# disp.plot(cmap='Blues', values_format='d')
# plt.title("Confusion Matrix - CatBoost")
# plt.show()



# from lightgbm import LGBMClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
# import matplotlib.pyplot as plt

# # # Features and target
# # X = df.drop(columns=['Personality'])
# # y = df['Personality']

# # Train-test split
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42, stratify=y
# )

# # Train LightGBM model
# model = LGBMClassifier(random_state=42)
# model.fit(X_train, y_train)

# # Predictions
# y_pred = model.predict(X_test)
# y_proba = model.predict_proba(X_test)[:, 1]

# # Metrics
# acc = accuracy_score(y_test, y_pred)
# f1 = f1_score(y_test, y_pred)
# auc = roc_auc_score(y_test, y_proba)

# print(f"Accuracy: {acc:.4f}")
# print(f"F1 Score: {f1:.4f}")
# print(f"AUC Score: {auc:.4f}")

# # Confusion matrix
# cm = confusion_matrix(y_test, y_pred)
# disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Extrovert', 'Introvert'])

# # Plot
# plt.figure(figsize=(6, 4))
# disp.plot(cmap='Blues', values_format='d')
# plt.title("Confusion Matrix - LightGBM")
# plt.show()



# from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
# import matplotlib.pyplot as plt

# # Features and target
# # X = df.drop(columns=['Personality'])
# # y = df['Personality']

# # Train-test split
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42, stratify=y
# )

# # Train Logistic Regression model
# model = LogisticRegression(max_iter=1000, random_state=42)
# model.fit(X_train, y_train)

# # Predictions
# y_pred = model.predict(X_test)
# y_proba = model.predict_proba(X_test)[:, 1]

# # Metrics
# acc = accuracy_score(y_test, y_pred)
# f1 = f1_score(y_test, y_pred)
# auc = roc_auc_score(y_test, y_proba)

# print(f"Accuracy: {acc:.4f}")
# print(f"F1 Score: {f1:.4f}")
# print(f"AUC Score: {auc:.4f}")

# # Confusion matrix
# cm = confusion_matrix(y_test, y_pred)
# disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Extrovert', 'Introvert'])

# # Plot
# plt.figure(figsize=(6, 4))
# disp.plot(cmap='Blues', values_format='d')
# plt.title("Confusion Matrix - Logistic Regression")
# plt.show()



from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np

# Features and target
X = df.drop(columns=['Personality'])
y = df['Personality']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train Logistic Regression model
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)

# Predict probabilities
y_proba = model.predict_proba(X_test)[:, 1]

# Try different thresholds
thresholds = np.arange(0.1, 0.95, 0.01)
scores = []

for thresh in thresholds:
    y_pred_thresh = (y_proba >= thresh).astype(int)
    f1 = f1_score(y_test, y_pred_thresh)
    scores.append((thresh, f1))

# Get best threshold based on F1 score
best_thresh, best_f1 = max(scores, key=lambda x: x[1])
y_pred_final = (y_proba >= best_thresh).astype(int)

# Final metrics
acc = accuracy_score(y_test, y_pred_final)
auc = roc_auc_score(y_test, y_proba)

print(f"Best Threshold: {best_thresh:.2f}")
print(f"Accuracy: {acc:.4f}")
print(f"F1 Score: {best_f1:.4f}")
print(f"AUC Score: {auc:.4f}")

# Confusion matrix
cm = confusion_matrix(y_test, y_pred_final)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Extrovert', 'Introvert'])

# Plot
plt.figure(figsize=(6, 4))
disp.plot(cmap='Blues', values_format='d')
plt.title(f"Confusion Matrix - Logistic Regression (Thresh={best_thresh:.2f})")
plt.show()



from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np

# Features and target
X = df.drop(columns=['Personality'])
y = df['Personality']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Build pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=4)),
    ('logreg', LogisticRegression(max_iter=1000, random_state=42))
])

# Fit pipeline
pipeline.fit(X_train, y_train)

# Predict probabilities
y_proba = pipeline.predict_proba(X_test)[:, 1]

# Threshold tuning
thresholds = np.arange(0.1, 0.95, 0.01)
scores = []

for thresh in thresholds:
    y_pred_thresh = (y_proba >= thresh).astype(int)
    f1 = f1_score(y_test, y_pred_thresh)
    scores.append((thresh, f1))

# Best threshold
best_thresh, best_f1 = max(scores, key=lambda x: x[1])
y_pred_final = (y_proba >= best_thresh).astype(int)

# Final metrics
acc = accuracy_score(y_test, y_pred_final)
auc = roc_auc_score(y_test, y_proba)

print(f"Best Threshold: {best_thresh:.2f}")
print(f"Accuracy: {acc:.4f}")
print(f"F1 Score: {best_f1:.4f}")
print(f"AUC Score: {auc:.4f}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_final)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Extrovert', 'Introvert'])

plt.figure(figsize=(6, 4))
disp.plot(cmap='Blues', values_format='d')
plt.title(f"Confusion Matrix - LogisticRegression + PCA2 (Thresh={best_thresh:.2f})")
plt.show()



pipeline.fit(X, y)


# Make a copy of dt to avoid modifying original
X_dt = dt.copy()

# Predict probabilities using trained pipeline
dt_proba = pipeline.predict_proba(X_dt)[:, 1]

# Use optimized threshold to generate final predictions
preds = (dt_proba >= best_thresh).astype(int)





# model.fit(X, y)


# preds=model.predict(dt)


# # Drop target column from test set if present
# X_test_real = dt.copy()

# # Predict probabilities
# y_test_proba = model.predict_proba(X_test_real)[:, 1]

# # Apply the optimized threshold
# preds = (y_test_proba >= best_thresh).astype(int)





# Map back to label names
label_map = {0: 'Extrovert', 1: 'Introvert'}
final_labels = [label_map[p] for p in preds]

# Create submission DataFrame
sub = pd.DataFrame({
    'id': ids,
    'Personality': final_labels
})


sub


# Count plot for df
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='Personality')
plt.title('Personality Distribution in df')
plt.xlabel('Personality')
plt.ylabel('Count')
plt.tight_layout()
plt.show()


plt.figure(figsize=(6, 4))
sns.countplot(data=sub, x='Personality')
plt.title('Predicted Personality Distribution in sub')
plt.xlabel('Personality')
plt.ylabel('Count')
plt.tight_layout()
plt.show()


sub.to_csv("submission.csv",index=False)

