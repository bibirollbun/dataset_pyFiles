import seaborn as sns 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
from collections import Counter
import joblib
import time
import csv
import warnings
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import random
from sklearn.model_selection import StratifiedKFold
from pathlib import Path
from datetime import datetime




train_df = pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv')
sample_submission = pd.read_csv('/kaggle/input/playground-series-s5e6/sample_submission.csv')


print("shape:", train_df.shape)
print(train_df.head())
print(train_df.tail())
print(train_df.dtypes)
print("\n Missing Values:")
print(train_df.isnull().sum())


print(train_df['Fertilizer Name'].unique())


print(train_df.columns)


train_df.describe()


# Target Distribution âž” Fertilizer Name
print("\nðŸ”¸ Fertilizer Name Count:\n", train_df['Fertilizer Name'].value_counts())


crop_types = sorted(train_df['Crop Type'].unique())


# Visualize Target Distribution
plt.figure(figsize=(10,6))
sns.countplot(x='Fertilizer Name', data=train_df, order=train_df['Fertilizer Name'].value_counts().index)
plt.xticks(rotation=45)
plt.title('Fertilizer Name Distribution')
plt.show()



# Plot for Categorical Columns âž” Crop Type & Soil Type
categorical_cols = ['Crop Type', 'Soil Type']

for col in categorical_cols:
    plt.figure(figsize=(10,5))
    sns.countplot(x=col, data=train_df, order=train_df[col].value_counts().index)
    plt.xticks(rotation=45)
    plt.title(f'{col} Distribution')
    plt.show()



# Categorical Columns 
for col in categorical_cols:
    plt.figure(figsize=(12,6))
    sns.countplot(x=col, hue='Fertilizer Name', data=train_df)
    plt.xticks(rotation=45)
    plt.title(f'{col} vs Fertilizer Name')
    plt.show()


fertilizer_counts = train_df['Fertilizer Name'].value_counts()

# Create the pie chart
plt.figure(figsize=(8, 8))
plt.pie(fertilizer_counts, 
        labels=fertilizer_counts.index, 
        autopct='%1.1f%%', 
        startangle=140, 
        colors=['#0584E5', '#75E6DA', '#FFC300', '#FF5733', '#C70039', '#900C3F'])

plt.title('Distribution of Fertilizer Names')
plt.axis('equal')  # Ensures pie is a circle
plt.show()


numerical_cols = ['Temparature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 
                  'Phosphorous']

# 2 rows Ã— 3 columns
rows = 2
cols = 3

plt.figure(figsize=(25, 10))  # Width Ã— Height

for i, col in enumerate(numerical_cols):
    plt.subplot(rows, cols, i + 1)
    plt.hist(train_df[col], bins=30, color='purple', edgecolor='black', alpha=0.7)
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()



warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

cols = ['Temparature', 'Humidity', 'Moisture', 'Nitrogen', 'Phosphorous', 'Potassium', 'Fertilizer Name']
train_df_small = train_df[cols].sample(4000, random_state=42)

# Create customized pairplot
plot = sns.pairplot(
    train_df_small,
    hue='Fertilizer Name',
    corner=True,
    height=2.5,          # Adjust individual plot size (default: 2.5)
    aspect=1.2,          # Aspect ratio of each plot
    plot_kws={'alpha': 0.6, 's': 30, 'edgecolor': 'k'},  # Style markers
    palette='tab10'       #Color palette
)

# Add subtitle
plot.fig.suptitle("Customized Pairplot - Fertilizer Dataset (Sample 1000)", fontsize=18, y=1.02)
plt.show()


# Correlation Heatmap âž” Numerical Features
plt.figure(figsize=(12,8))
sns.heatmap(train_df[numerical_cols].corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix (Numerical Features)')
plt.show()



rows = 2
cols = 3

plt.figure(figsize=(20, 12))  # Adjust as needed

for i, col in enumerate(numerical_cols):
    plt.subplot(rows, cols, i + 1)  # rows x cols layout
    sns.boxplot(y=train_df[col])
    plt.title(f'Boxplot of {col}')

plt.tight_layout()
plt.show()


numerical_cols = ['Temparature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 
                  'Phosphorous']

fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(20, 12))  # 2 rows Ã— 3 columns

for idx, col in enumerate(numerical_cols):
    row = idx // 3
    col_pos = idx % 3
    ax = axes[row, col_pos]
    
    sns.boxplot(x='Fertilizer Name', y=col, data=train_df, ax=ax)
    ax.set_title(f'{col} vs Fertilizer Name')
    ax.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.suptitle('Boxplot of Numerical Features vs Fertilizer Name', fontsize=20, y=1.03)
plt.show()



# Generate All Unique 2-Column Combinations â†’ Using itertools
pairs = list(itertools.combinations(numerical_cols, 2))
n_plots = len(pairs)

# Calculate number of rows and columns dynamically
n_cols = 3  # You can adjust this (e.g., 3 or 4)
n_rows = (n_plots // n_cols) + (n_plots % n_cols > 0)

# Create Figure
fig = plt.figure(figsize=(5 * n_cols, 4 * n_rows))

# Loop through each column pair and create a subplot
for i, (x, y) in enumerate(pairs, start=1):
    ax = fig.add_subplot(n_rows, n_cols, i)
    ax.scatter(train_df[x], train_df[y], alpha=0.5, edgecolors='black')
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(f"{x} vs {y}")

plt.tight_layout()
plt.suptitle("Scatter Plots of All Numerical Column Pairs", fontsize=18, y=1.02)
plt.show()



# Numerical columns
numerical_cols = ['Temparature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium',
                  'Phosphorous']

# Create subplots with 2 rows and 3 columns
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(22, 12))

for idx, col in enumerate(numerical_cols):
    row = idx // 3
    col_pos = idx % 3
    ax = axes[row, col_pos]
    
    sns.violinplot(x='Fertilizer Name', y=col, data=train_df, inner='box', palette='Set2', ax=ax)
    ax.set_title(f'{col} Distribution by Fertilizer Name')
    ax.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.suptitle('Violin Plot of Numerical Features vs Fertilizer Name', fontsize=20, y=1.03)
plt.show()



# Step 1: Group by Fertilizer Name and calculate mean
feature_means = train_df.groupby('Fertilizer Name')[['Temparature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous']].mean()

# Step 2: Setup subplot grid (2 rows Ã— 3 columns)
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(18, 10))

# Step 3: Loop through each feature and plot
for idx, col in enumerate(feature_means.columns):
    row = idx // 3
    col_pos = idx % 3
    ax = axes[row, col_pos]

    sns.barplot(x=feature_means.index, y=feature_means[col], palette='viridis', ax=ax)
    ax.set_title(f'Average {col} by Fertilizer Type')
    ax.set_ylabel(f'Mean {col}')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)

# Step 4: Final layout adjustments
plt.tight_layout()
plt.suptitle('Average Feature Values by Fertilizer Type', fontsize=20, y=1.03)
plt.show()



le_target = LabelEncoder()
train_df['target'] = le_target.fit_transform(train_df['Fertilizer Name'])

for df in [train_df, test_df]:
    df['Crop_Code'] = df['Crop Type'].astype('category').cat.codes
    df['Soil_Code'] = df['Soil Type'].astype('category').cat.codes

features = ['Temparature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous', 'Crop_Code', 'Soil_Code']
X = train_df[features]
X_test = test_df[features]
y = train_df['target']




categorical_cols = ['Soil Type', 'Crop Type']
label_encoders = {}

# Apply encoding to train and test datasets before creating feature sets
for col in categorical_cols:
    le = LabelEncoder()
    train_df[col] = le.fit_transform(train_df[col])
    test_df[col] = le.transform(test_df[col])  # use transform here to keep encoding consistent
    label_encoders[col] = le

# Now create feature sets X and X_test using encoded columns
features = ['Temparature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous',
            'Soil Type', 'Crop Type']
X = train_df[features]
X_test = test_df[features]

# Encode target
le_target = LabelEncoder()
y = le_target.fit_transform(train_df['Fertilizer Name'])



for df in [train_df, test_df]:
    df['Crop_Code'] = df['Crop Type'].astype('category').cat.codes
    df['Soil_Code'] = df['Soil Type'].astype('category').cat.codes

features = ['Temparature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous', 'Crop_Code', 'Soil_Code']
X = train_df[features]
X_test = test_df[features]
y = train_df['target']


# Define MAP@3 metric function
def mapk(actual, predicted, k=3):
    def apk(a, p, k):
        if a in p[:k]:
            return 1.0 / (p[:k].index(a) + 1)
        return 0.0
    return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])

# Initialize Stratified K-Fold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Initialize out-of-fold and test predictions arrays
oof_preds = np.zeros((len(train_df), len(le_target.classes_)))
test_preds = np.zeros((len(test_df), len(le_target.classes_)))
models = []
fold_scores = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    print(f"\nTraining fold {fold + 1}...")
    try:
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = RandomForestClassifier(
            n_estimators=500,
            max_depth=10,
            random_state=42 + fold,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        models.append(model)

        val_preds = model.predict_proba(X_val)
        oof_preds[val_idx] = val_preds
        test_preds += model.predict_proba(X_test) / skf.n_splits

        top3 = np.argsort(val_preds, axis=1)[:, -3:][:, ::-1]
        score = mapk(y_val.tolist(), top3.tolist(), k=3)
        fold_scores.append(score)
        print(f"Fold {fold + 1} MAP@3: {score:.5f}")
    except Exception as e:
        print(f"Error in fold {fold + 1}: {e}")



from sklearn.metrics import (
    log_loss, roc_auc_score, matthews_corrcoef,
    precision_score, recall_score, f1_score, confusion_matrix
)

# Get predicted class labels (most probable class)
y_pred_labels = np.argmax(oof_preds, axis=1)

# 1. Log Loss
logloss = log_loss(y, oof_preds)
print(f"Log Loss: {logloss:.5f}")

# 2. ROC-AUC Score (One-vs-Rest)
try:
    y_onehot = np.eye(len(np.unique(y)))[y]
    roc_auc = roc_auc_score(y_onehot, oof_preds, multi_class='ovr')
    print(f"ROC-AUC Score (OvR): {roc_auc:.5f}")
except:
    print("ROC-AUC not supported due to missing class probabilities.")

# 3. Matthews Correlation Coefficient (MCC)
mcc = matthews_corrcoef(y, y_pred_labels)
print(f"MCC Score: {mcc:.5f}")

# 4. Precision, Recall, F1 (Macro Averaged)
precision = precision_score(y, y_pred_labels, average='macro')
recall = recall_score(y, y_pred_labels, average='macro')
f1 = f1_score(y, y_pred_labels, average='macro')

print(f"Precision (Macro): {precision:.5f}")
print(f"Recall (Macro): {recall:.5f}")
print(f"F1 Score (Macro): {f1:.5f}")



def confusion_matrix_scratch(y_true, y_pred, labels):
    """
    y_true : list or array of true labels (encoded as integers)
    y_pred : list or array of predicted labels (encoded as integers)
    labels : list of all unique class labels (integers)
    
    Returns confusion matrix as 2D numpy array
    """
    n = len(labels)
    matrix = np.zeros((n, n), dtype=int)
    
    label_to_index = {label: i for i, label in enumerate(labels)}
    
    for t, p in zip(y_true, y_pred):
        i = label_to_index[t]
        j = label_to_index[p]
        matrix[i][j] += 1
        
    return matrix

# Suppose y_true and y_pred are your true and predicted labels (as integers)
# Suppose le_target.classes_ gives label names (fertilizer names)

labels = list(range(len(le_target.classes_)))  # integers 0 to N-1
class_names = le_target.classes_

cm = confusion_matrix_scratch(y_true=y, y_pred=y_pred_labels, labels=labels)

plt.figure(figsize=(10,8))
sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu', xticklabels=class_names, yticklabels=class_names)

plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix (Scratch Implementation)')
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()



def binarize_labels(y, num_classes):
    # Convert integer labels to one-hot encoding
    return np.eye(num_classes)[y]

def compute_tpr_fpr(y_true_bin, y_score, thresholds):
    tpr = []
    fpr = []
    for thresh in thresholds:
        y_pred = (y_score >= thresh).astype(int)
        TP = np.sum((y_pred == 1) & (y_true_bin == 1))
        FP = np.sum((y_pred == 1) & (y_true_bin == 0))
        TN = np.sum((y_pred == 0) & (y_true_bin == 0))
        FN = np.sum((y_pred == 0) & (y_true_bin == 1))
        tpr.append(TP / (TP + FN + 1e-10))  # True Positive Rate
        fpr.append(FP / (FP + TN + 1e-10))  # False Positive Rate
    return np.array(fpr), np.array(tpr)

def trapezoidal_auc(fpr, tpr):
    # Sort FPR and TPR in ascending order of FPR
    order = np.argsort(fpr)
    fpr_sorted = fpr[order]
    tpr_sorted = tpr[order]
    auc = 0.0
    for i in range(1, len(fpr_sorted)):
        auc += (fpr_sorted[i] - fpr_sorted[i-1]) * (tpr_sorted[i] + tpr_sorted[i-1]) / 2
    return auc

# Assuming you have these from your model
# y = true labels (shape: [num_samples])
# oof_preds = predicted probabilities (shape: [num_samples, num_classes])
# le_target = your LabelEncoder used on classes

num_classes = len(le_target.classes_)
y_bin = binarize_labels(y, num_classes)

thresholds = np.linspace(0, 1, 100)

plt.figure(figsize=(10, 8))

for i, class_name in enumerate(le_target.classes_):
    fpr, tpr = compute_tpr_fpr(y_bin[:, i], oof_preds[:, i], thresholds)
    auc_score = trapezoidal_auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{class_name} (AUC = {auc_score:.3f})')

plt.plot([0,1], [0,1], 'k--', label='Random Chance')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Multi-class ROC Curve (Scratch Implementation)')
plt.legend(loc='lower right')
plt.grid(alpha=0.3)
plt.show()



top3_test_preds = np.argsort(test_preds, axis=1)[:, -3:][:, ::-1]
top3_labels = le_target.inverse_transform(top3_test_preds.ravel()).reshape(top3_test_preds.shape)

submission = pd.DataFrame({
    'id': test_df['id'],
    'Fertilizer Name': [' '.join(row) for row in top3_labels]
})
submission.to_csv('submission.csv', index=False)
print("âœ… Submission file created successfully: submission.csv")



submission = pd.read_csv('submission.csv')

print("First 5 rows of submission:")
print(submission.head())


# After training loop and fold_scores calculation
best_fold = np.argmax(fold_scores)  # Index of fold with highest MAP@3 score

# Save best model (based on MAP@3 score)
best_model = models[best_fold]
joblib.dump(best_model, 'fertilizer_model.pkl')
print("Model saved as fertilizer_model.pkl")

# Save label encoder (used for inverse prediction)
joblib.dump(le_target, 'label_encoder.pkl')
print("Label encoder saved as label_encoder.pkl")

joblib.dump(le_target, 'target_encoder.pkl')
print("Label encoder saved as target_encoder.pkl")

print(" Model and encoders saved successfully!")

