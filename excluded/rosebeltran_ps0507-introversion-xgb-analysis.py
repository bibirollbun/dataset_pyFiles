import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.model_selection import KFold, StratifiedKFold, RepeatedStratifiedKFold
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

from xgboost import XGBClassifier
from catboost import CatBoostClassifier

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


train = pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv")
original = pd.read_csv("/kaggle/input/extrovert-vs-introvert-behavior-data/personality_dataset.csv")

train.head()


train.info()


test.info()


original.info()


train.describe()


test.describe()


original.describe()


# Drop the 'id' column
X = train.drop(columns=['id', 'Personality'])

# Do the same for test set
X_test = test.drop(columns='id')

# Extract the target column
y = train['Personality']

X.head()


# Encode target labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Output labels are now numbers
y_encoded[:10]


# Just check
X.info()


# Combine the two datasets
combined = pd.concat([X, X_test], axis=0)

combined.info()


# Count our nulls
missing_counts = combined.isnull().sum()
missing_counts = missing_counts[missing_counts > 0]  
missing_counts = missing_counts.sort_values(ascending=False)
missing_counts


plt.figure(figsize=(10, 6))
sns.barplot(
    x=missing_counts.values,
    y=missing_counts.index,
    palette="viridis"
)
plt.xlabel("Number of Missing Values")
plt.ylabel("Features")
plt.title("Missing Values per Column in Combined Dataset")
plt.tight_layout()
plt.show()



numerical = X.select_dtypes(include=['float64']).columns.tolist()
categorical = X.select_dtypes(include=['object']).columns.tolist()

print(numerical)
print(categorical)


# Just checking the correlation of our two categorical variables

from scipy.stats import chi2_contingency

def cramers_v(x, y):
    confusion_matrix = pd.crosstab(x, y)
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1)) / (n-1))  # bias correction
    rcorr = r - ((r-1)**2 / (n-1))
    kcorr = k - ((k-1)**2 / (n-1))
    return np.sqrt(phi2corr / min((kcorr-1), (rcorr-1)))



cramers_v(combined['Stage_fear'], combined['Drained_after_socializing'])


cramers_results = pd.DataFrame(index=categorical, columns=categorical)

for col1 in categorical:
    for col2 in categorical:
        cramers_results.loc[col1, col2] = cramers_v(combined[col1], combined[col2])

cramers_results = cramers_results.astype(float)

plt.figure(figsize=(8, 6))
sns.heatmap(cramers_results, annot=True, cmap="Greens", fmt=".3f")
plt.title("Cramér's V Correlation Between Categorical Variables")
plt.tight_layout()
plt.show()



# List of column pairs where col_A has missing and col_B can help
A = 'Stage_fear'
B = 'Drained_after_socializing'
fill_pairs = [(A, B), (B, A)]

for col_a, col_b in fill_pairs:
    mask = combined[col_a].isna() & combined[col_b].notna()
    combined.loc[mask, col_a] = combined.loc[mask, col_b]
    print(f"Filled {mask.sum()} missing values in {col_a} using {col_b}")



for col in categorical:
    combined[col] = combined[col].fillna("Sometimes")

combined.head(5)


combined.describe()


# Check if null values are gone for categoricals
combined.info()



print("\n*** Replacement Values per Column *****\n")

for col in numerical:
    metric = original[col].median()
    combined[col] = combined[col].fillna(metric)
    print(col, metric)

print("\n\n")
combined.head(5)



# Check if null values are gone for numericals
combined.info()


# Final check for entire input dataset
combined.isnull().values.any() == False


# Turn strings into boolean before modeling
combined = pd.get_dummies(combined, columns=categorical, drop_first=False)

combined.info()


# Preprocessing done, we can split back, I'm running out of names
# Train: X_processed
# Test: X_test_processed

X_processed = combined.iloc[:len(X)].reset_index(drop=True)
X_test_processed = combined.iloc[len(X):].reset_index(drop=True)

X_processed.head()


X_processed.info()


X_test_processed.info()


# First, hold out a percentage for final validation
X_train, X_val, y_train, y_val = train_test_split(
    X_processed, y_encoded,
    test_size=0.2,
    stratify=y_encoded,       
    random_state=42
)


skf = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=42)

fold_scores = []
fold = 1
models = []

for train_idx, val_idx in skf.split(X_train, y_train):
    # Split by index
    X_tr, X_val_fold = X_train.iloc[train_idx].copy(), X_train.iloc[val_idx].copy()
    y_tr, y_val_fold = y_train[train_idx], y_train[val_idx]
    
    # Define model
    model = XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=4,
        colsample_bytree=0.8,
        subsample=0.8,
        early_stopping=10,
        eval_metric='logloss',
        random_state=42,
        enable_categorical=False,
        use_label_encoder=False,
        objective='binary:logistic',
    )
    
    # Fit and predict
    model.fit(X_tr, y_tr,
              eval_set=[(X_val_fold, y_val_fold)],
              early_stopping_rounds=10,
              verbose=0
             )
    
    models.append(model)
    y_pred = model.predict(X_val_fold)
    
    # Evaluate
    acc = accuracy_score(y_val_fold, y_pred)
    fold_scores.append(acc)
    print(f"Fold {fold:2d} Accuracy: {acc:.5f}")
    fold += 1

# Summary stats
print("\nCross-validation summary:")
print("Mean Accuracy: {:.5f}".format(np.mean(fold_scores)))
print("Std Dev      : {:.5f}".format(np.std(fold_scores)))




# Predict on holdout
all_preds = np.zeros(X_val.shape[0])

for model in models:
    # Use probability prediction for ensemble
    y_proba = model.predict_proba(X_val)[:, 1]
    all_preds += y_proba

# Average probabilities
avg_pred = all_preds / len(models)

# Convert to final binary predictions (0 or 1)
final_pred = (avg_pred >= 0.5).astype(int)

# Evaluate
print("Holdout Accuracy: {:.5f}\n".format(accuracy_score(y_val, final_pred)))
print(classification_report(y_val, final_pred, digits=5))



# Boolean mask for incorrect predictions
errors = final_pred != y_val

# Create a DataFrame to inspect
error_df = X_val.copy()
error_df['TrueLabel'] = y_val
error_df['Predicted'] = final_pred
error_df['Probability'] = avg_pred
error_df['Personality'] = le.inverse_transform(y_val)

error_cases = error_df[errors]
print(f"\nTotal Errors: {len(error_cases)}\n")
error_cases.head(10)


plt.figure(figsize=(8, 5))
sns.histplot(error_cases['Probability'], bins=200, kde=True, color='tomato')
plt.title("Distribution of Predicted Probabilities")
plt.xlabel("Predicted Probability")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()



# Most confident wrong predictions
error_cases.sort_values(by='Probability', ascending=False).head(10)


# Most borderline errors (near 0.5 threshold)
error_cases = error_df[errors].copy() 
error_cases['DistanceFromThreshold'] = np.abs(error_cases['Probability'] - 0.5)
borderline_errors = error_cases.sort_values(by='DistanceFromThreshold')
borderline_errors.head(10)


# Dataset dictionary for looping
datasets = {
    'Train': train,
    'Errors': error_cases
}

# Radar setup
labels = numerical
num_vars = len(labels)
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1]  # loop back

# Plot setup
fig, axs = plt.subplots(1, 2, figsize=(18, 6), subplot_kw=dict(polar=True))
colors = {'Introvert': 'skyblue', 'Extrovert': 'salmon'}

for ax, (name, df) in zip(axs, datasets.items()):
    grouped = df.groupby("Personality")[numerical].mean()

    for personality in grouped.index:
        values = grouped.loc[personality].tolist()
        values += values[:1]  # close the loop

        ax.plot(angles, values, label=personality, color=colors[personality], linewidth=2)
        ax.fill(angles, values, alpha=0.2, color=colors[personality])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_title(f"{name} Dataset", fontsize=14, pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1))

plt.suptitle("Radar Plot of Numerical Means by Personality (Across Datasets)", fontsize=16, y=1.05)
plt.tight_layout()
plt.show()


# Confusion matrix
cm = confusion_matrix(y_val, final_pred)
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

# Plot
plt.figure(figsize=(8, 6))
sns.heatmap(cm_norm, annot=True, fmt=".5f", cmap="Greens", 
            xticklabels=le.classes_, yticklabels=le.classes_,
           )

plt.title("Normalized Confusion Matrix", fontsize=16)
plt.xlabel("Predicted Label", fontsize=12)
plt.ylabel("True Label", fontsize=12)
plt.xticks(rotation=0)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()



# Get importances
importances = pd.Series(model.feature_importances_, index=X_train.columns)

# Sort and reset index for seaborn
imp_df = importances.sort_values(ascending=False).reset_index()
imp_df.columns = ['Feature', 'Importance']

# Plot
plt.figure(figsize=(10, 6))
sns.barplot(data=imp_df, x='Importance', y='Feature', palette='viridis')

plt.title('Feature Importances (XGBoost)', fontsize=14)
plt.xlabel('Importance Score')
plt.ylabel('Feature')
plt.tight_layout()
plt.show()



# Train on 100% of the train data: X_processed, y_encoded

# Define model
model2 = XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=4,
    colsample_bytree=0.8,
    subsample=0.8,
    eval_metric='logloss',
    random_state=42,
    enable_categorical=False,
    use_label_encoder=False,
    objective='binary:logistic',
)

# Fit and predict
model2.fit(X_processed, y_encoded,
          verbose=0
         )

y_proba = model2.predict_proba(X_test_processed)[:, 1]

print(y_proba.shape)


# Convert to final binary predictions (0 or 1)
final_pred = (y_proba >= 0.5).astype(int)

submission = pd.read_csv("/kaggle/input/playground-series-s5e7/sample_submission.csv")
submission['Personality'] = le.inverse_transform(final_pred) 

submission.to_csv("submission.csv", index=False)
print("Submission file saved as 'submission.csv'")

submission.head()


submission['Personality'].value_counts()

