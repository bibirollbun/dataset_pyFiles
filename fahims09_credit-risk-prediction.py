!pip install imbalanced-learn==0.10.1


# Basic Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from collections import Counter
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.impute import KNNImputer
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.preprocessing import LabelBinarizer

import warnings
warnings.filterwarnings("ignore")
sns.set(style="whitegrid")


import xgboost
import sklearn
import imblearn

print("scikit-learn version:", sklearn.__version__)
print("imbalanced-learn version:", imblearn.__version__)
print("XGboost version:", xgboost.__version__)


# Load the dataset
df = pd.read_csv('/kaggle/input/GiveMeSomeCredit/cs-training.csv')
df.drop('Unnamed: 0', axis=1, inplace=True)  # Drop index column

# Rename target for readability
df.rename(columns={"SeriousDlqin2yrs": "Default"}, inplace=True)

# Quick overview
df.head()


for each in df.columns:
    print(each)


df['Default'].value_counts()


df.info()


df.describe().T


specified_rows = df[df['age']==109.0]
specified_rows


df.isnull().sum()


df.dtypes


# Make a copy so original is untouched
df_dropna = df.dropna()


print(f"Original shape: {df.shape}")
print(f"After dropping missing: {df_dropna.shape}")


plt.figure(figsize=(6,4))
sns.countplot(data=df_dropna, x='Default', palette='Set2')
plt.title('Loan Default Distribution')
plt.xticks([0, 1], ['Non-Default (0)', 'Default (1)'])
plt.ylabel('Count')
plt.show()



df_dropna


import math

def plot_histograms(df, cols, cols_per_row=3):
    total = len(cols)
    rows = math.ceil(total / cols_per_row)

    plt.figure(figsize=(5*cols_per_row, 4*rows))
    for i, col in enumerate(cols):
        plt.subplot(rows, cols_per_row, i + 1)
        sns.histplot(df[col], kde=True, bins=30, color='orange')
        plt.title(f'Distribution: {col}')
    plt.tight_layout()
    plt.show()



df_dropna.describe().T


specified_rows = df_dropna['MonthlyIncome'][df_dropna['MonthlyIncome']==0]
len(specified_rows)


num_cols = ['RevolvingUtilizationOfUnsecuredLines', 'age', 'NumberOfTime30-59DaysPastDueNotWorse', 'DebtRatio',
            'MonthlyIncome', 'NumberOfOpenCreditLinesAndLoans',
            'NumberOfTimes90DaysLate', 'NumberRealEstateLoansOrLines', 'NumberOfTime60-89DaysPastDueNotWorse',
            'NumberOfDependents']
plot_histograms(df_dropna, num_cols, cols_per_row=3)


df_dropna[num_cols].skew().sort_values(ascending=False)


def boxplot_grid(cols, df):
    rows = (len(cols) + 2) // 3
    plt.figure(figsize=(18, 4*rows))
    for i, col in enumerate(cols):
        plt.subplot(rows, 3, i + 1)
        sns.boxplot(x=df[col], color='lightcoral')
        plt.title(f'Boxplot: {col}')
    plt.tight_layout()
    plt.show()

boxplot_grid(num_cols, df_dropna)


def cap_outliers_iqr(df, columns):
    df_capped = df.copy()
    for col in columns:
        Q1 = df_capped[col].quantile(0.25)
        Q3 = df_capped[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        total = len(df_capped[col][df_capped[col] > upper]) + len(df_capped[col][df_capped[col] < lower])
        print(f'Total number of outliers in {col} was: {total}')
        df_capped[col] = np.where(df_capped[col] < lower, lower,
                                  np.where(df_capped[col] > upper, upper, df_capped[col]))
    return df_capped

# Apply
df_iqr_capped = cap_outliers_iqr(df_dropna, num_cols)


df_iqr_capped.describe().T


plot_histograms(df_iqr_capped, num_cols, cols_per_row=3)


num_cols_for_iqr = ['RevolvingUtilizationOfUnsecuredLines', 'age', 'DebtRatio',
            'MonthlyIncome', 'NumberOfOpenCreditLinesAndLoans',
            'NumberRealEstateLoansOrLines',
            'NumberOfDependents']


df_iqr_capped = cap_outliers_iqr(df_dropna, num_cols_for_iqr)
df_iqr_capped


plot_histograms(df_iqr_capped, num_cols_for_iqr, cols_per_row=3)


df = df_iqr_capped.copy()
features = num_cols


def plot_bivariate_boxplots(df, features, target='Default'):
    rows = (len(features) + 2) // 3
    plt.figure(figsize=(18, 5 * rows))
    
    for i, feature in enumerate(features):
        plt.subplot(rows, 3, i + 1)
        sns.boxplot(x=target, y=feature, data=df, palette='coolwarm')
        plt.title(f'{feature} vs {target}')
        plt.tight_layout()

plot_bivariate_boxplots(df, features)



def plot_kde_bivariate(df, features, target='Default', rows=3, cols_per_row=3):
    total = len(features)
    cols = cols_per_row
    rows = (total + cols - 1) // cols
    
    plt.figure(figsize=(6 * cols, 4 * rows))

    for i, col in enumerate(features):
        plt.subplot(rows, cols, i + 1)
        for label in df[target].unique():
            sns.kdeplot(df[df[target] == label][col], label=f'{target}={label}', fill=True, common_norm=False)
        plt.title(f'{col} vs {target}')
        plt.legend()
        plt.tight_layout()

    plt.show()

plot_kde_bivariate(df=df, features=features)




# Correlation heatmap
correlation = df.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation[['Default']].sort_values(by='Default', ascending=False), 
            annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation with Target Variable')
plt.show()



selected_cols = ['age', 'DebtRatio', 'MonthlyIncome', 'NumberOfOpenCreditLinesAndLoans', 'Default']
sns.pairplot(df[selected_cols], hue='Default', corner=True, plot_kws={'alpha': 0.9})
plt.tight_layout()
plt.show()



# === Step 1: Load the training data ===
df = pd.read_csv('/kaggle/input/GiveMeSomeCredit/cs-training.csv')
df.drop('Unnamed: 0', axis=1, inplace=True)  # Drop index column

# Rename target for readability
df.rename(columns={"SeriousDlqin2yrs": "Default"}, inplace=True)

df


# === Step 1: Drop all the rows having missing values ===
df_clean = df.dropna()


# === Step 2: Separate features and target ===
X = df_clean.drop('Default', axis=1)
y = df_clean['Default']


# === Step 3: Train-test split ===
# Perform 70:30 train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Check the shapes
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_train distribution:\n{y_train.value_counts(normalize=True)}")
print(f"y_test distribution:\n{y_test.value_counts(normalize=True)}")



# === Step 4: Apply IQR capping ===
def cap_outliers_with_train_iqr(X_train, X_test, columns):
    X_train_capped = X_train.copy()
    X_test_capped = X_test.copy()
    
    for col in columns:
        Q1 = X_train[col].quantile(0.25)
        Q3 = X_train[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        X_train_capped[col] = np.where(X_train_capped[col] < lower, lower,
                                       np.where(X_train_capped[col] > upper, upper, X_train_capped[col]))
        X_test_capped[col] = np.where(X_test_capped[col] < lower, lower,
                                      np.where(X_test_capped[col] > upper, upper, X_test_capped[col]))

    return X_train_capped, X_test_capped


# Select specific columns
num_cols_for_iqr = ['RevolvingUtilizationOfUnsecuredLines', 'age', 'DebtRatio',
            'MonthlyIncome', 'NumberOfOpenCreditLinesAndLoans',
            'NumberRealEstateLoansOrLines',
            'NumberOfDependents']


# Apply
X_train, X_test = cap_outliers_with_train_iqr(X_train, X_test, num_cols_for_iqr)


# === Step 5: Define pipelines ===
pipelines = {
    'Logistic Regression': Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(max_iter=1000, random_state=42))
    ]),
    'Random Forest': Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(random_state=42))
    ]),
    'XGBoost': Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42))
    ]),
    'LightGBM': Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LGBMClassifier(verbose=-1, random_state=42))
    ]),
    'AdaBoost': Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', AdaBoostClassifier(random_state=42))
    ]),
    'ANN': Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42))
    ])
}



from sklearn import set_config
set_config(display='diagram')
pipelines


def evaluate_models(pipelines, X_train, y_train, X_test, y_test):
    
    # Define metric functions
    metrics = {
        'Accuracy': accuracy_score,
        'Precision': lambda y_true, y_pred: precision_score(y_true, y_pred, average='macro'),
        'Recall': lambda y_true, y_pred: recall_score(y_true, y_pred, average='macro'),
        'F1 Score': lambda y_true, y_pred: f1_score(y_true, y_pred, average='macro'),
        'AUC-ROC': None  # Handled separately
    }

    results = {metric: {} for metric in metrics}

    # Binarize for ROC AUC
    lb = LabelBinarizer()
    lb.fit(y_train)

    for model_name, pipeline in pipelines.items():
        print(f'\nğŸ“Œ Evaluating: {model_name}')
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        # Print accuracy and classification report
        print(f'ğŸ”� Accuracy: {accuracy_score(y_test, y_pred):.2f}')
        print(classification_report(y_test, y_pred))

        # Plot confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap='Blues')
        plt.title(f'{model_name} - Confusion Matrix')
        plt.grid(False)
        plt.show()

        # Standard metrics
        for metric_name, metric_func in metrics.items():
            if metric_name != 'AUC-ROC':
                results[metric_name][model_name] = metric_func(y_test, y_pred)

        # AUC-ROC (if applicable)
        if hasattr(pipeline.named_steps['classifier'], "predict_proba"):
            y_proba = pipeline.predict_proba(X_test)
            y_test_bin = lb.transform(y_test)

            if y_proba.shape[1] == 2:
                auc = roc_auc_score(y_test_bin, y_proba[:, 1])
            else:
                auc = roc_auc_score(y_test_bin, y_proba, average='weighted', multi_class='ovr')

            results['AUC-ROC'][model_name] = auc
        else:
            results['AUC-ROC'][model_name] = None
    
        print('----------------------------------------------------------------------------')
    
    return results



results = evaluate_models(pipelines, X_train, y_train, X_test, y_test)


results_df = pd.DataFrame(results)
results_df = results_df.T 
results_df.round(3)


# Load the train dataset
df = pd.read_csv('/kaggle/input/GiveMeSomeCredit/cs-training.csv')
df.drop('Unnamed: 0', axis=1, inplace=True)  # Drop index column

# Rename target for readability
df.rename(columns={"SeriousDlqin2yrs": "Default"}, inplace=True)

df


# === Step 1: Load and copy the training data ===
df_original = df.copy() 

# === Step 2: Separate features and target ===
X = df_original.drop(columns='Default')
y = df_original['Default']


# === Step 3: Train-test split ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)


# === Step 4: Apply KNN Imputer ===
imputer = KNNImputer(n_neighbors=5)

# Fit on training data and transform both train and test
X_train_imputed = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)
X_test_imputed = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)


# === Step 5: Apply IQR capping ===
X_train_capped, X_test_capped = cap_outliers_with_train_iqr(X_train_imputed, X_test_imputed, num_cols_for_iqr)


# === Step 6: Apply SMOTE on training data ===

# Check class distribution before
print("Before SMOTE:", Counter(y_train))

smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_capped, y_train)

# Check class distribution after
print("After SMOTE:", Counter(y_train_resampled))

# Visualization before SMOTE operation
plt.figure(figsize= (6, 4))
sns.countplot(x=y_train, palette='Set2')
plt.title('Class Distribution Before Upsampling')
plt.xlabel("Default")
plt.ylabel("Count")
plt.show()

# Visualization after SMOTE operation
plt.figure(figsize = (6, 4))
sns.countplot(x=y_train_resampled, palette='Set2')
plt.title("Class Distribution After Upsampling")
plt.xlabel("Default")
plt.ylabel("Count")
plt.show()


# === Step 7: Define pipelines ===
pipelines = {
    'Logistic Regression': Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(max_iter=1000, random_state=42))
    ]),
    'Random Forest': Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(random_state=42))
    ]),
    'XGBoost': Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42))
    ]),
    'LightGBM': Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LGBMClassifier(verbose=-1, random_state=42))
    ]),
    'AdaBoost': Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', AdaBoostClassifier(random_state=42))
    ]),
    'ANN': Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42))
    ])
}



results = evaluate_models(pipelines, X_train_resampled, y_train_resampled, X_test_capped, y_test)


results_df_experiment_2a = pd.DataFrame(results)
results_df_experiment_2a = results_df_experiment_2a.T 
results_df_experiment_2a.round(3)





# === Step 5: Apply Undersampling operation ===

from imblearn.under_sampling import TomekLinks
from collections import Counter

# Check class distribution before
print("Before Tomek Links:", Counter(y_train))

# Initialize Tomek Links
tl = TomekLinks(sampling_strategy='auto')  # 'auto' removes majority class examples

# Apply on training data
X_train_resampled, y_train_resampled = tl.fit_resample(X_train_capped, y_train)

# Check class distribution after
print("After Tomek Links:", Counter(y_train_resampled))

# Visualization before Tomek operation
plt.figure(figsize= (6, 4))
sns.countplot(x=y_train, palette='Set2')
plt.title('Class Distribution Before Downsampling')
plt.xlabel("Default")
plt.ylabel("Count")
plt.show()

# Visualization after class balancing
plt.figure(figsize = (6, 4))
sns.countplot(x=y_train_resampled, palette='Set2')
plt.title("Class Distribution After Downsampling")
plt.xlabel("Default")
plt.ylabel("Count")
plt.show()


results = evaluate_models(pipelines, X_train_resampled, y_train_resampled, X_test_capped, y_test)


results_df_experiment_2b = pd.DataFrame(results)
results_df_experiment_2b = results_df_experiment_2b.T 
results_df_experiment_2b.round(3)


# === Step 4: Apply SMOTETomek on the training data ===

from imblearn.combine import SMOTETomek

# Check class distribution before
print("Before SMOTETomek:", Counter(y_train))

smote_tomek = SMOTETomek(n_jobs=-1, random_state=42)
X_train_resampled, y_train_resampled = smote_tomek._fit_resample(X_train_capped, y_train)

# Check class distribution after
print("After SMOTETomek:", Counter(y_train_resampled))


# Visualization before class balancing
plt.figure(figsize= (6, 4))
sns.countplot(x=y_train, palette='Set2')
plt.title('Class Distribution Before Balancing')
plt.xlabel("Default")
plt.ylabel("Count")
plt.show()


# Visualization after class balancing
plt.figure(figsize = (6, 4))
sns.countplot(x=y_train_resampled, palette='Set2')
plt.title("Class Distribution After SMOTETomek")
plt.xlabel("Default")
plt.ylabel("Count")
plt.show()


results = evaluate_models(pipelines, X_train_resampled, y_train_resampled, X_test_capped, y_test)


results_df_experiment_2c = pd.DataFrame(results)
results_df_experiment_2c = results_df_experiment_2c.T 
results_df_experiment_2c.round(3)


# Load the train dataset
df = pd.read_csv('/kaggle/input/GiveMeSomeCredit/cs-training.csv')
df.drop('Unnamed: 0', axis=1, inplace=True)  # Drop index column

# Rename target for readability
df.rename(columns={"SeriousDlqin2yrs": "Default"}, inplace=True)

df


# === Step 1: Load and copy the training data ===
df_original = df.copy() 

# === Step 2: Separate features and target ===
X = df_original.drop(columns='Default')
y = df_original['Default']


# === Step 3: Train-test split ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)


# === Step 4: Apply KNN Imputer ===
imputer = KNNImputer(n_neighbors=5)

# Fit on training data and transform both train and test
X_train_imputed = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)
X_test_imputed = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)


# === Step 5: Apply IQR capping ===
X_train_capped, X_test_capped = cap_outliers_with_train_iqr(X_train_imputed, X_test_imputed, num_cols_for_iqr)


# === Step 4: Apply SMOTETomek on the training data ===

from imblearn.combine import SMOTETomek
from collections import Counter

# Check class distribution before
print("Before SMOTETomek:", Counter(y_train))

smote_tomek = SMOTETomek(n_jobs=-1, random_state=42)
X_train_resampled, y_train_resampled = smote_tomek._fit_resample(X_train_capped, y_train)

# Check class distribution after
print("After SMOTETomek:", Counter(y_train_resampled))


# Visualization before class balancing
plt.figure(figsize= (6, 4))
sns.countplot(x=y_train, palette='Set2')
plt.title('Class Distribution Before Balancing')
plt.xlabel("Default")
plt.ylabel("Count")
plt.show()


# Visualization after class balancing
plt.figure(figsize = (6, 4))
sns.countplot(x=y_train_resampled, palette='Set2')
plt.title("Class Distribution After SMOTETomek")
plt.xlabel("Default")
plt.ylabel("Count")
plt.show()


from sklearn.feature_selection import RFE
from sklearn.model_selection import GridSearchCV

# Define the base model
rf = RandomForestClassifier(random_state=42)

# Set up the RFE with estimator and a placeholder for number of features
rfe = RFE(estimator=rf)

# Define parameter grid for number of features to select
param_grid = {
    'n_features_to_select': list(range(1, 11))  # From 1 to 10 features
}

# Wrap RFE with GridSearchCV
grid_search = GridSearchCV(
    estimator=rfe,
    param_grid=param_grid,
    scoring='f1',  # You can change this to 'f1', 'roc_auc', etc.
    cv=5,
    verbose=1,
    n_jobs=-1
)

# Fit GridSearchCV to find optimal number of features
grid_search.fit(X_train_resampled, y_train_resampled)

# Get the best number of features and feature mask
best_rfe = grid_search.best_estimator_
selected_features = X_train.columns[best_rfe.support_]

# Display selected features
print(f"Best number of features: {grid_search.best_params_['n_features_to_select']}")
print("Selected Features:")
print(selected_features)


from sklearn.inspection import permutation_importance

# === Step 1: Train AdaBoost on preprocessed data ===
model = AdaBoostClassifier(random_state=42)
model.fit(X_train_resampled, y_train_resampled)

# === Step 2: Apply Permutation Importance ===
result = permutation_importance(
    model, 
    X_test_capped, 
    y_test, 
    n_repeats=30, 
    random_state=42, 
    scoring='f1',  # or 'accuracy', 'roc_auc', etc.
    n_jobs=-1
)

# === Step 3: Show feature importance ===
importance_df = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance Mean': result.importances_mean,
    'Importance Std': result.importances_std
}).sort_values(by='Importance Mean', ascending=False)

importance_df




# === Step 4: Plot the importance ===
plt.figure(figsize=(10,6))
plt.barh(importance_df['Feature'], importance_df['Importance Mean'], xerr=importance_df['Importance Std'])
plt.gca().invert_yaxis()
plt.title('Permutation Importance (AdaBoost)')
plt.xlabel('Mean Importance Score')
plt.tight_layout()
plt.show()


# First convert X_train_resampled back to DataFrame using original X_train's column names
X_train_resampled = pd.DataFrame(X_train_resampled, columns=X_train.columns)


# Define the top 8 features based on Permutation Importance
top_8_features = [
    'NumberOfTimes90DaysLate',
    'RevolvingUtilizationOfUnsecuredLines',
    'NumberOfTime30-59DaysPastDueNotWorse',
    'NumberOfTime60-89DaysPastDueNotWorse',
    'age',
    'NumberRealEstateLoansOrLines',
    'NumberOfDependents',
    'DebtRatio'
]

# Subset training and test datasets with only the top 8 features
X_train_top8 = X_train_resampled[top_8_features]
X_test_top8 = X_test_capped[top_8_features]


results = evaluate_models(pipelines, X_train_top8, y_train_resampled, X_test_top8, y_test)


results_df_experiment_3 = pd.DataFrame(results)
results_df_experiment_3 = results_df_experiment_3.T 
results_df_experiment_3.round(3)

