import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math


train_df = pd.read_csv("/kaggle/input/playground-series-s5e12/train.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e12/test.csv")


train_df.head()


test_df.head()


train_df.describe()


train_df.isna().sum() / len(train_df) * 100


for col in train_df.columns:
    print(col + ": " + str(train_df[col].nunique()))


target = 'diagnosed_diabetes'
features = [col for col in train_df.columns if col not in [target, 'id']]

# 2. Calculate grid size
cols = 3
rows = 8

fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 5))
axes = axes.flatten()

for i, col in enumerate(features):
    # For Categorical data: Bar chart
    if train_df[col].dtype == 'object' or train_df[col].nunique() < 10:
        sns.countplot(data=train_df, x=col, hue=target, ax=axes[i])
    # For Numerical data: Boxplot or Histograms
    else:
        sns.histplot(data=train_df, x=col, hue='diagnosed_diabetes', 
                 element='step', kde=True, ax=axes[i], palette='viridis')
        # sns.histplot(data=train_df, x=target, y=col, ax=axes[i])
    
    axes[i].set_title(f'{col} vs {target}')

# 3. Clean up empty subplots and layout
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()


def draw_heatmap(dataset):
    corr_matrix = dataset.corr(numeric_only=True)
    
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
    plt.figure(figsize=(14, 10))
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f", 
                cmap='coolwarm', vmin=-1, vmax=1, linewidths=0.5)
    plt.title('Correlation Matrix (Numerical Features vs Target)')
    plt.show()

draw_heatmap(train_df)


correlations = train_df.corr(numeric_only=True)['diagnosed_diabetes'].sort_values(ascending=False)
print("Correlation Strength with Diabetes Diagnosis:")
print(correlations)


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_transformations(df, original_col):
    # 1. Prepare the data
    data = df.copy()
    data['Log'] = np.log(data[original_col] + 1) # +1 avoids log(0)
    data['Reciprocal'] = 1 / (data[original_col] + 1)
    data['Sqrt'] = np.sqrt(data[original_col])
    data['Exp (1/1.2)'] = data[original_col]**(1/1.2)
    
    # 2. Setup the Grid
    list_of_transforms = [original_col, 'Log', 'Reciprocal', 'Sqrt', 'Exp (1/1.2)']
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.flatten()
    
    # 3. Loop through and Plot
    for i, title in enumerate(list_of_transforms):
        sns.histplot(data[title], kde=True, ax=axes[i], color='royalblue')
        axes[i].set_title(f'Transformation: {title}', fontsize=14, fontweight='bold')
        axes[i].set_xlabel('')
        axes[i].set_ylabel('Frequency')
        
    # Remove the empty subplot
    fig.delaxes(axes[5])
    
    plt.tight_layout()
    plt.suptitle(f'Diagnostic Analysis for: {original_col}', fontsize=20, y=1.05)
    plt.show()


plot_transformations(train_df, 'age')


plot_transformations(train_df, 'ldl_cholesterol')


plot_transformations(train_df, 'physical_activity_minutes_per_week')


CATEGORICAL = train_df.select_dtypes(include=['object']).columns.to_list()
CATEGORICAL


from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import OrdinalEncoder

# 2. Encode strings to integers (Required for the function to work)
encoder = OrdinalEncoder()
X_encoded = encoder.fit_transform(train_df[CATEGORICAL])

# 3. Calculate MI Scores
# discrete_features=True tells the model these integers represent categories
mi_scores = mutual_info_classif(X_encoded, train_df['diagnosed_diabetes'], discrete_features=True, random_state=42)

# 4. Visualize
mi_results = pd.Series(mi_scores, index=CATEGORICAL).sort_values(ascending=False)
mi_results.plot(kind='barh', title='Mutual Information Scores')
plt.show()

print(mi_results)


from scipy.stats import chi2_contingency

# Loop through categories and test against target
chi2_dict = {}
for col in CATEGORICAL:
    # Create the contingency table
    contingency = pd.crosstab(train_df[col], train_df['diagnosed_diabetes'])
    chi2, p, dof, expected = chi2_contingency(contingency)
    chi2_dict[col] = p

# Sort by P-value (Lowest is most useful)
chi2_results = pd.Series(chi2_dict).sort_values()
print("Chi-Square P-Values (Look for < 0.05):")
print(chi2_results)


USEFUL_CATEGORICAL = ['education_level', 'ethnicity', 'income_level', 'employment_status', 'gender']


for col in USEFUL_CATEGORICAL:
    print(col + ": " + str(train_df[col].nunique()))


NUMERICAL = correlations[correlations.abs() >= 0.1].index.to_list()
# NUMERICAL = correlations.index.to_list()
NUMERICAL.remove('diagnosed_diabetes')
NUMERICAL


def feature_engineering(dataset):
    # Numerical 
    df_filtered = dataset[NUMERICAL].copy()
    df_filtered['age_exp'] = df_filtered['age']**(1/1.2)
    df_filtered['physical_activity_minutes_per_week_log'] = np.log(df_filtered['physical_activity_minutes_per_week'] + 1)

    # Categorical Logic
    income_map = {'Low': 0, 'Lower-Middle': 1, 'Middle': 2, 'Upper-Middle': 3, 'High': 4}
    edu_map = {'No formal': 0, 'Highschool': 1, 'Graduate': 2, 'Postgraduate': 3}

    # df_filtered['income_level'] = dataset['income_level'].map(income_map)
    # df_filtered['education_level'] = dataset['education_level'].map(edu_map)

    # cols_to_ohe = ['ethnicity', 'employment_status', 'gender']
    # cols_to_ohe = ['ethnicity', 'gender']
    # df_ohe = pd.get_dummies(dataset[cols_to_ohe], prefix=cols_to_ohe, dtype=int)

    # Interaction
    # df_filtered['age_exp_X_bmi'] = df_filtered['age_exp'] * df_filtered['bmi']
    df_filtered['age_exp_X_bmi'] = (df_filtered['age_exp'] - df_filtered['age_exp'].mean()) * (df_filtered['bmi'] - df_filtered['bmi'].mean())
    # df_filtered['bmi_X_physical_activity'] = df_filtered['bmi'] * df_filtered['physical_activity_minutes_per_week']
    df_filtered['bmi_X_physical_activity'] = (df_filtered['age_exp'] - df_filtered['age_exp'].mean()) * (df_filtered['physical_activity_minutes_per_week'] - df_filtered['physical_activity_minutes_per_week'].mean())
    # df_filtered['income_X_education'] = df_filtered['income_level'] * df_filtered['education_level']

    # Final cleanup
    # X = pd.concat([df_filtered, df_ohe], axis=1)
    X = df_filtered
    X = X.drop(columns=['age', 'physical_activity_minutes_per_week'], errors='ignore')
    
    target = 'diagnosed_diabetes'
    if target in dataset.columns:
        return X, dataset[target]
    return X


def test_feature_engineering():
    tmp_X, tmp_y = feature_engineering(train_df)
    tmp_df = pd.concat([tmp_X, pd.DataFrame(tmp_y, columns=['diagnosed_diabetes'])], axis=1)
    draw_heatmap(tmp_df)

test_feature_engineering()


from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, log_loss, classification_report, confusion_matrix, roc_auc_score


kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(len(train_df))
best_iterations = []


X = train_df.iloc[:, :-1]
y = train_df.iloc[:, -1]

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    train_fold = train_df.iloc[train_idx]
    val_fold   = train_df.iloc[val_idx]

    X_train, y_train = feature_engineering(train_fold)
    X_val, y_val     = feature_engineering(val_fold)

    X_val = X_val.reindex(columns=X_train.columns, fill_value=0)

    model = XGBClassifier(
        n_estimators=5000,
        learning_rate=0.01,
        max_depth=5,
        random_state=42,
        scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),   
        enable_categorical=True,
        
        reg_lambda=10,
        reg_alpha=5,
        
        eval_metric="logloss",
        early_stopping_rounds=50
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )

    best_iterations.append(model.best_iteration)
    # Save OOF predictions
    probs = model.predict_proba(X_val)[:, 1]
    oof_preds[val_idx] = probs

    preds = model.predict(X_val)

    # Calculate fold-specific scores
    fold_auc = roc_auc_score(y_val, probs)
    fold_acc = accuracy_score(y_val, preds)
    fold_ll  = log_loss(y_val, probs)
    
    print(f"Fold {fold+1} AUC: {fold_auc:.4f} | Best Iteration: {model.best_iteration} | Acc: {fold_acc:.4f} | LogLoss: {fold_ll:.4f}")

# Final CV Score (Outside the loop!)
total_cv_auc = roc_auc_score(y, oof_preds)
print(f"\nOverall CV ROC-AUC: {total_cv_auc:.4f}")



from sklearn.metrics import ConfusionMatrixDisplay

oof_labels = (oof_preds >= 0.5).astype(int)

total_cv_auc = roc_auc_score(y, oof_preds)
total_cv_acc = accuracy_score(y, oof_labels)
total_cv_ll  = log_loss(y, oof_preds)

print(f"\n{'='*30}")
print(f"OVERALL CV RESULTS")
print(f"{'='*30}")
print(f"ROC-AUC:  {total_cv_auc:.4f}")
print(f"Accuracy: {total_cv_acc:.4f}")
print(f"Log Loss: {total_cv_ll:.4f}")

print(classification_report(train_df[target], oof_labels))

fig, ax = plt.subplots(figsize=(8, 6))
ConfusionMatrixDisplay.from_predictions(train_df[target], oof_labels, cmap='Blues', ax=ax)
plt.title("OOF Confusion Matrix")
plt.show()


X, y = feature_engineering(train_df)
avg_n_estimators = int(np.mean(best_iterations))

final_model = XGBClassifier(
    n_estimators=avg_n_estimators,
    learning_rate=0.01,
    max_depth=5,
    random_state=42,
    scale_pos_weight=(y == 0).sum() / (y == 1).sum(),
    enable_categorical=True,
    
    reg_lambda=10,
    reg_alpha=5,
    
    eval_metric="logloss"
)

final_model.fit(
    X, y
)


# If using a Tree-based model (RandomForest, XGBoost, etc.)
import matplotlib.pyplot as plt

# 1. Create a Series with feature names and their importance scores
feat_importances = pd.Series(final_model.feature_importances_, index=X.columns)

# 2. Sort the values (highest at the top)
feat_importances = feat_importances.sort_values(ascending=True)

# 3. Plot it
plt.figure(figsize=(10, 4))
feat_importances.plot(kind='barh')
plt.xscale('log')
plt.title('Feature Importances (Log scale)')
plt.xlabel('Relative Importance')
plt.tight_layout()
plt.show()


test_filtered = feature_engineering(test_df)
test_filtered.head()


preds = final_model.predict(test_filtered)
preds


submission = pd.DataFrame({
    'id': test_df['id'],
    'diagnosed_diabetes': preds
})


submission.to_csv('submission.csv', index=False)

