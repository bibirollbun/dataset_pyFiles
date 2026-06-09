import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import StackingClassifier
import warnings
warnings.filterwarnings('ignore')


train_df = pd.read_csv('/kaggle/input/playground-series-s5e11/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e11/test.csv')

print("Train shape:", train_df.shape)
print("Test shape:", test_df.shape)
print("\nTrain columns:", train_df.columns.tolist())
print("\nMissing values in train:", train_df.isnull().sum().sum())
print("Missing values in test:", test_df.isnull().sum().sum())

print("\nFirst few rows:")
print(train_df.head())


def create_features(df):
    df = df.copy()
    
    # Creating new features
    df['income_to_loan_ratio'] = df['annual_income'] / (df['loan_amount'] + 1)
    df['debt_burden'] = df['annual_income'] * df['debt_to_income_ratio']
    df['affordability_ratio'] = (df['annual_income'] / 12) / (df['loan_amount'] * df['interest_rate'] / 1200 + 1)
    df['credit_income_ratio'] = df['credit_score'] / df['annual_income']
    
    # Creating risk score based on multiple factors
    df['risk_score'] = (
        df['debt_to_income_ratio'] * 0.3 + 
        (800 - df['credit_score']) / 800 * 0.3 + 
        df['interest_rate'] / 25 * 0.2 +
        (df['loan_amount'] / df['annual_income']) * 0.2
    )
    
    # Extracting subgrade as numerical feature
    if 'grade_subgrade' in df.columns:
        df['grade'] = df['grade_subgrade'].str[0]
        df['subgrade_num'] = df['grade_subgrade'].str[1].astype(int)
    
    # Creating employment stability feature
    employment_mapping = {
        'Unemployed': 0,
        'Student': 1,
        'Self-employed': 2,
        'Employed': 3,
        'Retired': 2
    }
    df['employment_stability'] = df['employment_status'].map(employment_mapping)
    
    # Education level encoding
    education_mapping = {
        'High School': 1,
        'Other': 2,
        'Bachelor\'s': 3,
        'Master\'s': 4,
        'PhD': 5
    }
    df['education_num'] = df['education_level'].map(education_mapping)
    
    return df

# Applying feature engineering
train_df_eng = create_features(train_df)
test_df_eng = create_features(test_df)

print("New features created:")
new_features = ['income_to_loan_ratio', 'debt_burden', 'affordability_ratio', 
               'credit_income_ratio', 'risk_score', 'employment_stability', 'education_num']
print(new_features)


def preprocess_data(train_df, test_df):
    # Selecting features for modeling
    feature_columns = [
        'annual_income', 'debt_to_income_ratio', 'credit_score', 'loan_amount', 'interest_rate',
        'income_to_loan_ratio', 'debt_burden', 'affordability_ratio', 'credit_income_ratio', 
        'risk_score', 'employment_stability', 'education_num', 'subgrade_num'
    ]
    
    categorical_cols = ['gender', 'marital_status', 'loan_purpose', 'grade']
    
    # Combining train and test for consistent encoding
    combined = pd.concat([train_df, test_df], axis=0)
    
    # Labeling encode categorical variables
    label_encoders = {}
    for col in categorical_cols:
        if col in combined.columns:
            le = LabelEncoder()
            combined[col] = le.fit_transform(combined[col].astype(str))
            label_encoders[col] = le
    
    # Spliting back
    train_processed = combined.iloc[:len(train_df)].copy()
    test_processed = combined.iloc[len(train_df):].copy()
    
    # Preparing final feature set
    all_features = feature_columns + categorical_cols
    
    X_train = train_processed[all_features]
    y_train = train_processed['loan_paid_back']
    X_test = test_processed[all_features]
    
    # Scale numerical features
    scaler = StandardScaler()
    numerical_features_to_scale = [f for f in all_features if f not in categorical_cols]
    X_train[numerical_features_to_scale] = scaler.fit_transform(X_train[numerical_features_to_scale])
    X_test[numerical_features_to_scale] = scaler.transform(X_test[numerical_features_to_scale])
    
    return X_train, y_train, X_test, scaler, label_encoders

X_train, y_train, X_test, scaler, label_encoders = preprocess_data(train_df_eng, test_df_eng)

print("Training features shape:", X_train.shape)
print("Test features shape:", X_test.shape)
print("Features used:", X_train.columns.tolist())


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

# Splitting data for validation
X = X_train
y = y_train

X_train_split, X_val, y_train_split, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training samples: {X_train_split.shape[0]}")
print(f"Validation samples: {X_val.shape[0]}")

# Simplified models with faster training
xgb_model = XGBClassifier(
    n_estimators=100, 
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric='logloss',
    n_jobs=-1  # Use all cores
)

cat_model = CatBoostClassifier(
    iterations=100, 
    depth=6,
    learning_rate=0.1,
    random_state=42,
    verbose=False,
    thread_count=-1  # Use all cores
)

rf_model = RandomForestClassifier(
    n_estimators=50, 
    max_depth=10,
    random_state=42,
    n_jobs=-1  # Use all cores
)

print("Training base models...")

print("Training XGBoost...", end=" ")
xgb_model.fit(X_train_split, y_train_split)
xgb_val_pred = xgb_model.predict_proba(X_val)[:, 1]
print("✓")

print("Training CatBoost...", end=" ")
cat_model.fit(X_train_split, y_train_split)
cat_val_pred = cat_model.predict_proba(X_val)[:, 1]
print("✓")

print("Training Random Forest...", end=" ")
rf_model.fit(X_train_split, y_train_split)
rf_val_pred = rf_model.predict_proba(X_val)[:, 1]
print("✓")

# Creating meta-features for stacking
meta_features_val = np.column_stack([xgb_val_pred, cat_val_pred, rf_val_pred])

# Simple meta-model (Logistic Regression)
meta_model = LogisticRegression(random_state=42, C=1.0)
meta_model.fit(meta_features_val, y_val)

print("Meta-model trained successfully!")

# Evaluating base models
print("\nBase Model Performance (Validation AUC):")
print(f"XGBoost: {roc_auc_score(y_val, xgb_val_pred):.4f}")
print(f"CatBoost: {roc_auc_score(y_val, cat_val_pred):.4f}")
print(f"Random Forest: {roc_auc_score(y_val, rf_val_pred):.4f}")

# Ensemble performance
ensemble_val_pred = meta_model.predict_proba(meta_features_val)[:, 1]
print(f"Stacking Ensemble: {roc_auc_score(y_val, ensemble_val_pred):.4f}")


# Retraining base models on full training data
print("\nRetraining models on full training data...")

print("Training XGBoost on full data...", end=" ")
xgb_model.fit(X_train, y_train)
print("✓")

print("Training CatBoost on full data...", end=" ")
cat_model.fit(X_train, y_train)
print("✓")

print("Training Random Forest on full data...", end=" ")
rf_model.fit(X_train, y_train)
print("✓")

print("\nGenerating test predictions...")

xgb_test_pred = xgb_model.predict_proba(X_test)[:, 1]
cat_test_pred = cat_model.predict_proba(X_test)[:, 1]
rf_test_pred = rf_model.predict_proba(X_test)[:, 1]

# Creating meta-features for test set
meta_features_test = np.column_stack([xgb_test_pred, cat_test_pred, rf_test_pred])

# Final ensemble predictions
final_predictions = meta_model.predict_proba(meta_features_test)[:, 1]

print("All predictions generated!")


# -------------------------------
# CatBoost feature importance
# -------------------------------
cat_importance = pd.DataFrame({
    'Feature': X.columns,
    'CatBoost': cat_model.get_feature_importance()
})
cat_importance['CatBoost'] = cat_importance['CatBoost'] / cat_importance['CatBoost'].max()

# -------------------------------
# XGBoost feature importance
# -------------------------------
xgb_importance = pd.DataFrame({
    'Feature': X.columns,
    'XGBoost': xgb_model.feature_importances_
})
xgb_importance['XGBoost'] = xgb_importance['XGBoost'] / xgb_importance['XGBoost'].max()

# -------------------------------
# Combine CatBoost and XGBoost
# -------------------------------
combined_importance = pd.merge(cat_importance, xgb_importance, on='Feature')
combined_importance['Average'] = (combined_importance['CatBoost'] + combined_importance['XGBoost']) / 2
combined_importance = combined_importance.sort_values('Average', ascending=False)

# -------------------------------
# Plot Top 15 Features
# -------------------------------
plt.figure(figsize=(12, 8))
sns.barplot(
    data=combined_importance.head(15), 
    x='Average', 
    y='Feature', 
    palette='viridis', 
    edgecolor='black'
)
plt.title('Top 15 Feature Importance - Ensemble (CatBoost + XGBoost)', fontsize=16, weight='bold')
plt.xlabel('Normalized Importance Score', fontsize=12)
plt.ylabel('')
plt.grid(axis='x', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

# -------------------------------
# Display Top 10 Features
# -------------------------------
print("Ensemble Feature Importance (CatBoost + XGBoost):")
display(combined_importance.head(10))


from sklearn import tree
from matplotlib import pyplot as plt
import textwrap


est = rf_model.estimators_[4]  


plt.figure(figsize=(20, 10))
tree.plot_tree(est, feature_names=X.columns, max_depth=3, filled=True, fontsize=10)
plt.show()


r_text = tree.export_text(est, feature_names=list(X.columns), max_depth=3)
print(r_text)



from xgboost import to_graphviz
dot = to_graphviz(xgb_model, num_trees=0)  # raíz del árbol 0
dot.render('xgb_tree', format='png', cleanup=True)  # guarda imagen legible



# import shap

# explainer = shap.Explainer(xgb_model)  
# shap_values = explainer(X)  # X es tu dataset de entrada

# shap.summary_plot(shap_values, X)

# shap.plots.waterfall(shap_values[0])



# Checking target distribution
plt.figure(figsize=(8, 6))
sns.countplot(x='loan_paid_back', data=train_df)
plt.title('Target Variable Distribution')
plt.show()

print("Target distribution:")
print(train_df['loan_paid_back'].value_counts(normalize=True))

# Checking correlation with numerical features
numerical_features = ['annual_income', 'debt_to_income_ratio', 'credit_score', 'loan_amount', 'interest_rate']
plt.figure(figsize=(10, 8))
corr_matrix = train_df[numerical_features + ['loan_paid_back']].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Matrix')
plt.show()

# Analyzing categorical features
categorical_features = ['gender', 'marital_status', 'education_level', 'employment_status', 'loan_purpose', 'grade_subgrade']
for col in categorical_features:
    print(f"\n{col} distribution:")
    print(train_df[col].value_counts())


plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
loan_paid_counts = train_df['loan_paid_back'].value_counts()
colors = ['#ff6b6b', '#51cf66']
plt.pie(loan_paid_counts.values, labels=['Not Paid (0)', 'Paid (1)'], autopct='%1.1f%%', 
        colors=colors, startangle=90)
plt.title('Loan Repayment Distribution')

plt.subplot(1, 3, 2)
# Distribution by loan purpose
purpose_paid = train_df.groupby('loan_purpose')['loan_paid_back'].mean().sort_values(ascending=False)
sns.barplot(y=purpose_paid.index, x=purpose_paid.values, palette='viridis')
plt.title('Repayment Rate by Loan Purpose')
plt.xlabel('Repayment Rate')

plt.subplot(1, 3, 3)
# Distribution by employment status
employment_paid = train_df.groupby('employment_status')['loan_paid_back'].mean().sort_values(ascending=False)
sns.barplot(y=employment_paid.index, x=employment_paid.values, palette='rocket')
plt.title('Repayment Rate by Employment Status')
plt.xlabel('Repayment Rate')

plt.tight_layout()
plt.show()


plt.figure(figsize=(16, 12))

plt.subplot(2, 3, 1)
sns.boxplot(x='loan_paid_back', y='credit_score', data=train_df, palette=colors)
plt.title('Credit Score vs Loan Repayment')

plt.subplot(2, 3, 2)
sns.boxplot(x='loan_paid_back', y='annual_income', data=train_df, palette=colors)
plt.title('Annual Income vs Loan Repayment')

plt.subplot(2, 3, 3)
sns.boxplot(x='loan_paid_back', y='debt_to_income_ratio', data=train_df, palette=colors)
plt.title('Debt-to-Income Ratio vs Loan Repayment')

plt.subplot(2, 3, 4)
sns.boxplot(x='loan_paid_back', y='loan_amount', data=train_df, palette=colors)
plt.title('Loan Amount vs Loan Repayment')

plt.subplot(2, 3, 5)
sns.boxplot(x='loan_paid_back', y='interest_rate', data=train_df, palette=colors)
plt.title('Interest Rate vs Loan Repayment')

plt.tight_layout()
plt.show()


plt.figure(figsize=(16, 6))

# risk segments based on credit score and debt ratio
train_df_eng['risk_segment'] = pd.cut(train_df_eng['credit_score'], bins=[0, 600, 700, 800, 850], 
                                    labels=['Poor', 'Fair', 'Good', 'Excellent'])
train_df_eng['debt_level'] = pd.cut(train_df_eng['debt_to_income_ratio'], bins=[0, 0.1, 0.2, 0.3, 1],
                                  labels=['Low', 'Medium', 'High', 'Very High'])

plt.subplot(1, 2, 1)
risk_repayment = train_df_eng.groupby('risk_segment')['loan_paid_back'].mean()
sns.barplot(x=risk_repayment.index, y=risk_repayment.values, palette='RdYlGn_r')
plt.title('Repayment Rate by Credit Score Segment')
plt.ylabel('Repayment Rate')
plt.xticks(rotation=45)

plt.subplot(1, 2, 2)
debt_repayment = train_df_eng.groupby('debt_level')['loan_paid_back'].mean()
sns.barplot(x=debt_repayment.index, y=debt_repayment.values, palette='RdYlGn_r')
plt.title('Repayment Rate by Debt Level')
plt.ylabel('Repayment Rate')
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

# Cleaning up temporary columns
train_df_eng.drop(['risk_segment', 'debt_level'], axis=1, inplace=True, errors='ignore')


plt.figure(figsize=(15, 10))

plt.subplot(2, 2, 1)
sns.violinplot(x='loan_paid_back', y='credit_score', data=train_df, palette=['red', 'black'])
plt.title('Credit Score Distribution by Repayment Status')

plt.subplot(2, 2, 2)
sns.violinplot(x='loan_paid_back', y='annual_income', data=train_df, palette=['red', 'black'])
plt.title('Income Distribution by Repayment Status')

plt.subplot(2, 2, 3)
sns.violinplot(x='loan_paid_back', y='debt_to_income_ratio', data=train_df, palette=['red', 'black'])
plt.title('Debt Ratio Distribution by Repayment Status')

plt.subplot(2, 2, 4)
sns.violinplot(x='loan_paid_back', y='interest_rate', data=train_df, palette=['red', 'black'])
plt.title('Interest Rate Distribution by Repayment Status')

plt.tight_layout()
plt.show()


sample_df = train_df.sample(1000, random_state=42) 

# pairplot with hue
sns.pairplot(sample_df[['annual_income', 'credit_score', 'debt_to_income_ratio', 
                       'loan_amount', 'loan_paid_back']], 
             hue='loan_paid_back', palette=['red', 'black'], 
             diag_kind='kde', plot_kws={'alpha': 0.6})
plt.suptitle('Feature Relationships Colored by Repayment Status', y=1.02)
plt.show()


plt.figure(figsize=(15, 10))

plt.subplot(2, 2, 1)
sns.kdeplot(data=train_df, x='credit_score', hue='loan_paid_back', palette=['red', 'black'], fill=True, alpha=0.6)
plt.title('Credit Score Distribution by Repayment Status')
plt.axvline(x=670, color='black', linestyle='--', alpha=0.5, label='Good Credit Threshold')
plt.legend()

plt.subplot(2, 2, 2)
sns.kdeplot(data=train_df, x='annual_income', hue='loan_paid_back', palette=['red', 'black'], fill=True, alpha=0.6)
plt.title('Income Distribution by Repayment Status')
plt.xlabel('Annual Income ($)')

plt.subplot(2, 2, 3)
sns.kdeplot(data=train_df, x='debt_to_income_ratio', hue='loan_paid_back', palette=['red', 'black'], fill=True, alpha=0.6)
plt.title('Debt-to-Income Ratio Distribution by Repayment Status')

plt.subplot(2, 2, 4)
sns.kdeplot(data=train_df, x='interest_rate', hue='loan_paid_back', palette=['red', 'black'], fill=True, alpha=0.6)
plt.title('Interest Rate Distribution by Repayment Status')

plt.tight_layout()
plt.show()


plt.figure(figsize=(12, 6))

# creating employment status analysis
employment_analysis = train_df.groupby('employment_status').agg({
    'loan_paid_back': ['count', 'mean']
}).round(3)
employment_analysis.columns = ['total_loans', 'repayment_rate']

# sorting by repayment rate
employment_analysis = employment_analysis.sort_values('repayment_rate')

# horizontal stacked bar
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Bar 1: Total loans
ax1.barh(employment_analysis.index, employment_analysis['total_loans'], color='skyblue', alpha=0.7)
ax1.set_xlabel('Number of Loans')
ax1.set_title('Loan Volume by Employment Status')
ax1.grid(alpha=0.3, axis='x')

# Bar 2: Repayment rate
colors = plt.cm.RdYlGn(employment_analysis['repayment_rate'])
ax2.barh(employment_analysis.index, employment_analysis['repayment_rate'], color=colors, alpha=0.7)
ax2.set_xlabel('Repayment Rate')
ax2.set_title('Repayment Performance by Employment Status')
ax2.set_xlim(0, 1)
ax2.grid(alpha=0.3, axis='x')

# adding value labels
for i, (idx, row) in enumerate(employment_analysis.iterrows()):
    ax1.text(row['total_loans'] + 10, i, f"{row['total_loans']}", va='center')
    ax2.text(row['repayment_rate'] + 0.02, i, f"{row['repayment_rate']:.1%}", va='center')

plt.tight_layout()
plt.show()


# submission file
submission_df = pd.DataFrame({
    'id': test_df['id'],
    'loan_paid_back': final_predictions,
})

print("Prediction Summary:")
print(f"Mean probability: {final_predictions.mean():.4f}")

submission_df.to_csv('/kaggle/working/submission.csv', index=False)

print(submission_df.head())

