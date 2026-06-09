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


# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# ML libraries
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# Set display options
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("âœ… All libraries imported successfully!")


# Load the datasets
data_path = r"/kaggle/input/playground-series-s5e11/"

# Since train.csv is large, we'll use chunking to load it
print("Loading training data...")
train_df = pd.read_csv(f"{data_path}//train.csv")
print(f"âœ… Training data loaded: {train_df.shape}")

print("\nLoading test data...")
test_df = pd.read_csv(f"{data_path}//test.csv")
print(f"âœ… Test data loaded: {test_df.shape}")

print("\nLoading sample submission...")
sample_sub = pd.read_csv(f"{data_path}//sample_submission.csv")
print(f"âœ… Sample submission loaded: {sample_sub.shape}")


# Display basic information
print("=" * 80)
print("TRAINING DATA OVERVIEW")
print("=" * 80)
print(f"\nShape: {train_df.shape}")
print(f"\nFirst few rows:")
display(train_df.head(10))

print("\n" + "=" * 80)
print("DATA TYPES AND MISSING VALUES")
print("=" * 80)
info_df = pd.DataFrame({
    'Data Type': train_df.dtypes,
    'Non-Null Count': train_df.count(),
    'Null Count': train_df.isnull().sum(),
    'Null Percentage': (train_df.isnull().sum() / len(train_df) * 100).round(2)
})
display(info_df)

print("\n" + "=" * 80)
print("STATISTICAL SUMMARY")
print("=" * 80)
display(train_df.describe())


# Check target variable distribution
print("=" * 80)
print("TARGET VARIABLE DISTRIBUTION")
print("=" * 80)
target_dist = train_df['loan_paid_back'].value_counts()
target_pct = train_df['loan_paid_back'].value_counts(normalize=True) * 100

print(f"\nLoan Paid Back Distribution:")
print(f"  0 (Not Paid): {target_dist[0]:,} ({target_pct[0]:.2f}%)")
print(f"  1 (Paid): {target_dist[1]:,} ({target_pct[1]:.2f}%)")

# Visualize target distribution
fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=('Count Distribution', 'Percentage Distribution'),
    specs=[[{'type': 'bar'}, {'type': 'pie'}]]
)

fig.add_trace(
    go.Bar(x=['Not Paid', 'Paid'], y=target_dist.values, 
           marker_color=['#EF553B', '#00CC96'],
           text=target_dist.values, textposition='auto'),
    row=1, col=1
)

fig.add_trace(
    go.Pie(labels=['Not Paid', 'Paid'], values=target_dist.values,
           marker_colors=['#EF553B', '#00CC96'],
           textinfo='label+percent'),
    row=1, col=2
)

fig.update_layout(title_text="Loan Payback Distribution", height=400, showlegend=False)
fig.show()


# Analyze numerical features
numerical_cols = ['annual_income', 'debt_to_income_ratio', 'credit_score', 'loan_amount', 'interest_rate']
categorical_cols = ['gender', 'marital_status', 'education_level', 'employment_status', 'loan_purpose', 'grade_subgrade']

# Distribution of numerical features
fig = make_subplots(
    rows=2, cols=3,
    subplot_titles=numerical_cols,
    vertical_spacing=0.12,
    horizontal_spacing=0.1
)

for idx, col in enumerate(numerical_cols):
    row = idx // 3 + 1
    col_idx = idx % 3 + 1
    
    fig.add_trace(
        go.Histogram(x=train_df[col], name=col, nbinsx=50, 
                     marker_color='lightblue', opacity=0.7),
        row=row, col=col_idx
    )
    
    fig.update_xaxes(title_text=col, row=row, col=col_idx)
    fig.update_yaxes(title_text="Frequency", row=row, col=col_idx)

fig.update_layout(height=600, title_text="Distribution of Numerical Features", showlegend=False)
fig.show()


# Box plots to detect outliers and compare by target
fig = make_subplots(
    rows=2, cols=3,
    subplot_titles=numerical_cols,
    vertical_spacing=0.12,
    horizontal_spacing=0.1
)

for idx, col in enumerate(numerical_cols):
    row = idx // 3 + 1
    col_idx = idx % 3 + 1
    
    for target_val in [0, 1]:
        fig.add_trace(
            go.Box(y=train_df[train_df['loan_paid_back'] == target_val][col],
                   name=f'Paid={target_val}',
                   marker_color='#EF553B' if target_val == 0 else '#00CC96',
                   showlegend=(idx == 0)),
            row=row, col=col_idx
        )
    
    fig.update_yaxes(title_text=col, row=row, col=col_idx)

fig.update_layout(height=600, title_text="Numerical Features by Loan Payback Status")
fig.show()


# Correlation analysis
print("=" * 80)
print("CORRELATION ANALYSIS")
print("=" * 80)

# Calculate correlation matrix
corr_data = train_df[numerical_cols + ['loan_paid_back']].corr()

# Create interactive heatmap
fig = go.Figure(data=go.Heatmap(
    z=corr_data.values,
    x=corr_data.columns,
    y=corr_data.columns,
    colorscale='RdBu',
    zmid=0,
    text=corr_data.values.round(2),
    texttemplate='%{text}',
    textfont={"size": 10},
    colorbar=dict(title="Correlation")
))

fig.update_layout(
    title="Correlation Heatmap of Numerical Features",
    height=600,
    width=800
)
fig.show()

# Show correlations with target
print("\nCorrelations with Target Variable (loan_paid_back):")
target_corr = corr_data['loan_paid_back'].sort_values(ascending=False)
for feature, corr_val in target_corr.items():
    if feature != 'loan_paid_back':
        print(f"  {feature:25s}: {corr_val:7.4f}")


# Analyze categorical features
print("=" * 80)
print("CATEGORICAL FEATURES ANALYSIS")
print("=" * 80)

for col in categorical_cols:
    print(f"\n{col.upper()}:")
    value_counts = train_df[col].value_counts()
    print(value_counts.head(10))
    
# Create visualizations for categorical features
fig = make_subplots(
    rows=2, cols=3,
    subplot_titles=categorical_cols,
    specs=[[{'type': 'bar'}, {'type': 'bar'}, {'type': 'bar'}],
           [{'type': 'bar'}, {'type': 'bar'}, {'type': 'bar'}]],
    vertical_spacing=0.15,
    horizontal_spacing=0.1
)

for idx, col in enumerate(categorical_cols):
    row = idx // 3 + 1
    col_idx = idx % 3 + 1
    
    value_counts = train_df[col].value_counts().head(10)
    
    fig.add_trace(
        go.Bar(x=value_counts.index, y=value_counts.values,
               marker_color='lightcoral', name=col),
        row=row, col=col_idx
    )
    
    fig.update_xaxes(title_text=col, tickangle=-45, row=row, col=col_idx)
    fig.update_yaxes(title_text="Count", row=row, col=col_idx)

fig.update_layout(height=700, title_text="Distribution of Categorical Features", showlegend=False)
fig.show()


# Analyze loan payback rate by categorical features
print("=" * 80)
print("LOAN PAYBACK RATE BY CATEGORICAL FEATURES")
print("=" * 80)

fig = make_subplots(
    rows=2, cols=3,
    subplot_titles=categorical_cols,
    vertical_spacing=0.15,
    horizontal_spacing=0.1
)

for idx, col in enumerate(categorical_cols):
    row = idx // 3 + 1
    col_idx = idx % 3 + 1
    
    # Calculate payback rate by category
    payback_rate = train_df.groupby(col)['loan_paid_back'].agg(['mean', 'count']).reset_index()
    payback_rate = payback_rate.sort_values('mean', ascending=False).head(10)
    
    fig.add_trace(
        go.Bar(x=payback_rate[col], y=payback_rate['mean'] * 100,
               marker_color='mediumseagreen',
               text=payback_rate['mean'].apply(lambda x: f'{x*100:.1f}%'),
               textposition='auto',
               name=col),
        row=row, col=col_idx
    )
    
    fig.update_xaxes(title_text=col, tickangle=-45, row=row, col=col_idx)
    fig.update_yaxes(title_text="Payback Rate (%)", row=row, col=col_idx)

fig.update_layout(height=700, title_text="Loan Payback Rate by Categories", showlegend=False)
fig.show()

# Print detailed statistics
for col in categorical_cols:
    print(f"\n{col.upper()}:")
    stats = train_df.groupby(col)['loan_paid_back'].agg(['mean', 'count', 'sum']).reset_index()
    stats.columns = [col, 'Payback_Rate', 'Total_Loans', 'Paid_Back_Count']
    stats['Payback_Rate'] = stats['Payback_Rate'] * 100
    stats = stats.sort_values('Payback_Rate', ascending=False)
    print(stats.head(10).to_string(index=False))


# Create a copy for feature engineering
train_fe = train_df.copy()
test_fe = test_df.copy()

print("=" * 80)
print("FEATURE ENGINEERING")
print("=" * 80)

# 1. Create income to loan ratio
train_fe['income_to_loan_ratio'] = train_fe['annual_income'] / (train_fe['loan_amount'] + 1)
test_fe['income_to_loan_ratio'] = test_fe['annual_income'] / (test_fe['loan_amount'] + 1)

# 2. Create credit score bins
def categorize_credit_score(score):
    if score < 580:
        return 'Poor'
    elif score < 670:
        return 'Fair'
    elif score < 740:
        return 'Good'
    elif score < 800:
        return 'Very Good'
    else:
        return 'Excellent'

train_fe['credit_category'] = train_fe['credit_score'].apply(categorize_credit_score)
test_fe['credit_category'] = test_fe['credit_score'].apply(categorize_credit_score)

# 3. Create debt burden feature
train_fe['debt_burden'] = train_fe['debt_to_income_ratio'] * train_fe['annual_income']
test_fe['debt_burden'] = test_fe['debt_to_income_ratio'] * test_fe['annual_income']

# 4. Extract grade from grade_subgrade
train_fe['grade'] = train_fe['grade_subgrade'].str[0]
test_fe['grade'] = test_fe['grade_subgrade'].str[0]

# 5. Create high risk flag
train_fe['high_risk'] = ((train_fe['credit_score'] < 650) & 
                          (train_fe['debt_to_income_ratio'] > 0.3)).astype(int)
test_fe['high_risk'] = ((test_fe['credit_score'] < 650) & 
                         (test_fe['debt_to_income_ratio'] > 0.3)).astype(int)

# 6. Loan to income percentage
train_fe['loan_to_income_pct'] = (train_fe['loan_amount'] / train_fe['annual_income']) * 100
test_fe['loan_to_income_pct'] = (test_fe['loan_amount'] / test_fe['annual_income']) * 100

# 7. Interest rate category
train_fe['interest_category'] = pd.cut(train_fe['interest_rate'], 
                                        bins=[0, 8, 12, 16, 20], 
                                        labels=['Low', 'Medium', 'High', 'Very High'])
test_fe['interest_category'] = pd.cut(test_fe['interest_rate'], 
                                       bins=[0, 8, 12, 16, 20], 
                                       labels=['Low', 'Medium', 'High', 'Very High'])

print(f"\nâœ… Created {len(train_fe.columns) - len(train_df.columns)} new features")
print(f"\nNew features: {[col for col in train_fe.columns if col not in train_df.columns]}")
print(f"\nNew shape: {train_fe.shape}")


# Encode categorical variables
print("=" * 80)
print("ENCODING CATEGORICAL VARIABLES")
print("=" * 80)

# Separate features and target
X = train_fe.drop(['id', 'loan_paid_back'], axis=1)
y = train_fe['loan_paid_back']
X_test = test_fe.drop(['id'], axis=1)

# Identify categorical columns
cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
print(f"\nCategorical columns to encode: {cat_cols}")

# Label encoding for categorical variables
label_encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    X_test[col] = le.transform(X_test[col].astype(str))
    label_encoders[col] = le

print(f"\nâœ… Encoded {len(cat_cols)} categorical features")

# Check for any missing values
print(f"\nMissing values in training set: {X.isnull().sum().sum()}")
print(f"Missing values in test set: {X_test.isnull().sum().sum()}")

# Display final feature set
print(f"\nFinal feature count: {X.shape[1]}")
print(f"\nFeature names:")
print(X.columns.tolist())


# Split data into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("=" * 80)
print("DATA SPLIT")
print("=" * 80)
print(f"\nTraining set: {X_train.shape}")
print(f"Validation set: {X_val.shape}")
print(f"Test set: {X_test.shape}")

# Scale numerical features
scaler = StandardScaler()
num_features = ['annual_income', 'debt_to_income_ratio', 'credit_score', 'loan_amount', 
                'interest_rate', 'income_to_loan_ratio', 'debt_burden', 'loan_to_income_pct']

X_train[num_features] = scaler.fit_transform(X_train[num_features])
X_val[num_features] = scaler.transform(X_val[num_features])
X_test[num_features] = scaler.transform(X_test[num_features])

print("\nâœ… Features scaled successfully")


# Train multiple models
print("=" * 80)
print("TRAINING MULTIPLE MODELS")
print("=" * 80)

models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
    'XGBoost': XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss'),
    'LightGBM': LGBMClassifier(n_estimators=100, random_state=42, verbose=-1),
    'CatBoost': CatBoostClassifier(n_estimators=100, random_state=42, verbose=False)
}

results = {}

for name, model in models.items():
    print(f"\nTraining {name}...")
    
    # Train model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_val)
    y_pred_proba = model.predict_proba(X_val)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_val, y_pred)
    precision = precision_score(y_val, y_pred)
    recall = recall_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)
    roc_auc = roc_auc_score(y_val, y_pred_proba)
    
    results[name] = {
        'model': model,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'predictions': y_pred,
        'predictions_proba': y_pred_proba
    }
    
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    print(f"  ROC-AUC:   {roc_auc:.4f}")

print("\nâœ… All models trained successfully!")


# Model Comparison
print("=" * 80)
print("MODEL COMPARISON SUMMARY")
print("=" * 80)

comparison_df = pd.DataFrame({
    'Model': list(results.keys()),
    'Accuracy': [results[m]['accuracy'] for m in results.keys()],
    'Precision': [results[m]['precision'] for m in results.keys()],
    'Recall': [results[m]['recall'] for m in results.keys()],
    'F1-Score': [results[m]['f1_score'] for m in results.keys()],
    'ROC-AUC': [results[m]['roc_auc'] for m in results.keys()]
})

comparison_df = comparison_df.sort_values('ROC-AUC', ascending=False)
display(comparison_df)

# Visualize model comparison
fig = go.Figure()

metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
for metric in metrics:
    fig.add_trace(go.Bar(
        name=metric,
        x=comparison_df['Model'],
        y=comparison_df[metric],
        text=comparison_df[metric].apply(lambda x: f'{x:.4f}'),
        textposition='auto',
    ))

fig.update_layout(
    title="Model Performance Comparison",
    xaxis_title="Model",
    yaxis_title="Score",
    barmode='group',
    height=500,
    legend=dict(x=0.7, y=1)
)
fig.show()

# Identify best model
best_model_name = comparison_df.iloc[0]['Model']
print(f"\nğŸ�† Best Model: {best_model_name} (ROC-AUC: {comparison_df.iloc[0]['ROC-AUC']:.4f})")


# Feature Importance Analysis (using best tree-based model)
print("=" * 80)
print("FEATURE IMPORTANCE ANALYSIS")
print("=" * 80)

# Get feature importance from best model
best_model = results[best_model_name]['model']

if hasattr(best_model, 'feature_importances_'):
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': best_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print(f"\nTop 15 Important Features ({best_model_name}):")
    display(feature_importance.head(15))
    
    # Visualize feature importance
    fig = go.Figure(go.Bar(
        x=feature_importance.head(15)['Importance'],
        y=feature_importance.head(15)['Feature'],
        orientation='h',
        marker_color='steelblue',
        text=feature_importance.head(15)['Importance'].apply(lambda x: f'{x:.4f}'),
        textposition='auto'
    ))
    
    fig.update_layout(
        title=f"Top 15 Feature Importances - {best_model_name}",
        xaxis_title="Importance",
        yaxis_title="Feature",
        height=500,
        yaxis={'categoryorder': 'total ascending'}
    )
    fig.show()
else:
    print(f"\n{best_model_name} does not provide feature importances.")


# Create comprehensive stakeholder dashboard
from plotly.subplots import make_subplots

print("=" * 80)
print("CREATING STAKEHOLDER DASHBOARD")
print("=" * 80)

# Dashboard Data Preparation
dashboard_data = train_df.copy()

# 1. Overall Statistics
total_loans = len(dashboard_data)
total_paid = dashboard_data['loan_paid_back'].sum()
total_unpaid = total_loans - total_paid
payback_rate = (total_paid / total_loans) * 100

total_loan_amount = dashboard_data['loan_amount'].sum()
avg_loan_amount = dashboard_data['loan_amount'].mean()
avg_interest_rate = dashboard_data['interest_rate'].mean()
avg_credit_score = dashboard_data['credit_score'].mean()

print("\nğŸ“Š KEY METRICS:")
print(f"   Total Loans: {total_loans:,}")
print(f"   Loans Paid Back: {total_paid:,} ({payback_rate:.2f}%)")
print(f"   Loans Not Paid: {total_unpaid:,} ({100-payback_rate:.2f}%)")
print(f"   Total Loan Amount: ${total_loan_amount:,.2f}")
print(f"   Average Loan Amount: ${avg_loan_amount:,.2f}")
print(f"   Average Interest Rate: {avg_interest_rate:.2f}%")
print(f"   Average Credit Score: {avg_credit_score:.0f}")


# DASHBOARD PANEL 1: Executive Summary
fig1 = make_subplots(
    rows=2, cols=2,
    subplot_titles=('Loan Status Distribution', 'Loan Amount Distribution',
                    'Credit Score Distribution', 'Interest Rate Distribution'),
    specs=[[{'type': 'pie'}, {'type': 'histogram'}],
           [{'type': 'histogram'}, {'type': 'histogram'}]],
    vertical_spacing=0.12,
    horizontal_spacing=0.15
)

# Loan Status
fig1.add_trace(
    go.Pie(labels=['Not Paid', 'Paid'], 
           values=[total_unpaid, total_paid],
           marker_colors=['#EF553B', '#00CC96'],
           hole=0.3,
           textinfo='label+percent+value',
           textfont_size=12),
    row=1, col=1
)

# Loan Amount
fig1.add_trace(
    go.Histogram(x=dashboard_data['loan_amount'], nbinsx=50,
                 marker_color='lightblue', name='Loan Amount'),
    row=1, col=2
)

# Credit Score
fig1.add_trace(
    go.Histogram(x=dashboard_data['credit_score'], nbinsx=50,
                 marker_color='lightgreen', name='Credit Score'),
    row=2, col=1
)

# Interest Rate
fig1.add_trace(
    go.Histogram(x=dashboard_data['interest_rate'], nbinsx=50,
                 marker_color='lightcoral', name='Interest Rate'),
    row=2, col=2
)

fig1.update_layout(
    title_text="ğŸ“Š EXECUTIVE SUMMARY DASHBOARD",
    height=700,
    showlegend=False
)

fig1.update_xaxes(title_text="Loan Amount ($)", row=1, col=2)
fig1.update_xaxes(title_text="Credit Score", row=2, col=1)
fig1.update_xaxes(title_text="Interest Rate (%)", row=2, col=2)

fig1.show()


# DASHBOARD PANEL 2: Risk Analysis
fig2 = make_subplots(
    rows=2, cols=2,
    subplot_titles=('Payback Rate by Credit Score Range', 
                    'Payback Rate by Employment Status',
                    'Payback Rate by Loan Purpose',
                    'Payback Rate by Education Level'),
    specs=[[{'type': 'bar'}, {'type': 'bar'}],
           [{'type': 'bar'}, {'type': 'bar'}]],
    vertical_spacing=0.15,
    horizontal_spacing=0.12
)

# Credit Score Range
credit_bins = pd.cut(dashboard_data['credit_score'], bins=[0, 600, 650, 700, 750, 850])
credit_payback = dashboard_data.groupby(credit_bins)['loan_paid_back'].mean() * 100
fig2.add_trace(
    go.Bar(x=credit_payback.index.astype(str), y=credit_payback.values,
           marker_color='steelblue', text=credit_payback.values.round(1),
           texttemplate='%{text}%', textposition='auto'),
    row=1, col=1
)

# Employment Status
emp_payback = dashboard_data.groupby('employment_status')['loan_paid_back'].mean() * 100
fig2.add_trace(
    go.Bar(x=emp_payback.index, y=emp_payback.values,
           marker_color='mediumseagreen', text=emp_payback.values.round(1),
           texttemplate='%{text}%', textposition='auto'),
    row=1, col=2
)

# Loan Purpose
purpose_payback = dashboard_data.groupby('loan_purpose')['loan_paid_back'].mean().sort_values(ascending=False).head(10) * 100
fig2.add_trace(
    go.Bar(x=purpose_payback.index, y=purpose_payback.values,
           marker_color='coral', text=purpose_payback.values.round(1),
           texttemplate='%{text}%', textposition='auto'),
    row=2, col=1
)

# Education Level
edu_payback = dashboard_data.groupby('education_level')['loan_paid_back'].mean() * 100
fig2.add_trace(
    go.Bar(x=edu_payback.index, y=edu_payback.values,
           marker_color='gold', text=edu_payback.values.round(1),
           texttemplate='%{text}%', textposition='auto'),
    row=2, col=2
)

fig2.update_layout(
    title_text="ğŸ�¯ RISK ANALYSIS DASHBOARD",
    height=700,
    showlegend=False
)

fig2.update_yaxes(title_text="Payback Rate (%)", row=1, col=1)
fig2.update_yaxes(title_text="Payback Rate (%)", row=1, col=2)
fig2.update_yaxes(title_text="Payback Rate (%)", row=2, col=1)
fig2.update_yaxes(title_text="Payback Rate (%)", row=2, col=2)

fig2.update_xaxes(tickangle=-45, row=2, col=1)
fig2.update_xaxes(tickangle=-45, row=2, col=2)

fig2.show()


# DASHBOARD PANEL 3: Financial Insights
fig3 = make_subplots(
    rows=2, cols=2,
    subplot_titles=('Average Loan Amount by Grade', 
                    'Interest Rate vs Credit Score',
                    'Debt-to-Income Ratio Distribution',
                    'Annual Income Distribution'),
    specs=[[{'type': 'bar'}, {'type': 'scatter'}],
           [{'type': 'box'}, {'type': 'violin'}]],
    vertical_spacing=0.15,
    horizontal_spacing=0.12
)

# Average Loan by Grade
grade_loan = dashboard_data.groupby('grade_subgrade')['loan_amount'].mean().sort_index()
fig3.add_trace(
    go.Bar(x=grade_loan.index, y=grade_loan.values,
           marker_color='purple', name='Avg Loan'),
    row=1, col=1
)

# Interest Rate vs Credit Score (sample for performance)
sample_data = dashboard_data.sample(min(10000, len(dashboard_data)), random_state=42)
fig3.add_trace(
    go.Scatter(x=sample_data['credit_score'], 
               y=sample_data['interest_rate'],
               mode='markers',
               marker=dict(color=sample_data['loan_paid_back'],
                         colorscale=['red', 'green'],
                         size=5, opacity=0.5),
               name='Loans'),
    row=1, col=2
)

# Debt-to-Income by Payback Status
for status in [0, 1]:
    fig3.add_trace(
        go.Box(y=dashboard_data[dashboard_data['loan_paid_back'] == status]['debt_to_income_ratio'],
               name=f'Paid={status}',
               marker_color='#EF553B' if status == 0 else '#00CC96'),
        row=2, col=1
    )

# Annual Income Distribution
fig3.add_trace(
    go.Violin(y=dashboard_data['annual_income'],
              box_visible=True,
              meanline_visible=True,
              fillcolor='lightseagreen',
              opacity=0.6,
              name='Income'),
    row=2, col=2
)

fig3.update_layout(
    title_text="ğŸ’° FINANCIAL INSIGHTS DASHBOARD",
    height=700,
    showlegend=True
)

fig3.update_xaxes(title_text="Grade", tickangle=-45, row=1, col=1)
fig3.update_xaxes(title_text="Credit Score", row=1, col=2)
fig3.update_yaxes(title_text="Average Loan Amount ($)", row=1, col=1)
fig3.update_yaxes(title_text="Interest Rate (%)", row=1, col=2)
fig3.update_yaxes(title_text="Debt-to-Income Ratio", row=2, col=1)
fig3.update_yaxes(title_text="Annual Income ($)", row=2, col=2)

fig3.show()


# DASHBOARD PANEL 4: Demographic Analysis
fig4 = make_subplots(
    rows=2, cols=2,
    subplot_titles=('Payback Rate by Gender', 
                    'Payback Rate by Marital Status',
                    'Loan Count by Gender',
                    'Loan Count by Marital Status'),
    specs=[[{'type': 'bar'}, {'type': 'bar'}],
           [{'type': 'pie'}, {'type': 'pie'}]],
    vertical_spacing=0.15,
    horizontal_spacing=0.15
)

# Payback by Gender
gender_payback = dashboard_data.groupby('gender')['loan_paid_back'].mean() * 100
fig4.add_trace(
    go.Bar(x=gender_payback.index, y=gender_payback.values,
           marker_color=['#FF69B4', '#4169E1', '#FFD700'],
           text=gender_payback.values.round(1),
           texttemplate='%{text}%', textposition='auto'),
    row=1, col=1
)

# Payback by Marital Status
marital_payback = dashboard_data.groupby('marital_status')['loan_paid_back'].mean() * 100
fig4.add_trace(
    go.Bar(x=marital_payback.index, y=marital_payback.values,
           marker_color=['#32CD32', '#FF6347', '#9370DB'],
           text=marital_payback.values.round(1),
           texttemplate='%{text}%', textposition='auto'),
    row=1, col=2
)

# Loan Count by Gender
gender_count = dashboard_data['gender'].value_counts()
fig4.add_trace(
    go.Pie(labels=gender_count.index, values=gender_count.values,
           marker_colors=['#FF69B4', '#4169E1', '#FFD700'],
           textinfo='label+percent'),
    row=2, col=1
)

# Loan Count by Marital Status
marital_count = dashboard_data['marital_status'].value_counts()
fig4.add_trace(
    go.Pie(labels=marital_count.index, values=marital_count.values,
           marker_colors=['#32CD32', '#FF6347', '#9370DB'],
           textinfo='label+percent'),
    row=2, col=2
)

fig4.update_layout(
    title_text="ğŸ‘¥ DEMOGRAPHIC ANALYSIS DASHBOARD",
    height=700,
    showlegend=False
)

fig4.update_yaxes(title_text="Payback Rate (%)", row=1, col=1)
fig4.update_yaxes(title_text="Payback Rate (%)", row=1, col=2)

fig4.show()


# DASHBOARD PANEL 5: Model Performance Dashboard
fig5 = make_subplots(
    rows=2, cols=2,
    subplot_titles=('Model Performance Comparison', 
                    'Confusion Matrix - Best Model',
                    'ROC Curve Comparison',
                    'Feature Importance (Top 10)'),
    specs=[[{'type': 'bar'}, {'type': 'heatmap'}],
           [{'type': 'scatter'}, {'type': 'bar'}]],
    vertical_spacing=0.15,
    horizontal_spacing=0.15
)

# Model Performance Comparison
fig5.add_trace(
    go.Bar(x=comparison_df['Model'], y=comparison_df['ROC-AUC'],
           marker_color='mediumpurple',
           text=comparison_df['ROC-AUC'].apply(lambda x: f'{x:.4f}'),
           textposition='auto'),
    row=1, col=1
)

# Confusion Matrix
from sklearn.metrics import confusion_matrix
best_preds = results[best_model_name]['predictions']
cm = confusion_matrix(y_val, best_preds)
fig5.add_trace(
    go.Heatmap(z=cm, x=['Predicted 0', 'Predicted 1'],
               y=['Actual 0', 'Actual 1'],
               colorscale='Blues',
               text=cm, texttemplate='%{text}',
               textfont={"size": 16}),
    row=1, col=2
)

# ROC Curves (simplified - just best model)
from sklearn.metrics import roc_curve
fpr, tpr, _ = roc_curve(y_val, results[best_model_name]['predictions_proba'])
fig5.add_trace(
    go.Scatter(x=fpr, y=tpr, mode='lines',
               name=best_model_name,
               line=dict(color='green', width=2)),
    row=2, col=1
)
fig5.add_trace(
    go.Scatter(x=[0, 1], y=[0, 1], mode='lines',
               name='Random',
               line=dict(color='red', dash='dash')),
    row=2, col=1
)

# Feature Importance (if available)
if hasattr(best_model, 'feature_importances_'):
    top_features = feature_importance.head(10)
    fig5.add_trace(
        go.Bar(x=top_features['Importance'], y=top_features['Feature'],
               orientation='h',
               marker_color='teal',
               text=top_features['Importance'].apply(lambda x: f'{x:.3f}'),
               textposition='auto'),
        row=2, col=2
    )

fig5.update_layout(
    title_text="ğŸ¤– MODEL PERFORMANCE DASHBOARD",
    height=800,
    showlegend=True
)

fig5.update_xaxes(title_text="Model", row=1, col=1)
fig5.update_xaxes(title_text="False Positive Rate", row=2, col=1)
fig5.update_yaxes(title_text="ROC-AUC Score", row=1, col=1)
fig5.update_yaxes(title_text="True Positive Rate", row=2, col=1)
fig5.update_yaxes(categoryorder='total ascending', row=2, col=2)

fig5.show()


# Generate predictions using the best model
print("=" * 80)
print("GENERATING TEST SET PREDICTIONS")
print("=" * 80)

# Use best model to predict on test set
test_predictions = best_model.predict(X_test)
test_predictions_proba = best_model.predict_proba(X_test)[:, 1]

# Create submission file
submission = pd.DataFrame({
    'id': test_df['id'],
    'loan_paid_back': test_predictions
})

print(f"\nâœ… Generated predictions for {len(submission)} test samples")
print(f"\nPrediction distribution:")
print(f"  Predicted Not Paid (0): {(test_predictions == 0).sum():,} ({(test_predictions == 0).sum()/len(test_predictions)*100:.2f}%)")
print(f"  Predicted Paid (1): {(test_predictions == 1).sum():,} ({(test_predictions == 1).sum()/len(test_predictions)*100:.2f}%)")

# Display first few predictions
print(f"\nFirst 10 predictions:")
display(submission.head(10))

# Save submission file
submission_path = f"/kaggle/working/submission.csv"
submission.to_csv(submission_path, index=False)
print(f"\nâœ… Submission file saved to: {submission_path}")


# Generate comprehensive insights report
print("=" * 80)
print("ğŸ“‹ KEY INSIGHTS AND RECOMMENDATIONS")
print("=" * 80)

insights = []

# 1. Overall Performance
insights.append({
    'Category': 'Overall Performance',
    'Insight': f'Overall loan payback rate is {payback_rate:.2f}%',
    'Recommendation': 'Focus on improving collection strategies for the remaining loans'
})

# 2. Credit Score Impact
credit_corr = train_df[['credit_score', 'loan_paid_back']].corr().iloc[0, 1]
insights.append({
    'Category': 'Credit Score',
    'Insight': f'Credit score shows correlation of {credit_corr:.4f} with loan payback',
    'Recommendation': 'Implement stricter credit score requirements for high-risk applicants (score < 650)'
})

# 3. Employment Status
emp_stats = dashboard_data.groupby('employment_status')['loan_paid_back'].agg(['mean', 'count'])
best_emp = emp_stats['mean'].idxmax()
worst_emp = emp_stats['mean'].idxmin()
insights.append({
    'Category': 'Employment Status',
    'Insight': f'{best_emp} status has highest payback rate ({emp_stats.loc[best_emp, "mean"]*100:.1f}%), {worst_emp} has lowest ({emp_stats.loc[worst_emp, "mean"]*100:.1f}%)',
    'Recommendation': f'Prioritize loans to {best_emp} applicants and require additional collateral for {worst_emp}'
})

# 4. Loan Purpose
purpose_stats = dashboard_data.groupby('loan_purpose')['loan_paid_back'].mean().sort_values(ascending=False)
best_purpose = purpose_stats.index[0]
worst_purpose = purpose_stats.index[-1]
insights.append({
    'Category': 'Loan Purpose',
    'Insight': f'{best_purpose} loans have highest payback rate ({purpose_stats.iloc[0]*100:.1f}%), {worst_purpose} lowest ({purpose_stats.iloc[-1]*100:.1f}%)',
    'Recommendation': f'Encourage {best_purpose} loans and carefully evaluate {worst_purpose} applications'
})

# 5. Debt-to-Income Ratio
dti_corr = train_df[['debt_to_income_ratio', 'loan_paid_back']].corr().iloc[0, 1]
insights.append({
    'Category': 'Debt-to-Income Ratio',
    'Insight': f'DTI ratio has correlation of {dti_corr:.4f} with payback (negative impact)',
    'Recommendation': 'Set maximum DTI threshold at 30% for new loan approvals'
})

# 6. Model Performance
insights.append({
    'Category': 'Predictive Model',
    'Insight': f'Best model ({best_model_name}) achieves {comparison_df.iloc[0]["ROC-AUC"]:.4f} ROC-AUC score',
    'Recommendation': 'Deploy this model for automated loan approval decisions with human oversight'
})

# Display insights
insights_df = pd.DataFrame(insights)
for idx, row in insights_df.iterrows():
    print(f"\n{idx + 1}. {row['Category'].upper()}")
    print(f"   ğŸ“Š Insight: {row['Insight']}")
    print(f"   ğŸ’¡ Recommendation: {row['Recommendation']}")

print("\n" + "=" * 80)
print("âœ… ANALYSIS COMPLETE!")
print("=" * 80)

