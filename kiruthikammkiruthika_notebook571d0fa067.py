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


"""
churn_capstone.py
Full end-to-end script for "Customer Churn Prediction" capstone.
Prereqs:
    pip install pandas numpy matplotlib seaborn scikit-learn xgboost shap joblib
Place 'telco_churn.csv' (Kaggle Telco Customer Churn) in same folder, OR the script will create & use a synthetic demo dataset.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, ConfusionMatrixDisplay

# -------------------------
# Utility: create synthetic demo dataset (if no Kaggle file)
# -------------------------
def create_synthetic_telco_demo(path='telco_churn_demo.csv', n=1000, random_state=42):
    np.random.seed(random_state)
    df = pd.DataFrame({
        'customerID': [f'ID{i:06d}' for i in range(n)],
        'gender': np.random.choice(['Male','Female'], n),
        'SeniorCitizen': np.random.choice([0,1], n, p=[0.85,0.15]),
        'Partner': np.random.choice(['Yes','No'], n),
        'Dependents': np.random.choice(['Yes','No'], n),
        'tenure': np.random.randint(0, 72, n),
        'PhoneService': np.random.choice(['Yes','No'], n, p=[0.9,0.1]),
        'MultipleLines': np.random.choice(['No phone service','No','Yes'], n, p=[0.05,0.6,0.35]),
        'InternetService': np.random.choice(['DSL','Fiber optic','No'], n, p=[0.35,0.45,0.2]),
        'OnlineSecurity': np.random.choice(['Yes','No','No internet service'], n),
        'OnlineBackup': np.random.choice(['Yes','No','No internet service'], n),
        'DeviceProtection': np.random.choice(['Yes','No','No internet service'], n),
        'TechSupport': np.random.choice(['Yes','No','No internet service'], n),
        'StreamingTV': np.random.choice(['Yes','No','No internet service'], n),
        'StreamingMovies': np.random.choice(['Yes','No','No internet service'], n),
        'Contract': np.random.choice(['Month-to-month','One year','Two year'], n, p=[0.6,0.2,0.2]),
        'PaperlessBilling': np.random.choice(['Yes','No'], n),
        'PaymentMethod': np.random.choice(['Electronic check','Mailed check','Bank transfer (automatic)','Credit card (automatic)'], n),
        'MonthlyCharges': np.round(np.random.uniform(18,120,size=n),2)
    })
    df['TotalCharges'] = (df['tenure'] * df['MonthlyCharges']).round(2)
    df.loc[df['tenure']==0, 'TotalCharges'] = ''
    churn_prob = (df['Contract']=='Month-to-month')*0.2 + (df['tenure']<6)*0.15 + (df['MonthlyCharges']>80)*0.1
    churn = np.random.rand(n) < churn_prob
    df['Churn'] = np.where(churn, 'Yes', 'No')
    df.to_csv(path, index=False)
    print(f"Created synthetic demo CSV at: {path} (shape={df.shape})")
    return df

# -------------------------
# Load the dataset
# -------------------------
def load_data(path='telco_churn.csv'):
    if os.path.exists(path):
        print(f"Loading dataset from {path}")
        df = pd.read_csv(path)
    else:
        print(f"{path} not found — creating a synthetic demo dataset.")
        df = create_synthetic_telco_demo(path='telco_churn_demo.csv')
    return df

# -------------------------
# Basic EDA prints + plots
# -------------------------
def quick_eda(df, show_plots=True):
    print("\n--- DATA INFO ---")
    print(df.info())
    print("\n--- MISSING VALUES ---")
    print(df.isna().sum())
    if 'Churn' in df.columns:
        print("\n--- TARGET DISTRIBUTION ---")
        print(df['Churn'].value_counts(normalize=True))
    if show_plots:
        plt.figure(figsize=(12,5))
        if 'Contract' in df.columns and 'Churn' in df.columns:
            plt.subplot(1,2,1)
            sns.countplot(data=df, x='Contract', hue='Churn')
            plt.title('Contract type vs Churn')
        if 'MonthlyCharges' in df.columns and 'Churn' in df.columns:
            plt.subplot(1,2,2)
            try:
                sns.kdeplot(data=df, x='MonthlyCharges', hue='Churn', common_norm=False)
            except Exception:
                sns.violinplot(data=df, x='Churn', y='MonthlyCharges')
            plt.title('MonthlyCharges distribution by Churn')
        plt.tight_layout()
        plt.show()

# -------------------------
# Preprocess, feature engineering
# -------------------------
def preprocess_and_split(df, target_col='Churn'):
    # Clean TotalCharges if present
    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = df['TotalCharges'].replace(' ', np.nan)
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    # Drop id
    if 'customerID' in df.columns:
        df = df.drop(columns=['customerID'])
    # Create tenure groups
    if 'tenure' in df.columns:
        df['tenure_group'] = pd.cut(df['tenure'], bins=[-1,6,12,24,48,72], labels=['0-6','7-12','13-24','25-48','49-72'])
    # Target binary
    df[target_col + 'Flag'] = df[target_col].map({'Yes':1, 'No':0})
    target_flag = target_col + 'Flag'
    X = df.drop(columns=[target_col, target_flag])
    y = df[target_flag].astype(int)
    # Identify numeric & categorical
    num_cols = X.select_dtypes(include=['int64','float64']).columns.tolist()
    cat_cols = X.select_dtypes(include=['object','category','bool']).columns.tolist()
    # ensure no target-like columns in X
    num_cols = [c for c in num_cols if c != target_flag]
    return X, y, num_cols, cat_cols

# -------------------------
# Build pipeline and train
# -------------------------
def build_pipelines(num_cols, cat_cols):
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='Missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))
    ])
    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, num_cols),
        ('cat', categorical_transformer, cat_cols)
    ])
    # Pipelines
    pipe_lr = Pipeline(steps=[('pre', preprocessor),
                              ('clf', LogisticRegression(max_iter=2000))])
    pipe_rf = Pipeline(steps=[('pre', preprocessor),
                              ('clf', RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1))])
    return pipe_lr, pipe_rf

# -------------------------
# Train, evaluate and save
# -------------------------
def train_and_evaluate(pipe_lr, pipe_rf, X_train, X_test, y_train, y_test, save_model_path='churn_model_rf.joblib'):
    # Fit logistic
    pipe_lr.fit(X_train, y_train)
    y_pred_lr = pipe_lr.predict(X_test)
    y_proba_lr = pipe_lr.predict_proba(X_test)[:,1]
    print("\n=== Logistic Regression ===")
    print(classification_report(y_test, y_pred_lr))
    print("ROC AUC (LogReg):", roc_auc_score(y_test, y_proba_lr))

    # Fit random forest
    pipe_rf.fit(X_train, y_train)
    y_pred_rf = pipe_rf.predict(X_test)
    y_proba_rf = pipe_rf.predict_proba(X_test)[:,1]
    print("\n=== Random Forest ===")
    print(classification_report(y_test, y_pred_rf))
    print("ROC AUC (RF):", roc_auc_score(y_test, y_proba_rf))

    # Confusion matrix (RF)
    ConfusionMatrixDisplay.from_estimator(pipe_rf, X_test, y_test)
    plt.title('Random Forest Confusion Matrix')
    plt.show()

    # ROC curves
    plt.figure(figsize=(6,5))
    fpr_rf, tpr_rf, _ = roc_curve(y_test, y_proba_rf)
    plt.plot(fpr_rf, tpr_rf, label=f'RandomForest (AUC={roc_auc_score(y_test, y_proba_rf):.3f})')
    fpr_lr, tpr_lr, _ = roc_curve(y_test, y_proba_lr)
    plt.plot(fpr_lr, tpr_lr, label=f'LogReg (AUC={roc_auc_score(y_test, y_proba_lr):.3f})')
    plt.plot([0,1],[0,1],'--', color='grey')
    plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate'); plt.legend(); plt.title('ROC Curves')
    plt.show()

    # Feature importance mapping (for RF)
    try:
        pre = pipe_rf.named_steps['pre']
        clf = pipe_rf.named_steps['clf']
        # get names
        num_names = num_cols
        cat_names = []
        if hasattr(pre.named_transformers_['cat'].named_steps['onehot'], 'get_feature_names_out'):
            cat_names = list(pre.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(cat_cols))
        else:
            cat_names = list(pre.named_transformers_['cat'].named_steps['onehot'].get_feature_names(cat_cols))
        feat_names = num_names + cat_names
        importances = clf.feature_importances_
        feat_imp = pd.DataFrame({'feature':feat_names, 'importance':importances}).sort_values('importance', ascending=False).head(30)
        print("\nTop feature importances (Random Forest):")
        print(feat_imp)
    except Exception as e:
        print("Could not retrieve feature importances:", e)

    # Save RF pipeline
    joblib.dump(pipe_rf, save_model_path)
    print(f"Saved Random Forest pipeline to {save_model_path}")

    return pipe_rf

# -------------------------
# Optional: SHAP explainability (requires shap)
# -------------------------
def run_shap(pipe_rf, X_test, max_display=20):
    try:
        import shap
        pre = pipe_rf.named_steps['pre']
        clf = pipe_rf.named_steps['clf']
        # transform test set
        X_test_trans = pre.transform(X_test)
        # get feature names
        if hasattr(pre.named_transformers_['cat'].named_steps['onehot'], 'get_feature_names_out'):
            cat_names = list(pre.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(cat_cols))
        else:
            cat_names = list(pre.named_transformers_['cat'].named_steps['onehot'].get_feature_names(cat_cols))
        feat_names = num_cols + cat_names
        explainer = shap.TreeExplainer(clf)
        shap_values = explainer.shap_values(X_test_trans[:500])
        # summary plot
        shap.summary_plot(shap_values, X_test_trans[:500], feature_names=feat_names, max_display=max_display)
    except Exception as e:
        print("SHAP not available or failed:", e)

# -------------------------
# Main run
# -------------------------
if __name__ == '__main__':
    # Load
    df = load_data('telco_churn.csv')
    print("Data shape:", df.shape)
    # EDA
    quick_eda(df, show_plots=True)
    # Preprocess and split
    X, y, num_cols, cat_cols = preprocess_and_split(df, target_col='Churn')
    print("\nNumeric columns:", num_cols)
    print("Categorical columns:", cat_cols)
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
    # Build
    pipe_lr, pipe_rf = build_pipelines(num_cols, cat_cols)
    # Train & evaluate
    trained_rf = train_and_evaluate(pipe_lr, pipe_rf, X_train, X_test, y_train, y_test, save_model_path='churn_model_rf.joblib')
    # Optional SHAP
    run_shap(trained_rf, X_test)
    print("\nAll done. Deliverables:\n - churn_model_rf.joblib (saved model)\n - Use this script or the notebook for step-by-step exploration.")

