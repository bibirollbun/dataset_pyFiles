# Install required packages if not already installed
import subprocess
import sys

def install_package(package):
    try:
        __import__(package)
        print(f"{package} is already installed")
    except ImportError:
        print(f"Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# List of required packages
packages = ['numpy', 'pandas', 'scikit-learn', 'lightgbm', 'matplotlib', 'seaborn', 'scipy']

for package in packages:
    install_package(package)

print("All packages are ready!")


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import RobustScaler
from sklearn.impute import KNNImputer
from sklearn.feature_selection import RFECV
from sklearn.preprocessing import PolynomialFeatures
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import f1_score, classification_report, make_scorer
from lightgbm import LGBMClassifier
from scipy.optimize import minimize_scalar

import warnings
warnings.filterwarnings('ignore')

# For plotting
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

print("All libraries imported successfully!")


# Check if files exist first
import os

if os.path.exists('/kaggle/input/diabetes-prediction-from-medicalrecords/train.csv') and os.path.exists('/kaggle/input/diabetes-prediction-from-medicalrecords/test.csv'):
    train_df = pd.read_csv('/kaggle/input/diabetes-prediction-from-medicalrecords/train.csv')
    test_df = pd.read_csv('/kaggle/input/diabetes-prediction-from-medicalrecords/test.csv')
    print(f'Train: {train_df.shape}, Test: {test_df.shape}')
    print(f'Columns: {list(train_df.columns)}')
else:
    print("Error: train.csv and/or test.csv not found in the current directory.")
    print("Please make sure the data files are in the same folder as this notebook.")


# Initial Exploration
print("\nTraining Data Info:\n")
train_df.info()
print("\nData Description:\n", train_df.describe())


# Check for missing values
print("\nMissing Values:\n", train_df.isnull().sum())

# # Remove duplicates
data_cleaned = train_df.drop_duplicates()

# # Remove constant columns and reassign to train_df
non_constant_columns = data_cleaned.loc[:, data_cleaned.nunique() > 1].columns
train_df = data_cleaned[non_constant_columns]

# Check for missing values
# # Summary of numerical and categorical features
print("\nNumerical Features:\n", train_df.describe())
print("\nCategorical Features:\n", train_df.select_dtypes(include='object').nunique())
print("\nMissing Values:\n", train_df.isnull().sum())


# Summary of numerical and categorical features
print("\nNumerical Features:\n", train_df.describe())
print("\nCategorical Features:\n", train_df.select_dtypes(include='object').nunique())


# Correlation heatmap
plt.figure(figsize=(8, 4))
numeric_data = train_df.select_dtypes(include=['number'])  # Filter numeric columns
correlation_matrix = numeric_data.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()


# Scatter plot for numerical relationships
numerical_features = ['Pregnancies', 'Glucose', 'BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']
sns.pairplot(train_df, vars=numerical_features,hue='Outcome')
plt.show()


def plotHistogramModern(values, label, feature, title):
    """
    Create histogram plots with hue-based coloring for different outcomes
    """
    plt.figure(figsize=(12, 6))
    sns.set_style("whitegrid")
    
    # Create histogram with hue
    sns.histplot(data=values, x=feature, hue=label, kde=False, stat='density')
    
    plt.title(title)
    plt.xlabel(feature)
    plt.ylabel('Proportion')
    plt.legend()
    plt.show()

# Only plot if data is loaded
if 'train_df' in locals():
    features_to_plot = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
    
    for feature in features_to_plot:
        plotHistogramModern(train_df, "Outcome", feature, f'{feature} vs Diagnosis (Blue = Healthy; Orange = Diabetes)')


def create_features(df, g=140, b=30, bp=80, a=45):
    df = df.copy()
    zero_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    for col in zero_cols:
        nz = df[df[col] != 0][col]
        if len(nz) > 0:
            df.loc[df[col] == 0, col] = np.random.uniform(nz.min(), nz.max(), (df[col] == 0).sum())
    df['BMI_Age'] = df['BMI'] * df['Age']
    df['Glucose_BMI'] = df['Glucose'] * df['BMI']
    df['Glucose_Age'] = df['Glucose'] * df['Age']
    df['Insulin_Glucose'] = df['Insulin'] * df['Glucose']
    df['Glucose_to_Insulin'] = df['Glucose'] / (df['Insulin'] + 1)
    df['BMI_to_Age'] = df['BMI'] / (df['Age'] + 1)
    df['Glucose_squared'] = df['Glucose'] ** 2
    df['BMI_squared'] = df['BMI'] ** 2
    df['Age_squared'] = df['Age'] ** 2
    df['Age_group'] = pd.cut(df['Age'], bins=[0, 30, 40, 50, 100], labels=[0, 1, 2, 3]).astype(float)
    df['BMI_category'] = pd.cut(df['BMI'], bins=[0, 18.5, 25, 30, 100], labels=[0, 1, 2, 3]).astype(float)
    df['Glucose_category'] = pd.cut(df['Glucose'], bins=[0, 100, 125, 200], labels=[0, 1, 2]).astype(float)
    df['High_Glucose'] = (df['Glucose'] > g).astype(int)
    df['High_BMI'] = (df['BMI'] > b).astype(int)
    df['High_BP'] = (df['BloodPressure'] > bp).astype(int)
    df['Older_Age'] = (df['Age'] > a).astype(int)
    df['Risk_Score'] = df['High_Glucose'].fillna(0) + df['High_BMI'].fillna(0) + df['High_BP'].fillna(0) + df['Older_Age'].fillna(0)
    return df

def tune_thresholds(df, y):
    best_score, best_p = 0, {'g': 140, 'b': 30, 'bp': 80, 'a': 45}
    for g in [130, 140, 150]:
        for b in [28, 30, 32]:
            for bp in [75, 80, 85]:
                for a in [40, 45, 50]:
                    tmp = create_features(df, g, b, bp, a)
                    X = tmp.drop(['Outcome', 'Id'], axis=1)
                    Xtr, Xv, ytr, yv = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
                    imp = KNNImputer(n_neighbors=5)
                    Xtr_i, Xv_i = imp.fit_transform(Xtr), imp.transform(Xv)
                    m = LogisticRegression(random_state=42, max_iter=1000).fit(Xtr_i, ytr)
                    score = f1_score(yv, m.predict(Xv_i))
                    if score > best_score:
                        best_score, best_p = score, {'g': g, 'b': b, 'bp': bp, 'a': a}
    return best_p

# Only run if data is available
if 'train_df' in locals():
    print("Tuning feature engineering thresholds...")
    bp = tune_thresholds(train_df, train_df['Outcome'])
    train_enh = create_features(train_df, **bp)
    test_enh = create_features(test_df, **bp)
    print(f"Best parameters: {bp}")
else:
    print("Skipping feature engineering - no data loaded")


if 'train_enh' in locals():
    X = train_enh.drop(['Outcome', 'Id'], axis=1)
    y = train_enh['Outcome']
    X_test = test_enh.drop(['Id'], axis=1)
    test_ids = test_enh['Id']
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"Data prepared: Training set {X_train.shape}, Validation set {X_val.shape}")
else:
    print("Skipping data preparation - no enhanced features available")


def tune_knn(X, y):
    best_score, best_k = 0, 5
    for k in [3, 5, 7, 10, 15]:
        imp = KNNImputer(n_neighbors=k)
        Xt = imp.fit_transform(X)
        Xtr, Xv, ytr, yv = train_test_split(Xt, y, test_size=0.2, random_state=42, stratify=y)
        score = f1_score(yv, LogisticRegression(random_state=42, max_iter=1000).fit(Xtr, ytr).predict(Xv))
        if score > best_score:
            best_score, best_k = score, k
    return KNNImputer(n_neighbors=best_k)

if 'X' in locals():
    print("Optimizing KNN imputation...")
    knn_imp = tune_knn(X, y)
    X_imp = pd.DataFrame(knn_imp.fit_transform(X), columns=X.columns)
    X_test_imp = pd.DataFrame(knn_imp.transform(X_test), columns=X_test.columns)
    X_train_imp = X_imp.iloc[X_train.index]
    X_val_imp = X_imp.iloc[X_val.index]
    print("KNN imputation completed")
else:
    print("Skipping KNN imputation - no data available")


if 'X_train_imp' in locals():
    print("Performing feature selection...")
    selector = LogisticRegression(random_state=42, max_iter=1000)
    rfe = RFECV(estimator=selector, step=1, cv=3, scoring='f1')
    rfe.fit(X_train_imp, y_train)
    sel = X.columns[rfe.support_]
    X_train_sel = X_train_imp[sel]
    X_val_sel = X_val_imp[sel]
    X_test_sel = X_test_imp[sel]
    print(f"Selected {len(sel)} features from {len(X.columns)} original features")
else:
    print("Skipping feature selection - no imputed data available")


if 'X_train_sel' in locals():
    print("Creating polynomial features...")
    poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    X_train_poly = poly.fit_transform(X_train_sel)
    X_val_poly = poly.transform(X_val_sel)
    X_test_poly = poly.transform(X_test_sel)
    
    print("Selecting best polynomial features...")
    ps = RFECV(estimator=selector, step=5, cv=3, scoring='f1', min_features_to_select=20)
    X_train_fin = ps.fit_transform(X_train_poly, y_train)
    X_val_fin = ps.transform(X_val_poly)
    X_test_fin = ps.transform(X_test_poly)
    print(f"Final feature count: {X_train_fin.shape[1]}")
else:
    print("Skipping polynomial features - no selected features available")


if 'X_train_fin' in locals():
    print("Scaling features...")
    scaler = RobustScaler()
    X_train_sc = scaler.fit_transform(X_train_fin)
    X_val_sc = scaler.transform(X_val_fin)
    X_test_sc = scaler.transform(X_test_fin)
    y_train_res = y_train  # Not using SMOTE
    print("Feature scaling completed")
else:
    print("Skipping feature scaling - no polynomial features available")


# Correct for Outcome=1 data imbalance
def apply_random_oversampling(X_train_fin, y_train):
    try:
        from imblearn.over_sampling import RandomOverSampler
        
        print("Original class distribution:")
        print(Counter(y_train))
        
        # Apply Random Oversampling
        ros = RandomOverSampler(random_state=42)
        X_train_balanced, y_train_balanced = ros.fit_resample(X_train_fin, y_train)
        
        print("\nAfter Random Oversampling - class distribution:")
        print(Counter(y_train_balanced))
        
        return X_train_balanced, y_train_balanced
    
    except ImportError:
        print("Please install imbalanced-learn: pip install imbalanced-learn")
        return X_train_fin, y_train
    
apply_random_oversampling(X_train_sc, y_train_res)


 # Convert to DataFrame with target using REAL feature names
try:
    # Get actual feature names from the RFECV selector
    feature_names = ps.get_feature_names_out()
except:
    # Fallback to generic names if ps not available
    feature_names = [f'feature_{i}' for i in range(X_train_sc.shape[1])]

X_train_df = pd.DataFrame(X_train_sc, columns=feature_names)
X_train_df['Outcome'] = y_train

print(f"Using feature names: {feature_names[:5]}...")  # Show first 5

# Select subset of features for plotting
selected_features = feature_names[:min(8, len(feature_names))]
    
# Create pairplot-style visualization
sns.pairplot(X_train_df, vars=selected_features, hue='Outcome', 
                plot_kws={'alpha': 0.6}, diag_kind='hist')
plt.show()


if 'X_train_sc' in locals():
    print("Training and tuning models...")
    
    rf_p = {'n_estimators': [200, 300, 500], 'max_depth': [10, 15, 20, None], 
            'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 4], 
            'max_features': ['sqrt', 'log2', 0.8]}
    gb_p = {'n_estimators': [200, 300, 500], 'max_depth': [3, 5, 7], 
            'learning_rate': [0.01, 0.05, 0.1], 'subsample': [0.8, 0.9, 1.0]}
    lg_p = {'n_estimators': [200, 300, 500], 'max_depth': [6, 8, 10], 
            'learning_rate': [0.01, 0.05, 0.1], 'num_leaves': [31, 50, 100], 
            'feature_fraction': [0.8, 0.9, 1.0]}

    def tune_m(m, pg, X, y):
        s = RandomizedSearchCV(m, pg, scoring=make_scorer(f1_score), cv=3, n_iter=20, random_state=42, n_jobs=-1)
        s.fit(X, y)
        return s.best_estimator_

    print("Training Random Forest...")
    t_rf = tune_m(RandomForestClassifier(random_state=42, class_weight='balanced'), rf_p, X_train_sc, y_train_res)
    
    print("Training Gradient Boosting...")
    t_gb = tune_m(GradientBoostingClassifier(random_state=42), gb_p, X_train_sc, y_train_res)
    
    print("Training LightGBM...")
    t_lg = tune_m(LGBMClassifier(random_state=42, verbose=-1, class_weight='balanced'), lg_p, X_train_sc, y_train_res)
    
    print("Model training completed")
else:
    print("Skipping model training - no scaled data available")


if 't_rf' in locals():
    print("Calibrating models...")
    c_rf = CalibratedClassifierCV(t_rf, method='sigmoid', cv=5).fit(X_train_sc, y_train_res)
    c_gb = CalibratedClassifierCV(t_gb, method='sigmoid', cv=5).fit(X_train_sc, y_train_res)
    c_lg = CalibratedClassifierCV(t_lg, method='sigmoid', cv=5).fit(X_train_sc, y_train_res)
    print("Model calibration completed")
else:
    print("Skipping model calibration - no trained models available")


if 'c_rf' in locals():
    print("Evaluating individual models...")
    
    # Evaluate individual models
    res = {}
    for n, m in [('RF', c_rf), ('GB', c_gb), ('LG', c_lg)]:
        f1 = f1_score(y_val, m.predict(X_val_sc))
        res[n] = {'model': m, 'f1': f1}
    
    res_df = pd.DataFrame({k: {'F1': v['f1']} for k, v in res.items()}).T.sort_values('F1', ascending=False)
    print("Individual Model Results:")
    print(res_df)
    
    # Create ensembles
    print("\nCreating ensemble models...")
    top3 = res_df.index.tolist()[:3]
    
    # Voting Ensemble
    vc = VotingClassifier(estimators=[(n, res[n]['model']) for n in top3], voting='soft', weights=[3, 2, 1])
    vc.fit(X_train_sc, y_train_res)
    f1_v = f1_score(y_val, vc.predict(X_val_sc))
    print(f'Voting F1: {f1_v:.4f}')
    
    # Stacking Ensemble
    sc = StackingClassifier(estimators=[(n, res[n]['model']) for n in top3], 
                           final_estimator=LogisticRegression(random_state=42, max_iter=1000), cv=5)
    sc.fit(X_train_sc, y_train_res)
    f1_s = f1_score(y_val, sc.predict(X_val_sc))
    print(f'Stacking F1: {f1_s:.4f}')
    
    # Select best model
    all_res = {**{k: v['f1'] for k, v in res.items()}, 'Voting': f1_v, 'Stacking': f1_s}
    best_name = max(all_res, key=all_res.get)
    best_m = vc if best_name == 'Voting' else sc if best_name == 'Stacking' else res[best_name]['model']
    
    print(f"\nBest model: {best_name} with F1 score: {all_res[best_name]:.4f}")
else:
    print("Skipping model evaluation - no calibrated models available")


if 'best_m' in locals():
    print("Optimizing classification threshold...")
    
    def opt_thresh(yt, yp):
        def obj(t):
            return -f1_score(yt, (yp >= t).astype(int))
        r = minimize_scalar(obj, bounds=(0.1, 0.9), method='bounded')
        return r.x, -r.fun
    
    ypp = best_m.predict_proba(X_val_sc)[:, 1]
    opt_t, opt_f1 = opt_thresh(y_val, ypp)
    final_t = opt_t if opt_f1 > all_res[best_name] else 0.5
    
    print(f'Optimized threshold: {final_t:.4f}, F1: {max(opt_f1, all_res[best_name]):.4f}')
    
    # Generate final predictions
    print("Generating final predictions...")
    test_proba = best_m.predict_proba(X_test_sc)[:, 1]
    test_pred = (test_proba >= final_t).astype(int)
    submission = pd.DataFrame({'Id': test_ids, 'Outcome': test_pred})
    
    # Save to local file instead of Kaggle working directory
    submission.to_csv('/kaggle/working/sample_submission.csv', index=False)
    print('Submission saved to sample_submission.csv!')
    print("Prediction distribution:")
    print(submission['Outcome'].value_counts())
else:
    print("Skipping predictions - no trained model available")

