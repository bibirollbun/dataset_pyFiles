import os 
os.listdir('/kaggle/input/')


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import warnings

warnings.filterwarnings("ignore")

df_train_id = pd.read_csv('/kaggle/input/ieee-fraud-detection/train_identity.csv')
df_train_txn = pd.read_csv('/kaggle/input/ieee-fraud-detection/train_transaction.csv')
df_test_id = pd.read_csv('/kaggle/input/ieee-fraud-detection/test_identity.csv')
df_test_txn = pd.read_csv('/kaggle/input/ieee-fraud-detection/test_transaction.csv')

df_sample = pd.read_csv('/kaggle/input/ieee-fraud-detection/sample_submission.csv')

print("Full Train Set")
df = pd.merge(df_train_txn, df_train_id, on = 'TransactionID', how = 'left' )
df.head()



num_cols = df.select_dtypes(include = ['int64', 'float64']).drop('isFraud', axis=1).columns.tolist()

corr_matrix = df[num_cols].sample(n=10000, random_state=42).corr(method='spearman').abs()

upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

to_drop = [column for column in upper.columns if any(upper[column] > 0.9)]
df_reduced = df.drop(columns=to_drop)

print(to_drop)

print(f"Dropped {len(to_drop)} correlated features.")




from sklearn.model_selection import train_test_split

X = df_reduced.drop('isFraud', axis = 1)
y = df_reduced['isFraud']

X_temp, X_test, y_temp, y_test = train_test_split(
    X, y , test_size = 0.2, random_state = 42, stratify = y
)

X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp , test_size = 0.25, random_state = 42, stratify = y_temp
)

print("Train:", X_train.shape, y_train.shape)
print("Validation:", X_val.shape, y_val.shape)
print("Test:", X_test.shape, y_test.shape)


from scipy.stats import zscore
from sklearn.preprocessing import StandardScaler, RobustScaler, PowerTransformer, OrdinalEncoder
from category_encoders import TargetEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

num_cols = X_train.select_dtypes(include = ['int64', 'float64']).columns.tolist()
cat_cols = X_train.select_dtypes(include =['object']).columns.tolist()

skews = X_train[num_cols].apply(lambda x: x.skew(skipna=True)).fillna(0)

normal_features = skews[abs(skews) < 1].index.tolist()        
moderate_features = skews[(abs(skews) >= 1) & (abs(skews) < 5)].index.tolist()   
heavy_features = skews[abs(skews) >= 5].index.tolist()      

print(f"Normal features: {len(normal_features)}")
print(f"Moderate features: {len(moderate_features)}")
print(f"Heavy features: {len(heavy_features)}")



for col in moderate_features:
    lower, upper = X_train[col].quantile([0.01, 0.99])
    X_train[col] = X_train[col].clip(lower, upper)


def make_preprocessor(model_name):
    standard_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    robust_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', RobustScaler())
    ])

    power_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('power', PowerTransformer(method='yeo-johnson', standardize=True)),
        ('scaler', RobustScaler())
    ])

    cat_pipe_tree = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
    ])

    cat_pipe_target = Pipeline([
        ('imputer', SimpleImputer(strategy = 'most_frequent')),
        ('encoder', TargetEncoder())
    ])

    if model_name in ['LogisticRegression', 'SVM(RBF)', 'KNN', 'NaiveBayes']:
        cat_pipe = cat_pipe_target
    else:
        cat_pipe = cat_pipe_tree

    preprocessor = ColumnTransformer([
        ('standard', standard_pipe, normal_features),
        ('robust', robust_pipe, moderate_features),
        ('power', power_pipe, heavy_features),
        ('cat', cat_pipe, cat_cols)
    ], remainder='drop', sparse_threshold=0)  

    return preprocessor
    
print("\nSkew summary before transformations:")
print(skews.describe())



# preprocessor.fit(X_train)

# X_train_transf = preprocessor.transform(X_train)
# X_val_transf = preprocessor.transform(X_val)

# print("Transformed shapes:", X_train_transf.shape, X_val_transf.shape)

# print("Any NaNs?", np.isnan(X_train_transf).any(), np.isnan(X_val_transf).any())
# print("Any infs?", np.isinf(X_train_transf).any(), np.isinf(X_val_transf).any())
# print("Max absolute value:", np.max(np.abs(X_train_transf)))



from sklearn.model_selection import learning_curve
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
# from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
import joblib


models = {
    'LogisticRegression': LogisticRegression(max_iter=2000, random_state=42),
    'RandomForest': RandomForestClassifier(random_state=42, n_estimators=200),
    'GradientBoosting': GradientBoostingClassifier(random_state=42),
    'AdaBoost': AdaBoostClassifier(random_state=42),
    # 'SVM (RBF)': SVC(probability=True, random_state=42),
    'NaiveBayes': GaussianNB(),
    'KNN': KNeighborsClassifier(),
    'XGBoost': XGBClassifier(
        random_state=42, eval_metric='logloss', use_label_encoder=False,
        tree_method='hist', n_estimators=300
    )
}

results = []

for name, model in models.items():
    preprocessor = make_preprocessor(name)
    clf_pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])

    clf_pipe.fit(X_train, y_train)
    y_val_pred = clf_pipe.predict(X_val)
    y_val_proba = clf_pipe.predict_proba(X_val)[:, 1] if hasattr(clf_pipe[-1], 'predict_proba') else y_val_pred

    acc = accuracy_score(y_val, y_val_pred)
    f1 = f1_score(y_val, y_val_pred)
    auc = roc_auc_score(y_val, y_val_proba)

    print(f"\n {name}")
    print(f"Accuracy: {acc:.4f} | F1: {f1:.4f} | ROC-AUC: {auc:.4f}")

    cm = confusion_matrix(y_val, y_val_pred)
    ConfusionMatrixDisplay(cm).plot(cmap='Blues', values_format='d')
    plt.title(f"{name} - Confusion Matrix")
    plt.show()

    results.append({'model': name, 'val_accuracy': acc, 'val_f1': f1, 'val_auc': auc})
    OUTPUT_DIR = "/kaggle/working/"
    joblib.dump(clf_pipe, os.path.join(OUTPUT_DIR, f"{name}_pipeline.joblib"))

results_df = pd.DataFrame(results).sort_values(by='val_auc', ascending=False)
print("\n Final Results:")
print(results_df)





from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt

def plot_learning_curve(estimator, X, y, title, scoring='f1_weighted', cv=5):
    train_sizes, train_scores, val_scores = learning_curve(
        estimator, X, y,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 8),  
        random_state=42,
        shuffle=True
    )

    train_mean = np.mean(train_scores, axis=1)
    train_std  = np.std(train_scores, axis=1)
    val_mean   = np.mean(val_scores, axis=1)
    val_std    = np.std(val_scores, axis=1)

    plt.figure(figsize=(8, 5))
    plt.plot(train_sizes, train_mean, 'o-', color='blue', label='Training score')
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std,
                     alpha=0.15, color='blue')

    plt.plot(train_sizes, val_mean, 'o-', color='green', label='Validation score')
    plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std,
                     alpha=0.15, color='green')

    plt.title(title, fontsize=14, pad=15)
    plt.xlabel('Training Set Size')
    plt.ylabel(scoring.replace('_', ' ').title())
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


import joblib
import os

OUTPUT_DIR = '/kaggle/working'
model_files = [
    'RandomForest_pipeline.joblib',
    'XGBoost_pipeline.joblib',
    'GradientBoosting_pipeline.joblib'
]

models = {}
for fname in model_files:
    fpath = os.path.join(OUTPUT_DIR, fname)
    
    if not os.path.exists(fpath):
        raise FileNotFoundError(f"Model not found: {fpath}")
    
    model = joblib.load(fpath)
    
    key = fname.replace('_pipeline.joblib', '').lower()
    models[key] = model
    
    print(f"Loaded {fname} → models['{key}']")


rf_pipe  = models['randomforest']
xgb_pipe = models['xgboost']
gb_pipe  = models['gradientboosting']


plot_learning_curve(rf_pipe, X_train, y_train, "RandomForest Learning Curve")
plot_learning_curve(xgb_pipe, X_train, y_train, "XGBoost Learning Curve")
plot_learning_curve(gb_pipe, X_train, y_train, "GradientBoosting Learning Curve")



X_train.shape


from xgboost import XGBClassifier

if hasattr(xgb_pipe, 'named_steps'):  
    model = xgb_pipe.named_steps['model']
    if isinstance(model, XGBClassifier):
        model.set_params(tree_method='gpu_hist', predictor='gpu_predictor', gpu_id=0)
        print(" XGBoost set to use GPU")


import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline

xgb_params = {
    "n_estimators": np.logspace(1, 4, 6, dtype=int),
    "max_depth": [1, 3, 5, 7, 9],
    "learning_rate": np.logspace(-4, -1, 6)
}


search_pipe = RandomizedSearchCV(
    estimator=xgb_pipe,
    param_distributions={
        "model__n_estimators": xgb_params["n_estimators"],
        "model__max_depth": xgb_params["max_depth"],
        "model__learning_rate": xgb_params["learning_rate"],
    },
    n_iter=20,          
    cv=3,                
    scoring="f1",
    n_jobs=-1,           
    verbose=2,
    random_state=42
)

search_pipe.fit(X_train, y_train)


