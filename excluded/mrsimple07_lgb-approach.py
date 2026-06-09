# === PART 1: LOADING DATA & BASIC EDA ===
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ✅ Load Kaggle competition data
train = pd.read_csv("/kaggle/input/playground-series-s5e8/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e8/test.csv")
submission = pd.read_csv("/kaggle/input/playground-series-s5e8/sample_submission.csv")

# ✅ Load external bank marketing dataset (semicolon separated)
external = pd.read_csv("/kaggle/input/bank-marketing-dataset-full/bank-full.csv", sep=';')

# ✅ Map external target to binary with extra weight indicator
external['y'] = external['y'].map({'yes': 1, 'no': 0})
external['weight'] = np.where(external['y'] == 1, 2.0, 1.0)  # Double weight for minority class

# ✅ Combine Kaggle train with external dataset for larger training set
train['weight'] = 1.0  # Standard weight for original data
combined_train = pd.concat([train, external], axis=0, ignore_index=True)

print("Train shape:", train.shape)
print("Test shape:", test.shape)
print("External shape:", external.shape)
print("Combined Train Shape:", combined_train.shape)
print("\nTarget distribution:")
print(combined_train['y'].value_counts(normalize=True))

# ✅ Quick target distribution plot
plt.figure(figsize=(6,4))
sns.countplot(x='y', data=combined_train)
plt.title('Target Variable Distribution')
plt.show()

# ✅ Quick look at numeric features
num_cols = ['age','balance','day','duration','campaign','pdays','previous']
plt.figure(figsize=(15,10))
for i,col in enumerate(num_cols,1):
    plt.subplot(3,3,i)
    sns.histplot(combined_train[col], kde=True)
    plt.title(f'Distribution of {col}')
plt.tight_layout()
plt.show()


from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import mutual_info_classif
from sklearn.decomposition import PCA
import numpy as np

all_data = pd.concat([train.drop(['y', 'weight'], axis=1), test]).reset_index(drop=True)

def create_enhanced_features(df):
    # First encode categorical columns that will be used in numerical operations
    df['default_encoded'] = df['default'].map({'no': 0, 'yes': 1})
    df['housing_encoded'] = df['housing'].map({'no': 0, 'yes': 1})
    df['loan_encoded'] = df['loan'].map({'no': 0, 'yes': 1})
    
    # ====== Basic Numerical Transformations ======
    df['balance_to_age'] = df['balance'] / (df['age'] + 1)
    df['balance_to_duration'] = df['balance'] / (df['duration'] + 1)
    df['age_times_balance'] = df['age'] * df['balance']
    df['duration_per_contact'] = df['duration'] / (df['campaign'] + 1)
    
    # ====== Contact History Features ======
    df['has_been_contacted'] = (df['pdays'] != -1).astype(int)
    df['contact_success'] = ((df['pdays'] != -1) & (df['poutcome'] == 'success')).astype(int)
    df['contact_failure'] = ((df['pdays'] != -1) & (df['poutcome'] == 'failure')).astype(int)
    df['contact_other'] = ((df['pdays'] != -1) & (~df['poutcome'].isin(['success', 'failure']))).astype(int)
    df['contact_attempt_ratio'] = df['previous'] / (df['campaign'] + 1)
    
    # ====== Time-Based Features ======
    month_map = {'jan':1, 'feb':2, 'mar':3, 'apr':4, 'may':5, 'jun':6,
                 'jul':7, 'aug':8, 'sep':9, 'oct':10, 'nov':11, 'dec':12}
    df['month_num'] = df['month'].map(month_map)
    df['is_quarter_end'] = df['month_num'].isin([3, 6, 9, 12]).astype(int)
    df['day_of_week'] = df['day'] % 7
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    
    # ====== Lagged Features (without target leakage) ======
    # Sort by client and time variables first
    df = df.sort_values(['id', 'month_num', 'day'])
    
    # Previous campaign statistics
    df['prev_campaign_count'] = df.groupby('id')['campaign'].shift(1).fillna(0)
    df['prev_duration_avg'] = df.groupby('id')['duration'].transform(
        lambda x: x.shift(1).expanding().mean().fillna(0))
    
    # Time since last contact
    df['days_since_last_contact'] = df.groupby('id')['day'].transform(
        lambda x: x.diff().fillna(0))
    
    # ====== Advanced Interaction Features ======
    # Financial interactions - now using encoded values
    df['financial_stability'] = (df['balance'] / (df['age'] + 1)) * (1 - df['default_encoded'])
    df['credit_engagement'] = (df['loan_encoded'] + df['housing_encoded']) * df['previous']
    
    # Temporal interactions
    df['seasonal_contact_intensity'] = (df['month_num'] % 12) * df['campaign']
    df['weekend_cellular'] = ((df['day_of_week'] >= 5) & (df['contact'] == 'cellular')).astype(int)
    
    # Job-risk interactions
    job_risk_map = {
        'unemployed': 1.5, 'student': 1.2, 'retired': 0.8, 
        'management': 0.5, 'technician': 0.7, 'services': 1.0,
        'blue-collar': 1.1, 'entrepreneur': 0.9, 'housemaid': 1.0,
        'unknown': 1.0, 'self-employed': 0.8, 'admin.': 0.7
    }
    df['job_risk_score'] = df['job'].map(job_risk_map).fillna(1.0) * np.log1p(df['balance'].abs())
    
    # ====== Polynomial Features ======
    df['age_squared'] = df['age'] ** 2
    df['balance_log'] = np.log1p(np.abs(df['balance'])) * np.sign(df['balance'])
    df['campaign_squared'] = df['campaign'] ** 2
    
    # ====== Advanced Binning ======
    df['age_bin'] = pd.cut(df['age'], bins=[0, 20, 30, 40, 50, 60, 70, 100], 
                          labels=['0-20', '20-30', '30-40', '40-50', '50-60', '60-70', '70+'])
    df['balance_bin'] = pd.cut(df['balance'], bins=[-5000, -1000, 0, 1000, 5000, 10000, 20000, 50000, 100000],
                              labels=['<-1k', '-1k-0', '0-1k', '1k-5k', '5k-10k', '10k-20k', '20k-50k', '50k+'])
    df['duration_bin'] = pd.cut(df['duration'], bins=[0, 60, 120, 180, 240, 300, 600, 1200, 5000],
                               labels=['0-1m', '1-2m', '2-3m', '3-4m', '4-5m', '5-10m', '10-20m', '20m+'])
    
    # ====== Categorical Interactions ======
    df['housing_loan'] = df['housing'].astype(str) + '_' + df['loan'].astype(str)
    df['marital_education'] = df['marital'].astype(str) + '_' + df['education'].astype(str)
    df['job_marital'] = df['job'].astype(str) + '_' + df['marital'].astype(str)
    
    # ====== Campaign Features ======
    df['campaign_bin'] = pd.cut(df['campaign'], bins=[0, 1, 2, 3, 5, 10, 20, 50],
                               labels=['1', '2', '3', '4-5', '6-10', '11-20', '20+'])
    df['is_first_contact'] = (df['campaign'] == 1).astype(int)
    df['contact_frequency'] = df['campaign'] / (df['age'] / 30 + 1)  # Contacts per month of age
    
    # Drop temporary encoded columns
    df = df.drop(['default_encoded', 'housing_encoded', 'loan_encoded'], axis=1)
    
    return df
all_data = create_enhanced_features(all_data)

# Separate back into train and test
train_processed = all_data.iloc[:len(train)].copy()
test_processed = all_data.iloc[len(train):].copy()
train_processed['y'] = train['y']
train_processed['weight'] = train['weight']

# Identify feature types
num_cols = [col for col in all_data.columns if all_data[col].dtype in ['int64', 'float64'] 
            and col not in ['id', 'y', 'weight']]
cat_cols = [col for col in all_data.columns if all_data[col].dtype == 'object' 
            or isinstance(all_data[col].dtype, pd.CategoricalDtype)]

# Label encoding for categorical features
for col in cat_cols:
    le = LabelEncoder()
    le.fit(all_data[col])
    train_processed[col] = le.transform(train_processed[col])
    test_processed[col] = le.transform(test_processed[col])

# Standard scaling for numerical features
scaler = StandardScaler()
scaler.fit(all_data[num_cols])
train_processed[num_cols] = scaler.transform(train_processed[num_cols])
test_processed[num_cols] = scaler.transform(test_processed[num_cols])

# Final feature selection
features = num_cols + cat_cols
X = train_processed[features]
y = train_processed['y']
sample_weights = train_processed['weight']
X_test = test_processed[features]

# Feature importance analysis
mi = mutual_info_classif(X, y)
mi_df = pd.DataFrame({'feature': features, 'mi_score': mi}).sort_values('mi_score', ascending=False)
plt.figure(figsize=(10, 8))
sns.barplot(x='mi_score', y='feature', data=mi_df.head(20))
plt.title('Top 20 Features by Mutual Information')
plt.show()

# PCA visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
plt.figure(figsize=(10, 8))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y, alpha=0.5)
plt.title('PCA Visualization')
plt.show()



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier,
    AdaBoostClassifier, BaggingClassifier, VotingClassifier, HistGradientBoostingClassifier
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# Prepare data
X_all = X.copy()
y_all = y.copy()
X_test_all = X_test.copy()

# Class distribution
class_counts = np.bincount(y_all)
print(f"\nClass distribution: {dict(zip(np.unique(y_all), class_counts))}")
print(f"Class ratio: {class_counts[0] / class_counts[1]:.2f}:1")

# KFold setup
n_folds = 5
kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

# Model zoo for binary classification
models = {
    'LGBM': LGBMClassifier(
        objective='binary',
        metric='binary_logloss',
        # device='gpu',
        # gpu_use_dp=True,
        n_estimators=2000,
        learning_rate=0.05,
        random_state=42,
        verbose=-1
    ),
    # 'CatBoost': CatBoostClassifier(
    #     # task_type="GPU",
    #     cat_features=cat_cols,
    #     loss_function='Logloss',
    #     verbose=0,
    #     n_estimators=10000,
    #     learning_rate=0.05,
    #     random_state=42
    # ),
    # 'XGBoost': XGBClassifier(
    #     objective='binary:logistic',
    #     eval_metric='logloss',
    #     tree_method='gpu_hist',
    #     use_label_encoder=False,
    #     random_state=42
    # ),
    # 'GradientBoosting': GradientBoostingClassifier(n_estimators=300, random_state=42),
    # 'HistGB': HistGradientBoostingClassifier(
    #     loss='log_loss',             # For binary classification (default)
    #     learning_rate=0.05,           # Learning rate
    #     max_iter=3000,                # Number of boosting iterations
    #     max_depth=7,                 # Max depth of trees
    #     class_weight='balanced',     # Automatically adjust weights inversely proportional to class frequencies
    #     l2_regularization=1.0,       # Regularization to prevent overfitting
    #     max_leaf_nodes=31,           # Max leaf nodes per tree
    #     min_samples_leaf=20,         # Min samples in a leaf
    #     random_state=42),
    # 'Voting': VotingClassifier(estimators=[
    #     ('lr', LogisticRegression(max_iter=1000, solver='lbfgs', random_state=42)),
    #     ('rf', RandomForestClassifier(random_state=42)),
    #     ('cb', CatBoostClassifier(verbose=0, random_state=42))
    # ], voting='soft')
}




# Prepare submission frame
submission_df = pd.DataFrame()
submission_df['id'] = test["id"]

# Track results
results = []

# Train all models
for name, model in models.items():
    print(f"\n==== {name} ====")
    cv_scores = []
    oof_preds = np.zeros(len(X_all))
    test_preds = np.zeros(len(X_test_all))

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_all, y_all)):
        X_train, X_val = X_all.iloc[train_idx], X_all.iloc[val_idx]
        y_train, y_val = y_all.iloc[train_idx], y_all.iloc[val_idx]
        # Check if model supports sample_weight (e.g., HistGradientBoostingClassifier)
        if name.lower().startswith('HistGB'):
            weight_1 = (y_train == 0).sum() / (y_train == 1).sum()
            sample_weight = np.where(y_train == 1, weight_1, 1.0)
            model.fit(X_train, y_train, sample_weight=sample_weight)
        else: 
            model.fit(X_train, y_train)
        val_probs = model.predict_proba(X_val)[:, 1]
        val_preds = (val_probs > 0.5).astype(int)

        score = roc_auc_score(y_val, val_probs)
        cv_scores.append(score)
        oof_preds[val_idx] = val_probs

        print(f"Fold {fold + 1} AUC: {score:.4f}")
        print(classification_report(y_val, val_preds))

        cm = confusion_matrix(y_val, val_preds)
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'{name} - Fold {fold + 1} Confusion Matrix')
        plt.tight_layout()
        plt.show()

        test_preds += model.predict_proba(X_test_all)[:, 1] / n_folds

    mean_auc = np.mean(cv_scores)
    std_auc = np.std(cv_scores)
    print(f"\n{name} CV AUC: {mean_auc:.5f} ± {std_auc:.5f}")

    results.append((name, mean_auc, std_auc))
    submission_df[name] = test_preds

# Show model comparison
results_df = pd.DataFrame(results, columns=["Model", "Mean AUC", "Std AUC"])
print("\n==== Model AUC Summary ====")
print(results_df.sort_values("Mean AUC", ascending=False))

# Save best model predictions
top_model = results_df.sort_values("Mean AUC", ascending=False).iloc[0]['Model']
submission_df['y'] = submission_df[top_model]
submission_df[['id', 'y']].to_csv('/kaggle/working/submission.csv', index=False)
print(f"\nSubmission saved using top model: {top_model}")


