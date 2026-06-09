import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score, classification_report
from sklearn.model_selection import StratifiedKFold
from lightgbm import LGBMClassifier


train = pd.read_csv("/kaggle/input/icr-identify-age-related-conditions/train.csv")
test = pd.read_csv("/kaggle/input/icr-identify-age-related-conditions/test.csv")
sample_submission = pd.read_csv("/kaggle/input/icr-identify-age-related-conditions/sample_submission.csv")


train.head()


train.shape


test.shape


train.info()


train.describe()


train['Class'].value_counts(normalize=True)


#missing values
missing = train.isnull().sum().sort_values(ascending=False)
missing[missing > 0]


num_cols = train.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_cols = train.select_dtypes(include=['object']).columns.tolist()


print("Numeric columns:", len(num_cols))
print("Categorical columns:", len(cat_cols))


print("Missing values:\n", train.isnull().sum().sort_values(ascending=False).head(10))


target = 'Class'
if target in num_cols:
    num_cols.remove(target)

train[num_cols] = train[num_cols].fillna(train[num_cols].median())
test[num_cols] = test[num_cols].fillna(train[num_cols].median())


for col in cat_cols:
    mode_val = train[col].mode()[0]
    train[col] = train[col].fillna(mode_val)
    test[col] = test[col].fillna(mode_val)
    
print("Missing values after filling:", train.isnull().sum().sum())


#Class Distribution
sns.countplot(x='Class', data=train)
plt.title('Class distribution')
plt.show()


#Numeric Distribution
train[num_cols].hist(figsize=(20, 16), bins=30)
plt.suptitle('Numeric Distribution', fontsize=14)
plt.show()


#Checking skewness
skew_val = train[num_cols].skew().sort_values(ascending=False)
print("Top 15 skewed:\n", skew_val.head(15))


#Correlation with class
correlation = train[num_cols].corrwith(train['Class']).sort_values(ascending=False)
print("Top 10 correlation:\n", correlation.head(10))
print("\nLowest 10 correlation:\n", correlation.tail(10))


corr_matrix = train[num_cols].corr()


#heatmap visualization
plt.figure(figsize=(20,15))
sns.heatmap(corr_matrix, cmap="coolwarm", center=0)
plt.title("Correlation Heatmap:", fontsize=14)
plt.show()


#Let's find which are the highly correlated pair
threshold=0.90
high_corr = [
    (col1, col2, corr_matrix.loc[col1, col2])
    for col1 in corr_matrix.columns
    for col2 in corr_matrix.columns
    if (col1 != col2 and abs(corr_matrix.loc[col1, col2]) > threshold)
]
print(f"Highly Correlated Pairs (>|{threshold}|):")
for pair in high_corr[:10]:
    print(pair)


X = train.drop(columns=['Class'])
y = train['Class']


X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


# preprocessing - numeric and categorical columns
num_pre = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

cat_pre = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])


#Combined Preprocessing
preprocessor = ColumnTransformer(
    transformers = [
        ('num', num_pre, num_cols),
        ('cat', cat_pre, cat_cols)
    ]
)


print("Numeric Columns:", len(num_cols))
print("Categorical Columns:", len(cat_cols))


lr_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42))
])

lr_model.fit(X_train, y_train)


y_pred_prob = lr_model.predict_proba(X_valid)[:,1]
y_pred = (y_pred_prob > 0.5).astype(int)


print("Validation Log-Loss:", log_loss(y_valid, y_pred_prob))
print("\nValidation ROC-AUC:", roc_auc_score(y_valid, y_pred))
print("\nClassification Report:\n", classification_report(y_valid,y_pred))


skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


fold_scores = []
for fold, (train_idx, valid_idx) in enumerate(skf.split(X, y), 1):
    X_train_fold = X.iloc[train_idx]
    X_valid_fold = X.iloc[valid_idx]
    y_train_fold = y.iloc[train_idx]
    y_valid_fold = y.iloc[valid_idx]

    lr_model.fit(X_train_fold, y_train_fold)
    y_pred_proba = lr_model.predict_proba(X_valid_fold)[:, 1]

    loss = log_loss(y_valid_fold, y_pred_proba)
    auc = roc_auc_score(y_valid_fold, y_pred_proba)

    fold_scores.append((loss, auc))
    print(f"Fold {fold}: LogLoss={loss:.5f}, AUC={auc:.5f}")


all_losses = []
all_aucs = []
for f in fold_scores:
    loss_value = f[0]
    all_losses.append(loss_value)
    auc_value = f[1]
    all_aucs.append(auc_value)

mean_loss = np.mean(all_losses)
mean_auc = np.mean(all_aucs)

print("Average CV Log Loss:", round(mean_loss, 5))
print("Average CV AUC:", round(mean_auc, 5))


# Handling the issue of ValueError: X has 551 features, but LGBMClassifier is expecting 550 (Column Transformer Issue)
preprocessor.fit(X)
X_processed = preprocessor.transform(X)


lgbm_model = LGBMClassifier(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=-1,
    num_leaves=31,
    random_state=42,
    n_jobs=-1,
    force_col_wise=True,
    verbosity=-1   # disables all LightGBM warnings
)


fold_scores = []

for fold, (train_idx, valid_idx) in enumerate(skf.split(X_processed, y), 1):
    X_train_fold, X_valid_fold = X_processed[train_idx], X_processed[valid_idx]
    y_train_fold, y_valid_fold = y.iloc[train_idx], y.iloc[valid_idx]
    
    lgbm_model.fit(X_train_fold, y_train_fold)
    y_pred_proba = lgbm_model.predict_proba(X_valid_fold)[:, 1]
    
    loss = log_loss(y_valid_fold, y_pred_proba)
    auc = roc_auc_score(y_valid_fold, y_pred_proba)
    
    fold_scores.append((loss, auc))
    print(f"Fold {fold}: LogLoss={loss:.5f}, AUC={auc:.5f}")

# Average performance
mean_loss = np.mean([f[0] for f in fold_scores])
mean_auc = np.mean([f[1] for f in fold_scores])
print("\nAverage CV LogLoss:", round(mean_loss, 5))
print("Average CV AUC:", round(mean_auc, 5))


preprocessor.fit(X)
X_processed = preprocessor.transform(X)
X_test_processed = preprocessor.transform(test)


log_reg = LogisticRegression(
    max_iter=1000,
    class_weight='balanced',
    random_state=42
)

log_reg.fit(X_processed, y)
log_reg_pred = log_reg.predict_proba(X_test_processed)[:, 1]


lgbm = LGBMClassifier(
    n_estimators=1000,
    learning_rate=0.05,
    num_leaves=31,
    random_state=42,
    n_jobs=-1,
    force_col_wise=True,
    verbosity=-1
)

lgbm.fit(X_processed, y)
lgbm_pred = lgbm.predict_proba(X_test_processed)[:, 1]


# Blend prediction
blend_pred = (log_reg_pred + lgbm_pred) / 2


# Creating submission
# Logistic Regression submission
submission_lr = sample_submission.copy()
submission_lr['class_1'] = log_reg_pred
submission_lr['class_0'] = 1 - log_reg_pred
submission_lr.to_csv("submission_logreg.csv", index=False)

#LightGBM submission
submission_lgbm = sample_submission.copy()
submission_lgbm['class_1'] = lgbm_pred
submission_lgbm['class_0'] = 1 - lgbm_pred
submission_lgbm.to_csv("submission_lgbm.csv", index=False)

# Blended submission
submission_blend = sample_submission.copy()
submission_blend['class_1'] = blend_pred
submission_blend['class_0'] = 1 - blend_pred
submission_blend.to_csv("submission.csv", index=False)

print("Submissions saved:")
print("submission_logreg.csv")
print("submission_lgbm.csv")
print("submission.csv")

