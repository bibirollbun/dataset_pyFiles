# 1. Import Libraries
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# For warnings
import warnings
warnings.filterwarnings('ignore')



test_categorical =pd.read_excel("/kaggle/input/widsdatathon2025/TEST/TEST_CATEGORICAL.xlsx")
test_categorical.head()


test_connectome=pd.read_csv("/kaggle/input/widsdatathon2025/TEST/TEST_FUNCTIONAL_CONNECTOME_MATRICES.csv")
test_connectome.head()


test_quantitative=pd.read_excel("/kaggle/input/widsdatathon2025/TEST/TEST_QUANTITATIVE_METADATA.xlsx")
test_quantitative.head()


train_solutions= pd.read_excel("/kaggle/input/widsdatathon2025/TRAIN_NEW/TRAINING_SOLUTIONS.xlsx")
train_solutions


train_categorical=pd.read_excel("/kaggle/input/widsdatathon2025/TRAIN_NEW/TRAIN_CATEGORICAL_METADATA_new.xlsx")
train_categorical.head()


train_connectome = pd.read_csv("/kaggle/input/widsdatathon2025/TRAIN_NEW/TRAIN_FUNCTIONAL_CONNECTOME_MATRICES_new_36P_Pearson.csv")
train_connectome .head()


train_quantitative= pd.read_excel("/kaggle/input/widsdatathon2025/TRAIN_OLD/TRAIN_QUANTITATIVE_METADATA.xlsx")
train_quantitative.head()


sample_submission=pd.read_excel("/kaggle/input/widsdatathon2025/SAMPLE_SUBMISSION.xlsx")
sample_submission


# ============================
# 1. Imports
# ============================
import pandas as pd
import numpy as np

# Models and preprocessing
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from lightgbm import LGBMClassifier

# Evaluation
from sklearn.metrics import accuracy_score, classification_report

# Visualization
import seaborn as sns
import matplotlib.pyplot as plt

# Ignore warnings
import warnings
warnings.filterwarnings('ignore')



# Test Data
path_test_cat = "/kaggle/input/widsdatathon2025/TEST/TEST_CATEGORICAL.xlsx"
path_test_conn = "/kaggle/input/widsdatathon2025/TEST/TEST_FUNCTIONAL_CONNECTOME_MATRICES.csv"
path_test_quant = "/kaggle/input/widsdatathon2025/TEST/TEST_QUANTITATIVE_METADATA.xlsx"

# Train Data
train_solutions = "/kaggle/input/widsdatathon2025/TRAIN_NEW/TRAINING_SOLUTIONS.xlsx"
path_train_cat = "/kaggle/input/widsdatathon2025/TRAIN_NEW/TRAIN_CATEGORICAL_METADATA_new.xlsx"
path_train_conn = "/kaggle/input/widsdatathon2025/TRAIN_NEW/TRAIN_FUNCTIONAL_CONNECTOME_MATRICES_new_36P_Pearson.csv"
path_train_quant = "/kaggle/input/widsdatathon2025/TRAIN_OLD/TRAIN_QUANTITATIVE_METADATA.xlsx"

# Ø¶Ø¨Ø· Ø¨Ø¹Ø¶ Ø§Ù„Ø®ÙŠØ§Ø±Ø§Øª Ù„Ù„Ø¹Ø±Ø¶
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
path_sample_sub = "/kaggle/input/widsdatathon2025/SAMPLE_SUBMISSION.xlsx"

# Ù‚Ø±Ø§Ø¡Ø© Ø§Ù„Ù…Ù„Ù�Ø§Øª
train_cat = pd.read_excel(path_train_cat)
train_quant = pd.read_excel(path_train_quant)
train_conn = pd.read_csv(path_train_conn)
test_cat = pd.read_excel(path_test_cat)
test_quant = pd.read_excel(path_test_quant)
test_conn = pd.read_csv(path_test_conn)
sample_sub = pd.read_excel(path_sample_sub)
train_solutions = pd.read_excel(train_solutions)
# Ø¯Ø§Ù„Ø© Ù…Ø³Ø§Ø¹Ø¯Ø© Ù„Ø¹Ù…Ù„ Ù…Ù„Ø®Øµ Ø³Ø±ÙŠØ¹ Ù„Ø£ÙŠ DataFrame
def quick_summary(df, name="Data"):
    print(f"\n{'='*50}")
    print(f"Summary for {name}:")
    print(f"Shape: {df.shape}")
    print("Data Types:")
    print(df.dtypes.value_counts())
    print("\nMissing Values (top 10 columns):")
    print(df.isnull().sum().sort_values(ascending=False).head(10))
    print(f"{'='*50}\n")

# ØªØ·Ø¨ÙŠÙ‚ Ø§Ù„Ø¯Ø§Ù„Ø© Ø¹Ù„Ù‰ Ø¬Ù…ÙŠØ¹ Ø§Ù„Ù…Ù„Ù�Ø§Øª
quick_summary(train_cat, "Train Categorical")
quick_summary(train_quant, "Train Quantitative")
quick_summary(train_conn, "Train Connectome")
quick_summary(test_cat, "Test Categorical")
quick_summary(test_quant, "Test Quantitative")
quick_summary(test_conn, "Test Connectome")

# Ù†Ø¸Ø±Ø© Ø®Ø§ØµØ© Ø¹Ù„Ù‰ ØªÙˆØ²ÙŠØ¹ Ø§Ù„Ù‡Ø¯Ù� (ADHD_Outcome)
if 'ADHD_Outcome' in train_cat.columns:
    target_counts = train_cat['ADHD_Outcome'].value_counts(normalize=True)
    print("Target Distribution (Train Categorical - ADHD_Outcome):")
    print(target_counts)
    # Ø±Ø³Ù… Ø§Ù„ØªÙˆØ²ÙŠØ¹
    plt.figure(figsize=(6,4))
    sns.barplot(x=target_counts.index, y=target_counts.values)
    plt.title('ADHD Outcome Distribution')
    plt.xlabel('Class')
    plt.ylabel('Percentage')
    plt.show()



# ------------------- Cleaning Categorical Data -------------------

# Ø§Ù„Ù…Ø´Ø§Ø±ÙƒÙŠÙ† Ù†Ù�Ø³Ù‡Ù… Ø¨ÙƒÙ„ Ø§Ù„Ù…Ù„Ù�Ø§Øª Ù�Ù„Ø§Ø²Ù… Ù†Ø±ØªØ¨ IDs
train_cat['participant_id'] = train_cat['participant_id'].astype(str)
test_cat['participant_id'] = test_cat['participant_id'].astype(str)

# Ø§Ù„Ù‚ÙŠÙ… Ø§Ù„Ù†ØµÙŠØ© NaNs --> Ù†Ø­Ø· Ø¨Ø¯Ù„Ù‡Ø§ "Unknown"
cat_cols = ['Barratt_Barratt_P2_Occ', 'Barratt_Barratt_P2_Edu', 
            'PreInt_Demos_Fam_Child_Race', 'PreInt_Demos_Fam_Child_Ethnicity',
            'Barratt_Barratt_P1_Occ', 'Barratt_Barratt_P1_Edu', 
            'MRI_Track_Scan_Location']

for col in cat_cols:
    #train_cat[col] = train_cat[col].fillna('Unknown')
    train_cat[col] = train_cat[col].dropna()
    test_cat[col] = test_cat[col].dropna()
    #test_cat[col] = test_cat[col].fillna('Unknown')

# ------------------- Cleaning Quantitative Data -------------------

# participant_id Ø§Ù„Ù‰ str Ø¹Ø´Ø§Ù† Ø§Ù„Ø¯Ù…Ø¬ Ù„Ø§Ø­Ù‚Ø§Ù‹
train_quant['participant_id'] = train_quant['participant_id'].astype(str)
test_quant['participant_id'] = test_quant['participant_id'].astype(str)

# Ø§Ù„Ø£Ø¹Ù…Ø¯Ø© Ø§Ù„Ø±Ù‚Ù…ÙŠØ© NaNs --> Ù†Ø¹ÙˆØ¶ Ø¨Ø§Ù„Ù…ÙŠØ¯ÙŠØ§Ù† (Ø£ÙƒØ«Ø± Ø£Ù…Ø§Ù† Ù…Ø¹ outliers)
quant_cols = [col for col in train_quant.columns if col != 'participant_id']

for col in quant_cols:
    median_value = train_quant[col].median()
    train_quant[col] = train_quant[col].fillna(median_value)

for col in quant_cols:
    median_value = test_quant[col].median()
    test_quant[col] = test_quant[col].fillna(median_value)

# ------------------- Cleaning Connectome Data -------------------

# participant_id Ø§Ù„Ù‰ str
train_conn['participant_id'] = train_conn['participant_id'].astype(str)
test_conn['participant_id'] = test_conn['participant_id'].astype(str)

# connectome Ù†Ø¸ÙŠÙ� (Ù…Ø§Ù�ÙŠÙ‡ missing), Ù�Ù‚Ø· Ù†ØªØ£ÙƒØ¯ Ù…Ù† Ø§Ù„Ø¯Ø§ØªØ§ ØªØ§ÙŠØ¨
conn_cols = [col for col in train_conn.columns if col != 'participant_id']

train_conn[conn_cols] = train_conn[conn_cols].astype(float)
test_conn[conn_cols] = test_conn[conn_cols].astype(float)

print("âœ… Data Cleaning Done!")



import seaborn as sns
import matplotlib.pyplot as plt

# ---------------- Age Distribution ----------------
if 'MRI_Track_Age_at_Scan' in train_quant.columns:
    plt.figure(figsize=(8,5))
    sns.histplot(train_quant['MRI_Track_Age_at_Scan'], kde=True, bins=30, color='skyblue')
    plt.title('Age at MRI Scan Distribution')
    plt.xlabel('Age')
    plt.ylabel('Frequency')
    plt.grid()
    plt.show()

# ---------------- Correlation Heatmap ----------------
plt.figure(figsize=(10,8))
corr = train_quant.drop(columns=['participant_id']).corr()
sns.heatmap(corr, cmap='coolwarm', center=0, annot=False)
plt.title('Correlation Heatmap - Quantitative Features')
plt.show()

# ---------------- EHQ_EHQ_Total Distribution ----------------
if 'EHQ_EHQ_Total' in train_quant.columns:
    plt.figure(figsize=(8,5))
    sns.histplot(train_quant['EHQ_EHQ_Total'], kde=True, bins=30, color='salmon')
    plt.title('EHQ_EHQ_Total Distribution')
    plt.xlabel('EHQ_EHQ_Total Value')
    plt.ylabel('Frequency')
    plt.grid()
    plt.show()

    # Ù†Ø´ÙŠÙƒ Ù„Ùˆ Ù�ÙŠÙ‡ Ù‚ÙŠÙ… Ø³Ø§Ù„Ø¨Ø©
    negatives = train_quant[train_quant['EHQ_EHQ_Total'] < 0]
    print(f"Ø¹Ø¯Ø¯ Ø§Ù„Ù‚ÙŠÙ… Ø§Ù„Ø³Ø§Ù„Ø¨Ø© Ù�ÙŠ EHQ_EHQ_Total: {len(negatives)}")





# ----------------- Merge Train Data -----------------

# Ø¯Ù…Ø¬ Ø§Ù„Ø¨ÙŠØ§Ù†Ø§Øª Ø¨Ù†Ø§Ø¡Ù‹ Ø¹Ù„Ù‰ participant_id
train_full = train_cat.merge(train_quant, on='participant_id', how='left')
train_full = train_full.merge(train_conn, on='participant_id', how='left')

print(f"Train Full Shape: {train_full.shape}")

# ----------------- Merge Test Data -----------------

test_full = test_cat.merge(test_quant, on='participant_id', how='left')
test_full = test_full.merge(test_conn, on='participant_id', how='left')

print(f"Test Full Shape: {test_full.shape}")


train_solutions 


# ----------------- Ù�ØµÙ„ Ø§Ù„Ù‡Ø¯Ù� y -----------------
# Ø­Ù�Ø¸ target
y = train_solutions[["ADHD_Outcome","Sex_F"]]

# Ø­Ø°Ù� target Ùˆ participant_id Ù…Ù† X
X = train_full.drop(columns=['participant_id'])
X_test = test_full.drop(columns=['participant_id'])

print(f"Train Features Shape: {X.shape}")
print(f"Test Features Shape: {X_test.shape}")



import xgboost as xgb
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score


# Encode Ø§Ù„Ø£Ø¹Ù…Ø¯Ø© object
for col in X.columns:
    if X[col].dtype == 'object':
        le = LabelEncoder()
        full_col_data = pd.concat([X[col], X_test[col]], axis=0)
        le.fit(full_col_data.astype(str))
        X[col] = le.transform(X[col].astype(str))
        X_test[col] = le.transform(X_test[col].astype(str))

# Fill missing values temporarily (for pipeline later)
X.fillna(-999, inplace=True)
X_test.fillna(-999, inplace=True)


# Train & Validation
X_train, X_val, y_train, y_val = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)



# ----------------- Build Pipeline -----------------
# Ù…ÙˆØ¯ÙŠÙ„ XGBoost
xgb_model = xgb.XGBClassifier(
    n_estimators=100,
    learning_rate=0.01,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    #verbosity=0,
   # tree_method='hist'  # fast with larger datasets
)

# Ø¨Ù†Ø§Ø¡ Pipeline
pipeline = Pipeline([
 #   ('imputer', SimpleImputer(strategy='median')),  # Ù†Ù…Ù„Ø£ Ø§Ù„Ù†ÙˆØ§Ù‚Øµ Ø¨Ø§Ù„Ù…ÙŠØ¯ÙŠØ§Ù†
    ('scaler', StandardScaler()),                  # Ù†Ø³ÙƒÙ„ Ø§Ù„Ù�ÙŠØªØ´Ø±Ø§Øª
    ('classifier', MultiOutputClassifier(xgb_model))  # Ù…ÙˆØ¯ÙŠÙ„ Ù…ØªØ¹Ø¯Ø¯ Ø§Ù„Ø£Ù‡Ø¯Ø§Ù�
])

# ----------------- Train Pipeline -----------------

pipeline.fit(X_train, y_train)


# ----------------- Validation -----------------

# ØªÙˆÙ‚Ø¹ Ø¹Ù„Ù‰ Validation
val_preds = pipeline.predict_proba(X_val)

# MultiOutput gives list of arrays 
val_preds_proba = np.vstack([pred[:,1] for pred in val_preds]).T

#  ROC AUC for each target 
auc_scores = []
for i in range(y.shape[1]):
    auc = roc_auc_score(y_val.iloc[:, i], val_preds_proba[:, i])
    auc_scores.append(auc)
    print(f"ğŸ”¥ AUC for Target {y.columns[i]}: {auc:.4f}")

print(f"\nğŸš€ Average AUC: {np.mean(auc_scores):.4f}")

# ----------------- Predict on Test Data -----------------

#  Test pred
test_preds = pipeline.predict_proba(X_test)


#submission to  be prerapred
submission = sample_sub.copy()
for idx, col in enumerate(submission.columns[1:]):  # Ù†Ù�ØªØ±Ø¶ Ø£ÙˆÙ„ Ø¹Ù…ÙˆØ¯ participant_id
    submission[col] = test_preds[idx][:, 1]  # Ø§Ù„Ø§Ø­ØªÙ…Ø§Ù„ Ù„Ù„ØµÙ� Ø§Ù„Ø£ÙˆÙ„ Ù„ÙƒÙ„ Ù‡Ø¯Ù�

submission.to_csv('xgboost_multilabel_submission.csv', index=False)
print("\nâœ… Submission file saved as 'xgboost_multilabel_submission.csv'")


