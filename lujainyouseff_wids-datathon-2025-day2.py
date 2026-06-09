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

# ضبط بعض الخيارات للعرض
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
path_sample_sub = "/kaggle/input/widsdatathon2025/SAMPLE_SUBMISSION.xlsx"

# قراءة الملفات
train_cat = pd.read_excel(path_train_cat)
train_quant = pd.read_excel(path_train_quant)
train_conn = pd.read_csv(path_train_conn)
test_cat = pd.read_excel(path_test_cat)
test_quant = pd.read_excel(path_test_quant)
test_conn = pd.read_csv(path_test_conn)
sample_sub = pd.read_excel(path_sample_sub)
train_solutions = pd.read_excel(train_solutions)


# quick_summary
def quick_summary(df, name="Data"):
    print(f"\n{'='*50}")
    print(f"Summary for {name}:")
    print(f"Shape: {df.shape}")
    print("Data Types:")
    print(df.dtypes.value_counts())
    print("\nMissing Values (top 10 columns):")
    print(df.isnull().sum().sort_values(ascending=False).head(10))
    print(f"{'='*50}\n")

# 
quick_summary(train_cat, "Train Categorical")
quick_summary(train_quant, "Train Quantitative")
quick_summary(train_conn, "Train Connectome")
quick_summary(test_cat, "Test Categorical")
quick_summary(test_quant, "Test Quantitative")
quick_summary(test_conn, "Test Connectome")

# (ADHD_Outcome)
if 'ADHD_Outcome' in train_cat.columns:
    target_counts = train_cat['ADHD_Outcome'].value_counts(normalize=True)
    print("Target Distribution (Train Categorical - ADHD_Outcome):")
    print(target_counts)
    plt.figure(figsize=(6,4))
    sns.barplot(x=target_counts.index, y=target_counts.values)
    plt.title('ADHD Outcome Distribution')
    plt.xlabel('Class')
    plt.ylabel('Percentage')
    plt.show()



# IDs common 
train_cat['participant_id'] = train_cat['participant_id'].astype(str)
test_cat['participant_id'] = test_cat['participant_id'].astype(str)

#  NaNs -->  "Unknown"
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

# participant_id to str
train_quant['participant_id'] = train_quant['participant_id'].astype(str)
test_quant['participant_id'] = test_quant['participant_id'].astype(str)

#  numerical cols NaNs 
quant_cols = [col for col in train_quant.columns if col != 'participant_id']

for col in quant_cols:
    median_value = train_quant[col].median()
    train_quant[col] = train_quant[col].fillna(median_value)

for col in quant_cols:
    median_value = test_quant[col].median()
    test_quant[col] = test_quant[col].fillna(median_value)

# ------------------- Cleaning Connectome Data -------------------

# participant_id to str
train_conn['participant_id'] = train_conn['participant_id'].astype(str)
test_conn['participant_id'] = test_conn['participant_id'].astype(str)

# check missing connectome 
conn_cols = [col for col in train_conn.columns if col != 'participant_id']

train_conn[conn_cols] = train_conn[conn_cols].astype(float)
test_conn[conn_cols] = test_conn[conn_cols].astype(float)

print("✅ Data Cleaning Done!")



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

    # نشيك لو فيه قيم سالبة
    negatives = train_quant[train_quant['EHQ_EHQ_Total'] < 0]
    print(f"num of negative values EHQ_EHQ_Total: {len(negatives)}")



import seaborn as sns
import matplotlib.pyplot as plt

# ---------------- Distribution BEFORE Cleaning ----------------
plt.figure(figsize=(8,5))
sns.histplot(train_quant['EHQ_EHQ_Total'], kde=True, bins=30, color='salmon')
plt.title('EHQ_EHQ_Total Distribution - Before Cleaning')
plt.xlabel('EHQ_EHQ_Total Value')
plt.ylabel('Frequency')
plt.grid()
plt.show()

# ---------------- Print Number of Negative Values ----------------
negatives = train_quant[train_quant['EHQ_EHQ_Total'] < 0]
print(f"num of neg values before: {len(negatives)}")

# ---------------- Fix Negative Values ----------------
# fill negative values with 0
train_quant['EHQ_EHQ_Total'] = train_quant['EHQ_EHQ_Total'].apply(lambda x: x if x >= 0 else 0)

# ---------------- Distribution AFTER Cleaning ----------------
plt.figure(figsize=(8,5))
sns.histplot(train_quant['EHQ_EHQ_Total'], kde=True, bins=30, color='green')
plt.title('EHQ_EHQ_Total Distribution - After Cleaning')
plt.xlabel('EHQ_EHQ_Total Value')
plt.ylabel('Frequency')
plt.grid()
plt.show()

# ---------------- Print Number of Negative Values After ----------------
negatives_after = train_quant[train_quant['EHQ_EHQ_Total'] < 0]
print(f"num of neg values after: {len(negatives_after)} ✅")



# ----------------- Merge Train Data -----------------

# merging on  participant_id
train_full = train_cat.merge(train_quant, on='participant_id', how='left')
train_full = train_full.merge(train_conn, on='participant_id', how='left')

print(f"Train Full Shape: {train_full.shape}")

# ----------------- Merge Test Data -----------------

test_full = test_cat.merge(test_quant, on='participant_id', how='left')
test_full = test_full.merge(test_conn, on='participant_id', how='left')

print(f"Test Full Shape: {test_full.shape}")


train_solutions.head()


# target
y = train_solutions[["ADHD_Outcome","Sex_F"]]

# target & participant_id from X
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


# Encode object cols
for col in X.columns:
    if X[col].dtype == 'object':
        le = LabelEncoder()
        full_data = pd.concat([X[col], X_test[col]], axis=0).astype(str)
        le.fit(full_data)
        X[col] = le.transform(X[col].astype(str))
        X_test[col] = le.transform(X_test[col].astype(str))

# Fill missing values temporarily (for pipeline later)
X.fillna(-999, inplace=True)
X_test.fillna(-999, inplace=True)


# Train & Validation
X_train, X_val, y_train, y_val = train_test_split(X, y, stratify=y['ADHD_Outcome'], test_size=0.2, random_state=42)



import xgboost as xgb
import numpy as np
import pandas as pd

# ندرّب موديل بسيط على كل الداتا
temp_model = xgb.XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6,
    random_state=42,
    n_jobs=-1,
    tree_method='hist'
)
temp_model.fit(X_train, y_train['ADHD_Outcome'])  # نركز مثلاً أولاً على ADHD فقط

feature_importances = temp_model.feature_importances_

importance_df = pd.DataFrame({
    'feature': X_train.columns,
    'importance': feature_importances
}).sort_values(by='importance', ascending=False)

print(importance_df.head(10))

top_features = importance_df['feature'].head(500).tolist()

X_train_selected = X_train[top_features]
X_val_selected = X_val[top_features]
X_test_selected = X_test[top_features]


# Calculate scale for ADHD_Outcome
adhd_neg = (y_train['ADHD_Outcome'] == 0).sum()
adhd_pos = (y_train['ADHD_Outcome'] == 1).sum()
scale_adhd = adhd_neg / adhd_pos

# Calculate scale for Sex_F
sex_neg = (y_train['Sex_F'] == 0).sum()
sex_pos = (y_train['Sex_F'] == 1).sum()
scale_sex = sex_neg / sex_pos

print(f"Scale for ADHD_Outcome: {scale_adhd:.2f}")
print(f"Scale for Sex_F: {scale_sex:.2f}")



# ADHD_Outcome model
xgb_adhd = xgb.XGBClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_adhd,
    random_state=42,
    n_jobs=-1,
    tree_method='hist'
)

# Sex_F model
xgb_sex = xgb.XGBClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_sex,
    random_state=42,
    n_jobs=-1,
    tree_method='hist'
)



multi_output_model = MultiOutputClassifier(
    estimator=xgb.XGBClassifier(), n_jobs=-1
)

# We will replace each estimator manually
multi_output_model.estimators_ = [xgb_adhd, xgb_sex]



pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),  
    ('scaler', StandardScaler()),                  
    ('classifier', multi_output_model) 
])
pipeline.fit(X_train, y_train)


pipeline.fit(X_train_selected, y_train)


from sklearn.metrics import classification_report, accuracy_score

# Predict
y_val_pred = pipeline.predict(X_val_selected)

# Evaluate ADHD Outcome
print("ADHD Outcome Evaluation:")
print(classification_report(y_val['ADHD_Outcome'], y_val_pred[:, 0]))

# Evaluate Sex_F
print("\nSex_F Evaluation:")
print(classification_report(y_val['Sex_F'], y_val_pred[:, 1]))



print("ADHD Outcome Accuracy:", accuracy_score(y_val['ADHD_Outcome'], y_val_pred[:, 0]))
print("Sex_F Accuracy:", accuracy_score(y_val['Sex_F'], y_val_pred[:, 1]))



# Predict on test data
y_test_pred = pipeline.predict(X_test_selected)

# Create a DataFrame
submission = pd.DataFrame({
    'participant_id': test_full['participant_id'],
    'ADHD_Outcome': y_test_pred[:, 0], 
    'Sex_F': y_test_pred[:, 1]
})

# Save to CSV
submission.to_csv('submission.csv', index=False)
print("✅ Submission file saved!")


