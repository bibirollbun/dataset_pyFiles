# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import missingno as msno #data cleaning
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import roc_curve, auc, ConfusionMatrixDisplay



# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


df = pd.read_csv("/kaggle/input/seleksi-academya-data-science-2025/train.csv")
df.head()


testdf = pd.read_csv("/kaggle/input/seleksi-academya-data-science-2025/test.csv")
testdf.head()


df.info()  
df.describe()  
df.isnull().sum() 


df.hist(figsize=(20, 10), color = 'grey')
plt.show()


for col in df.select_dtypes(include="object").columns:
    plt.figure(figsize=(5, 3))
    sns.countplot(y=df[col])
    plt.title(f"Distribusi {col}")
    plt.show()


numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
numerical_cols.remove('id')

for col in numerical_cols:
    plt.figure(figsize=(8, 4))
    sns.boxplot(x=df[col])
    plt.title(f"Boxplot of {col}")
    plt.show()


corr_matrix = df[numerical_cols].corr()


corr_matrix = df[numerical_cols].corr()
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.3f')
plt.title('Heatmap Korelasi Fitur Numerik')
plt.show()


df.duplicated().sum() # Menampilkan jumlah duplicated values pada train data


testdf.duplicated().sum() # Menampilkan jumlah duplicated values pada test data


df.replace("", pd.NA, inplace=True)  # Ganti string kosong dengan NA
df.replace([np.inf, -np.inf], np.nan, inplace=True)  # Ganti inf dengan NaN
df.fillna(0, inplace=True)  # Ganti NaN dengan nilai 0


testdf.replace("", pd.NA, inplace=True)
testdf.replace([np.inf, -np.inf], np.nan, inplace=True)
testdf.fillna(0, inplace=True)  # Ganti NaN dengan nilai 0


df.isna().sum()  # Menampilkan jumlah missing values di setiap kolom train data


testdf.isna().sum() # Menampilkan jumlah missing values di setiap kolom test data


# Mengganti missing values dengan asumsi skenario terburuk
replace_dict = {
    "having_ip_address": "No",
    "shortining_service": "Yes",
    "having_at_symbol": "No",
    "prefix_suffix": "No",
    "favicon": "No",
    "port": "Yes",
    "https_token": "No",
    "sfh": "Suspecious",
    "submitting_to_email": "Yes",
    "abnormal_url": "Yes",
    "on_mouseover": "No",
    "rightclick": "Disabled",
    "popupwindow": "No",
    "iframe": "Yes",
    "dnsrecord": "No",
    "google_index": "No",
}


df.fillna(replace_dict, inplace=True)


testdf.fillna(replace_dict, inplace=True)


# Convert categorical values into numerical representation
convert_categorical = {
    "Yes": 1, "No": 0,
    "Suspicious": 0, "Low": 1, "High": 2,
    "Phishing": 0, "Legitimate": 1,
    "Disabled": 0, "Enabled": 1
}

# Apply conversion to the dataset
df.replace(convert_categorical, inplace=True)
pd.set_option('future.no_silent_downcasting', True)

# Separate features (X) and target (y)
X = df.drop(columns=["Result"])  # Remove target column from features
y = df["Result"].astype(int)  # Convert target to int (0/1)


# Apply One-Hot Encoding for categorical columns
X = pd.get_dummies(X, dtype=int)

# Ensure all columns are numeric (category encoding for any remaining objects)
for col in X.select_dtypes(include=['object']).columns:
    X[col] = X[col].astype('category').cat.codes


# XGBBoost model for handling imbalance data

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Compute class imbalance ratio
class_counts = np.bincount(y_train)
scale_pos_weight = class_counts[0] / class_counts[1] if class_counts[1] > 0 else 1

print(f"Scale Pos Weight: {scale_pos_weight:.2f}")

# Initialize XGBoost classifier with imbalance handling
xgb_clf = XGBClassifier(
    scale_pos_weight=scale_pos_weight,  # Handle class imbalance
    random_state=42,
    eval_metric="logloss",
    use_label_encoder=False
)

# 1. Check class distribution
plt.figure(figsize=(6, 4))
sns.countplot(x=y_train, palette="viridis")
plt.title("Class Distribution After Handling Imbalance")
plt.xlabel("Class (0: Phishing, 1: Legitimate)")
plt.ylabel("Count")
plt.show()


# Drop 'id' column as it's not useful for modeling
df.drop(columns=['id'], inplace=True, errors='ignore')

# Feature Engineering
# 1. Count special characters in URLs
df["special_char_count"] = df[["having_ip_address", "shortining_service", "having_at_symbol"]].sum(axis=1)

# 2. Domain length as a separate feature
df["domain_length"] = df["url_length"]

# 3. HTTPS token presence
df["is_https"] = df["https_token"].map({1: 1, 0: 0})

# 4. URL Shortening service used
df["is_shortened"] = df["shortining_service"].map({1: 1, 0: 0})

# 5. Presence of pop-ups and iframes
df["has_popup"] = df["popupwindow"].map({1: 1, 0: 0})
df["has_iframe"] = df["iframe"].map({1: 1, 0: 0})

# 6. Phishing Risk Score (Basic weighted sum)
df["phishing_risk_score"] = (
    df["special_char_count"] * 0.2 +
    df["is_https"] * -0.3 +  # HTTPS presence reduces risk
    df["is_shortened"] * 0.4 +
    df["has_popup"] * 0.3 +
    df["has_iframe"] * 0.3
)

# Fill missing values with 0
df.fillna(0, inplace=True)

df.to_csv("feature_engineered_train.csv", index=False)
print("Feature Engineering Completed! New dataset saved.")








