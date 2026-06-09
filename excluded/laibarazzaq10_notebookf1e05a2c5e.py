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


import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from lightgbm import LGBMClassifier
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical

# ğŸ§ª Reproducibility
def set_seed(seed=42):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    random.seed(seed)

set_seed()

# ğŸ“¥ Load Data
train = pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv")
submission = pd.read_csv("/kaggle/input/playground-series-s5e7/sample_submission.csv")

# ğŸ�¯ Target encoding
le_target = LabelEncoder()
train["Personality"] = le_target.fit_transform(train["Personality"])
X_raw = train.drop(columns=["id", "Personality"])
X_test_raw = test.drop(columns=["id"])
y = train["Personality"]

# ğŸ�·ï¸� Encode categoricals
def label_encode_all(train_df, test_df):
    for col in train_df.select_dtypes(include="object").columns:
        le = LabelEncoder()
        combined = pd.concat([train_df[col], test_df[col]])
        le.fit(combined)
        train_df[col] = le.transform(train_df[col])
        test_df[col] = le.transform(test_df[col])
    return train_df, test_df

X_encoded, X_test_encoded = label_encode_all(X_raw.copy(), X_test_raw.copy())

# ğŸ§½ Impute + Scale
imp = SimpleImputer(strategy="mean")
sc = StandardScaler()
X_clean = pd.DataFrame(sc.fit_transform(imp.fit_transform(X_encoded)), columns=X_encoded.columns)
X_test_clean = pd.DataFrame(sc.transform(imp.transform(X_test_encoded)), columns=X_test_encoded.columns)

# â�• Basic Feature Engineering
for df in [X_clean, X_test_clean]:
    df["mean"] = df.mean(axis=1)
    df["std"] = df.std(axis=1)
    df["range"] = df.max(axis=1) - df.min(axis=1)

# â�• PCA for compression
pca = PCA(n_components=3, random_state=42)
X_pca = pca.fit_transform(X_clean)
X_test_pca = pca.transform(X_test_clean)

# â�• KMeans clustering
kmeans = KMeans(n_clusters=3, random_state=42, n_init="auto")
X_kmeans = kmeans.fit_predict(X_clean)
X_test_kmeans = kmeans.predict(X_test_clean)

# ğŸ”€ Final feature matrices
X_final = np.hstack([X_clean, X_pca, X_kmeans.reshape(-1, 1)])
X_test_final = np.hstack([X_test_clean, X_test_pca, X_test_kmeans.reshape(-1, 1)])

# ğŸ“Š Train/Val Split
X_train, X_val, y_train, y_val = train_test_split(X_final, y, stratify=y, test_size=0.2, random_state=42)

# ğŸŒ³ LightGBM
lgbm = LGBMClassifier(n_estimators=200, max_depth=4, learning_rate=0.1,
                      subsample=0.9, colsample_bytree=0.9, random_state=42)
lgbm.fit(X_train, y_train)
val_preds = lgbm.predict(X_val)

print("\nğŸ“Š Validation Report:\n", classification_report(y_val, val_preds, target_names=le_target.classes_))

# ğŸ”® Final Prediction
test_preds = lgbm.predict(X_test_final)

# ğŸ’¾ Submission
submission["Personality"] = le_target.inverse_transform(test_preds)
submission.to_csv("submission.csv", index=False)
print("\nâœ… Submission saved as 'submission.csv'")
print(submission.head())





