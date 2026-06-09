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


df=pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv")
dt=pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv")
samp=pd.read_csv("/kaggle/input/playground-series-s5e7/sample_submission.csv")



df.info()


dt.info()


df=df.drop(columns=['id'])
ids=dt['id']
dt=dt.drop(columns=['id'])


df.head(2)


dt.head(2)


import pandas as pd
import numpy as np

for df_ in [df, dt]:
    for col in df_.columns:
        if df_[col].dtype == 'object':  # Object column
            mode_val = df_[col].mode(dropna=True)[0]
            df_[col] = df_[col].fillna(mode_val)
        elif pd.api.types.is_numeric_dtype(df_[col]):
            mean_val = df_[col].mean()
            df_[col] = df_[col].fillna(mean_val).astype(int)



df.head()


dt.head()


import seaborn as sns
import matplotlib.pyplot as plt
from itertools import combinations

sns.set(style='whitegrid')

# Numeric features
numeric_cols = [
    'Time_spent_Alone', 'Social_event_attendance', 'Going_outside',
    'Friends_circle_size', 'Post_frequency'
]

# Boxplots: each numeric feature vs target hue
for col in numeric_cols:
    plt.figure(figsize=(6, 4))
    sns.boxplot(data=df, x=col, hue='Personality')
    plt.title(f'{col} by Personality')
    plt.tight_layout()
    plt.show()

# Count plots for binary/categorical features with hue
for col in ['Stage_fear', 'Drained_after_socializing']:
    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x=col, hue='Personality')
    plt.title(f'{col} Count by Personality')
    plt.tight_layout()
    plt.show()

# Scatter plots for every pair of numeric columns with target hue
for col1, col2 in combinations(numeric_cols, 2):
    plt.figure(figsize=(6, 4))
    sns.scatterplot(data=df, x=col1, y=col2, hue='Personality', alpha=0.7)
    plt.title(f'{col1} vs {col2} by Personality')
    plt.tight_layout()
    plt.show()



from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

for dataset in [df, dt]:
    for col in dataset.columns:
        if dataset[col].dtype == 'O':
            if dataset is df and col == 'Personality':
                continue
            dataset[col] = le.fit_transform(dataset[col])



df['Personality'] = df['Personality'].map({'Extrovert': 0, 'Introvert': 1})



from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier

# Features and target
X = df.drop(columns=['Personality'])
y = df['Personality']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# XGBoost
xgb = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
xgb.fit(X_train, y_train)
xgb_pred = xgb.predict(X_test)
print("XGBoost:\n", classification_report(y_test, xgb_pred))

# CatBoost
catb = CatBoostClassifier(verbose=0)
catb.fit(X_train, y_train)
catb_pred = catb.predict(X_test)
print("CatBoost:\n", classification_report(y_test, catb_pred))

# LightGBM
lgbm = LGBMClassifier()
lgbm.fit(X_train, y_train)
lgbm_pred = lgbm.predict(X_test)
print("LightGBM:\n", classification_report(y_test, lgbm_pred))



from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
print("KNN:\n", classification_report(y_test, knn.predict(X_test)))



svc = SVC(kernel='rbf',probability=True)
svc.fit(X_train, y_train)

print("SVM-RBF:\n", classification_report(y_test, svc.predict(X_test)))



from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Prepare data
X = df.drop(columns=['Personality'])
y = df['Personality']

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA to 2D
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Train-test split on PCA data
X_train, X_test, y_train, y_test = train_test_split(
    X_pca, y, test_size=0.2, stratify=y, random_state=42
)

# Train classifier on PCA data
model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
model.fit(X_train, y_train)

# Predict and report
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Create meshgrid for decision surface
x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500),
                     np.linspace(y_min, y_max, 500))

# Predict on meshgrid
grid = np.c_[xx.ravel(), yy.ravel()]
Z = model.predict(grid).reshape(xx.shape)

# Plot
plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, alpha=0.3, cmap='tab10')
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y, palette='tab10', edgecolor='k')
plt.title("Decision Surface with PCA + XGBoost")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.legend(title='Personality')
plt.tight_layout()
plt.show()



svc.fit(X, y)


# Predict
preds = svc.predict(dt)

# Map numeric predictions to labels
label_map = {0: 'Extrovert', 1: 'Introvert'}
preds = [label_map[p] for p in preds]



sub=pd.DataFrame({
    'id' : ids,
    'Personality' : preds
})


sub


# Count plot for df
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='Personality')
plt.title('Personality Distribution in df')
plt.xlabel('Personality')
plt.ylabel('Count')
plt.tight_layout()
plt.show()


plt.figure(figsize=(6, 4))
sns.countplot(data=sub, x='Personality')
plt.title('Predicted Personality Distribution in sub')
plt.xlabel('Personality')
plt.ylabel('Count')
plt.tight_layout()
plt.show()



sub.to_csv("submission.csv",index=False)

