#%%capture
# !pip uninstall -y numpy pandas scikit-learn imbalanced-learn scipy
# !pip install numpy==1.26.4 pandas==2.1.4 scipy==1.11.4
# !pip install scikit-learn==1.3.2 imbalanced-learn==0.11.0



import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import  gc, time

from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import LabelEncoder
from scipy import sparse
from sklearn.metrics import accuracy_score

SEED = 42

# utility
def seed_everything(seed=SEED):
    np.random.seed(seed)
seed_everything()

import warnings
warnings.filterwarnings('ignore')


train = pd.read_csv("/kaggle/input/playground-series-s5e12/train.csv")

train



test = pd.read_csv('/kaggle/input/playground-series-s5e12/test.csv')

test


train.info()


train.isnull().sum()


for x in train.columns:
    print(f'{train[x].value_counts()}\n')


train.info()


top_user = train['age'].value_counts().nlargest(10).reset_index()
top_user.columns = ['age', 'count']
px.bar(
    top_user, 
    x='age', y='count', 
    title="Top 10 persons age into train df",
    color='count', 
    color_continuous_scale='Viridis'
).show()
plt.show()


px.histogram(
    train, 
    x='gender', 
    nbins=20,
    title="Gender Distribution",
    color_discrete_sequence=['teal']
).show()
plt.show()


fig = px.pie(
    train.groupby("ethnicity").size().reset_index(name="count"),
    names="ethnicity",
    values="count",
    title="Pie chart Distribution for ethnicity",
    color_discrete_sequence=px.colors.qualitative.Pastel
)
fig.show()

plt.show()


cols = ['diet_score','sleep_hours_per_day','screen_time_hours_per_day','heart_rate']

for col in cols:
    px.histogram(
        train, 
        x=col, 
        nbins=20,
        title=f"{col} Distribution",
        color_discrete_sequence=['teal']
    ).show()

plt.show()


train["income_level"].value_counts().plot(kind="bar")
plt.title("Bar distribution by income_level")
plt.xlabel("income_level")
plt.ylabel("count")
plt.show()


px.histogram(
    train,
    x="smoking_status",
    title="Histogram distribution by income_level"
).show()



fig = px.pie(
    train.groupby("employment_status").size().reset_index(name="count"),
    names="employment_status",
    values="count",
    title="Pie chart Distribution for employment_status",
    color_discrete_sequence=px.colors.qualitative.Pastel
)
fig.show()

plt.show()


px.histogram(
    train,
    nbins=20,
    x="diagnosed_diabetes",
    title="Histogram distribution by diagnosed_diabetes"
).show()


from sklearn.metrics import  confusion_matrix, classification_report, make_scorer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,VotingClassifier, StackingClassifier
from sklearn.neighbors import KNeighborsClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
import joblib

from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler , LabelEncoder
from sklearn.pipeline import Pipeline


df = train.copy()
df= df.drop(['id'],axis=1)
cat_cols = df.select_dtypes(include=['object','category']).columns  # pick categorical columns
label_encoder = LabelEncoder()

label_encoders = {}

for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

df.head()


numeric_features = list(df.select_dtypes(include=['int64', 'float64']).columns)


Q1 = df[numeric_features].quantile(0.25)
Q3 = df[numeric_features].quantile(0.75)
IQR = Q3 - Q1

outlier = df[((df[numeric_features] < (Q1 - 1.5 * IQR)) | (df[numeric_features] > (Q3 + 1.5 * IQR))).any(axis=1)]

df1 = df[~((df[numeric_features] < (Q1 - 1.5 * IQR)) | (df[numeric_features] > (Q3 + 1.5 * IQR))).any(axis=1)]


print(f'Outlier found : {outlier.shape}')


print(f'Dataset after Outlier removed : {df1.shape}')


df1['diagnosed_diabetes'].value_counts()


X = df1.drop(columns=['diagnosed_diabetes'],axis=1)
y = df1['diagnosed_diabetes']
 
X.describe()


# from imblearn.over_sampling import SMOTE

# smote = SMOTE(random_state=42)

# X_balanced, y_balanced = smote.fit_resample(X, y)

# X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, test_size=0.3, random_state=42)

# scaler_d = StandardScaler()

# X_train_sc = scaler_d.fit_transform(X_train)

# X_test_sc = scaler_d.transform(X_test)


#y_balanced['diagnosed_diabetes'].value_counts()


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=22)

scaler_d = StandardScaler()

X_train_sc = scaler_d.fit_transform(X_train)

X_test_sc = scaler_d.transform(X_test)


base_models = [
    ("xgb", XGBClassifier(
        n_estimators=700, learning_rate=0.05, max_depth=5,
        subsample=0.9, colsample_bytree=0.9,
    )),

    ("lgb", LGBMClassifier(
        n_estimators=700, learning_rate=0.05,
        num_leaves=31, subsample=0.8, colsample_bytree=0.9, verbosity=-1
    )),

    ("cat", CatBoostClassifier(
        iterations=600,
        learning_rate=0.05,
        depth=6,
        verbose=False
    )),

]


meta_model = CatBoostClassifier(
    iterations=500,
    depth=6,
    learning_rate=0.05,
    verbose=False
)


stack_model = StackingClassifier(
    estimators=base_models,
    final_estimator=meta_model,
    stack_method='predict_proba',
    passthrough=True,
    n_jobs=-1
)


stack_model.fit(X_train_sc, y_train)

y_pred = stack_model.predict_proba(X_test_sc)[:,1]
print("Stacking + CatBoost AUC:",roc_auc_score(y_test, y_pred))


test = pd.read_csv('/kaggle/input/playground-series-s5e12/test.csv')

test


ids = test['id']
test = test.drop(['id'],axis=1)

for col in cat_cols:
    le = label_encoders[col]
    test[col] = test[col].astype(str).apply(
        lambda x: le.transform([x])[0] if x in le.classes_ else -1
    )



test_sc = scaler_d.transform(test)




test['gender'].value_counts()


test_pred = stack_model.predict_proba(test_sc)[:,1]


# Create V5 submission
submissions = pd.DataFrame({
    'id': ids,
    'diagnosed_diabetes': test_pred
})

submissions.to_csv('submission.csv', index=False)
print("Submission saved: submission.csv")


submissions.head(10)




