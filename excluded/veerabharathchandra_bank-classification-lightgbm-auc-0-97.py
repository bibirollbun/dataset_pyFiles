import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, RocCurveDisplay
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.base import clone
import warnings
warnings.filterwarnings('ignore')


train_df = pd.read_csv('/kaggle/input/playground-series-s5e8/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e8/test.csv')


train_df.head()


test_df.head()


y =train_df['y']
y.head()


train_df.drop(columns ='y', inplace = True)


train_df


test_df.head()


train_df.shape


test_df.shape


train_df.info()


train_df.describe()


test_df.info()


test_df.describe()


train_df.isnull().sum()


train_df.duplicated().sum()


num_col = [col for col in train_df.columns if train_df[col].dtype != 'object' and col != 'id']
print(f"Numerical columns : {num_col}")
for col in num_col:
    plt.figure(figsize=(5,4))
    sns.boxplot(x=train_df[col], color ='limegreen')  # y='y' groups by target
    plt.title(f"Boxplot of {col}")
    plt.show()


for col in num_col:
    plt.figure(figsize=(5,4))
    sns.histplot(x=train_df[col], kde =True, color = "salmon",bins =20)  
    plt.title(f"Boxplot of {col}")
    plt.show()


corr = train_df[num_col].corr()
plt.figure(figsize=(10,8))
sns.heatmap(corr, annot=True, cmap="coolwarm", center=0)
plt.title("Correlation Matrix (Numerical Features)")
plt.show()


cat_col = train_df.select_dtypes(include = 'O').columns
print(f"Categorical columns : {cat_col}")

for col in cat_col:
    plt.figure(figsize=(10,5))
    sns.countplot(x=train_df[col], color = "grey")  
    plt.title(f"Count plot of {col}")
    plt.xticks(rotation=45, ha="right") 
    plt.show()
    plt.tight_layout()


for col in cat_col:
  print(f"Column Name : {train_df[col].value_counts()}\n\n")


for col in cat_col:
  print(f"{col}:{train_df[col].unique()}\n")


# Define encoding groups

label_col = ['default', 'housing', 'loan']
ohe_col=['job', 'marital', 'contact','poutcome']
ord_col = ['education', 'month']


edu_map = [['primary','secondary' ,'tertiary','unknown']]
month_map = [['jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec']]
label_map=[['no','yes']]


# Pipelines
ohe_pipe = Pipeline(steps =[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("ohe", OneHotEncoder(handle_unknown="ignore"))
])

label_pipe = Pipeline(steps=[
    ("imputer",SimpleImputer(strategy = 'most_frequent')),
    ("ord",OrdinalEncoder(categories=label_map*len(label_col)))
])

ord_pipe = Pipeline(steps=[
    ("imputer",SimpleImputer(strategy = 'most_frequent')),
    ("ord",OrdinalEncoder(categories=edu_map+month_map))
])

num_pipe = Pipeline(steps=[
    ("imputer",SimpleImputer(strategy = 'median')),
    ("scaler",StandardScaler())
])


preprocessor = ColumnTransformer(transformers=[
    ("ohe", ohe_pipe, ohe_col),
    ("ord", ord_pipe, ord_col),
    ("label", label_pipe, label_col),
    ("num",num_pipe,num_col)], remainder="drop")


preprocessor


X=train_df.copy()
X.head()


X_train,X_val,y_train,y_val = train_test_split(X,y,test_size=0.2, stratify = y, random_state =42)


print(f"X_train shape = {X_train.shape}")
print(f"X_val shape = {X_val.shape}")
print(f"y_train shape = {y_train.shape}")
print(f"y_val shape = {y_val.shape}")


# models
lr= LogisticRegression(max_iter = 1000,class_weight = 'balanced')


lgbm = LGBMClassifier(n_estimators=1000,
    learning_rate=0.05,
    num_leaves=63,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    class_weight="balanced"
)


xgb = XGBClassifier(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=6,
    min_child_weight=6,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1,
    objective="binary:logistic",
    eval_metric="auc",
    tree_method="hist",
    random_state=42,
    scale_pos_weight=(1 - y.mean())/y.mean()
)


# pack your estimators
models = {
    "LogisticRegression": lr,
    "XGBoost": xgb,
    "LightGBM": lgbm            
}

results = []
probas = {}

plt.figure(figsize=(8,5))
for name, est in models.items():
    # fresh pipeline for each model
    pipe = Pipeline([("prep", preprocessor), ("clf", clone(est))])
    pipe.fit(X_train, y_train)

    val_proba = pipe.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, val_proba)
    results.append({"model": name, "val_auc": auc})
    probas[name] = (pipe, val_proba)  
    
    RocCurveDisplay.from_predictions(y_val, val_proba, name=name)
    plt.title(f"Validation ROC Curve for {name}")
plt.show()




# rank models
score_df = pd.DataFrame(results).sort_values("val_auc", ascending=False).reset_index(drop=True)
print("\nValidation ROC-AUC ranking:")
print(score_df)




# pick best
best_name = score_df.loc[0, "model"]
best_pipe, best_val_proba = probas[best_name]
print(f"\nBest model on validation: {best_name} (AUC={score_df.loc[0, 'val_auc']:.5f})")




# ===== Fit best model on ALL training data and predict test =====
best_model = Pipeline([("prep", preprocessor), ("clf", clone(models[best_name]))])
best_model.fit(X, y)

y_test_pred_proba = best_model.predict_proba(test_df)[:, 1]




# save submission
submission = pd.DataFrame({'id': test_df['id'],'y': y_test_pred_proba})
submission.to_csv("submission.csv", index=False)
print("\nSaved submission.csv using best model:", best_name)



submission.head(5)

