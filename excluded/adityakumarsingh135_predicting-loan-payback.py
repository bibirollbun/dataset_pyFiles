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


import warnings 
# import missingno as msno
import seaborn as sns 
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')


train = pd.read_csv('/kaggle/input/playground-series-s5e11/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e11/test.csv')



print(train.shape, test.shape)


df = train.copy()


df.head()




num_features = df.select_dtypes(include =['int64', 'float64'])
categorical_features = pd.concat([df.select_dtypes(exclude = ['int64', 'float64']), df['loan_paid_back']], axis = 1)
target ='loan_paid_back'

print('numeric features:', num_features.columns)
print('categorical_features: ', categorical_features.columns)



for feature in num_features[:-1]:
    sns.histplot(
        data=df,
        x=feature,
        kde=True,
        hue=target,
        bins=30
    )
    plt.title(f'Distribution: {feature}')
    plt.xticks(rotation=45)
    plt.grid(axis='y')
    plt.legend(title='Is loan paid back?', labels=['Yes', 'No'])
    plt.show()


# now we we ill observe the categorical features relation with loan_paid_back

for feature in categorical_features[:-1]:
    sns.countplot(
        data=df,
        x=feature,
        hue=target,
    )
    plt.title(f'Distribution: {feature}')
    plt.xticks(rotation=45)
    plt.grid(axis='y')
    plt.legend(title='Is loan paid back?', labels=['No', 'Yes'])
    plt.show()


# check for outliers 
for feature in num_features:
    sns.boxplot(data=df, x=feature)
    plt.title(f'Boxplot (outliers): {feature}')
    plt.show()


from scipy.stats import ttest_ind, chi2_contingency


def ttest(features):
    loan_paid = df[df['loan_paid_back'] == 1][features]
    loan_default = df[df['loan_paid_back']== 0][features]
    
    #apply ttest
    t_stat, p_val =ttest_ind(loan_paid, loan_default, equal_var = False)
    print(features)
    print("t-statistic: ", t_stat)
    print("p-value: ", p_val)


for feature in num_features:
    ttest(feature)


# same thing goes for categorical features chi^2 test
def chi2_test(features):
    observed = pd.crosstab(index=df['loan_paid_back'], columns=df[features])
    chi2, p_val, dof, expected = chi2_contingency(observed) 

    print(features)    
    print("Chi² statistic:", chi2)
    print("p-value:", p_val)




for features in categorical_features:
    chi2_test(features)


!pip install category_encoders



!pip install xgboost



!pip install lightgbm



!pip install catboost



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold, cross_validate, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from category_encoders import TargetEncoder

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import VotingClassifier 
import warnings
warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", None)
sns.set(style="whitegrid")


train = pd.read_csv('/kaggle/input/playground-series-s5e11/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e11/test.csv')
original = pd.read_csv('/kaggle/input/loan-prediction-dataset-2025/loan_dataset_20000.csv')


# Add a 'dataset' column to track source
train['dataset'] = 'train'
test['dataset'] = 'test'

original['dataset'] = 'train'



# Combine train and test datasets for unified preprocessing
df = pd.concat([train, test, original], axis=0).reset_index(drop=True)

print("Dataset shape:", df.shape)
df


df.shape


df.info()


numeric_cols = df.select_dtypes(include = ['float64', 'int64']).columns.tolist()
cat_cols = df.select_dtypes(include=['object', 'bool']).columns.tolist()

print(numeric_cols)
print(cat_cols)


for col in cat_cols:
    print(f"\nUnique values in '{col}':")
    print(df[col].value_counts())


plt.figure(figsize=(8,6))
sns.heatmap(df[['annual_income', 'debt_to_income_ratio', 'credit_score', 'loan_amount', 'interest_rate', 'loan_paid_back']].corr(), annot= True , cmap='coolwarm', fmt='.2f')
plt.title('correlation Heatmap', fontsize = 14)
plt.show()



# Split into train/test sets
train_df = df[df["dataset"] == "train"].copy()
test_df = df[df["dataset"] == "test"].copy()

print(f"Training samples: {len(train_df)}")
print(f"Test samples: {len(test_df)}")
print(f"Missing target values: {train_df['loan_paid_back'].isna().sum()}")

# Separate features and target
X_train = train_df.drop(["id", "loan_paid_back", "dataset"], axis=1)
y_train = train_df["loan_paid_back"]

X_test = test_df.drop(["id", "loan_paid_back", "dataset"], axis=1)

num_cols = [
    "annual_income",
    "debt_to_income_ratio",
    "credit_score",
    "loan_amount"
]


cate_cols = [
    "gender",
    "marital_status",
    "education_level",
    "employment_status",
    "loan_purpose",
    "grade_subgrade"
]


bool_cols = []  

# Columns to encode using Target Encoding (categorical + bools)
#interest_rate is numeric but sometimes its treated like category 
cols_to_encode = cate_cols + bool_cols + ["interest_rate"]

# ColumnTransformer for preprocessing
#columnstransforming it alows you to transform different columns using different methods 
#
preprocessor = ColumnTransformer(
    transformers=[
        #takes all categorical columns applies target encoding it helps to prevent overfitting 
        ("target_enc", TargetEncoder(cols=cols_to_encode, smoothing=25.0), cols_to_encode),
        # standardScaler converts number around 0 and varriance 1 
        ("scaler", StandardScaler(), num_cols)
    ],
   
    remainder="drop"
)


models ={
     "CatBoost": CatBoostClassifier(
        iterations=3000,
        learning_rate=0.03,
        depth=8,
        loss_function='Logloss',
        eval_metric='AUC',
        random_seed=42,
        verbose=0,
        auto_class_weights='Balanced',
        l2_leaf_reg=5
     ),
    "XGBoost": XGBClassifier(
        objective='binary:logistic',
        eval_metric='auc',
        learning_rate=0.01,
        max_depth=6,
        min_child_weight=3,
        colsample_bytree=0.3,
        subsample=0.6,
        reg_alpha=0.5,
        reg_lambda=2.0,
        n_estimators=10000,
        random_state=42,
        n_jobs=-1,
        tree_method='hist',
        device="cuda"
    )
}

#ensemble model
models["ensemble_all"] = VotingClassifier(
    estimators=[
        # ("LightGBM", models["LightGBM"]),
        ("CatBoost", models["CatBoost"]),
        ("XGBoost", models["XGBoost"])
    ],
    voting='soft',
    weights=[1,1]
)




#cv 
kfold = StratifiedKFold(n_splits= 5, shuffle=True, random_state=42)
cv_results = {}

scoring= {
    "Accuracy": "accuracy",
    "Precision": "precision",
    "Recall": "recall",
    "F1": "f1",
    "ROC_AUC": "roc_auc"
}

for name, model in models.items():
    print(name)

    #pipeline 
    pipeline =Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    #cv
    cv_scores = cross_validate(
        pipeline,
        X_train,
        y_train,
        cv=kfold,
        scoring= scoring,
        n_jobs = -1
    )
    cv_results[name] = {metric: np.mean(scores) for metric,scores in cv_scores.items() if "test_" in metric}

    print(f"Accuracy:  {cv_results[name]['test_Accuracy']:.4f}")
    print(f"Precision: {cv_results[name]['test_Precision']:.4f}")
    print(f"Recall:    {cv_results[name]['test_Recall']:.4f}")
    print(f"F1-score:  {cv_results[name]['test_F1']:.4f}")
    print(f"ROC-AUC:   {cv_results[name]['test_ROC_AUC']:.4f}")



results_df = pd.DataFrame({
    model: {
        "Accuracy": cv_results[model]["test_Accuracy"],
        "Precision": cv_results[model]["test_Precision"],
        "Recall": cv_results[model]["test_Recall"],
        "F1": cv_results[model]["test_F1"],
        "ROC_AUC": cv_results[model]["test_ROC_AUC"]
    } for model in cv_results.keys()
}).T.round(4)


print(results_df)


best_model_name = results_df['ROC_AUC'].idxmax()
best_model = models[best_model_name]

print(best_model_name)
final_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", best_model)
])

final_pipeline.fit(X_train, y_train)


test_pred = final_pipeline.predict_proba(X_test)[:, 1]


submission = pd.DataFrame({
    'id': test_df['id'].values,
    'loan_paid_back': test_pred
})

submission.to_csv('submission.csv', index=False)



submission.head()


submission['id'] = submission['id'].astype(int)


submission.head()




