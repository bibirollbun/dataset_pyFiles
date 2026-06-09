import numpy as np 
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, RepeatedStratifiedKFold


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))



df_train = pd.read_csv("/kaggle/input/playground-series-s4e1/train.csv")
original_data = pd.read_csv("/kaggle/input/bank-customer-churn-prediction/Churn_Modelling.csv")
df_test = pd.read_csv("/kaggle/input/playground-series-s4e1/test.csv")
submission = pd.read_csv("/kaggle/input/playground-series-s4e1/sample_submission.csv")


df_train.shape


df_train.head()


original_data.head()


df_test.head()


submission.head()


df_train.isna().sum()


df_train.duplicated().sum()


original_data.isna().sum()


original_data.duplicated().sum()


df_train = df_train.drop("id", axis=1)
original_data = original_data.drop("RowNumber", axis=1)


original_data.head()


df_train = pd.concat([df_train, original_data], axis=0)


df_train.shape


df_train.duplicated().sum()


df_train.isna().sum()


df_train = df_train.drop_duplicates()


df_train.duplicated().sum()


df_train = df_train.dropna()


df_train.isna().sum()


from sklearn.feature_extraction.text import TfidfVectorizer
import gc


# we return same text, because here we dont do any tokenization because surnames are usually not sentences and regular english language
def dummy(text):
    return text

vectorizer = TfidfVectorizer(ngram_range=(3, 5), lowercase=False, sublinear_tf=True, analyzer = 'word',
    tokenizer = dummy,
    preprocessor = dummy,
    token_pattern = None, 
    strip_accents='unicode',
    max_features=1000
)

vectorizer.fit(df_train["Surname"])

vocab = vectorizer.vocabulary_

vectorizer = TfidfVectorizer(ngram_range=(3, 5), lowercase=False, sublinear_tf=True, vocabulary=vocab,
                            analyzer = 'word',
                            tokenizer = dummy,
                            preprocessor = dummy,
                            token_pattern = None, strip_accents='unicode', max_features=1000
                            )



train_surnames = vectorizer.fit_transform(df_train["Surname"])
test_surnames = vectorizer.transform(df_test["Surname"])

# we free the space by removing reference to the object vectorizer and use garbage collector
# to remove the unused space from the memory
del vectorizer
gc.collect()


from sklearn.decomposition import PCA

pca = PCA(n_components=10)
tfidf_train_pca = pca.fit_transform(train_surnames.toarray())
tfidf_test_pca = pca.transform(test_surnames.toarray())

pca_columns = [f'Surname_PCA_{i+1}' for i in range(10)]
df_train_pca = pd.DataFrame(tfidf_train_pca, columns=pca_columns)
df_test_pca = pd.DataFrame(tfidf_test_pca, columns=pca_columns)


df_train_pca.shape


df_train.shape


df_train.reset_index(drop=True, inplace=True)
df_train_pca.reset_index(drop=True, inplace=True)

df_train = pd.concat([df_train, df_train_pca], axis="columns")


df_train.head()


df_test.reset_index(drop=True, inplace=True)
df_test_pca.reset_index(drop=True, inplace=True)

df_test = pd.concat([df_test, df_test_pca], axis="columns")


df_train = df_train.drop("Surname", axis=1)
df_test  = df_test.drop("Surname", axis=1)


df_train.head()


numeric_data = df_train.select_dtypes(include=[np.number])
categorical_data = df_train.select_dtypes(exclude=[np.number])


numeric_data.head()


categorical_data.head()


# categorical_data = categorical_data.drop("Surname", axis=1)
categorical_data.head()


corr = numeric_data.corr()
corr.style.background_gradient(cmap='coolwarm')


# df_train = df_train.drop("Surname", axis=1)

test_ids = df_test["id"]
# df_test = df_test.drop(["Surname", "id"], axis=1)
df_test = df_test.drop(["id"], axis=1)


df_test.head()


df_train.head()


enc = LabelEncoder()

categorical_features = ["Geography", "Gender", "Tenure", "HasCrCard", "IsActiveMember"]

for cat_feat in categorical_features:
    df_train[cat_feat] = enc.fit_transform(df_train[cat_feat])
    df_test[cat_feat] = enc.transform(df_test[cat_feat])



df_train.head()


df_test.head()


# by https://www.kaggle.com/code/chinmayadatt/notebook-analysing-bank-churn-dataset
def add_new_features(df):
    df['Geo_Gender'] = df['Geography'] + df['Gender'] + 10
    df['AgeGroup'] = df['Age'] // 10 * 10
    df['IsSenior'] = df['Age'].apply(lambda x: 1 if x >= 60 else 0)
    df['QualityOfBalance'] = pd.cut(df['Balance'], bins=[-1,100,1000,10000,50000,1000000], labels=['VeryLow', 'Low', 'Medium','High','Highest'])
    df['QualityOfBalance'].replace(['VeryLow', 'Low', 'Medium','High','Highest'],[0,1,2,3,4], inplace=True)
    df['Balance_to_Salary_Ratio'] = df['Balance'] / df['EstimatedSalary']
    df['CreditScoreTier'] = pd.cut(df['CreditScore'], bins=[0, 650, 750, 850], labels=['Low', 'Medium', 'High'])
    df['CreditScoreTier'].replace(['Low', 'Medium', 'High'],[0, 1, 2], inplace=True)
    df['IsActive_by_CreditCard'] = df['HasCrCard'] * df['IsActiveMember']
    df['Products_Per_Tenure'] =  df['Tenure'] / df['NumOfProducts']
    df['Customer_Status'] = df['Tenure'].apply(lambda x:0 if x < 2 else 1)
    return df


df_train = add_new_features(df_train)
df_test  = add_new_features(df_test)


df_train.head()


df_train = df_train.astype({
    'QualityOfBalance': int,
    'CreditScoreTier': int
})
df_test = df_test.astype({
    'QualityOfBalance': int,
    'CreditScoreTier': int
})


df_train.describe()


# numeric_data      = df_train.drop(["Geography", "Gender", "Tenure", "HasCrCard", "IsActiveMember", "Geo_Gender", "IsSenior", "QualityOfBalance", "CreditScoreTier", "IsActive_by_CreditCard", "Products_Per_Tenure", "Customer_Status", "Exited"], axis=1)
# numeric_data_test = df_test.drop(["Geography", "Gender", "Tenure", "HasCrCard", "IsActiveMember", "Geo_Gender", "IsSenior", "QualityOfBalance", "CreditScoreTier", "IsActive_by_CreditCard", "Products_Per_Tenure", "Customer_Status"], axis=1)

# numeric_data      = numeric_data.drop("CustomerId", axis=1)
# numeric_data_test = numeric_data_test.drop("CustomerId", axis=1)
# numeric_data.head()


# numeric_data.head()


# numeric_data_test.head()


# scaler = StandardScaler()

# scaled_numerical_data_train = scaler.fit_transform(numeric_data)
# scaled_numerical_data_test  = scaler.transform(numeric_data_test)    


# scaled_numerical_data_train[0]


# pca_columns = [f'Surname_PCA_{i+1}' for i in range(10)]
# features = ["CreditScore", "Age", "Balance", "NumOfProducts", "EstimatedSalary"]
# for i in pca_columns:
#     features.append(i)

# for i, feat in enumerate(features):
#     l = []
    
#     for j in range(len(scaled_numerical_data_train)):
#         l.append(scaled_numerical_data_train[j][i])
    
#     df_train[feat] = l
    
#     l = []
    
#     for j in range(len(scaled_numerical_data_test)):
#         l.append(scaled_numerical_data_test[j][i])
    
#     df_test[feat]  = l


df_train.head()


df_test.head()


X = df_train.drop("Exited", axis=1)
y = df_train["Exited"]
X_test = df_test


lgbm = LGBMClassifier(**{  'objective'           : 'binary',
                           'boosting_type'       : 'gbdt',
                           'metric'              : "auc",
                           'random_state'        : 42,
                           'colsample_bytree'    : 0.56,
                           'subsample'           : 0.35,
                           'learning_rate'       : 0.05,
                           'max_depth'           : 8,
                           'n_estimators'        : 1000,
                           'num_leaves'          : 140,
                           'reg_alpha'           : 0.14,
                           'reg_lambda'          : 0.85,
                           'verbosity'           : -1, 
                          })
xgb  = XGBClassifier(**{  'objective'             : 'binary:logistic',
                          'eval_metric'           : "auc",
                          'random_state'          : 42,
                          'colsample_bytree'      : 0.25,
                          'learning_rate'         : 0.07,
                          'max_depth'             : 8,
                          'n_estimators'          : 800,                         
                          'reg_alpha'             : 0.09,
                          'reg_lambda'            : 0.70,
                          'min_child_weight'      : 22,
                          'verbosity'             : 0,
                         })
cat  = CatBoostClassifier(**{
                         'iterations'            : 10000,
                         'objective'             : 'Logloss',
                         'eval_metric'           : "AUC",
                         'early_stopping_rounds' : 1000,
                         'bagging_temperature'   : 0.1,
                         'colsample_bylevel'     : 0.88,
                         'iterations'            : 1000,
                         'learning_rate'         : 0.065,
                         'max_depth'             : 7,
                         'l2_leaf_reg'           : 1,
                         'min_data_in_leaf'      : 25,
                         'random_strength'       : 0.1, 
                         'max_bin'               : 100,
                         'verbose'               : 0,
                        })

vote = VotingClassifier(estimators=[('lgbm', lgbm), ('xgb', xgb), ('cat', cat)], voting='soft', weights=[2, 1, 1])

# Initialize an empty array to hold the submission predictions
submission_predictions = []

kf = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)

# save aucs
aucs = []
ind = 1

for train_index, test_index in kf.split(X, y):
    print(f"============== Working on fold #{ind} ================")
    X_train_kf, X_val_kf = X.iloc[train_index], X.iloc[test_index]
    y_train_kf, y_val_kf = y.iloc[train_index], y.iloc[test_index]

    print()
    print("               Fitting the voting model...              ")
    # Fit the model
    vote.fit(X_train_kf, y_train_kf)

    print()
    print("            Predicting on the validation data           ")
    # Predict probabilities for validation set
    y_pred_val = vote.predict_proba(X_val_kf)[:, 1]

    # Calculate AUC for validation set
    auc_val = roc_auc_score(y_val_kf, y_pred_val)
    print()
    print(f"           Validation ROC AUC Score: {auc_val}        ")
    
    aucs.append(auc_val)

    print()
    print("             Predicting on submission data...")
    # Predict probabilities for test set (df_test)
    y_pred_test = vote.predict_proba(X_test)[:, 1]
    submission_predictions.append(y_pred_test)
    
    print()
    print(f"                 Fold #{ind} finished !                ")
    
    ind+=1


print(f"Average ROC AUC Score: {sum(aucs) / len(aucs)}")

# Average predictions from different folds
avg_submission = pd.DataFrame(submission_predictions).mean(axis=0)

submission["Exited"] = avg_submission

# Save submission to CSV
submission.to_csv("submission.csv", index=False)

submission.head()

