import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from catboost import CatBoostClassifier
from sklearn.metrics import f1_score, confusion_matrix,classification_report, roc_auc_score, roc_curve, ConfusionMatrixDisplay
from sklearn.model_selection import learning_curve,RandomizedSearchCV, GridSearchCV,train_test_split 


import warnings
warnings.filterwarnings("ignore")


df = pd.read_csv("/kaggle/input/playground-series-s5e11/train.csv").drop(columns=  'id',axis = 1)
test = pd.read_csv("/kaggle/input/playground-series-s5e11/test.csv").drop(columns=  'id',axis = 1)
sub = pd.read_csv("/kaggle/input/playground-series-s5e11/sample_submission.csv")


df


df.describe()


df.info()


corr = df.corr(numeric_only=True)


sns.heatmap(corr,annot=True,cmap="coolwarm")


def preprocess_train_test(train_df, test_df):
    drop_cols = ["gender", "marital_status", "education_level"]
    train_df = train_df.drop(columns=drop_cols, axis=1)
    test_df = test_df.drop(columns=drop_cols, axis=1)
    
    # Ordinal encoding for employment_status and grade_subgrade
    ord_enc = OrdinalEncoder()
    train_df[["employment_status", "grade_subgrade"]] = ord_enc.fit_transform(
        train_df[["employment_status", "grade_subgrade"]]
    )
    test_df[["employment_status", "grade_subgrade"]] = ord_enc.transform(
        test_df[["employment_status", "grade_subgrade"]]
    )
    
    # Label encoding for loan_purpose
    lbl_enc = LabelEncoder()
    train_df["purpose_encoded"] = lbl_enc.fit_transform(train_df["loan_purpose"])
    test_df["purpose_encoded"] = lbl_enc.transform(test_df["loan_purpose"])
    
    # Drop original loan_purpose column (optional)
    train_df = train_df.drop(columns=["loan_purpose"])
    test_df = test_df.drop(columns=["loan_purpose"])
    
    return train_df, test_df



df,test = preprocess_train_test(df,test)


X_train,X_test,y_train,y_test = train_test_split(df.drop("loan_paid_back",axis = 1),df["loan_paid_back"],test_size=0.2,random_state=42)


model = CatBoostClassifier(
                iterations=1500,
                learning_rate=0.15,
                depth=4,
                class_weights = {0:3, 1:1} ,  
                eval_metric='AUC',
                l2_leaf_reg=9,
                random_seed=42,
                loss_function='Logloss',
                verbose=100
                    
        )
    
model.fit(X_train, y_train)


y_pred = model.predict(X_test)
y_pred_proba=model.predict_proba(X_test)[:,1]
score = roc_auc_score(y_test, y_pred_proba)
cm=confusion_matrix(y_test,y_pred)
importance = model.get_feature_importance()
importance_df= pd.DataFrame({
    'Feature': X_train.columns, 
    'Importance':importance
}).sort_values('Importance', ascending=False
)

print(classification_report(y_test, y_pred))
print(score)
print(cm)
print(importance_df.to_string(index=False))


predication = model.predict(test)


sub["loan_paid_back"] = predication


sub.to_csv("loan_paid_back.csv",index = False)




