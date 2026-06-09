import numpy as np # Linear Algebra
import pandas as pd # Data Processing, CSV file I/O (e.g. pd.read_csv)

#Visulation
import matplotlib.pyplot as plt 
import seaborn as sns

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier

from lightgbm import LGBMClassifier
import xgboost
from xgboost  import XGBClassifier

# Pre-Processing
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

# Hyperparameter Search
from sklearn.model_selection import RandomizedSearchCV

# Evaluate
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.metrics import accuracy_score,recall_score,precision_score,confusion_matrix, log_loss,classification_report

# Model Save-Load
import joblib

import warnings
warnings.filterwarnings("ignore")


palette= "YlOrRd"


df_train = pd.read_csv("/kaggle/input/ai-lab-turkiye-datathon-2025/train.csv")
df_test = pd.read_csv("/kaggle/input/ai-lab-turkiye-datathon-2025/test.csv")


df_train.head(2)


df_test.head(2)


def check_df(dataframe, size=5):
    print('\n\n--- Shape -----------------------------------------------------------------------')
    print(dataframe.shape)
    print('\n\n--- Types -----------------------------------------------------------------------')
    print(dataframe.dtypes)
    print('\n\n--- Head ------------------------------------------------------------------------')
    print(dataframe.head(size))
    print('\n\n--- Tail ------------------------------------------------------------------------')
    print(dataframe.tail(size))
    print('\n\n--- NA --------------------------------------------------------------------------')
    print(dataframe.isnull().sum())
    print('\n\n--- INFO --------------------------------------------------------------------------')
    print(dataframe.info())


check_df(df_train)


check_df(df_test)


target_col = 'Status'
num_cols = df_train.drop(columns=["id"]).select_dtypes(include=['float64', 'int64']).columns.tolist()
cat_cols = df_train.select_dtypes(include=['object', 'category']).columns.tolist()


def target_distribution(dataframe,target_col):
    
    d = dataframe[target_col].value_counts().to_frame(name="Count") 
    
    plt.figure(figsize=(10, 6))
    
    sns.barplot(x=d.index, y='Count', data=d, palette=palette) 
    
    plt.title('Frequency Distribution of the Status Variable', fontsize=16)
    plt.xlabel('Status Categories', fontsize=12)
    plt.ylabel('Frekans (Count)', fontsize=12)
    plt.grid(axis='y', linestyle='--') 
    plt.tight_layout() 
    
    plt.show()


target_distribution(df_train,target_col) # Target variable data distribution


# Data distributions of categorical columns
plt.figure(figsize=(12, 8))

for i, col in enumerate(cat_cols, 1):
    plt.subplot(3, 3, i)
    sns.countplot(data=df_train, x=col, palette=palette)
    plt.title(f"Distribution of {col}")

plt.tight_layout()
plt.show()


df_train.isnull().sum()


def clean_impute(train_df,test_df, target_col, numeric_strategy="knn", knn_k=5):

    if target_col in num_cols: num_cols.remove(target_col)
    if target_col in cat_cols: cat_cols.remove(target_col)

    imputer = None # This imputer will be used in the test data to be loaded later.

    if numeric_strategy == "median":
        medians = train_df[num_cols].median()
        train_df[num_cols] = train_df[num_cols].fillna(medians)
        test_df[num_cols] = test_df[num_cols].fillna(medians)

    elif numeric_strategy == "knn":
        imputer = KNNImputer(n_neighbors=knn_k)
        train_df[num_cols] = imputer.fit_transform(train_df[num_cols])
        test_df[num_cols]  = imputer.transform(test_df[num_cols])

    modes = train_df[cat_cols].mode().iloc[0]
    train_df[cat_cols] = train_df[cat_cols].fillna(modes)
    test_df[cat_cols] = test_df[cat_cols].fillna(modes)

    return train_df, test_df, imputer, modes #The mode and imputer will be used on different data sets.



train_filled,test_filled,imputer,modes = clean_impute(df_train,df_test,target_col)


train_filled.isnull().sum()


test_filled.isnull().sum()


df_corr = train_filled.copy()

cat_cols = df_corr.select_dtypes(include="object").columns

le = LabelEncoder()
for col in cat_cols:
    df_corr[col] = le.fit_transform(df_corr[col].astype(str))

corr_matrix = df_corr.corr()


target_corr = corr_matrix["Status"].sort_values(ascending=False)
target_corr


plt.figure(figsize=(12,14))
sns.heatmap(corr_matrix, fmt=".2f",linewidths=0.8, annot = True, cmap=palette)
plt.title("Correlation Matrix")
plt.show()


cat_cols = train_filled.select_dtypes(include="object").columns.drop("Status")

train_df = pd.get_dummies(train_filled, columns=cat_cols, drop_first=True)
test_df = pd.get_dummies(test_filled, columns=cat_cols, drop_first=True)


check_df(train_df)


check_df(test_df)


le_target = LabelEncoder()
train_df["Status"] = le_target.fit_transform(train_df["Status"])


train_df.head()


train_df.info()


X_train = train_df.drop(columns=["Status", "id"])
X_test = test_df.drop(columns=["id"])
y_train = train_df["Status"]

# Align
X_train, X_test = X_train.align(X_test, join="left", axis=1, fill_value=0)


print("X_train: {}".format(X_train.shape))
print("X_test: {}".format(X_test.shape))
print("y_train: {}".format(y_train.shape))


X_train[:2]


X_test[:2]


scaler = StandardScaler()

X_train_scaled = X_train.copy()
X_test_scaled  = X_test.copy()

X_train_scaled[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test_scaled[num_cols]  = scaler.transform(X_test[num_cols])


# To resolve the version mismatch issue;   
!pip install --upgrade scikit-learn==1.2.2
!pip install --upgrade imbalanced-learn==0.11.0


from imblearn.over_sampling import SMOTE


# SMOTE only train set
smote_scaled = SMOTE(random_state=42, n_jobs=1)
X_train_scaled_res, y_train_res = smote_scaled.fit_resample(X_train_scaled, y_train)

smote_unscaled = SMOTE(random_state=42, n_jobs=1)
X_train_unscaled_res, y_train_res = smote_unscaled.fit_resample(X_train, y_train)

# The result of two smote operations will be the same for y_train. 
# Therefore, no other variable has been defined.


print("X_train: {}".format(X_train_scaled_res.shape))
print("X_test: {}".format(X_train_unscaled_res.shape))
print("y_train: {}".format(y_train_res.shape))


df_res = pd.DataFrame({"Status": y_train_res})

target_distribution(df_res, target_col)


# Models to be trained with scaled data 
models_scaled = {
    "LogisticRegression": LogisticRegression(max_iter=2000, multi_class='multinomial'),
    "KNN": KNeighborsClassifier(),
    "SVC": SVC(probability=True),
    "MLP": MLPClassifier(max_iter=1000)
}

# Models to be trained with unscaled data.
models_raw = {
    "DecisionTree": DecisionTreeClassifier(),
    "RandomForest": RandomForestClassifier(n_estimators=100),
    "GradientBoosting": GradientBoostingClassifier(),
    "XGBoost": XGBClassifier(eval_metric='mlogloss'),
    "LightGBM": LGBMClassifier()
}


def evaluate_models(models, X, y, cv=5):
    results = {}
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)

    for name, model in models.items():

        y_pred = cross_val_predict(model, X, y, cv=kf)
        y_proba = cross_val_predict(model, X, y, cv=kf, method='predict_proba')

        acc = accuracy_score(y, y_pred)
        prec = precision_score(y, y_pred, average="weighted")
        rec = recall_score(y, y_pred, average="weighted")
        ll = log_loss(y, y_proba)

        results[name] = {
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "LogLoss": ll,
        }

    return pd.DataFrame(results).T



print("Scaled models (distance/gradient based):")
df_scaled_results = evaluate_models(models_scaled, X_train_scaled_res, y_train_res)
print(df_scaled_results)


print("\nRaw models (tree/boosting based):")
df_raw_results = evaluate_models(models_raw, X_train_unscaled_res, y_train_res)
print(df_raw_results)


print(df_raw_results)


model_params = {
    "LightGBM": {
        "model": LGBMClassifier(
            objective="multiclass",       
            num_class=3,
            metric="multi_logloss", 
            random_state=42,               
            n_jobs=1,                      
            verbose=-1                      
        ),
        "params": {
            "n_estimators": [200,300,500],   #default=100    
            "learning_rate": [0.05, 0.07, 0.1], # default=0.1
            "num_leaves": [15,31,63],       # default=31       
            "min_child_samples": [20,30,40],     #default=20   
            "subsample": [0.8, 1.0],
            "colsample_bytree": [0.8, 1.0]
        }
    },
    "XGBoost": {
        "model": XGBClassifier(
            tree_method='gpu_hist',
            predictor='gpu_predictor',
            objective="multi:softprob",# multi-class probability 
            num_class=3,    
            eval_metric="mlogloss",# log-loss
            random_state=42,             
            n_jobs=-1,                    
            verbosity=0                  
        ),
        "params": {
            "n_estimators": [100,300, 400, 500],   
            "learning_rate": [0.03, 0.05, 0.1,0.3], # default=0.3    
            "max_depth": [3,4,5,6],   # default=6          
            "subsample": [0.7, 0.8, 0.9,1], # default=1     
            "colsample_bytree": [0.8, 1.0], # default=1
            "gamma": [0, 0.05, 0.1],   # default=0  
            "reg_lambda": [1,3,5],   # L2 regularization
        }
    }

}


scores = []

for model_name, model_info in model_params.items():
    print(f"Running RandomizedSearchCV for {model_name}...")
    
    model = model_info["model"]
    param_grid = model_info["params"]
    
    # RandomizedSearchCV
    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_grid,
        n_iter=30,            
        scoring="neg_log_loss", 
        cv=5,
        verbose=1,
        n_jobs=1,
        random_state=42
    )
    
    random_search.fit(X_train_unscaled_res, y_train_res)
    
    scores.append([
        model_name,
        random_search.best_params_,
        -random_search.best_score_  
    ])


results_df = pd.DataFrame(scores, columns=["Model", "Best_Params", "CV_LogLoss"])
results_df = results_df.sort_values("CV_LogLoss")
results_df


results_df


results_df["Best_Params"][0],results_df["Best_Params"][1]


best_params = {'subsample': 0.8,
  'num_leaves': 63,
  'n_estimators': 300,
  'min_child_samples': 30,
  'learning_rate': 0.07,
  'colsample_bytree': 0.8}


best_model_final =  LGBMClassifier(
            objective="multiclass",       
            num_class=3,
            metric="multi_logloss", 
            random_state=42,                                    
            verbose=-1)


best_model_final.fit(X_train_unscaled_res, y_train_res)


joblib.dump(best_model_final, "/kaggle/working/best_model_final.pkl")
print("Model saved: best_model_final.pkl")


y_test_proba = best_model_final.predict_proba(X_test)
 
columns = ["Status_C", "Status_CL", "Status_D"]

proba_df = pd.DataFrame(y_test_proba, columns=columns)

proba_df["id"] = test_df["id"] 

proba_df = proba_df[["id", "Status_C", "Status_CL", "Status_D"]]

proba_df.to_csv("/kaggle/working/submission.csv", index=False)

print("submission.csv created.")


best_model_lgbm = joblib.load("/kaggle/working/best_model_final.pkl")
print("Model loaded successfully.")


# train set probability forecasts
y_pred_proba = best_model_lgbm.predict_proba(X_train)  
y_pred_class = best_model_lgbm.predict(X_train)       


y_pred_proba[:4]


y_pred_class[:4]


y_train[:4]


ll = log_loss(y_train, y_pred_proba)
acc = accuracy_score(y_train, y_pred_class)

print(f"LogLoss: {ll:.6f}")
print(f"Accuracy: {acc:.4f}")


columns = ["Status_C", "Status_CL", "Status_D"]

print(classification_report(y_train, y_pred_class, target_names=columns))


cm = confusion_matrix(y_train, y_pred_class)

plt.figure(figsize=(10,8))
sns.heatmap(cm, annot=True, fmt='d', cmap=sns.color_palette(palette), xticklabels=columns, yticklabels=columns)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()


# train set probability forecasts
y_pred_proba_smote = best_model_lgbm.predict_proba(X_train_unscaled_res)  
y_pred_class_smote = best_model_lgbm.predict(X_train_unscaled_res)   


ll_res = log_loss(y_train_res, y_pred_proba_smote)
acc_res = accuracy_score(y_train_res, y_pred_class_smote)

print(f"LogLoss: {ll_res:.6f}")
print(f"Accuracy: {acc_res:.4f}")


columns = ["Status_C", "Status_CL", "Status_D"]

print(classification_report(y_train_res, y_pred_class_smote, target_names=columns))


cm = confusion_matrix(y_train_res, y_pred_class_smote)

plt.figure(figsize=(10,8))
sns.heatmap(cm, annot=True, fmt='d', cmap=sns.color_palette(palette), xticklabels=columns, yticklabels=columns)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()


# Dataset load and EDA
df_second_test = pd.read_csv("/kaggle/input/cirrhosis-patient-survival-prediction/cirrhosis.csv")
check_df(df_second_test)


# Data Preprocessing
df_second_test = df_second_test.rename(columns={"ID": "id"}) # The ID column naming does not match; it has been corrected.


# The column order has been arranged.
train_order = df_train.columns

df_second_test = df_second_test.reindex(columns=train_order)

# column sequence check
print("df_second_test columns : ",df_second_test.columns)
print("df_train columns : ",train_order)


# nan imputation ;  imputer and modes have been calculated above.
def apply_imputation(new_test_df, num_cols, cat_cols, imputer, modes):
    new_test_df[num_cols] = imputer.transform(new_test_df[num_cols])
    new_test_df[cat_cols] = new_test_df[cat_cols].fillna(modes)
    return new_test_df

df_second_test = apply_imputation(df_second_test, num_cols, cat_cols, imputer, modes)
df_second_test.isnull().sum()


# one-hot-encoding
df_second_test = pd.get_dummies(df_second_test, columns=cat_cols, drop_first=True)
df_second_test.head(2)


# label encode 
le_test_target = LabelEncoder()
df_second_test["Status"] = le_test_target.fit_transform(df_second_test["Status"])


df_second_test.info()


# Trainâ€“Test Preparation: X and y

X_second_test = df_second_test.drop(columns=["Status", "id"])
y_second_test = df_second_test["Status"]

# Align ; equal feature
_, X_second_test = X_train.align(X_second_test, join="left", axis=1, fill_value=0)


print("X_second_test: {}".format(X_second_test.shape))
print("y_second_test: {}".format(y_second_test.shape))


# Load model 
best_model_lgbm = joblib.load("/kaggle/input/liver-cirrhosis-classification-final-model/other/default/1/best_model_final.pkl")
print("Model loaded successfully.")


# probability forecasts
y_pred_proba_second = best_model_lgbm.predict_proba(X_second_test)  
y_pred_class_second = best_model_lgbm.predict(X_second_test) 


ll_second = log_loss(y_second_test, y_pred_proba_second)
acc_second = accuracy_score(y_second_test, y_pred_class_second)

print(f"LogLoss: {ll_second:.6f}")
print(f"Accuracy: {acc_second:.4f}")


columns = ["Status_C", "Status_CL", "Status_D"]

print(classification_report(y_second_test, y_pred_class_second, target_names=columns))


cm = confusion_matrix(y_second_test, y_pred_class_second)

plt.figure(figsize=(10,8))
sns.heatmap(cm, annot=True, fmt='d', cmap=sns.color_palette(palette), xticklabels=columns, yticklabels=columns)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

