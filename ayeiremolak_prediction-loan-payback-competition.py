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


train_df= pd.read_csv("/kaggle/input/playground-series-s5e11/train.csv")
test_df=pd.read_csv("/kaggle/input/playground-series-s5e11/test.csv")


train_df.head()


test_df.head()


print("missing values in train dataset\n", train_df.isnull().sum())


print("missing values in test dataset\n", test_df.isnull().sum())


print("\ntarget value distribution\n", train_df["loan_paid_back"].value_counts())


print("\ntarget value (%)\n", train_df["loan_paid_back"].value_counts(normalize=True)*100)


train_df.describe()


import seaborn as sns
import matplotlib.pyplot as plt



plt.figure(figsize=(10,6))

sns.boxplot(x="loan_paid_back",y="credit_score",data=train_df)

plt.title("Credit score distribution ")
plt.show()


plt.figure(figsize=(10, 6))
sns.violinplot(x='loan_paid_back', y='credit_score', data=train_df)
plt.title('Violin (1 vs. 0)')
plt.show()


plt.figure(figsize=(10,6))
sns.violinplot(x="loan_paid_back", y="annual_income", data=train_df)
plt.title("income vs loan paid back")
plt.show()


plt.figure(figsize=(10,6))
sns.violinplot(x="loan_paid_back", y="loan_amount", data=train_df)
plt.title("loan amount vs loan paid back")
plt.show()


plt.figure(figsize=(12,7))

sns.barplot(x="education_level",y="loan_paid_back",data=train_df)
plt.title('Average Repayment Rates by Education Level')
plt.xlabel("education level")
plt.ylabel("average repayment rates")
plt.xticks(rotation=45)
plt.show()


plt.figure(figsize=(12,7))

sns.barplot(x="marital_status",y="loan_paid_back",data=train_df)
plt.title('Average Repayment Rates by marital status')
plt.xlabel("marital statu")
plt.ylabel("average repayment rates")
plt.show()


numeric_col=train_df.select_dtypes(include=["number"]).drop(columns=["id"])

corr_matrix=numeric_col.corr()

plt.figure(figsize=(12,10))

sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Heatmap in numeric columns')
plt.show()


from sklearn.model_selection import train_test_split

x=train_df.drop(columns=["loan_paid_back","id"])
y=train_df["loan_paid_back"]

x_train,x_val,y_train,y_val=train_test_split(x,y,test_size=0.2,random_state=42,stratify=y)


x_train.shape


x_val.shape


numeric_features=x_train.select_dtypes(include=["int64","float64"]).columns.tolist()

categorical_features=x_train.select_dtypes(include=["object"]).columns.tolist()


print(numeric_features)
print(categorical_features)


from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


numeric_transformer=Pipeline(steps=[("scaler",StandardScaler())])

categorical_transformer=Pipeline(steps=[("encoder",OneHotEncoder(handle_unknown="ignore"))])

preprocessor=ColumnTransformer(transformers=[("num",numeric_transformer,numeric_features),
                                           ("cat",categorical_transformer,categorical_features)])


from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score

model=LGBMClassifier(random_state=42, class_weight="balanced",n_jobs=-1)

full_pipeline=Pipeline(steps=[("preprocessor",preprocessor),("model",model)])

print("Training the model")

full_pipeline.fit(x_train,y_train)

y_pred_probs=full_pipeline.predict_proba(x_val)[:,1]

score=roc_auc_score(y_val,y_pred_probs)
print(score)


test_id=test_df["id"]
x_test=test_df.drop("id",axis=1)

test_prediction_probs=full_pipeline.predict_proba(x_test)[:,1]

submission_df=pd.DataFrame({"id":test_id,"loan_paid_back":test_prediction_probs})

submission_df.to_csv("submission.csv",index=False)

print(submission_df.head())


!pip install optuna


import optuna



def objective(trial):
    
    # 1. define parameters
    params = {
        # One of the most influential parameters: Tree complexity
        'num_leaves': trial.suggest_int('num_leaves', 31, 127),
        # learning rate
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        # tree number
        'n_estimators': trial.suggest_int('n_estimators', 200, 1000, step=100),
        # To avoid overfitting (random selection of features)
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        # To avoid overfitting (random selection of subsample)
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        # -(Regularization)
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 0.5),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 0.5),
        
        # constant 
        'random_state': 42,
        'n_jobs': -1,
        'class_weight': 'balanced',
        'verbose': -1 
    }
    
    # 2. define model
    model2 = LGBMClassifier(**params)
    
    # 3. Assemble and train the pipeline
    clf = Pipeline(steps=[('preprocessor', preprocessor),
                          ('model2', model2)])
    
    clf.fit(x_train, y_train)
    
    # 4. Skoru hesapla (ROC AUC)
    y_pred_probs2 = clf.predict_proba(x_val)[:, 1]
    roc_auc = roc_auc_score(y_val, y_pred_probs2)
    
    return roc_auc



study = optuna.create_study(direction='maximize')


print("Hyperparameter Optimization Launched with Optuna...")

study.optimize(objective, n_trials=50, show_progress_bar=True)


print("\n==============================================")
print(f"Best ROC AUC Score: {study.best_value:.5f}")
print("Best parameters:")
print(study.best_params)
print("==============================================")




# Optuna's best parameters
best_params = {
    'num_leaves': 105, 
    'learning_rate': 0.022269003946117104, 
    'n_estimators': 900, 
    'colsample_bytree': 0.5027355162928253, 
    'subsample': 0.7999279820651789, 
    'reg_alpha': 0.1859335105045552, 
    'reg_lambda': 0.13049523844962085,
    # constant
    'random_state': 42,
    'n_jobs': -1,
    'class_weight': 'balanced',
    'verbose': -1
}

final_model = LGBMClassifier(**best_params)

final_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', final_model)
])

print("final model training...")
final_pipeline.fit(x_train, y_train) 

y_pred_probs_final = final_pipeline.predict_proba(x_val)[:, 1]
score_final = roc_auc_score(y_val, y_pred_probs_final)
print(f"Kontrol Skoru: {score_final:.5f}") 






# final pipeline predictions
test_predictions_probs2 = final_pipeline.predict_proba(x_test)[:, 1]

# (submission.csv)
submission2_df = pd.DataFrame({
    'id': test_id,
    'loan_paid_back': test_predictions_probs2
})

submission2_df.to_csv('submission_v2.csv', index=False)

submission_df.head()





preprocessor_fitted = final_pipeline.named_steps['preprocessor']
feature_names_out = preprocessor_fitted.get_feature_names_out()
# LightGBM modeli, final_pipeline içindeki 'model' adımıdır
lgbm_model = final_pipeline.named_steps['model']

# Feature importance skorlarını al
feature_importances = pd.DataFrame({
    'Feature': feature_names_out,
    'Importance': lgbm_model.feature_importances_
})

# Önem sırasına göre sırala
feature_importances = feature_importances.sort_values(by='Importance', ascending=False)

# Dosyayı dışa aktar
feature_importances.to_csv('feature_importance.csv', index=False)

print("✅ Feature_importance.csv ready for Power BI!")


print(feature_importances.head(5))


final_risk_df = x_val.copy()

# 3. Gerçek sonucu (y_val) ve tahmin olasılığını yeni sütun olarak ekle
# y_val bir Series olduğu için .values ile eklenmesi gerekir.
final_risk_df['True_Label'] = y_val.values  # Gerçek sonuç (0 veya 1)
final_risk_df['Predicted_Risk'] = y_pred_probs_final # Tahmin olasılığı (0.00 - 1.00)

# 4. Dosyayı kaydet
final_risk_df.to_csv('final_risk_data.csv', index=False)

print("✅ final_risk_data.csv is ready!")
final_risk_df.head()


final_risk_df.dtypes




