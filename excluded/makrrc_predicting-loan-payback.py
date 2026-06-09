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


from sklearn.preprocessing import LabelEncoder


train = pd.read_csv('/kaggle/input/playground-series-s5e11/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e11/test.csv')


oringin = pd.read_csv("/kaggle/input/loan-prediction-dataset-2025/loan_dataset_20000.csv")


train.head()


train.describe()


train.isna().sum()


train.dtypes


train['gender'].unique()


train[train['gender'] == 'Other']


objList = train.select_dtypes(include = "object").columns
objList


le = LabelEncoder()
def CodificarLabelEncode(Dataset, columns, LabelEncoder):
    LabelEncoder.fit(Dataset[columns])
    Dataset['le_'+columns] = le.transform(Dataset[columns])
    return Dataset


labelLists = ['marital_status', 'education_level']


# for labelList in labelLists:
#     train = CodificarLabelEncode(train, labelList, le)

# train


from sklearn.preprocessing import OrdinalEncoder


# def CodificarOrdinalEncoder( Dataset, columns, OrdinalEncoder ):
#     OrdinalEncoder.fit(Dataset[columns])
#     Dataset['oe_'+columns] = le.transform(Dataset[[columns]])
#     return Dataset    


oe = OrdinalEncoder()

def CodificarOrdinalEncoder(dataset, column, encoder):
    """
    Codifica uma coluna categórica usando OrdinalEncoder e
    adiciona uma nova coluna codificada ao DataFrame.
    """
    # O erro estava aqui: precisamos passar [[column]] para garantir 2D
    encoder.fit(dataset[[column]])  
    dataset['oe_' + column] = encoder.transform(dataset[[column]])
    return dataset


ordinalLists = ['marital_status', 'education_level']


for ordinalList in ordinalLists:
    train = CodificarOrdinalEncoder(train, ordinalList, oe)


for ordinalList in ordinalLists:
    test = CodificarOrdinalEncoder(test, ordinalList, oe)


train['employment_status'].unique()


train['gender'].unique()


def get_dummies1(dataset, column):
    result = ''
    for elemento in column:
        print(elemento)
        temp = pd.get_dummies(dataset[elemento])
        dataset = pd.concat([dataset, temp], axis=1)
    return dataset    


train = get_dummies1(train, ['employment_status','gender'])
train


test = get_dummies1(test, ['employment_status','gender'])
test


train['Other']


train[train['Other'] == True]


train = train.replace({True:1,False:0})


test = test.replace({True:1,False:0})


test.columns


oringin.columns


train = train[['id', 'annual_income', 'debt_to_income_ratio', 'credit_score', 
       'loan_amount', 'interest_rate', 'gender', 'marital_status',
       'education_level', 'employment_status', 'loan_purpose',
       'grade_subgrade', 'oe_marital_status',
       'oe_education_level', 'Employed', 'Retired', 'Self-employed', 'Student',
       'Unemployed', 'Female', 'Male', 'Other', 'loan_paid_back']]


# train = train[['id','Unemployed','Employed','Self-employed',
#              'debt_to_income_ratio','credit_score','Student','interest_rate',
#              'loan_amount','Retired', 'loan_paid_back']]


test = test[['id', 'annual_income', 'debt_to_income_ratio', 'credit_score', 
       'loan_amount', 'interest_rate', 'gender', 'marital_status',
       'education_level', 'employment_status', 'loan_purpose',
       'grade_subgrade', 'oe_marital_status',
       'oe_education_level', 'Employed', 'Retired', 'Self-employed', 'Student',
       'Unemployed', 'Female', 'Male', 'Other']]


# test = test[[
#     'id','Unemployed','Employed','Self-employed',
#              'debt_to_income_ratio','credit_score','Student','interest_rate',
#              'loan_amount','Retired'
#             ]]


len(train.columns)


len(test.columns)


y = train.iloc[:,-1]


y


y_test = test.iloc[:,test.columns != 'id']


y_test


x = train.iloc[:,train.columns != 'id']


x


x_test = test.iloc[:,:-1]
x_test


test


x = x[x.columns[x.dtypes != 'object']]


x_test = x_test[x_test.columns[x_test.dtypes != 'object']]


from sklearn import linear_model
from sklearn.inspection import permutation_importance

from sklearn.linear_model import RidgeCV

import matplotlib.pyplot as plt
import seaborn as sns


x.iloc[:,:-1]


from sklearn.ensemble import RandomForestClassifier


from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

from xgboost import XGBRegressor
from catboost import CatBoostRegressor

from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import LinearSVR

from sklearn.ensemble import StackingRegressor

from sklearn.ensemble import StackingClassifier

from sklearn.linear_model import LogisticRegression


# Define base estimators
estimators = [
    ('xgb', XGBClassifier(objective='binary:logistic')),
    ('lgb', LGBMClassifier()),
    ('cat', CatBoostClassifier(verbose=0)),
]


# Create and fit the StackingRegressor
stacking_classifier = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression(max_iter=2000)
)


# model.fit(x.iloc[:,:-1],y)


# result = permutation_importance(
#     model, x.iloc[:,:-1], y, n_repeats=10, random_state=42, n_jobs=2
# )


# # 📈 Criar DataFrame ordenado pela importância
# importances = pd.DataFrame({
#     "Feature": x.iloc[:,:-1].columns,
#     "Importance": result.importances_mean
# }).sort_values(by="Importance", ascending=False)


# # 🖼️ Plotar gráfico de barras
# plt.figure(figsize=(8, 5))
# plt.barh(importances["Feature"], importances["Importance"], color="royalblue")
# plt.xlabel("Importância média (Permutation Importance)")
# plt.ylabel("Variáveis (Features)")
# plt.title("Importância das Features — Linear Regression")
# plt.gca().invert_yaxis()  # Inverter eixo para mostrar a mais importante no topo
# plt.grid(alpha=0.3)
# plt.show()

# # 📋 Exibir tabela
# print(importances)


# from xgboost import XGBRegressor


# xgb_model = XGBRegressor()


x.head()


x


x.iloc[:,:-1]


# xgb_model.fit(x.iloc[:,:-1],y)


# result = permutation_importance(
#     xgb_model, x.iloc[:,:-1], y, n_repeats=10, random_state=42, n_jobs=2
# )


# # 📈 Criar DataFrame ordenado pela importância
# importances = pd.DataFrame({
#     "Feature": x.iloc[:,:-1].columns,
#     "Importance": result.importances_mean
# }).sort_values(by="Importance", ascending=False)


# # 🖼️ Plotar gráfico de barras
# plt.figure(figsize=(8, 5))
# plt.barh(importances["Feature"], importances["Importance"], color="royalblue")
# plt.xlabel("Importância média (Permutation Importance)")
# plt.ylabel("Variáveis (Features)")
# plt.title("Importância das Features — Linear Regression")
# plt.gca().invert_yaxis()  # Inverter eixo para mostrar a mais importante no topo
# plt.grid(alpha=0.3)
# plt.show()

# # 📋 Exibir tabela
# print(importances)





sns.heatmap(x.iloc[:,:-1].corr())


# sns.pairplot(train)


train


from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


scaler = StandardScaler()
scaled_data = scaler.fit_transform(x)
scaled_data


pca= PCA(n_components=3)
pca.fit(scaled_data)


transformed_data = pca.transform(scaled_data)
transformed_data


df_transformed_data = pd.DataFrame(transformed_data)


df_transformed_data


# sns.scatterplot(data=df_transformed_data)


# sns.pairplot(df_transformed_data)


# sns.histplot(df_transformed_data)


# sns.histplot(train)


train = pd.concat([train, df_transformed_data], axis=1)
train


# # Create a box plot
# plt.boxplot(df_transformed_data)
# plt.title("Box Plot using Matplotlib")
# # plt.ylabel("Value")
# plt.show()


# train.hist(figsize=(16, 12))


from sklearn.model_selection import train_test_split


y


X_train, X_test, y_train, y_test = train_test_split(x, y, stratify=y, random_state=42)


X_train.iloc[:,:-1].columns


X_train.columns


stacking_classifier.fit(X_train.iloc[:,:-1],y_train)





# Make predictions
y_pred = stacking_classifier.predict(X_test.iloc[:,:-1])
# model = RandomForestClassifier()


X_train.iloc[:,:-1]


oringin


oringin.head()


train.head()


colunas_comuns = oringin.columns.intersection(train.columns)
colunas_comuns


oringin[colunas_comuns]


oringin = oringin.loc[:,colunas_comuns]


for labelList in labelLists:
    oringin = CodificarLabelEncode(oringin, labelList, le)


oringin = get_dummies1(oringin , ['employment_status','gender'])


for ordinalList in ordinalLists:
    train = CodificarOrdinalEncoder(oringin, ordinalList, oe)


oringin


colunas_comuns = oringin.columns.intersection(train.columns)
colunas_comuns


oringin


origin_pred = oringin['loan_paid_back']


oringin = oringin[['annual_income', 'debt_to_income_ratio', 'credit_score', 'loan_amount',
       'interest_rate', 'oe_marital_status', 'oe_education_level', 'Employed',
       'Retired', 'Self-employed', 'Student', 'Unemployed', 'Female', 'Male',
       'Other']]


# model.fit(X_train.iloc[:,:-1],y_train)

# Define base estimators
estimators2 = [
    ('xgbc', XGBRegressor(n_estimators=2, learning_rate=0.1, max_depth=5)),
    ('rfc', RandomForestRegressor()),
    ('svc', LinearSVR())
]

# Define the final estimator
final_estimator = RidgeCV()


stacking_classifier.fit(oringin,origin_pred)


# Create and fit the StackingRegressor
stacking_regressor2 = StackingRegressor(
    estimators=estimators2,
    final_estimator=final_estimator,
    cv=5  # Use 5-fold cross-validation for training the final estimator
)


# Assuming X_test is your test features
y_pred0 = stacking_classifier.predict(X_train.iloc[:,:-1])


origin_pred


X_train.iloc[:,:-1]


# Assuming X_test is your test features
y_pred0 = stacking_classifier.predict(X_train.iloc[:,:-1])


residuals = y_train - y_pred0


# model2 = LinearRegression()


# from catboost import CatBoostRegressor


# pip install "tabpfn @ git+https://github.com/PriorLabs/TabPFN.git"


# xgb_model2 = XGBRegressor()


# pip install tabpfn


# !pip install /kaggle/input/pip-install-lifelines/autograd-1.7.0-py3-none-any.whl
# !pip install /kaggle/input/pip-install-lifelines/autograd-gamma-0.5.0.tar.gz
# !pip install /kaggle/input/pip-install-lifelines/interface_meta-1.3.0-py3-none-any.whl
# !pip install /kaggle/input/pip-install-lifelines/formulaic-1.0.2-py3-none-any.whl
# !pip install /kaggle/input/pip-install-lifelines/lifelines-0.30.0-py3-none-any.whl


# import huggingface_hub
# huggingface_hub.login()


# from tabpfn import TabPFNRegressor


# regressor_args = {
#     "device": 'cpu',
#     "n_estimators": 2
# }

# reg = TabPFNRegressor(**regressor_args, fit_mode="batched", ignore_pretraining_limits=True)


# To use TabPFN v2:
# clf = TabPFNClassifier.create_default_for_version(ModelVersion.V2)
# reg.fit(X_train.iloc[:,:-1], residuals)


stacking_regressor2.fit(X_train.iloc[:,:-1], residuals)


correction = stacking_regressor2.predict(X_train.iloc[:,:-1])


# y_pred_proba = stacking_regressor2.predict_proba(X_test.iloc[:,:-1])[:, 1]


# Soma das duas previsões
final_pred = y_pred0 + correction


from sklearn.metrics import mean_squared_error, roc_auc_score, r2_score


# y_pred_proba_ = StackingRegressor.predict_proba(X_train.iloc[:,:-1])[:, 1]


from sklearn.metrics import roc_curve, roc_auc_score, auc


def roc_auc(y_train, y_pred):
    # Plotting the ROC Curve
    fpr, tpr, thresholds = roc_curve(y_train, y_pred)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()


# roc_auc(y_train, y_pred_proba_)


# roc_auc(y_train, final_pred)


# Avaliação y_pred
rmse = np.sqrt(mean_squared_error(final_pred, y_train))
r2 = r2_score(final_pred, y_train)


print(f"RMSE: {rmse:.2f}")
print(f"R²: {r2:.3f}")


# # Evaluate performance
# accuracy = model.score(X_train.iloc[:,:-1],final_pred)
# print(f"Linear Regression Classifier Accuracy: {accuracy}")


x.columns != 'loan_paid_back'


test.columns


test = test[['id', 'annual_income', 'debt_to_income_ratio', 'credit_score', 
       'loan_amount', 'interest_rate', 'gender', 'marital_status',
       'education_level', 'employment_status', 'loan_purpose',
       'grade_subgrade', 'oe_marital_status',
       'oe_education_level', 'Employed', 'Retired', 'Self-employed', 'Student',
       'Unemployed', 'Female', 'Male', 'Other']]


x.columns


# test.loc[:,(x.columns != 'loan_paid_back') & (x.columns != 'id') ]


test = test[['id','annual_income', 'debt_to_income_ratio', 'credit_score', 'loan_amount',
       'interest_rate', 'oe_marital_status', 'oe_education_level', 'Employed',
       'Retired', 'Self-employed', 'Student', 'Unemployed', 'Female', 'Male',
       'Other']]


test_submit_columns = test.iloc[:, test.columns != 'id' ]


# test_submit_columns = test.loc[:,(x.columns != 'loan_paid_back') & (x.columns != 'id') ]


test_submit_columns.head()


# Assuming X_test is your test features
normal_y_pred = stacking_classifier.predict(test_submit_columns)


# Correção do segundo modelo
residuals_y_pred =  stacking_regressor2.predict(test_submit_columns)



final_submit = normal_y_pred + residuals_y_pred


final_submit.shape


final_submit.shape


y_test.shape


x.columns


test.columns


id = test['id']


test = test.drop(columns=['id'])


#Submission must be done within the standard requested by the challenge. 
#    In this case, the depression column must be submitted with 0 and 1.
#    This is why it is necessary to convert.

submission = final_submit
submission


## Submit notebooks to the challenge. Final


submission_final = pd.DataFrame({

        "id":id,

        "Premium Amount":submission

    })

submission_final.to_csv('submission.csv', index=False)


print(" Arquivo submission.csv pronto ")

