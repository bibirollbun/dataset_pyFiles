import pandas as pd
import warnings

warnings.filterwarnings('ignore')
#читаем данные
df = pd.read_csv("/kaggle/input/bank-churn-competition-by-ipii-hs-ex-mts/train.csv")

df.info()

df.head()


df = df.drop(columns=['id','CustomerId','Surname'],axis=1)


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

corr_matrix = df.corr(numeric_only=True)
plt.figure(figsize=(12, 8))  # tùy chỉnh kích thước
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
plt.show()


# Создаем и обучаем энкодер
from sklearn.preprocessing import OneHotEncoder

countries = ['Germany', 'France', 'Spain']
category_cols = ['Gender','HasCrCard','IsActiveMember']

encoder = OneHotEncoder(drop='first', sparse_output=False)
encoded_data = encoder.fit_transform(df[category_cols])


df_dum = pd.DataFrame(encoded_data, 
                        columns=encoder.get_feature_names_out(category_cols),
                        index=df.index)
df[category_cols] = df_dum
#Идея обучить различные модели для каждой страны
df_France = df[df['Geography'] == 'France']
df_Spain = df[df['Geography'] == 'Spain']
df_Germany = df[df['Geography'] == 'Germany']
df_Germany.drop(columns='Geography', inplace=True)
df_Spain.drop(columns='Geography', inplace=True)
df_France.drop(columns='Geography', inplace=True)


from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
from sklearn.ensemble import StackingClassifier, VotingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder

countries = ['Germany', 'France', 'Spain']
model_dict = {}
error = 0
for country in countries:
    drop_cols = ['Exited']
    if country == 'Germany':
        df = df_Germany
    elif country == 'France':
        df = df_France
    else:
        df = df_Spain


    X = df.drop(drop_cols, axis=1)
    y = df["Exited"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    logistic_pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('model', LogisticRegression(
            solver='liblinear',
            penalty='l1',
            C=0.1,
            class_weight='balanced',
            max_iter=200
        ))
    ])
    #Обучаем carboost изменяем число итераций в зависимости от объема обучающих данных
    cat_model = CatBoostClassifier(
        iterations=len(df) // 70,
        learning_rate=0.1,
        depth=5,
        loss_function='Logloss',
        verbose=100
    )
    model_dict[country] = cat_model

    cat_model.fit(X_train, y_train)

    y_pred = cat_model.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)
    mistake = cm[0][1] + cm[1][0]
    error += mistake
print(error)


#Повторяем для тестовой выборки
df_test = pd.read_csv('/kaggle/input/bank-churn-competition-by-ipii-hs-ex-mts/test.csv')

df_id = df_test["id"]

df_test = df_test.drop(columns=['id','CustomerId','Surname'], axis=1)
encoded_data = encoder.transform(df_test[category_cols])


df_dum = pd.DataFrame(encoded_data, 
                        columns=encoder.get_feature_names_out(category_cols),
                        index=df_test.index)
y_pred_proba = []
df_test[category_cols] = df_dum
for ind, st in df_test.iterrows():
    model = model_dict[st['Geography']]
    my_serial = df_test.iloc[ind].drop('Geography')
    proba_pred = model.predict_proba(my_serial)
    y_pred_proba.append(proba_pred[1])



#Сохраняем решение
submission_df = pd.DataFrame({
    "id": df_id,
    "Exited": y_pred_proba})
submission_df.to_csv("submission.csv", index=False)

