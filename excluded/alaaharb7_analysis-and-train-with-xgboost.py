import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="darkgrid",font_scale=1.5)
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

from sklearn.metrics import accuracy_score, precision_score, recall_score ,f1_score, confusion_matrix, classification_report


df = pd.read_csv(r'/kaggle/input/playground-series-s4e2/train.csv')
df.head()


df.shape


df.info()


df.describe()


df.isna().sum()


df.duplicated().sum()


categorical_cols = df.select_dtypes(include = "O").columns
numerical_cols = df.select_dtypes(include = "number").columns


for i in df[categorical_cols]:
    print(f'{i} ({df[i].nunique()}) => {df[i].unique()}')
    print('-' * 30)


df['NObeyesdad'].value_counts()


plt.figure(figsize = (15, 8))

ax = sns.countplot(data = df, x = 'NObeyesdad', palette = 'Set2')
for i in ax.containers:
    ax.bar_label(i, fontweight = 'black', size = 14)
plt.title("NObeyesdad ?",fontweight="black",size=20,pad=20)



df['Gender'].value_counts()


plt.figure(figsize = (8, 6))
ax = sns.countplot(data = df, x = 'Gender', palette = 'Set2')
for container in ax.containers:
    ax.bar_label(container, fontweight = 'black', size = 12)
plt.title('Number of Male & Female', size = 15)
plt.show()


df['Age'].describe()


plt.figure(figsize = (15, 8))
sns.kdeplot(data = df, x = 'Age', fill = True)
plt.title('Distribution of Age', size = 18)
plt.show()


df['Height'].describe()


plt.figure(figsize = (15, 8))
sns.histplot(data = df, x = 'Height', kde = True)
plt.title('Distribution of Height', size = 18)
plt.show()


df['Weight'].describe()


plt.figure(figsize = (15, 8))
sns.kdeplot(data = df, x = 'Weight', fill = True)
plt.title('Distribution of Weight', size = 18)
plt.show()


df['family_history_with_overweight'].value_counts()


plt.figure(figsize = (15, 6))

plt.subplot(1,2,1)
ax = sns.countplot(data = df, x = 'family_history_with_overweight', palette = 'Set2')
for i in ax.containers:
    ax.bar_label(i, fontweight = 'black', size = 14)
plt.title("family_history_with_overweight ?",fontweight="black",size=20,pad=20)


plt.subplot(1,2,2)
plt.pie(df['family_history_with_overweight'].value_counts(), autopct = '%1.1f%%', explode = [0.1, 0], labels = ['Yes', 'No'], colors = sns.set_palette('Set2'),textprops={"fontweight":"black"})
plt.title("family_history_with_overweight ?",fontweight="black",size=20,pad=20)
plt.show()


df['FAVC'].value_counts()


ax = sns.countplot(data = df, x = 'FAVC', palette = 'Set2')
for i in ax.containers:
    ax.bar_label(i, fontweight = 'black', size = 14)
plt.title("What is the number of FAVC in Data ?",fontweight="black",size=20,pad=20)
plt.show()


df['FCVC'].describe()


plt.figure(figsize = (15, 8))
sns.kdeplot(data = df, x = 'FCVC', fill = True, palette = 'Set2')
plt.title('Number of FCVC', size = 18)
plt.show()


df['NCP'].describe()


plt.figure(figsize = (15, 8))
sns.kdeplot(data = df, x = 'NCP', fill = True, palette = 'Set2')
plt.title('Number of Main Meals per Day', size = 18)
plt.show()


df['CAEC'].value_counts()


plt.figure(figsize = (15, 6))

plt.subplot(1,2,1)
ax = sns.countplot(data = df, x = 'CAEC', palette = 'Set2')
for i in ax.containers:
    ax.bar_label(i, fontweight = 'black', size = 14)
plt.title("Consumption of Food Between Meals ?",fontweight="black",size=20,pad=20)


plt.subplot(1,2,2)
plt.pie(df['CAEC'].value_counts(), autopct = '%1.1f%%', explode = [0.1, 0, 0,0], labels = df['CAEC'].value_counts().index, colors = sns.set_palette('Set2'),textprops={"fontweight":"black"})
plt.title("Consumption of Food Between Meals ?",fontweight="black",size=20,pad=20)
plt.show()


df['SMOKE'].value_counts()


plt.figure(figsize = (8, 6))
ax = sns.countplot(data = df, x = 'SMOKE', palette = 'Set2')
for i in ax.containers:
    ax.bar_label(i, fontweight = 'black', size = 14)
plt.title("What is Number of Is Smoke ?",fontweight="black",size=20,pad=20)
plt.show()


df['CH2O'].describe()


plt.figure(figsize = (15, 8))
sns.kdeplot(data = df, x = 'CH2O', fill = True, palette = 'Set2')
plt.title(' Consumption of Water Daily', size = 18)
plt.show()


df['SCC'].value_counts()


plt.figure(figsize = (8, 6))
plt.pie(df['SCC'].value_counts(), autopct = '%1.1f%%', explode = [0.1, 0], labels = ['No', 'Yes'], colors = sns.set_palette('Set2'),textprops={"fontweight":"black"})
plt.title("Number Of Poeple Monitor calorie consumption ",fontweight="black",size=20,pad=20)
plt.show()


df['FAF'].describe()


plt.figure(figsize = (15, 8))
sns.histplot(data = df, x = 'FAF')
plt.title('Distribution of FAF', size = 18)
plt.show()


df['TUE'].describe()


plt.figure(figsize = (15, 8))
sns.kdeplot(data = df, x = 'TUE', fill = True, palette = 'Set2')
plt.title('Distribution Of TUE', size = 18)
plt.show()


plt.figure(figsize = (15, 6))

plt.subplot(1,2,1)
ax = sns.countplot(data = df, x = 'CALC', palette = 'Set2')
for i in ax.containers:
    ax.bar_label(i, fontweight = 'black', size = 14)
plt.title(" CountPlot of CALC ?",fontweight="black",size=20,pad=20)


plt.subplot(1,2,2)
ax = sns.countplot(data = df, x = 'MTRANS', palette = 'Set2')
for i in ax.containers:
    ax.bar_label(i, fontweight = 'black', size = 14)
plt.xticks(rotation = 45)
plt.title("Count Plot of MTRANS ?",fontweight="black",size=20,pad=20)
plt.show()


plt.figure(figsize = (15, 10))
for ind, val in enumerate(['FAVC', 'Gender', 'SMOKE','SCC', 'CALC', 'family_history_with_overweight']):
    plt.subplot(2,3, ind + 1)
    sns.countplot(data = df, x = val, hue = 'NObeyesdad', palette = 'Set2')
    plt.legend(fontsize=10)
    plt.title(f'{val} vs NObeyesdad')

plt.tight_layout()
plt.show()


encoder = LabelEncoder()


for i in df[categorical_cols]:
    df[i] =  encoder.fit_transform(df[i])


df.info()


df_test = pd.read_csv(r'/kaggle/input/playground-series-s4e2/test.csv')
df_test.head()


test_cols = df_test.select_dtypes(include = 'O').columns


for i in df_test[test_cols]:
   df_test[i] =  encoder.fit_transform(df_test[i])


df_test.info()


scaler = StandardScaler()


df['Age'] = scaler.fit_transform(df[['Age']])
df['Weight'] = scaler.fit_transform(df[['Weight']])


df_test['Age'] = scaler.fit_transform(df_test[['Age']])
df_test['Weight'] = scaler.fit_transform(df_test[['Weight']])


plt.figure(figsize = (15, 6))

plt.subplot(1,2,1)
sns.kdeplot(data = df, x = 'Age',fill = True, palette = 'Set2')
plt.title("Distribution of Age",fontweight="black",size=20,pad=20)

plt.subplot(1,2,2)
sns.kdeplot(data = df, x = 'Weight', fill = True, palette = 'Set2')
plt.title("Distribution of Weight",fontweight="black",size=20,pad=20)
plt.show()


X = df.drop(['NObeyesdad', 'id'], axis = 1)
y = df['NObeyesdad']


X.shape


y.shape


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 42)


def train_model(model):
    model.fit(X_train, y_train)
    print(f'Train Score => {model.score(X_train, y_train)}')
    print(f'Test Score => {model.score(X_test, y_test)}')
    y_pred = model.predict(X_test)
    print(classification_report(y_pred, y_test))


lr = LogisticRegression()


train_model(lr)


svc = SVC()


train_model(svc)


knn = KNeighborsClassifier(n_neighbors = 5)


train_model(knn)


DT = DecisionTreeClassifier(max_depth = 9, random_state=42)


train_model(DT)


RF = RandomForestClassifier(n_estimators = 100)


train_model(RF)


gb = GradientBoostingClassifier(n_estimators = 150)


train_model(gb)


ada = AdaBoostClassifier(n_estimators = 100, estimator = RF, learning_rate = 0.1)


train_model(ada)


xgb = XGBClassifier(n_estimators = 300, max_depth = 3, learning_rate = 0.1)


train_model(xgb)


cat = CatBoostClassifier(iterations=300, learning_rate=0.3, depth=5)


train_model(cat)


params = {'n_estimators': [100, 200, 300], 
         'learning_rate': [0.1, 0.01, 0.001],
         'max_depth': [3, 6, 9]}

score = 'accuracy'


model_xgb = GridSearchCV(xgb, params, scoring = score, n_jobs = -1)
model_xgb.fit(X_train, y_train)
print(model_xgb.best_params_)
print(model_xgb.best_score_)


params = {
    'depth': [3, 5, 7],  # استخدم المعامل depth فقط
    'learning_rate': [0.01, 0.1, 0.3],
    'iterations': [100, 200, 300]
}


model_cat = GridSearchCV(cat, params, scoring = score, n_jobs = -1)
model_cat.fit(X_train, y_train)
print(model_cat.best_params_)
print(model_cat.best_score_)


test_X = df_test.drop('id', axis = 1)
pred_X = xgb.predict(test_X)


submission = pd.DataFrame({'id': df_test['id'], 'NObeyesdad': pred_X})
submission


submission.to_csv("submission.csv", index = False)




