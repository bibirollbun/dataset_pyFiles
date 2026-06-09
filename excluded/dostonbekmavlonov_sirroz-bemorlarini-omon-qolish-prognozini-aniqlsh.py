import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier



import pandas as pd

# Kaggle muhitida fayllarni yuklash
df_train = pd.read_csv('/kaggle/input/multiclassificationtask/train.csv')
df_test = pd.read_csv('/kaggle/input/multiclassificationtask/test.csv')
df_sub = pd.read_csv('/kaggle/input/multiclassificationtask/sample_submission.csv')

# Ma'lumotlarni ko'rib chiqish
df_train.head()



print(df_train.shape)
print(df_test.shape)
print(df_sub.shape)


categorical = ['Drug','Sex','Ascites','Hepatomegaly','Spiders','Edema','Status']
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
for i in categorical:
    df_train[i] = le.fit_transform(df_train[i])
df_train.head()


df_train.corrwith(df_train['Status']).abs().sort_values(ascending=False)


del_cat = ['id','Drug','Ascites','Spiders','Hepatomegaly']
train_cor = df_train.drop(del_cat,axis=1)

# replacing NaN to mean value
train_cor.fillna(df_train.mean(),inplace=True)
train_cor.head()


train_cor.isnull().sum() # final result


test_cor = df_test.drop(del_cat,axis=1)
id = df_test['id'].copy()
test_cor.isnull().sum()
test_cor.info()


# label encoder test
categ = ['Sex','Edema']
for i in categ:
    test_cor[i] = le.fit_transform(test_cor[i])
test_cor.head()

# fillna()
test_cor.fillna(test_cor.mean(),inplace=True)
test_cor.head()


#mixing train_cor elements
train_f = train_cor.sample(frac=1, random_state=42).reset_index(drop=True)
train_f['Status'].value_counts()

# drop Status = 3
train_f = train_f[train_f['Status'] != 3]
train_f['Status'].value_counts()



# train_set test_set
from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(train_f, test_size=0.1, random_state=42)
x_train = train_set.drop('Status',axis=1)
y_train = train_set['Status'].copy()
x_test = test_set.drop('Status',axis=1)
y_test = test_set['Status'].copy()


# standart scaler
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)
test_cor = sc.transform(test_cor)


# final data
print(x_train.shape) # train set
print(y_train.shape) # train set
print(x_test.shape) # test set
print(y_test.shape) # test set
print(test_cor.shape) # submission test set
print(id.shape) # submission id


y_test.value_counts()
y_train.value_counts()


from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression(C=10)
log_reg.fit(x_train,y_train)

# y_pred probability
y_pred = log_reg.predict_proba(x_test)


# best hyperparametrs
from sklearn.model_selection import GridSearchCV
param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000] }
grid_search = GridSearchCV(LogisticRegression(), param_grid, cv=5)
grid_search.fit(x_train, y_train)
grid_search.best_params_



# model evaluation with multiclass logariphimic loss
from sklearn.metrics import log_loss
mcll = log_loss(y_test,y_pred)
print("Multi-Class Lorgariphimic Loss: ", mcll)


from sklearn.tree import DecisionTreeClassifier
dt_clf = DecisionTreeClassifier(max_depth=4)
dt_clf.fit(x_train,y_train)

# y_pred probability
y_pred = dt_clf.predict_proba(x_test)


# best hyperparametr max_depth and n_estemators
from sklearn.model_selection import GridSearchCV
param_grid = {'max_depth': list(range(1,20))}
grid_search = GridSearchCV(DecisionTreeClassifier(), param_grid, cv=5)
grid_search.fit(x_train, y_train)
grid_search.best_params_


# model evaluation with multiclass logariphimic loss
from sklearn.metrics import log_loss
mcll = log_loss(y_test,y_pred)
print("Multi-Class Lorgariphimic Loss: ", mcll)


from sklearn.ensemble import RandomForestClassifier
rf_clf = RandomForestClassifier(n_estimators=100, max_depth=12)
rf_clf.fit(x_train,y_train)

# y_pred probability
y_pred = rf_clf.predict_proba(x_test)


# best hyperparametr
from sklearn.model_selection import GridSearchCV
param_grid = {'max_depth': list(range(1,20))}
grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=5)
grid_search.fit(x_train, y_train)
grid_search.best_params_


# model evaluation with multiclass logariphimic loss
from sklearn.metrics import log_loss
mcll = log_loss(y_test,y_pred)
print("Multi-Class Lorgariphimic Loss: ", mcll)


from sklearn.ensemble import GradientBoostingClassifier
xgb_clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=4)
xgb_clf.fit(x_train,y_train)

# y_pred probability
y_pred = xgb_clf.predict_proba(x_test)


# model evaluation with multiclass logariphimic loss
from sklearn.metrics import log_loss
mcll = log_loss(y_test,y_pred)
print("Multi-Class Lorgariphimic Loss: ", mcll)


# getting sample_submission1.csv file
y_pred_sub = xgb_clf.predict_proba(test_cor)
sub = pd.DataFrame(y_pred_sub, columns=['Status_C','Status_CL','Status_D'])
sub.insert(0,'id',id)
sub.to_csv('sample_submission1.csv',index=False)




