import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score 
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.linear_model import LogisticRegression
from scipy.stats import linregress
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))



train_data = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv')
test_data = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv')
sample_output = pd.read_csv('/kaggle/input/playground-series-s5e3/sample_submission.csv')


train_data.head()
train_data.drop('id',axis = 1,inplace = True)


train_data.info()


train_data.head()


def feature_creation(df):
    df['dewpoint_spread'] = df['maxtemp']-df['dewpoint']
    df['pressure_deviation'] = df['pressure']-np.mean(df['pressure'])
    df['humidity'] = df['humidity']-np.mean(df['humidity'])
    df['relative_humidity'] = 100 * (np.exp((17.625 * df['dewpoint']) / (243.04 + df['dewpoint'])) / np.exp((17.625 * df['mintemp']) / (243.04 + df['mintemp'])))
    return df

train_data = feature_creation(train_data)
test_data = feature_creation(test_data)


plt.figure(figsize = (8,8))
sns.heatmap(train_data.corr(method = 'spearman'),annot = True)
plt.show()



vif_data = pd.DataFrame()
features = train_data.iloc[:, :-1]  # Exclude the last column
features = features.drop(['temparature','dewpoint_spread'],axis = 1)
# Compute VIF
vif_data = pd.DataFrame()
vif_data["Feature"] = features.columns  # Only independent variables
vif_data["VIF"] = [variance_inflation_factor(features.values, i) for i in range(features.shape[1])]


vif_data.sort_values(by = ['VIF'],ascending = False)


target = train_data['rainfall']

# Compute p-values for each feature
p_values = {}
for feature in train_data.columns[:-1]:  # Exclude the target column
    _, _, _, p_value, _ = linregress(train_data[feature], target)
    p_values[feature] = p_value

# Convert to DataFrame
p_values_df = pd.DataFrame(list(p_values.items()), columns=['Feature', 'p-value'])
p_values_df['p-value'] = p_values_df['p-value'].astype(float).round(2)
print(p_values_df)


# ,'temparature','dewpoint_spread','winddirection','day'
X = train_data.drop(['rainfall','winddirection'],axis = 1)
y = train_data['rainfall']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3,random_state = 52)

# Best params so far
model = XGBClassifier(eval_metric='auc', use_label_encoder=False,n_estimators = 200,max_depth = 6,gamma = 0.6,subsample = 0.5,
                     reg_alpha=0.9,colsample_bytree=0.5)

# model = XGBClassifier(eval_metric='auc', use_label_encoder=False,n_estimators = 300,max_depth = 6,gamma = 0.6,subsample = 0.5,
#                      reg_alpha=1,colsample_bytree=0.7)

model.fit(X,y)


print(roc_auc_score(y_train,model.predict(X_train)))
roc_auc_score(y_test,model.predict(X_test))


id_1 = test_data.id
test_data.drop('id',axis = 1,inplace = True)
test_data.head()


res = model.predict_proba(test_data.drop(['winddirection'],axis =1))[:,1]
res = pd.DataFrame(res,columns = ['rainfall'])
res = pd.concat([res,id_1],axis = 1)
res = res[['id','rainfall']]
res.to_csv('submission_12.csv',index = False)

