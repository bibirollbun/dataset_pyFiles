import pandas  as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


train = pd.read_csv('/kaggle/input/exploring-predictive-health-factors/train.csv')
test = pd.read_csv('/kaggle/input/exploring-predictive-health-factors/test.csv')


train.shape


train.head()


sns.pairplot(train)


train.nunique()


train.isna().sum()


x_train = train
x_test = test


x_train['PCOS'] = list(map(lambda x: int(x=='Yes'), x_train['PCOS']))
x_train.head()


Age_feach = x_train.groupby('Age', as_index=False).agg({'Weight_kg':'mean','PCOS':'mean'})\
.rename(columns={'Weight_kg':'Age_Weight_kg','PCOS':'Age_PCOS'})
Hormonal_Imbalance_feach = x_train.groupby('Hormonal_Imbalance', as_index=False).agg({'Weight_kg':'mean','PCOS':'mean'})\
.rename(columns={'Weight_kg':'Hormonal_Imbalance_Weight_kg','PCOS':'Hormonal_Imbalance_PCOS'})
Hyperandrogenism_feach = x_train.groupby('Hyperandrogenism', as_index=False).agg({'Weight_kg':'mean','PCOS':'mean'})\
.rename(columns={'Weight_kg':'Hyperandrogenism_Weight_kg','PCOS':'Hyperandrogenism_PCOS'})
Hirsutism_feach = x_train.groupby('Hirsutism', as_index=False).agg({'Weight_kg':'mean','PCOS':'mean'})\
.rename(columns={'Weight_kg':'Hirsutism_Weight_kg','PCOS':'Hirsutism_PCOS'})
Conception_Difficulty_feach = x_train.groupby('Conception_Difficulty', as_index=False).agg({'Weight_kg':'mean','PCOS':'mean'})\
.rename(columns={'Weight_kg':'Conception_Difficulty_Weight_kg','PCOS':'Conception_Difficulty_PCOS'})
Insulin_Resistance_feach = x_train.groupby('Insulin_Resistance', as_index=False).agg({'Weight_kg':'mean','PCOS':'mean'})\
.rename(columns={'Weight_kg':'Insulin_Resistance_Weight_kg','PCOS':'Insulin_Resistance_PCOS'})
Exercise_Frequency_feach = x_train.groupby('Exercise_Frequency', as_index=False).agg({'Weight_kg':'mean','PCOS':'mean'})\
.rename(columns={'Weight_kg':'Exercise_Frequency_Weight_kg','PCOS':'Exercise_Frequency_PCOS'})
Exercise_Type_feach = x_train.groupby('Exercise_Type', as_index=False).agg({'Weight_kg':'mean','PCOS':'mean'})\
.rename(columns={'Weight_kg':'Exercise_Type_Weight_kg','PCOS':'Exercise_Type_PCOS'})
Exercise_Duration_feach = x_train.groupby('Exercise_Duration', as_index=False).agg({'Weight_kg':'mean','PCOS':'mean'})\
.rename(columns={'Weight_kg':'Exercise_Duration_Weight_kg','PCOS':'Exercise_Duration_PCOS'})
Sleep_Hours_feach = x_train.groupby('Sleep_Hours', as_index=False).agg({'Weight_kg':'mean','PCOS':'mean'})\
.rename(columns={'Weight_kg':'Sleep_Hours_Weight_kg','PCOS':'Sleep_Hours_PCOS'})
Exercise_Benefit_feach = x_train.groupby('Exercise_Benefit', as_index=False).agg({'Weight_kg':'mean','PCOS':'mean'})\
.rename(columns={'Weight_kg':'Exercise_Benefit_Weight_kg','PCOS':'Exercise_Benefit_PCOS'})


x_train = pd.merge(x_train, Age_feach, on='Age', how='left')
x_train = pd.merge(x_train, Hormonal_Imbalance_feach, on='Hormonal_Imbalance', how='left')
x_train = pd.merge(x_train, Hyperandrogenism_feach, on='Hyperandrogenism', how='left')
x_train = pd.merge(x_train, Hirsutism_feach, on='Hirsutism', how='left')
x_train = pd.merge(x_train, Conception_Difficulty_feach, on='Conception_Difficulty', how='left')
x_train = pd.merge(x_train, Insulin_Resistance_feach, on='Insulin_Resistance', how='left')
x_train = pd.merge(x_train, Exercise_Frequency_feach, on='Exercise_Frequency', how='left')
x_train = pd.merge(x_train, Exercise_Type_feach, on='Exercise_Type', how='left')
x_train = pd.merge(x_train, Exercise_Duration_feach, on='Exercise_Duration', how='left')
x_train = pd.merge(x_train, Sleep_Hours_feach, on='Sleep_Hours', how='left')
x_train = pd.merge(x_train, Exercise_Benefit_feach, on='Exercise_Benefit', how='left')


x_test = pd.merge(x_test, Age_feach, on='Age', how='left')
x_test = pd.merge(x_test, Hormonal_Imbalance_feach, on='Hormonal_Imbalance', how='left')
x_test = pd.merge(x_test, Hyperandrogenism_feach, on='Hyperandrogenism', how='left')
x_test = pd.merge(x_test, Hirsutism_feach, on='Hirsutism', how='left')
x_test = pd.merge(x_test, Conception_Difficulty_feach, on='Conception_Difficulty', how='left')
x_test = pd.merge(x_test, Insulin_Resistance_feach, on='Insulin_Resistance', how='left')
x_test = pd.merge(x_test, Exercise_Frequency_feach, on='Exercise_Frequency', how='left')
x_test = pd.merge(x_test, Exercise_Type_feach, on='Exercise_Type', how='left')
x_test = pd.merge(x_test, Exercise_Duration_feach, on='Exercise_Duration', how='left')
x_test = pd.merge(x_test, Sleep_Hours_feach, on='Sleep_Hours', how='left')
x_test = pd.merge(x_test, Exercise_Benefit_feach, on='Exercise_Benefit', how='left')


x_train.head()


sns.heatmap(x_train.iloc[:,14:].corr())


x_train.isna().sum()


x_train.Age = x_train.Age.fillna('Nan')
x_train.iloc[:, 4:14] = x_train.iloc[:, 4:14].fillna('Nan')


x_test.Age = x_test.Age.fillna('Nan')
x_test.iloc[:, 3:13]= x_test.iloc[:, 3:13].fillna('Nan')


x_train.Weight_kg = x_train.Weight_kg.fillna(0)
x_train.iloc[:, 14:] = x_train.iloc[:, 14:].fillna(0)


x_test.Weight_kg = x_test.Weight_kg.fillna(0)
x_test.iloc[:, 13:] = x_test.iloc[:, 13:].fillna(0)


x_train.isna().sum()


pip install sdv


from catboost import CatBoostClassifier
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_validate, KFold, ShuffleSplit
from sklearn.metrics import roc_auc_score
from sdv.single_table import GaussianCopulaSynthesizer, CTGANSynthesizer, TVAESynthesizer, CopulaGANSynthesizer
from sdv.metadata import Metadata


train_tr = x_train
test_tr = x_test


train_tr.head()


X_test = test_tr.drop(columns=['ID'])
#X_test = X_test.astype(str)
X_test.Weight_kg = X_test.Weight_kg.astype(float)


X_train = train_tr.drop(columns=['ID','PCOS'])
#X_train = X_train.astype(str)
X_train.Weight_kg = X_train.Weight_kg.astype(float)
Y_train = train.PCOS


data = pd.merge(X_train, Y_train, left_index=True, right_index=True)
data.head()


metadata =  Metadata.detect_from_dataframe(data)


syntizer = CopulaGANSynthesizer(metadata)


syntizer.fit(data)


syntez_data = syntizer.sample(num_rows=4000)


syntez_data.head()


new_data = pd.concat([data, syntez_data], ignore_index=True)


new_data.head()


X_train_syntez = new_data.drop(columns='PCOS')
Y_train_syntez = new_data['PCOS']


cros = KFold(n_splits=21, shuffle=True)


models = []


for i, j in enumerate(cros.split(X_train_syntez, Y_train_syntez)):
    models.append(CatBoostClassifier(iterations=2000, max_depth=8, \
learning_rate=0.005, cat_features=list([*list(X_train_syntez.columns)[2:12],'Age'])\
,logging_level='Silent').fit(X_train_syntez.loc[j[0]], Y_train_syntez.loc[j[0]]))
    try:
        print(roc_auc_score(Y_train_syntez.loc[j[1]],models[i].predict_proba(X_train_syntez.loc[j[1]])[:,1]))
    except Exception:
        print('warn')


predict = np.mean(np.array([models[i].predict_proba(X_train)[:,1] for i in range(len(models))]), axis=0)


roc_auc_score(Y_train, predict)


predict_sub = np.mean(np.array([models[i].predict_proba(X_test)[:,1] for i in range(len(models))]), axis=0)


submission = pd.DataFrame(np.column_stack([test.ID, predict_sub]), columns=['ID', 'PCOS'])
submission.ID = submission.ID.astype(int)


submission.head()


submission.to_csv('submission.csv', index=False)


syntizer.get_learned_distributions()




