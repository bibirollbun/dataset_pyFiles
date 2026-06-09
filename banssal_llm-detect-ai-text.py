
import numpy as np 
import pandas as pd 
import warnings
warnings.filterwarnings("ignore")
import os
pd.plotting.register_matplotlib_converters()
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
sns.set_style("dark") # Theme for plots as Dark
# sns.set_palette("viridis")
sns.color_palette("flare")
import tensorflow as tf
from catboost import CatBoostClassifier, Pool
from xgboost import XGBClassifier
from xgboost.callback import EarlyStopping
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, cross_validate, StratifiedKFold, RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
import optuna
import imblearn
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from catboost import Pool, CatBoostClassifier, cv


# Training and test datasets
train_essays1 = pd.read_csv("/kaggle/input/llm-detect-ai-generated-text/train_essays.csv")
train_essays2 = pd.read_csv("/kaggle/input/llm-generated-essays/ai_generated_train_essays.csv")
train_essays3 = pd.read_csv("/kaggle/input/daigt-external-dataset/daigt_external_dataset.csv")
test_essays = pd.read_csv("/kaggle/input/llm-detect-ai-generated-text/test_essays.csv")



test_essays.drop(["prompt_id"],axis=1,inplace=True)
train_essays1.drop(["prompt_id"],axis=1,inplace=True)
train_essays2.drop(["prompt_id"],axis=1,inplace=True)
train_essays4 = train_essays3[["id","source_text"]]
train_essays4["generated"] = 1
train_essays4.columns = ["id","text","generated"]
train_essays3 = train_essays3[["id","text"]]
train_essays3["generated"] = 0

train_essays = pd.concat([train_essays1,train_essays2,train_essays3,train_essays4])
train_essays.reset_index(drop=True,inplace=True)
train_essays.head(10)


# Parameters
VOCAB_SIZE = 10000
MAXLEN = 500

train_essays["oneHot"] = [tf.keras.preprocessing.text.one_hot(i,VOCAB_SIZE) for i in train_essays["text"]]
df = pd.DataFrame(tf.keras.utils.pad_sequences(train_essays["oneHot"],padding="pre",maxlen=MAXLEN))
train_essays = pd.concat((train_essays,df),axis=1)
train_essays.drop(["text","oneHot"],inplace=True,axis=1)
train_essays.set_index("id",inplace=True)
train_essays.head(10)


test_essays["oneHot"] = [tf.keras.preprocessing.text.one_hot(i,VOCAB_SIZE) for i in test_essays["text"]]
df = pd.DataFrame(tf.keras.utils.pad_sequences(test_essays["oneHot"],padding="pre",maxlen=MAXLEN))
test_essays = pd.concat((test_essays,df),axis=1)
test_essays.drop(["text","oneHot"],inplace=True,axis=1)
test_essays.set_index("id",inplace=True)
test_essays.head(10)


SEED = 6
X = train_essays.drop('generated', axis=1).values
y = train_essays['generated'].values

# LightGBM
lgbm = LGBMClassifier(random_state=SEED)
print('LGBM CV AUC:', cross_val_score(lgbm, X, y, cv=4, scoring='roc_auc').mean())

# XGBoost
xgb = XGBClassifier(random_state=SEED)
print('XGB CV AUC:', cross_val_score(xgb, X, y, cv=4, scoring='roc_auc').mean())


# Fit on full train, predict on test
lgbm.fit(X, y)
xgb.fit(X, y)
preds = (lgbm.predict_proba(test_essays)[:,1] + xgb.predict_proba(test_essays)[:,1]) / 2

submission = pd.DataFrame({'id': test_essays.index, 'generated': preds})
submission.to_csv('submission.csv', index=False)

