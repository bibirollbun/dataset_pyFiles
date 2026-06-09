# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load
import joblib
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


import pandas as pd
import numpy as npno
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn import tree
from sklearn.naive_bayes import MultinomialNB,GaussianNB
from sklearn.metrics import f1_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer
from sklearn.model_selection import GridSearchCV , RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import xgboost as xgb
import copy



train_df=pd.read_csv('/kaggle/input/cmi-bfrb-detection-knn-imputed-dataset/train_knn_imputed_columned.csv')
train_demo_df=pd.read_csv('/kaggle/input/cmi-detect-behavior-with-sensor-data/train_demographics.csv')
tot_train_df=pd.merge(train_df,train_demo_df,on='subject',how='left')
tot_train_df=tot_train_df[(tot_train_df['acc_x']<=30) & (tot_train_df['acc_x']>=-30)]
tot_train_df=tot_train_df[(tot_train_df['acc_y']<=20) & (tot_train_df['acc_y']>=-20)]
tot_train_df=tot_train_df[(tot_train_df['acc_z']<=30) & (tot_train_df['acc_z']>=-30)]


# train_df=pd.read_csv('/kaggle/input/cmi-detect-behavior-with-sensor-data/train.csv')
#train_demo_df=pd.read_csv('/kaggle/input/cmi-detect-behavior-with-sensor-data/train_demographics.csv')
# test_df=pd.read_csv('/kaggle/input/cmi-detect-behavior-with-sensor-data/test.csv')
# test_demo_df=pd.read_csv('/kaggle/input/cmi-detect-behavior-with-sensor-data/test_demographics.csv')


# #clean_df = train_df.dropna()
# first =pd.merge(train_df,train_demo_df,on='subject',how='left')


# tot_train_df=tot_train_df[(tot_train_df['acc_x']<=30) & (tot_train_df['acc_x']>=-30)]
# tot_train_df=tot_train_df[(tot_train_df['acc_y']<=20) & (tot_train_df['acc_y']>=-20)]
# tot_train_df=tot_train_df[(tot_train_df['acc_z']<=30) & (tot_train_df['acc_z']>=-30)]


#train_df = pd.read_csv('/kaggle/input/cmi-bfrb-detection-knn-imputed-dataset/train_knn_imputed_columned.csv')



#tot_train_df=pd.merge(train_df,train_demo_df,on='subject',how='left')


#tot_train_df= tot_train_df.sample(frac=0.3, random_state=2911)


X_data = tot_train_df.drop(columns=['row_id','sequence_type','subject','orientation','behavior','phase','gesture'])
target = tot_train_df["gesture"]



target


encoder_main = LabelEncoder()

scaler = StandardScaler()
scale_cols = ["acc_x","acc_y","acc_z","thm_1","thm_2","thm_3","thm_4","thm_5","norm_acc"]
imputer = KNNImputer(n_neighbors=3,metric="nan_euclidean",weights="distance")



y_data = encoder_main.fit_transform(target)


modelRF = RandomForestClassifier(criterion="log_loss",max_depth = 100,n_estimators=100)


modelXGB = xgb.XGBClassifier(
    tree_method='hist',
    device='cuda',
    learning_rate = 0.25,
    n_estimators = 265,
    max_depth=6,
    reg_lambda = 3,
    reg_alpha = 0.5,
    colsample_bytree=0.7,
    eval_metric = "mlogloss",
    early_stopping_rounds = 10,
    objective='multi:softmax' 
              
)


class dropCols(BaseEstimator,TransformerMixin):
    def fit(self,X_train,y_train):
        return self
    def transform(self,df):
        cols = ["row_id","age","subject","sequence_id"]
        for col in df.columns:
            if col in cols:
                df = df.drop(col,axis=1)
        return df  
class seqID_toStr (BaseEstimator,TransformerMixin):
    def fit(self,x,y):
        return self
    def transform(self,df):
        df = df.copy()
        if ("sequence_id" in df.columns): #& (df["sequence_id"].dtype != int):
            #df["sequence_id"] = df["sequence_id"].str.replace("SEQ_","")
            #df["sequence_id"] = df["sequence_id"].astype("category")
            df = df.drop(["sequence_id"],axis=1)
        return df
class insertCols(BaseEstimator,TransformerMixin):
    def fit(self,x,y):
        return self
    def transform(self,df):
        if "norm_acc" not in df.columns:
             df["norm_acc"] = df["acc_x"]**2 + df["acc_y"]**2 + df["acc_z"]**2
        if "norm_rot" not in df.columns:
            df["norm_rot"] = df["rot_x"]**2 + df["rot_y"]**2 + df["rot_z"]**2 + df["rot_w"]**2
        return df
class impute_df(BaseEstimator,TransformerMixin):
    def fit(self,X,y):
        return self
    def transform(self,df):
        arr = imputer.fit_transform(df)
        df = pd.DataFrame(arr,columns= df.columns)
        return df
class fill_na(BaseEstimator,TransformerMixin):
    def fit(self,X,y):
        return self
    def transform(self,df):
        df.iloc[0] = df.iloc[0].fillna(0)
        df = df.interpolate(method = "linear",axis=1)
        df = df.ffill(axis=1).bfill(axis=1)
        return df
class failsafe_na(BaseEstimator,TransformerMixin):
    def fit(self,X,y):
        return self
    def transform(self,X):
        X = X.fillna(0)
        return X
class debug(BaseEstimator,TransformerMixin):
    def fit(self,X,y):
        return self
    def transform(self,X):
        if "sequence_id" in X.columns:
            print(X.columns)
        return X
            
instance_drop_cols = dropCols()
instance_seqId_toStr = seqID_toStr()
instance_insertCols = insertCols()
instance_impute = impute_df()
instance_fill_na = fill_na()
instance_failsafe_na = failsafe_na()
debugger = debug()



process_pipeline_RF = Pipeline([("customColumnsDrop",instance_drop_cols),
                                ("seqIdToStr",instance_seqId_toStr),
                                ("failsafeForNan",instance_failsafe_na),
                                ("insertCols", instance_insertCols),
                                ("RandomForest",modelRF) 
                               ])




pipeline_validation = Pipeline([("customColumnsDrop",instance_drop_cols),
                                ("seqIdToStr",instance_seqId_toStr),
                                ("failsafeForNan",instance_failsafe_na),
                                ("insertCols", instance_insertCols),
                               ])


X_temp, X_test, y_temp, y_test = train_test_split(
    X_data, y_data, test_size=0.2, random_state=42
)


X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.2, random_state=42
)


X_train = pipeline_validation.fit_transform(X_train,y_train)
X_val = pipeline_validation.transform(X_val)



modelXGB.fit(X_train,y_train,eval_set=[(X_val, y_val)],verbose=True)


print("Best iteration:", modelXGB.best_iteration)


X_test = pipeline_validation.transform(X_test)
y_pred = modelXGB.predict(X_test)
score = f1_score(y_pred,y_test,average = "macro")
score


# process_pipeline_DT.fit(X_train,y_train)


# y_pred = process_pipeline_DT.predict(X_test)
# score = f1_score(y_pred,y_test,average = "macro")
# score


joblib.dump(encoder_main,"encoder_main.pkl")
joblib.dump(pipeline_validation,"pipeline_validation.pkl")
joblib.dump(modelXGB,"modelXGB.pkl")





