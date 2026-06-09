import os
import joblib
import pandas as pd
import polars as pl


import kaggle_evaluation.cmi_inference_server


import pandas as pd
import numpy as np
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
import xgboost as xgb
import copy



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
            



pipeline_validation = joblib.load("/kaggle/input/sarveshdt-cmi/pipeline_validation.pkl")
encoder_main = joblib.load("/kaggle/input/sarveshdt-cmi/encoder_main.pkl")
modelXGB = joblib.load("/kaggle/input/sarveshdt-cmi/modelXGB.pkl")




from tqdm import tqdm
def fillna_test(df):
    N_NEIGHBORS = 3
    sensor_cols = [col for col in df.columns if col.startswith(("acc", "rot", "thm", "tof"))]
    imputed_chunks = []
    for seq_id, group in tqdm(df.groupby("sequence_id")):
        sensor_data = group[sensor_cols]
        if sensor_data.isnull().all().any():
            imputed_group=group.copy()
            imputed_group[sensor_cols] = sensor_data.fillna(0)

        elif len(group) < N_NEIGHBORS :
            imputed_group = group.copy()
            imputed_group[sensor_cols] = sensor_data.fillna(sensor_data.mode())
        else:
            imputer = KNNImputer(n_neighbors=N_NEIGHBORS, weights="distance")
            imputed_vals = imputer.fit_transform(sensor_data)
            imputed_group = group.copy()
            imputed_group[sensor_cols] = imputed_vals

        imputed_chunks.append(imputed_group)
    df_imputed = pd.concat(imputed_chunks).sort_values(["sequence_id", "sequence_counter"])
    return df_imputed



def predict(sequence: pl.DataFrame, demographics: pl.DataFrame):
    sequence = sequence.to_pandas()
    demographics = demographics.to_pandas()

    X_test=pd.merge(sequence,demographics,on='subject',how='left')
    X_test = fillna_test(X_test)
    X_test = X_test.fillna(0)
    X_test = pipeline_validation.transform(X_test)
    preds = encoder_main.inverse_transform(modelXGB.predict(X_test))
    mode_pred = pd.Series(preds).mode()
   
    ans = mode_pred.iloc[0]
    return str(ans)


from kaggle_evaluation.cmi_inference_server import CMIInferenceServer

# Wrap your predict function
inference_server = CMIInferenceServer(predict)

# Run local test or real submission
if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    inference_server.serve()  # Runs on real submission
else:
    inference_server.run_local_gateway(
        data_paths=(
            "/kaggle/input/cmi-detect-behavior-with-sensor-data/test.csv",
            "/kaggle/input/cmi-detect-behavior-with-sensor-data/test_demographics.csv",
        )
    )




