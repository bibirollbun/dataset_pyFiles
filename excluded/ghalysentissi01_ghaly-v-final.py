import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


%%time
try:
    from lifelines.utils import concordance_index
except ModuleNotFoundError:
    print('Installing lifelines...')
    !pip install -q /kaggle/input/pip-install-lifelines/autograd-1.7.0-py3-none-any.whl
    !pip install -q /kaggle/input/pip-install-lifelines/autograd-gamma-0.5.0.tar.gz
    !pip install -q /kaggle/input/pip-install-lifelines/interface_meta-1.3.0-py3-none-any.whl
    !pip install -q /kaggle/input/pip-install-lifelines/formulaic-1.0.2-py3-none-any.whl
    !pip install -q /kaggle/input/pip-install-lifelines/lifelines-0.30.0-py3-none-any.whl


# !pip install lifelines
!pip install xgboost
# !pip install scikit-survival



from itertools import combinations
from random import choice
from typing import Any, Callable, Iterable

import matplotlib.pyplot as plt
import numpy as np
import optuna
from catboost import CatBoostClassifier, CatBoostRegressor
from lifelines import (
    CoxPHFitter,
    KaplanMeierFitter,
    LogLogisticAFTFitter,
    LogNormalAFTFitter,
    WeibullAFTFitter,
)
from lifelines.utils import concordance_index
from lightgbm import LGBMClassifier, LGBMRegressor
from numpy import array, concatenate, mean, median, nan, ndarray, sqrt, var
from pandas import DataFrame, Series, crosstab, get_dummies, isna, options, read_csv
from seaborn import FacetGrid, histplot, kdeplot
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
# from sksurv.ensemble import RandomSurvivalForest
from xgboost import XGBClassifier, XGBRegressor
import pandas as pd


options.mode.chained_assignment = None
IN_DRIVE = False
if IN_DRIVE:
  from google.colab import drive
  drive.mount('/content/drive')
  filepath      = '/content/drive/MyDrive/Ue_Machine_learning/'
else:
  filepath = '"/kaggle/input/equity-post-HCT-survival-predictions'


train_data = read_csv('/kaggle/input/equity-post-HCT-survival-predictions' + '/train.csv')
test_data  = read_csv('/kaggle/input/equity-post-HCT-survival-predictions' + '/test.csv')


print(f'Our dataset has {train_data.shape[0]} rows and {train_data.shape[1]} columns')
print(f'Our test dataset has {test_data.shape[0]} rows and {test_data.shape[1]} columns')
train_data.head()


missing_values = train_data.isnull().sum()
missing_values = missing_values[missing_values>0].sort_values(ascending=False)
print(missing_values)


efs0 = train_data[train_data['efs'] == 0]
efs1 = train_data[train_data['efs'] == 1]


kdeplot(train_data['efs_time'])


plt.figure(figsize=(7,5))
kdeplot(efs0['efs_time'], label='Échec (efs=0)', shade=True)
kdeplot(efs1['efs_time'], label='Succès (efs=1)', shade=True)
plt.xlabel('efs_time')
plt.ylabel('Density')
plt.legend()
plt.title('Distribution de efs_time selon efs')
plt.show()



train_data.groupby('efs')['efs_time'].describe()



plt.figure(figsize=(10,5))
kdeplot(train_data['donor_age'].dropna(), label='Âge du donneur', shade=True)
kdeplot(train_data['age_at_hct'].dropna(), label='Âge du receveur', shade=True)
plt.xlabel('Âge')
plt.ylabel('Densité')
plt.legend()
plt.title("Distribution de l'âge du donneur et du receveur")
plt.show()


# Calculer le nombre d'occurrences par groupe dans la colonne 'race_group'
group_counts = train_data['race_group'].value_counts()

plt.figure(figsize=(8, 8))
plt.pie(group_counts, labels=group_counts.index, autopct='%1.1f%%', startangle=90)
plt.title("Distribution des groupes raciaux")
plt.axis('equal')
plt.show()



X_train = train_data[[col for col in train_data.columns if col != "ID"]]
X_test  = test_data[[col for col in test_data.columns if col != "ID"]]


def remove_nan_lines(X_in: DataFrame):
    nan_counts = X_in.isnull().sum(axis=1)
    X_in = X_in[nan_counts <= 25]
    return X_in



pd.crosstab(X_train['hla_high_res_8'],X_train['hla_low_res_8'],dropna=False)


hla_high_columns = [
    "hla_match_c_high",
    "hla_high_res_8",
    "hla_high_res_6",
    "hla_high_res_10",
    "hla_match_dqb1_high",
    "hla_match_drb1_high",
    "hla_match_a_high",
    "hla_match_b_high",
]
hla_low_columns = [hla.replace('high','low') for hla in hla_high_columns]
hla_columns = [
    "hla_nmdp_6",
    *hla_high_columns,
    *hla_low_columns,
]
def fill_hla_na(X_in: DataFrame) -> DataFrame:
    for i in range(len(hla_low_columns)) :
        low = hla_low_columns[i]
        high = hla_high_columns[i]
        X_in[low]  = X_in[low].fillna(X_in[high])
        X_in[high] = X_in[high].fillna(X_in[low])
    return X_in


def fillna_two_hla(X_in: DataFrame, col_1: str, col_2: str) -> DataFrame:
  cross_tab = crosstab(X_in[col_1], X_in[col_2], dropna=False)

  most_frequent_value_col_1 = cross_tab.idxmax(axis=1)
  most_frequent_value_col_2 = cross_tab.idxmax(axis=0)

  X_in[col_1] = X_in[col_1].fillna(X_in[col_2].map(most_frequent_value_col_2))
  X_in[col_2] = X_in[col_2].fillna(X_in[col_1].map(most_frequent_value_col_1))
  return X_in

def fillna_all_combinations(X_in: DataFrame) -> DataFrame:
    X_in = fill_hla_na(X_in)  # Appliquer fillna_hla d'abord

    # Générer toutes les combinaisons possibles de colonnes HLA (prises 2 par 2)
    for col_1, col_2 in combinations(hla_columns, 2):
        X_in = fillna_two_hla(X_in, col_1, col_2)

    return X_in


mapping_tce_match = {
    'Permissive mismatched'         : 'Permissive',
    'Permissive'                    : 'Permissive',
    'Bi-directional non-permissive' : 'Non-permissive', # GvH and HvG
    'Fully matched'                 : 'Fully matched',
    'GvH non-permissive'            : 'GvH non-permissive',
    'HvG non-permissive'            : 'HvG non-permissive'
}
rename_tce_cols = ['tce_div_match', 'tce_match']

def rename_tce_match(X_in: DataFrame) -> DataFrame:
    for col in rename_tce_cols:
        X_in[col] = X_in[col].cat.rename_categories(mapping_tce_match)
    return X_in


mapping_fillna_tce_match = {
    'Permissive'         : 'Permissive',
    'Non-permissive'     :  choice(['GvH non-permissive', 'HvG non-permissive']),
    'Fully matched'      : 'Permissive',
    'GvH non-permissive' : 'GvH non-permissive',
    'HvG non-permissive' : 'HvG non-permissive',
}

def fillna_tce_match(X_in: DataFrame) -> DataFrame:
    for i in range(len(X_in)):
      tce_match     = X_in.loc[i,'tce_match']
      tce_div_match = X_in.loc[i,'tce_div_match']

      if isna(tce_match) and isna(tce_div_match):
        continue

      if isna(tce_match):
        if tce_div_match in mapping_fillna_tce_match:
          X_in.loc[i,'tce_match'] = mapping_fillna_tce_match[tce_div_match]

      if isna(tce_div_match):
        if tce_match in mapping_fillna_tce_match:
          X_in.loc[i, 'tce_div_match'] = mapping_fillna_tce_match[tce_match]

    return X_in


mapping_cyto_score = {
    'Poor'         : 'Poor',
    'Intermediate' : 'Intermediate',
    'Normal'       : 'Favorable',
    'Other'        : 'Intermediate',
    'Favorable'    : 'Favorable',
    'TBD'          : 'TBD',
    'Not tested'   : 'Not tested',
}

def fillna_cyto_score(X_in: DataFrame):
  X_in['cyto_score'] = X_in['cyto_score'].map(mapping_cyto_score)
  X_in['cyto_score'] = X_in['cyto_score'].fillna(X_in['cyto_score_detail'])
  X_in['cyto_score_detail'] = X_in['cyto_score_detail'].fillna(X_in['cyto_score'])
  return X_in


def reduce_year(X_in: DataFrame) -> DataFrame:
    X_in['year_hct']=X_in['year_hct'].replace(2020,2019) # only 4 rows
    X_in['year_hct'] = X_in['year_hct'] - 2000
    return X_in


def fill_karnofsky_score(X_in: DataFrame):
    X_in['karnofsky_score'] = X_in['karnofsky_score'].fillna(90)
    return X_in


def fillna_not_done(X_in: DataFrame):
    notDoneList = ["psych_disturb","diabetes","arrhythmia","renal_issue","pulm_severe","obesity","hepatic_severe","prior_tumor","peptic_ulcer","rheum_issue","hepatic_mild","cardiac","pulm_moderate"]
    for col in notDoneList:
        if 'Not done' not in X_in[col].cat.categories:
          X_in[col] = X_in[col].cat.add_categories('Not done')
        X_in[col] = X_in[col].fillna('Not done')
    return X_in


def add_new_feat(X_in: DataFrame): ##not added yet, to test one by one
    X_in['donor_age-age_at_hct']=X_in['donor_age']-X_in['age_at_hct']
    X_in['comorbidity_score+karnofsky_score']=X_in['comorbidity_score']+X_in['karnofsky_score']
    X_in['comorbidity_score-karnofsky_score']=X_in['comorbidity_score']-X_in['karnofsky_score']
    X_in['comorbidity_score*karnofsky_score']=X_in['comorbidity_score']*X_in['karnofsky_score']
    X_in['comorbidity_score/karnofsky_score']=X_in['comorbidity_score']/X_in['karnofsky_score']
    X_in['is_cyto_score_same'] = (X_in['cyto_score'] == X_in['cyto_score_detail']).astype(int)
    return X_in


X_train = train_data[[col for col in train_data.columns if col != "ID"]]
X_test  = test_data[[col for col in test_data.columns if col != "ID"]]

categorical_columns    = list(X_train.select_dtypes(object).columns)
X_train[categorical_columns] = X_train[categorical_columns].astype(str).astype('category')

categorical_columns = list(X_train.select_dtypes('category').columns)
numerical_columns   = list(set(X_train.columns) - set(categorical_columns) - set(["efs", "efs_time"]))

X_train[categorical_columns] = X_train[categorical_columns].astype(str).astype('category')
X_test[categorical_columns] = X_test[categorical_columns].astype(str).astype('category')
print(f'We have {len(categorical_columns)} categorical_columns')
print(f'We have {len(numerical_columns)}   numerical_columns')

races = list(X_train["race_group"].unique())
print(f'All races are : \n {races}')



def apply_feature_engineering (X: DataFrame) -> DataFrame:
  X = remove_nan_lines(X)
  X = fillna_all_combinations(X)
  # X = fillna_cyto_score(X)
  # X = fill_karnofsky_score(X)
  X = rename_tce_match(X)
  X = fillna_tce_match(X)
  X = fillna_not_done(X)
  X = reduce_year(X)
  # X = add_new_feat(X)

  return X

X_train_feat_added = apply_feature_engineering(X_train)
X_test_feat_added  = apply_feature_engineering(X_test)


def get_ohe(X_in: DataFrame, categories: list[str]) -> DataFrame:
    X_ohe = get_dummies(X_in, columns=categories, drop_first=False)
    invalid_col_names = [f for f in X_ohe.columns if "<" in f]
    return X_ohe.rename(columns={f: f.replace("<", "_inf_") for f in invalid_col_names})




X_train_ohe = get_ohe(X_train_feat_added.drop(columns=["efs", "efs_time"]), categorical_columns)
X_test_ohe  = get_ohe(X_test_feat_added, categorical_columns)

#we could have missing ohe columns in test
missing_cols = list(set(X_train_ohe.columns) - set(X_test_ohe.columns)-set(['race_group']))
# Add missing columns efficiently
X_test_ohe = pd.concat([X_test_ohe, pd.DataFrame(0, index=X_test_ohe.index, columns=missing_cols)], axis=1)
X_test_ohe = X_test_ohe[X_train_ohe.columns] # same order

non_numerical_columns = [col for col in X_train_ohe.columns if col not in numerical_columns]

imputer = SimpleImputer(strategy='median')
X_train_ohe_filled = imputer.fit_transform(X_train_ohe)
X_train_ohe_filled = pd.DataFrame(X_train_ohe_filled, columns=X_train_ohe.columns)

scaler = StandardScaler()
X_train_scaled = X_train_ohe_filled.copy()
X_train_scaled[numerical_columns] = scaler.fit_transform(X_train_ohe_filled[numerical_columns])
# X_train_scaled_num = pd.DataFrame(X_train_scaled_num, columns=numerical_columns, index=X_train_ohe.index)
# X_train_non_scaled = X_train_ohe_filled[non_numerical_columns]
# X_train_scaled = pd.concat([X_train_scaled_num, X_train_non_scaled], axis=1)

X_test_ohe_filled = imputer.transform(X_test_ohe)
X_test_ohe_filled = pd.DataFrame(X_test_ohe_filled, columns=X_test_ohe.columns)


X_test_scaled = X_test_ohe_filled.copy()
X_test_scaled[numerical_columns] = scaler.transform(X_test_ohe_filled[numerical_columns])
# X_test_scaled_num = pd.DataFrame(X_test_scaled_num, columns=numerical_columns, index=X_test_ohe.index)
# X_test_non_scaled = X_test_ohe_filled[non_numerical_columns]
# X_test_scaled = pd.concat([X_test_scaled_num, X_test_non_scaled], axis=1)



columns_to_add = ["race_group", "efs", "efs_time"]
for col in columns_to_add:
    X_train_scaled[col] = train_data[col]

X_test_scaled['race_group'] = test_data['race_group'].astype('category')


s = X_train_scaled['efs_time']

X_train_scaled['efs_time'] = (s - s.min()) / (s.max() - s.min())

X_train = X_train_scaled.copy()


X_train = X_train_scaled.copy()
X_test  = X_test_scaled.copy()


class GenericModel:

    def __init__(self, model: Any) -> None:
        self.model = model

    def fit_model(self, *args, **kwargs) -> None:
        raise NotImplementedError

    def get_risk_scores(self, data: DataFrame) -> Iterable[float]:
        raise NotImplementedError

    def __getattr__(self, attr_name: str) -> Any:
        return getattr(self.model, attr_name)


def get_c_index_by_race(
        x_in: DataFrame,
        model: GenericModel,
    ) -> float:
    race_c_indexes = []
    x_feats = [f for f in x_in.columns if f not in ("race_group")]
    for race in races:

        df_race = x_in[x_in["race_group"] == race]

        c_index_race = concordance_index(
                        df_race['efs_time'],
                        -model.get_risk_scores(df_race[x_feats]),
                        df_race['efs'])

        race_c_indexes.append(c_index_race)

    return float(mean(race_c_indexes)-sqrt(var(race_c_indexes)))

def cross_validate(train_df: DataFrame, model: GenericModel, k: int = 5) -> float:
    
    stratify_col = train_df["race_group"]

    kf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

    kfold_c_index_scores = []
    for train_index, val_index in kf.split(train_df, stratify_col):
        X_train_fold, X_val_fold = train_df.iloc[train_index], train_df.iloc[val_index]

        model.fit_model(X_train_fold)
        c_index = get_c_index_by_race(X_val_fold, model)

        kfold_c_index_scores.append(c_index)

    return mean(kfold_c_index_scores)





class CoxPHFitter_HCT(GenericModel):

    def __init__(self, model: CoxPHFitter) -> None:
        self.model = model

    def fit_model(self, data: DataFrame):
        # Cette colonne cause des problèmes
        feats = [f for f in data.columns if f not in ['race_group', 'gvhd_proph_FK+- others(not MMF,MTX)']]
        self.model.fit(data[feats], duration_col="efs_time", event_col="efs")

    def get_risk_scores(self, data: DataFrame) -> Series:
        return self.model.predict_partial_hazard(data)
    



# cph = CoxPHFitter_HCT(
#     CoxPHFitter(penalizer=0.1)
# )

# cross_validate(
#     X_train,
#     cph,
# )


# from lifelines import  NelsonAalenFitter
# class XGBRegressor_HCT(GenericModel):

#     def __init__(self, model: XGBRegressor) -> None:
#         self.model = model

#     def fit_model(self, data: DataFrame):
#         feats = [f for f in data.columns if f not in ['race_group', 'gvhd_proph_FK+- others(not MMF,MTX)', "efs", "efs_time"]]
#         y_nel = self.create_nelson(data)
#         self.model.fit(data[feats], y_nel, sample_weight=data["efs"])
        

#     def create_nelson(self,data):
#         data=data.copy()
#         naf = NelsonAalenFitter(nelson_aalen_smoothing=0)
#         naf.fit(durations=data['efs_time'], event_observed=data['efs'])
#         return naf.cumulative_hazard_at_times(data['efs_time']).values*-1
        
#     def get_risk_scores(self, data: DataFrame) -> Series:
#         feats = [f for f in data.columns if f not in ['race_group', 'gvhd_proph_FK+- others(not MMF,MTX)', "efs", "efs_time"]]
#         return self.model.predict(data[feats])



# # xbgbooster_params = {
# #     "objective": "survival:cox",
# #     "eval_metric": "cox-nloglik",
# #     "eta": 0.1,
# #     "max_depth": 3,
# #     "subsample": 0.8,
# #     "colsample_bytree": 0.8,
# #     'enable_categorical': True
# # }
# xbgbooster_params = {
#         "max_depth":4,  
#         "colsample_bytree":0.55,  
#         "subsample":0.8,  
#         "learning_rate":0.02,  
#         "enable_categorical":True,
#         "min_child_weight":80,
#         # "early_stopping_rounds":200,
#         "n_jobs":4,
# }
# xgb_regressor = XGBRegressor_HCT(
#     XGBRegressor(**xbgbooster_params)
# )
# cross_validate(
#     X_train,
#     xgb_regressor,
# )


# from lifelines import  NelsonAalenFitter
# def create_nelson(data):
#     data=data.copy()
#     naf = NelsonAalenFitter(nelson_aalen_smoothing=0)
#     naf.fit(durations=data['efs_time'], event_observed=data['efs'])
#     return naf.cumulative_hazard_at_times(data['efs_time']).values*-1

# def create_stratified_folds(data, target, n_splits=10):
#     data['fold'] = -1
#     # num_bins = int(np.floor(1 + np.log2(len(data))))  # Sturges' rule for binning
#     if (target!="race_group"):
#         data['bins'] = pd.qcut(data[target], q=50, duplicates='drop',labels=False)
#     data["bins"]=data["race_group"]
#     skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
#     for fold, (_, val_idx) in enumerate(skf.split(data, data['bins'])):
#         data.loc[val_idx, 'fold'] = fold
    
#     data = data.drop(columns=['bins'])
#     return data

# def xgb_training(X_train,X_test):
#     train=X_train
#     test=X_test
    
#     train["y_nel"] = create_nelson(train)
#     train.loc[train.efs == 0, "y_nel"] = (-(-train.loc[train.efs == 0, "y_nel"])**0.5)
#     FOLDS = 10
#     train=create_stratified_folds(train,"race_group",FOLDS)
#     train.columns = train.columns.astype(str)
#     test.columns = test.columns.astype(str)
#     train.columns = train.columns.str.replace(r"[<>\[\]]", "_", regex=True)
#     test.columns = test.columns.str.replace(r"[<>\[\]]", "_", regex=True)
    
#     RMV = ["ID","efs","efs_time","y","efs_time2",'y_nel', 'fold']
#     FEATURES = [c for c in train.columns if not c in RMV]
    
#     oof_xgb1 = np.zeros(len(train))
#     pred_xgb1 = np.zeros(len(test))
    
#     for i in range(FOLDS):
    
#         print("#"*25)
#         print(f"### Fold {i+1}")
#         print("#"*25)
        
#         x_train = train.loc[train.fold!=i,FEATURES].copy()
#         y_train = train.loc[train.fold!=i,"y_nel"]
#         x_valid = train.loc[train.fold==i,FEATURES].copy()
#         y_valid = train.loc[train.fold==i,"y_nel"]
    
#         x_train['race_group'] = x_train['race_group'].astype('category')
#         x_valid['race_group'] = x_valid['race_group'].astype('category')
#         x_test = test[FEATURES].copy()
    
#         model_xgb = XGBRegressor(
#             # device="cuda",
#             max_depth=4,  
#             colsample_bytree=0.55,  
#             subsample=0.8,  
#             n_estimators=5000,  
#             learning_rate=0.02,  
#             enable_categorical=True,
#             min_child_weight=80,
#             early_stopping_rounds=200,
#             n_jobs=4
#         )
#         model_xgb.fit(
#             x_train, y_train,
#             eval_set=[(x_valid, y_valid)],  
#             verbose=500 
#         )
    
#         # INFER OOF
#         oof_xgb1[train.index[train.fold==i]] = (model_xgb.predict(x_valid))
#         # INFER TEST
#         pred_xgb1 += (model_xgb.predict(x_test))

#         return pred_xgb1
    


# pred_xgb1 = xgb_training(X_train,X_test)



class AFTFitter_HCT(GenericModel):

    def __init__(self, model: WeibullAFTFitter | LogNormalAFTFitter | LogLogisticAFTFitter) -> None:
        self.model = model

    def fit_model(self, data: DataFrame):
        x_features = [f for f in data.columns if f != "race_group"]
        self.model.fit(data[x_features], duration_col="efs_time", event_col="efs")

    def get_risk_scores(self, data: DataFrame) -> Series:
        return self.model.predict_median(data)




X_train_aft = X_train.copy()
X_train_aft['efs_time'] = np.where(
    X_train_aft['efs_time'] == 0,
    0.00001,
    X_train_aft['efs_time']
)


# waft = AFTFitter_HCT(WeibullAFTFitter())


# cross_validate(
#     X_train_aft,
#     waft,
# )

# risk_scores = waft.get_risk_scores(X_test)


# lnaft = AFTFitter_HCT(LogNormalAFTFitter())

# cross_validate(
#     X_train_aft,
#     lnaft,
# )

# risk_scores = lnaft.get_risk_scores(X_test)


llaft = AFTFitter_HCT(LogLogisticAFTFitter())

cross_validate(
    X_train_aft,
    llaft,
)

risk_scores = llaft.get_risk_scores(X_test)


# class RandomSurvivalForest_HCT(GenericModel):

#     _exlude_cols = ("efs", "efs_time", "race_group")

#     def __init__(self, model: RandomSurvivalForest) -> None:
#         self.model = model
    
#     def fit_model(self, data: DataFrame):
#         y = array(
#             [(efs == 1, efs_time) for efs, efs_time in zip(data["efs"], data["efs_time"])],
#             dtype=[("event", bool), ("time", float)],
#         )
#         feats = self.get_x_feats(data.columns)
#         self.model.fit(data[feats], y)

#     def get_risk_scores(self, data: DataFrame) -> Series:
#         feats = self.get_x_feats(data.columns)
#         batch_size = 500
#         return concatenate([
#             self.model.predict(data[feats][i:i+batch_size]) for i in range(0, len(data), batch_size)
#         ])
    
#     def get_x_feats(self, columns: list[str]) -> list[str]:
#         return [f for f in columns if f not in self._exlude_cols]


# rsf_params = {
#     "n_estimators": 50,
#     "max_features": "sqrt",
#     "min_samples_split": 10,
#     "min_samples_leaf": 5,
#     "n_jobs": -1,
#     "random_state": 42,
# }

# rsf = RandomSurvivalForest_HCT(
#     RandomSurvivalForest(**rsf_params),
# )

# # 10 mins
# cross_validate(
#     X_train,
#     rsf
# )


class EnsembleHCT(GenericModel):

    def __init__(self, regressor: Any, classifier: Any) -> None:
        self.regressor = regressor
        self.classifier = classifier
        self.merger_function = None

    def fit_model(self, data: DataFrame) -> None:
        x_cols = [f for f in data.columns if f not in ('efs_time', 'efs', "race_group")]
        self.regressor.fit(data[x_cols], data['efs_time'])
        self.classifier.fit(data[x_cols], data['efs'])

    def get_risk_scores(self, data: DataFrame) -> Iterable[float]:
        x_cols = [f for f in data.columns if f not in ('efs_time', 'efs', "race_group")]
        self.reg_pred = self.regressor.predict(data[x_cols])
        self.cls_pred = self.classifier.predict_proba(data[x_cols])[:, 1]

        if self.merger_function is None:
            raise ValueError
        
        return self.merger_function(self.reg_pred, self.cls_pred)
    
    def set_merger_function(self, merger_function: Callable) -> None:
        self.merger_function = merger_function

    def __getattr__(self, attr_name: str) -> Any:
        return getattr(self.model, attr_name)
    


def get_rank(y_rsq: Series) -> Series:
    return Series(y_rsq).rank() / len(y_rsq)



def linear_model_merge(reg_pred: Series, cls_pred: Series, w1: float, w2: float) -> Series:
    return get_rank(w1 * reg_pred + w2 * cls_pred)

def linear_objective(trial, train_df, model):
    w1 = trial.suggest_float("w1", 0.1,5.0)
    w2 = trial.suggest_float("w2", 0.1, 5.0)

    model.set_merger_function(lambda x, y: linear_model_merge(x, y, w1, w2))
    
    return cross_validate(train_df, model)


# Pas de constant, pas de fonction objective.
def prod_merge(reg_pred, cls_pred):
    return get_rank(cls_pred + reg_pred)



def complex_merge(reg_pred, cls_pred, a, b, c):
    reg_scaled = a * np.log(1 + np.abs(reg_pred))
    cls_scaled = b * np.sqrt(np.abs(cls_pred))
    interaction_term = c * reg_scaled * cls_scaled
    return get_rank(reg_scaled + cls_scaled + interaction_term)

def complex_objective(trial, train_df, model):
    a = trial.suggest_float("a", 0.1, 5.0)
    b = trial.suggest_float("b", 0.1, 5.0)
    c = trial.suggest_float("c", 0.1, 5.0)

    model.set_merger_function(lambda x, y: complex_merge(x, y, a, b, c))

    return cross_validate(train_df, model)




xgb_ensemble_model = EnsembleHCT(
    XGBRegressor(objective="reg:squarederror", n_estimators=200),
    XGBClassifier(objective="binary:logistic", n_estimators=200),
)


# study = optuna.create_study(direction="maximize")
# study.optimize(lambda trial: linear_objective(trial, X_train, xgb_ensemble_model), n_trials=10)


# w1, w2 = study.best_params["w1"], study.best_params["w2"] 
# xgb_ensemble_model.set_merger_function(lambda x, y: linear_model_merge(x, y, w1, w2))
# cross_validate(X_train, xgb_ensemble_model)

# risk_scores = xgb_ensemble_model.get_risk_scores(X_test)


# xgb_ensemble_model.set_merger_function(prod_merge)
# cross_validate(X_train, xgb_ensemble_model)

# risk_scores = xgb_ensemble_model.get_risk_scores(X_test)


# study = optuna.create_study(direction="maximize")
# study.optimize(lambda trial: complex_objective(trial, X_train, xgb_ensemble_model), n_trials=10)

# a, b, c = study.best_params["a"], study.best_params["b"], study.best_params["c"] 
# xgb_ensemble_model.set_merger_function(lambda x, y: complex_merge(x, y, a, b, c))
# cross_validate(X_train, xgb_ensemble_model)

# risk_scores = xgb_ensemble_model.get_risk_scores(X_test)


catboost_ensemble_model = EnsembleHCT(
    CatBoostRegressor(iterations=200, depth=6, learning_rate=0.1, loss_function="RMSE"),
    CatBoostClassifier(iterations=200, depth=6, learning_rate=0.1, loss_function="Logloss"),
)


# study = optuna.create_study(direction="maximize")
# study.optimize(lambda trial: linear_objective(trial, X_train, catboost_ensemble_model), n_trials=10)

# w1, w2 = study.best_params["w1"], study.best_params["w2"] 
# catboost_ensemble_model.set_merger_function(lambda x, y: linear_model_merge(x, y, w1, w2))
# cross_validate(X_train, catboost_ensemble_model)

# risk_scores = catboost_ensemble_model.get_risk_scores(X_test)


# catboost_ensemble_model.set_merger_function(prod_merge)
# cross_validate(X_train, catboost_ensemble_model)

# risk_scores = catboost_ensemble_model.get_risk_scores(X_test)


# study = optuna.create_study(direction="maximize")
# study.optimize(lambda trial: complex_objective(trial, X_train, catboost_ensemble_model), n_trials=10)

# a, b, c = study.best_params["a"], study.best_params["b"], study.best_params["c"] 
# catboost_ensemble_model.set_merger_function(lambda x, y: complex_merge(x, y, a, b, c))
# cross_validate(X_train, catboost_ensemble_model)

# risk_scores = catboost_ensemble_model.get_risk_scores(X_test)


lgbm_ensemble_model = EnsembleHCT(
    LGBMRegressor(),
    LGBMClassifier(),
)

lgbm_rename_cols = {
   col: f"feature_{i}" for i, col in enumerate(X_train.columns) if col not in ('efs_time', 'efs', "race_group")
}


# study = optuna.create_study(direction="maximize")
# study.optimize(lambda trial: linear_objective(trial, X_train.rename(columns=lgbm_rename_cols), lgbm_ensemble_model), n_trials=10)

# w1, w2 = study.best_params["w1"], study.best_params["w2"] 
# lgbm_ensemble_model.set_merger_function(lambda x, y: linear_model_merge(x, y, w1, w2))
# cross_validate(X_train, lgbm_ensemble_model)

# risk_scores = lgbm_ensemble_model.get_risk_scores(X_test)


# lgbm_ensemble_model.set_merger_function(prod_merge)
# cross_validate(X_train.rename(columns=lgbm_rename_cols), lgbm_ensemble_model)

# risk_scores = lgbm_ensemble_model.get_risk_scores(X_test)


# study = optuna.create_study(direction="maximize")
# study.optimize(lambda trial: complex_objective(trial, X_train.rename(columns=lgbm_rename_cols), lgbm_ensemble_model), n_trials=10)

# a, b, c = study.best_params["a"], study.best_params["b"], study.best_params["c"] 
# lgbm_ensemble_model.set_merger_function(lambda x, y: complex_merge(x, y, a, b, c))
# cross_validate(X_train, lgbm_ensemble_model)

# risk_scores = lgbm_ensemble_model.get_risk_scores(X_test)


# from scipy.stats import rankdata 
# sub = pd.read_csv("/kaggle/input/equity-post-HCT-survival-predictions/sample_submission.csv")
# sub.prediction = rankdata(pred_xgb1)

# sub.to_csv("submission.csv",index=False)


# model = xgb_regressor
# prediction=model.get_risk_scores(X_test)
# submission= pd.DataFrame({
#     'ID': test_data['ID'],
#     'prediction': prediction
# })
# submission.to_csv('submission.csv', index=False)

# submission.head()


from scipy.stats import rankdata 
sub = pd.read_csv("/kaggle/input/equity-post-HCT-survival-predictions/sample_submission.csv")
sub.prediction = rankdata(risk_scores)

sub.to_csv("submission.csv",index=False)




