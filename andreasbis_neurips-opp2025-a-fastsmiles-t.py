!pip install /kaggle/input/neurips-opp2025-download-libraries/rdkit-2025.3.3-cp311-cp311-manylinux_2_28_x86_64.whl


!pip install \
  --no-index \
  --no-deps \
  --no-build-isolation \
  /kaggle/input/neurips-opp2025-download-libraries/mordred-1.2.0.tar.gz


import os
import joblib
import numbers
import warnings
from pathlib import Path
warnings.filterwarnings('ignore')


import numpy as np
import polars as pl
import pandas as pd
from rdkit import Chem
from mordred import Calculator
from mordred import descriptors


pd.options.display.max_columns = None


from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')


import lightgbm as lgb
from metric import score
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold


class CFG:
    
    ext1_data_path = Path('/kaggle/input/tc-smiles/Tc_SMILES.csv')
    ext2_data_path = Path('/kaggle/input/smiles-extra-data/JCIM_sup_bigsmiles.csv')
    ext3_data_path = Path('/kaggle/input/smiles-extra-data/data_dnst1.xlsx')
    ext4_data_path = Path('/kaggle/input/smiles-extra-data/data_tg3.xlsx')

    train_path = Path('/kaggle/input/neurips-open-polymer-prediction-2025/train.csv')
    test_path = Path('/kaggle/input/neurips-open-polymer-prediction-2025/test.csv')
    
    batch_size = 16384
    threshold = 0.80
    early_stop = 300
    n_splits = 5
    
    ctb_params = {
        'loss_function': 'MAE',
        'learning_rate': 0.03,
        'random_state': 42,
        'task_type': 'CPU',
        'reg_lambda': 4.0,
        'num_trees': 6000,
        'depth': 4
    }

    lgb_params = {
        'objective': 'regression',
        'min_child_samples': 32,
        'num_iterations': 6000,
        'learning_rate': 0.03,
        'extra_trees': True,
        'reg_lambda': 4.0,
        'reg_alpha': 0.1,
        'num_leaves': 64,
        'metric': 'mae',
        'max_depth': 4,
        'device': 'cpu',
        'max_bin': 128,
        'verbose': -1,
        'seed': 42
    }
    
    xgb_params = {
        'objective': 'reg:squarederror',
        'enable_categorical': True,
        'max_cat_to_onehot': 4,
        'min_child_weight': 32,
        'learning_rate': 0.03,
        'n_estimators': 6000,
        'eval_metric': 'mae',
        'max_leaves': 32,
        'device': 'cpu',
        'verbosity': 0,
        'max_depth': 4,
        'lambda': 4.0,
        'alpha': 0.1,
        'seed': 42,
    }


class FE:
    
    def __init__(
        self, 
        ext1_data_path, 
        ext2_data_path,
        ext3_data_path,
        ext4_data_path, 
        batch_size,
        threshold
    ):
        
        self._ext1_data_path = ext1_data_path
        self._ext2_data_path = ext2_data_path
        self._ext3_data_path = ext3_data_path
        self._ext4_data_path = ext4_data_path
        
        self._batch_size = batch_size        
        self._threshold = threshold
        
        self._ignore_3D = True   
        self._bad_features = None
        
        self._properties = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
        
    def _load_data(self, path):
        
        return pl.read_csv(path, batch_size=self._batch_size).to_pandas()
    
    def _to_canonical(self, smiles):
        
        try:
            mol = Chem.MolFromSmiles(smiles)
            return Chem.MolToSmiles(mol, canonical=True)
            
        except:
            return np.nan
    
    def _load(self, path, columns_dict, excel=False):

        if not excel:
            ext_data = pd.read_csv(path)

        else:
            ext_data = pd.read_excel(path)
        
        ext_data = ext_data.rename(columns=columns_dict)
        ext_data['SMILES'] = ext_data['SMILES'].apply(self._to_canonical)
        
        return ext_data

    def _merge(self, data, ext_data, target):
        
        shared_smiles = set(data['SMILES']) & set(ext_data['SMILES'])
        unique_smiles = set(ext_data['SMILES']) - set(data['SMILES'])
        
        populated = set(data.loc[data[target].notna(), 'SMILES'])
        shared_smiles -= populated
        
        for s in shared_smiles:
            value = ext_data.loc[ext_data['SMILES'] == s, target].iat[0]
            data.loc[data['SMILES'] == s, target] = value
        
        new_rows = ext_data[ext_data['SMILES'].isin(unique_smiles)]
        data = pd.concat([data, new_rows], axis=0).reset_index(drop=True)
        
        return data
    
    def _merge_with_ext1_data(self, df):
        
        ext_data = self._load(self._ext1_data_path, {'TC_mean': 'Tc'})        
        ext_data = ext_data[['SMILES', 'Tc']].groupby('SMILES', as_index=False)['Tc'].mean()

        df = self._merge(df, ext_data, 'Tc')
        
        return df

    def _merge_with_ext2_data(self, df):
        
        ext_data = self._load(self._ext2_data_path, {'Tg (C)': 'Tg'})
        ext_data = ext_data[['SMILES', 'Tg']] .groupby('SMILES', as_index=False)['Tg'].mean()

        df = self._merge(df, ext_data, 'Tg')
        
        return df

    def _merge_with_ext3_data(self, df):
        
        ext_data = self._load(self._ext3_data_path, {'density(g/cm3)': 'Density'}, excel=True)
        ext_data = ext_data[['SMILES', 'Density']]
        
        mask = ext_data['SMILES'].notnull() & ext_data['Density'].notnull() & (ext_data['Density'] != 'nylon')
        
        ext_data = ext_data[mask].copy()        
        ext_data['Density'] -= 0.118
        ext_data = ext_data.groupby('SMILES', as_index=False)['Density'].mean()

        df = self._merge(df, ext_data, 'Density')
        
        return df

    def _merge_with_ext4_data(self, df):
        
        ext_data = self._load(self._ext4_data_path, {'Tg [K]': 'Tg'}, excel=True)
        ext_data = ext_data[['SMILES', 'Tg']]      
        
        ext_data['Tg'] -= 273.15
        ext_data = ext_data.groupby('SMILES', as_index=False)['Tg'].mean()
        
        df = self._merge(df, ext_data,'Tg')
        
        return df
        
    def _merge_data(self, df):  
        
        df = self._merge_with_ext1_data(df)
        df = self._merge_with_ext2_data(df)
        df = self._merge_with_ext3_data(df)
        df = self._merge_with_ext4_data(df)      

        return df
    
    def _aggregate_smiles(self, df):
        
        mols = df['SMILES'].map(Chem.MolFromSmiles)
        calc = Calculator(descriptors, ignore_3D=self._ignore_3D)
        
        features = calc.pandas(mols).apply(pd.to_numeric, errors='coerce')
        
        if self._bad_features is None:
            self._bad_features = [
                c for c in features.columns 
                if features[c].nunique(dropna=False) <= 1 
                or features[c].isna().mean() > self._threshold
            ]
        
        return pd.concat([df, features], axis=1)
    
    def _cast_datatypes(self, df):
        
        for col in df.columns:
            
            non_null = df[col].dropna()
            if len(non_null) == 0:
                continue
            
            val = non_null.iloc[0]
            
            if isinstance(val, numbers.Integral):
                target = 'int32'
                
            elif isinstance(val, numbers.Real):
                target = 'float32'
                
            else:
                target = 'string'
                
            df[col] = df[col].astype(target)
        
        return df

    def _stats(self, df):
        
        unique_smiles = df['SMILES'].unique()
        print(f'Dataset contains {len(unique_smiles)} unique SMILES.\n') 
        
        for p in self._properties:                
            count = df[p].notna().sum()
            print(f'Target {p} has {count} non-missing values.')        
    
    def info(self, df):
        
        print(f'\nShape of dataframe: {df.shape}')
        mem = df.memory_usage().sum() / 1024**2
        print('Memory usage: {:.2f} MB\n'.format(mem))
        
        if 'Density' in df.columns: # Or any other target
            self._stats(df)
    
    def apply_fe(self, path):
        
        df = self._load_data(path)
        
        if df.shape[1] > 2:
            df = self._merge_data(df)
            
        df = self._aggregate_smiles(df)
        df = df.drop(columns=self._bad_features, errors='ignore')        
        df = self._cast_datatypes(df)        
        self.info(df)
        
        cat_cols = df.select_dtypes(include=['string']).columns.tolist()
        if cat_cols:
            df[cat_cols] = df[cat_cols].fillna('Missing').astype('category')
        
        return df, cat_cols


fe = FE(
    CFG.ext1_data_path, 
    CFG.ext2_data_path,
    CFG.ext3_data_path,
    CFG.ext4_data_path,
    CFG.batch_size,
    CFG.threshold
)


train_data, cat_cols = fe.apply_fe(CFG.train_path)


class MD:
    
    def __init__(
        self,
        train_data,
        cat_cols,
        ctb_params,
        lgb_params,
        xgb_params, 
        early_stop, 
        n_splits
    ):

        self._properties = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
        
        self.train_data = train_data
        self.cat_cols = cat_cols
        
        self._ctb_params = ctb_params
        self._lgb_params = lgb_params
        self._xgb_params = xgb_params
        
        self._early_stop = early_stop
        self._n_splits = n_splits

        self._ctb_oof_preds = []
        self._lgb_oof_preds = []
        self._xgb_oof_preds = []
        
    def _prepare_cv(self, df):
        
        oof_preds = np.zeros(len(df))
        
        cv = KFold(n_splits=self._n_splits, shuffle=True, random_state=42)

        return cv, oof_preds
        
    def train_model(self, target, title):
        
        df = self.train_data.dropna(subset=[target]).reset_index(drop=True)
        
        X = df.drop(['id'] + self._properties, axis=1)
        y = df[target]
        
        models = []
        
        cv, oof_preds = self._prepare_cv(X)
    
        for fold, (train_index, valid_index) in enumerate(cv.split(X, y)):
                
            X_train = X.iloc[train_index]
            X_valid = X.iloc[valid_index]
                
            y_train = y.iloc[train_index]
            y_valid = y.iloc[valid_index]
                        
            if title.startswith('CTB'):
                
                model = CatBoostRegressor(**self._ctb_params, verbose=0, cat_features=self.cat_cols)                        
                model.fit(
                    X_train,
                    y_train,
                    eval_set=(X_valid, y_valid),
                    early_stopping_rounds=self._early_stop, 
                    verbose=0
                )   
    
            elif title.startswith('LGB'):
                   
                model = lgb.LGBMRegressor(**self._lgb_params)                        
                model.fit(
                    X_train, 
                    y_train,  
                    eval_set=[(X_valid, y_valid)],
                    callbacks=[lgb.early_stopping(self._early_stop, verbose=0), lgb.log_evaluation(0)]
                )   
            
            elif title.startswith('XGB'):
                
                model = XGBRegressor(**self._xgb_params)                    
                model.fit(
                    X_train, 
                    y_train,
                    eval_set=[(X_valid, y_valid)],
                    early_stopping_rounds=self._early_stop,
                    verbose=False,
                )           
            
            models.append(model)
            oof_preds[valid_index] = model.predict(X_valid)
        
        return models, oof_preds
    
    def save_model(self, models, oof_preds, title, target):
        
        base_path = os.path.join('/kaggle/working', target, title)
        os.makedirs(base_path, exist_ok=True)
    
        models_path = os.path.join(base_path, 'Models')
        os.makedirs(models_path, exist_ok=True)
        
        for i, model in enumerate(models):
            model_path = os.path.join(models_path, f'{title.lower()}-fold-{i+1}.pkl')
            joblib.dump(model, model_path)
    
        name = title.lower().replace('-', '_')
    
        oof_preds = pd.DataFrame(oof_preds, columns=['oof_preds'])    
        oof_preds.to_csv(os.path.join(base_path, f'{name}_oof_preds.csv'), index=False)
    
    def train_and_save_model(self, title):
        
        for col in self.cat_cols:
            self.train_data[col] = self.train_data[col].astype('category')

        for target in self._properties:
            
            models, oof_preds = self.train_model(target, title)
            
            valid_mask = self.train_data[target].notna().values
            valid_indices = np.where(valid_mask)[0]
            
            full_oof_preds = np.full(len(self.train_data), np.nan, dtype=float)
            full_oof_preds[valid_indices] = oof_preds
            
            self.save_model(models, full_oof_preds, title, target)
            
            if title.startswith('CTB'):
                self._ctb_oof_preds.append(full_oof_preds)
                
            elif title.startswith('LGB'):
                self._lgb_oof_preds.append(full_oof_preds)
                
            elif title.startswith('XGB'):
                self._xgb_oof_preds.append(full_oof_preds)
        
        solution_data = self.train_data[['id'] + self._properties]
        
        if title.startswith('CTB'):
            oof_list = self._ctb_oof_preds
        
        elif title.startswith('LGB'):
            oof_list = self._lgb_oof_preds
        
        elif title.startswith('XGB'):
            oof_list = self._xgb_oof_preds
        
        oof_matrix = np.vstack(oof_list).T
        
        oof_data = pd.DataFrame(oof_matrix, columns=self._properties)
        oof_data['id'] = self.train_data['id'].values
        oof_data = oof_data[['id'] + self._properties]
        
        model_score = score(solution_data, oof_data, 'id')
        print(f'wMAE for {title}: {model_score:.3f}')


md = MD(
    train_data,
    cat_cols,
    CFG.ctb_params,
    CFG.lgb_params,
    CFG.xgb_params,
    CFG.early_stop,
    CFG.n_splits
)


md.train_and_save_model('CTB-A')


md.train_and_save_model('LGB-A')


md.train_and_save_model('XGB-A')

