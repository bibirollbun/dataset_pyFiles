!pip install /kaggle/input/pip-install-lifelines/autograd-1.7.0-py3-none-any.whl
!pip install /kaggle/input/pip-install-lifelines/autograd-gamma-0.5.0.tar.gz
!pip install /kaggle/input/pip-install-lifelines/interface_meta-1.3.0-py3-none-any.whl
!pip install /kaggle/input/pip-install-lifelines/formulaic-1.0.2-py3-none-any.whl
!pip install /kaggle/input/pip-install-lifelines/lifelines-0.30.0-py3-none-any.whl
import warnings
from pathlib import Path
warnings.filterwarnings('ignore')
import numpy as np
import polars as pl
import pandas as pd
import plotly.colors as pc
import plotly.express as px
import plotly.graph_objects as go
import lightgbm as lgb
from metric import score
from scipy.stats import rankdata 
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold
from lifelines import CoxPHFitter, KaplanMeierFitter, NelsonAalenFitter
from sklearn.preprocessing import StandardScaler

import plotly.io as pio
pio.renderers.default = 'iframe'
pd.options.display.max_columns = None


# Cell 1: Update CFG class
class CFG:
    train_path = Path('/kaggle/input/equity-post-HCT-survival-predictions/train.csv')
    test_path = Path('/kaggle/input/equity-post-HCT-survival-predictions/test.csv')
    subm_path = Path('/kaggle/input/equity-post-HCT-survival-predictions/sample_submission.csv')
    colorscale = 'Sunset'
    color = '#EADDCA'
    batch_size = 32768
    early_stop = 300
    penalizer = 0.01
    n_splits = 5

    # Original CatBoost params
    ctb_params = {
        'loss_function': 'RMSE',
        'learning_rate': 0.03,
        'random_state': 42,
        'task_type': 'GPU',
        'num_trees': 6000,
        # 'subsample': 0.85,
        'reg_lambda': 8.0,
        'depth': 8
    }

    # Original LightGBM params
    lgb_params = {
        'objective': 'regression',
        'min_child_samples': 20,
        'num_iterations': 6000,
        'learning_rate': 0.01,
        'extra_trees': True,
        'reg_lambda': 3.0,
        'reg_alpha': 0.1,
        'num_leaves': 64,
        'metric': 'rmse',
        'max_depth': 10,
        'device': 'gpu',
        'max_bin': 255,
        'verbose': -1,
        'seed': 42
    }

    # Cox model params
    cox1_params = {
        'grow_policy': 'Depthwise',
        'min_child_samples': 8,
        'loss_function': 'Cox',
        'learning_rate': 0.03,
        'random_state': 42,
        'task_type': 'CPU',
        'num_trees': 6000,
        'subsample': 0.6,  
        'reg_lambda': 8.0,
        'depth': 8,
        'bootstrap_type': 'Bernoulli',
    }

    cox2_params = {
        'grow_policy': 'Lossguide',
        'loss_function': 'Cox',
        'learning_rate': 0.03,
        'random_state': 42,
        'task_type': 'CPU',
        'num_trees': 6000,
        'subsample': 0.6,  
        'reg_lambda': 8.0,
        'num_leaves': 32,
        'depth': 8,
        'bootstrap_type': 'Bernoulli',
    }

    cox3_params = {
        'grow_policy': 'Depthwise',
        'min_child_samples': 16,
        'loss_function': 'Cox',
        'learning_rate': 0.02,
        'random_state': 42,
        'task_type': 'CPU',
        'num_trees': 7000,
        'subsample': 0.5,  
        'reg_lambda': 6.0,
        'depth': 10,
        'bootstrap_type': 'Bernoulli',
    }


# print(cat_cols)
cat_cols=['dri_score', 'psych_disturb', 'cyto_score', 'diabetes', 'tbi_status', 'arrhythmia', 'graft_type', 'vent_hist', 'renal_issue', 'pulm_severe', 'prim_disease_hct', 'cmv_status', 'tce_imm_match', 'rituximab', 'prod_type', 'cyto_score_detail', 'conditioning_intensity', 'ethnicity', 'obesity', 'mrd_hct', 'in_vivo_tcd', 'tce_match', 'hepatic_severe', 'prior_tumor', 'peptic_ulcer', 'gvhd_proph', 'rheum_issue', 'sex_match', 'race_group', 'hepatic_mild', 'tce_div_match', 'donor_related', 'melphalan_dose', 'cardiac', 'pulm_moderate']


class FE:
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.scaler = StandardScaler()

    def load_data(self, path):
        return pl.read_csv(path, batch_size=self.batch_size)
    def recalculate_hla_sums(self, df):     
        df = df.with_columns(
            (pl.col("hla_match_a_low").fill_null(0) + pl.col("hla_match_b_low").fill_null(0) + 
             pl.col("hla_match_drb1_high").fill_null(0)).alias("hla_nmdp_6"),
            
            (pl.col("hla_match_a_low").fill_null(0) + pl.col("hla_match_b_low").fill_null(0) + 
             pl.col("hla_match_drb1_low").fill_null(0)).alias("hla_low_res_6"),
            
            (pl.col("hla_match_a_high").fill_null(0) + pl.col("hla_match_b_high").fill_null(0) + 
             pl.col("hla_match_drb1_high").fill_null(0)).alias("hla_high_res_6"),
            
            (pl.col("hla_match_a_low").fill_null(0) + pl.col("hla_match_b_low").fill_null(0) + 
             pl.col("hla_match_c_low").fill_null(0) + pl.col("hla_match_drb1_low").fill_null(0)
            ).alias("hla_low_res_8"),
            
            (pl.col("hla_match_a_high").fill_null(0) + pl.col("hla_match_b_high").fill_null(0) + 
             pl.col("hla_match_c_high").fill_null(0) + pl.col("hla_match_drb1_high").fill_null(0)
            ).alias("hla_high_res_8"),
            
            (pl.col("hla_match_a_low").fill_null(0) + pl.col("hla_match_b_low").fill_null(0) + 
             pl.col("hla_match_c_low").fill_null(0) + pl.col("hla_match_drb1_low").fill_null(0) +
             pl.col("hla_match_dqb1_low").fill_null(0)).alias("hla_low_res_10"),
            
            (pl.col("hla_match_a_high").fill_null(0) + pl.col("hla_match_b_high").fill_null(0) + 
             pl.col("hla_match_c_high").fill_null(0) + pl.col("hla_match_drb1_high").fill_null(0) +
             pl.col("hla_match_dqb1_high").fill_null(0)).alias("hla_high_res_10"),
        )
        return df
    def cast_datatypes(self, df):
        num_cols = [
            'hla_high_res_8', 'hla_low_res_8', 'hla_high_res_6',
            'hla_low_res_6', 'hla_high_res_10', 'hla_low_res_10',
            'hla_match_dqb1_high', 'hla_match_dqb1_low',
            'hla_match_drb1_high', 'hla_match_drb1_low',
            'hla_nmdp_6', 'year_hct', 'hla_match_a_high',
            'hla_match_a_low', 'hla_match_b_high', 'hla_match_b_low',
            'hla_match_c_high', 'hla_match_c_low', 'donor_age',
            'age_at_hct', 'comorbidity_score', 'karnofsky_score',
            'efs', 'efs_time'
        ]

        for col in df.columns:
            if col in num_cols:
                df = df.with_columns(pl.col(col).fill_null(df[col].median()).cast(pl.Float32))  

            else:
                df = df.with_columns(pl.col(col).fill_null('Not Done').cast(pl.String))  

        return df.with_columns(pl.col('ID').cast(pl.Int32))

    def add_features(self, df):
        # Interactions
        log_transform_cols = ['age_at_hct', 'donor_age', 'comorbidity_score', 'karnofsky_score']
        for col in log_transform_cols:
            df[f"log_{col}"] = np.log1p(df[col])

        poly_features = ['age_at_hct', 'donor_age', 'karnofsky_score', 'comorbidity_score']
        for col in poly_features:
            df[f"{col}_squared"] = df[col] ** 2

        df['age_karnofsky'] = df['age_at_hct'] * df['karnofsky_score']
        df['age_comorbidity'] = df['age_at_hct'] * df['comorbidity_score']
        df['donor_recipient_age_diff'] = abs(df['donor_age'] - df['age_at_hct'])
        
        # Time-based features
        df['years_since_2000'] = df['year_hct'] - 2000
        
        # HLA match ratios
        df['hla_match_ratio'] = (df['hla_high_res_8'] + df['hla_low_res_8']) / 16
        
        # Polynomial features for important numerical columns
        df['age_squared'] = df['age_at_hct'] ** 2
        df['karnofsky_squared'] = df['karnofsky_score'] ** 2
        df['year_decade'] = (df['year_hct'] // 10) * 10 
        
        return df

    def normalize_features(self, df, train=True):
        num_cols = df.select_dtypes(include=['float32', 'float64', 'int32', 'int64']).columns
        num_cols = [col for col in num_cols if col not in ['ID', 'efs', 'efs_time']]
        
        if train:
            df[num_cols] = self.scaler.fit_transform(df[num_cols])
        else:
            df[num_cols] = self.scaler.transform(df[num_cols])
        
        return df

    def info(self, df):     
        print(f'\nShape of dataframe: {df.shape}')    
        mem = df.memory_usage().sum() / 1024**2
        print('Memory usage: {:.2f} MB\n'.format(mem))
        display(df.head())

    def apply_fe(self, path):
        df = self.load_data(path)
        df = self.cast_datatypes(df)
        df = df.to_pandas()
        df = self.add_features(df)
        df = self.normalize_features(df, train=True)
        self.info(df)
        
        # cat_cols = [col for col in df.columns if df[col].dtype == pl.String]
        return df, cat_cols


fe = FE(CFG.batch_size)
train_data, cat_cols = fe.apply_fe(CFG.train_path)
test_data, _ = fe.apply_fe(CFG.test_path)
print('2')


train_data['cyto_score'] = train_data['cyto_score'].replace({'Not done': 'Not tested'})
test_data['cyto_score'] = test_data['cyto_score'].replace({'Not done': 'Not tested'})


set(train_data['graft_type'])


print(train_data.columns)


class MD:
    def __init__(self, early_stop, penalizer, n_splits, color):
        self.early_stop = early_stop
        self.penalizer = penalizer
        self.n_splits = n_splits
        self.color = color

    def create_target1(self, data, cat_cols):
        cph_data = pd.get_dummies(data, columns=cat_cols, drop_first=True)
        cph = CoxPHFitter(penalizer=self.penalizer)
        cph.fit(cph_data, duration_col='efs_time', event_col='efs')
        data['target1'] = cph.predict_partial_hazard(cph_data)
        return data

    def create_target2(self, data):
        kmf = KaplanMeierFitter()
        kmf.fit(durations=data['efs_time'], event_observed=data['efs'])
        data['target2'] = kmf.survival_function_at_times(data['efs_time']).values
        return data

    def create_target3(self, data):
        naf = NelsonAalenFitter()
        naf.fit(durations=data['efs_time'], event_observed=data['efs'])
        data['target3'] = naf.cumulative_hazard_at_times(data['efs_time']).values
        data['target3'] = data['target3'] * -1
        return data

    def create_target4(self, data):
        data['target4'] = data.efs_time.copy()
        data.loc[data.efs == 0, 'target4'] *= -1
        return data

    def train_model(self, data, cat_cols, params, target, title):
        for col in cat_cols:
            data[col] = data[col].astype('category')
            
        X = data.drop(['ID', 'efs', 'efs_time', 'target1', 'target2', 'target3', 'target4'], axis=1)
        y = data[target]
        
        models, fold_scores = [], []
        
        # Use KFold 
        cv = KFold(n_splits=self.n_splits, shuffle=True, random_state=42)
        
        oof_preds = np.zeros(len(X))
        
        for fold, (train_index, valid_index) in enumerate(cv.split(X)):
            X_train = X.iloc[train_index]
            X_valid = X.iloc[valid_index]
            y_train = y.iloc[train_index]
            y_valid = y.iloc[valid_index]
            
            if title.startswith('LightGBM'):
                model = lgb.LGBMRegressor(**params)
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_valid, y_valid)],
                    eval_metric='rmse',
                    callbacks=[
                        lgb.early_stopping(self.early_stop, verbose=0),
                        lgb.log_evaluation(0)
                    ]
                )
                
            elif title.startswith('CatBoost'):
                model = CatBoostRegressor(**params, verbose=0, cat_features=cat_cols)
                model.fit(
                    X_train, y_train,
                    eval_set=(X_valid, y_valid),
                    early_stopping_rounds=self.early_stop,
                    verbose=0
                )
                
            models.append(model)
            oof_preds[valid_index] = model.predict(X_valid)
            
            y_true_fold = data.iloc[valid_index][['ID', 'efs', 'efs_time', 'race_group']].copy()
            y_pred_fold = data.iloc[valid_index][['ID']].copy()
            y_pred_fold['prediction'] = oof_preds[valid_index]
            
            fold_score = score(y_true_fold, y_pred_fold, 'ID')
            fold_scores.append(fold_score)
        
        y_true = data[['ID', 'efs', 'efs_time', 'race_group']].copy()
        y_pred = data[['ID']].copy()
        y_pred['prediction'] = oof_preds
        
        c_index_score = score(y_true.copy(), y_pred.copy(), 'ID')
        if target == 'target1':
            t = 'Cox Target'
        elif target == 'target2':
            t = 'Kaplan-Meier Target'
        elif target == 'target3':
            t = 'Nelson-Aalen Target'
        else:
            t = 'Cox Loss'
        print(f'\nOverall C-Index for {title} {t}: {c_index_score:.3f}\n')
        
        return models, oof_preds

    def infer_model(self, data, cat_cols, models):
        data = data.drop(['ID'], axis=1)
        for col in cat_cols:
            data[col] = data[col].astype('category')
        return np.mean([model.predict(data) for model in models], axis=0)


md = MD(CFG.early_stop, CFG.penalizer, CFG.n_splits, CFG.color)

# Create all targets
train_data = md.create_target1(train_data, cat_cols)
train_data = md.create_target2(train_data)
train_data = md.create_target3(train_data)
train_data = md.create_target4(train_data)
print('3')


train_data.head()


# Train CatBoost models19 10
print("Training CatBoost models...")
ctb1_models, _ = md.train_model(train_data, cat_cols, CFG.ctb_params, target='target1', title='CatBoost1')
print('1')
ctb2_models, _ = md.train_model(train_data, cat_cols, CFG.ctb_params, target='target2', title='CatBoost2')
ctb3_models, _ = md.train_model(train_data, cat_cols, CFG.ctb_params, target='target3', title='CatBoost3')

# Train LightGBM models
print("\nTraining LightGBM models...")
lgb1_models, _ = md.train_model(train_data, cat_cols, CFG.lgb_params, target='target1', title='LightGBM1')
lgb2_models, _ = md.train_model(train_data, cat_cols, CFG.lgb_params, target='target2', title='LightGBM2')
lgb3_models, _ = md.train_model(train_data, cat_cols, CFG.lgb_params, target='target3', title='LightGBM3')

# Train Cox models


# # Directory to save models
# import os
# import joblib  # For LightGBM

# output_dir = "trained_models"
# os.makedirs(output_dir, exist_ok=True)

# # Save CatBoost models
# print("Saving CatBoost models...")
# for idx, model in enumerate(ctb1_models):
#     model.save_model(f"{output_dir}/CatBoost1_model_fold{idx}.cbm")

# for idx, model in enumerate(ctb2_models):
#     model.save_model(f"{output_dir}/CatBoost2_model_fold{idx}.cbm")

# for idx, model in enumerate(ctb3_models):
#     model.save_model(f"{output_dir}/CatBoost3_model_fold{idx}.cbm")

# # Save LightGBM models
# print("\nSaving LightGBM models...")
# for idx, model in enumerate(lgb1_models):
#     joblib.dump(model, f"{output_dir}/LightGBM1_model_fold{idx}.pkl")

# for idx, model in enumerate(lgb2_models):
#     joblib.dump(model, f"{output_dir}/LightGBM2_model_fold{idx}.pkl")

# for idx, model in enumerate(lgb3_models):
#     joblib.dump(model, f"{output_dir}/LightGBM3_model_fold{idx}.pkl")



# import shutil
# from IPython.display import FileLink

# # Compress the models directory into a ZIP file
# shutil.make_archive("trained_models", "zip", output_dir)

# # Provide a download link
# FileLink("trained_models.zip")



# print("\nTraining Cox models...")
cox1_models, _ = md.train_model(train_data, cat_cols, CFG.cox1_params, target='target4', title='CatBoost')
cox2_models, _ = md.train_model(train_data, cat_cols, CFG.cox2_params, target='target4', title='CatBoost')
cox3_models, _ = md.train_model(train_data, cat_cols, CFG.cox3_params, target='target4', title='CatBoost')


# import os
# import joblib
# from catboost import CatBoostClassifier  # or CatBoostRegressor, depending on your use case

# # Directory containing the saved models
# output_dir = "/kaggle/input/trained-model"

# # Load CatBoost models
# print("Loading CatBoost models...")
# ctb1_models = []
# ctb2_models = []
# ctb3_models = []

# for idx in range(len([file for file in os.listdir(output_dir) if file.startswith("CatBoost1_model_fold")])):
#     model = CatBoostRegressor(verbose=0, cat_features=cat_cols)  # Replace with CatBoostRegressor if needed
#     model.load_model(f"{output_dir}/CatBoost1_model_fold{idx}.cbm")
#     ctb1_models.append(model)

# for idx in range(len([file for file in os.listdir(output_dir) if file.startswith("CatBoost2_model_fold")])):
#     model = CatBoostRegressor(verbose=0, cat_features=cat_cols)  # Replace with CatBoostRegressor if needed
#     model.load_model(f"{output_dir}/CatBoost2_model_fold{idx}.cbm")
#     ctb2_models.append(model)

# for idx in range(len([file for file in os.listdir(output_dir) if file.startswith("CatBoost3_model_fold")])):
#     model = CatBoostRegressor(verbose=0, cat_features=cat_cols)  # Replace with CatBoostRegressor if needed
#     model.load_model(f"{output_dir}/CatBoost3_model_fold{idx}.cbm")
#     ctb3_models.append(model)

# # Load LightGBM models
# print("\nLoading LightGBM models...")
# lgb1_models = []
# lgb2_models = []
# lgb3_models = []

# for idx in range(len([file for file in os.listdir(output_dir) if file.startswith("LightGBM1_model_fold")])):
#     model = joblib.load(f"{output_dir}/LightGBM1_model_fold{idx}.pkl")
#     lgb1_models.append(model)

# for idx in range(len([file for file in os.listdir(output_dir) if file.startswith("LightGBM2_model_fold")])):
#     model = joblib.load(f"{output_dir}/LightGBM2_model_fold{idx}.pkl")
#     lgb2_models.append(model)

# for idx in range(len([file for file in os.listdir(output_dir) if file.startswith("LightGBM3_model_fold")])):
#     model = joblib.load(f"{output_dir}/LightGBM3_model_fold{idx}.pkl")
#     lgb3_models.append(model)

# print("All models loaded successfully!")



if not ctb1_models:
    raise ValueError("The models list is empty. Ensure models are loaded correctly.")


# CatBoost predictions
ctb1_preds = md.infer_model(test_data, cat_cols, ctb1_models)
ctb2_preds = md.infer_model(test_data, cat_cols, ctb2_models)
ctb3_preds = md.infer_model(test_data, cat_cols, ctb3_models)
# # LightGBM predictions
lgb1_preds = md.infer_model(test_data, cat_cols, lgb1_models)
lgb2_preds = md.infer_model(test_data, cat_cols, lgb2_models)
lgb3_preds = md.infer_model(test_data, cat_cols, lgb3_models)
# Cox predictions


cox1_preds = md.infer_model(test_data, cat_cols, cox1_models)
cox2_preds = md.infer_model(test_data, cat_cols, cox2_models)
cox3_preds = md.infer_model(test_data, cat_cols, cox3_models)


#10:21
# Combine all predictions
preds = [
    ctb1_preds, ctb2_preds, ctb3_preds,
    lgb1_preds, lgb2_preds, lgb3_preds,
    cox1_preds, cox2_preds, cox3_preds
]

# Define weights based on model performance
weights = [
    3.0, 8.0, 8.0,  # CatBoost weights
    3.0, 4.0, 4.0,  # LightGBM weights
    3.0, 3.0, 3.0   # Cox weights 
]

# Create ranked predictions
ranked_preds = np.array([rankdata(p) for p in preds])
ensemble_preds = np.sum([w * p for w, p in zip(weights, ranked_preds)], axis=0)
print('5')


# Create submission
subm_data = pd.read_csv(CFG.subm_path)
subm_data['prediction'] = ensemble_preds
subm_data.to_csv('submission.csv', index=False)
display(subm_data.head())

