import warnings
from pathlib import Path
warnings.filterwarnings('ignore')


import numpy as np
import polars as pl
import pandas as pd
import plotly.colors as pc
import plotly.express as px
import plotly.graph_objects as go


import plotly.io as pio
pio.renderers.default = 'iframe'


pd.options.display.max_columns = None


import lightgbm as lgb
from typing import List
from typing import Union
from metric import score
from typing import Optional
from metric import scaling_error
from catboost import CatBoostRegressor
from metric import get_property_weights
from sklearn.model_selection import KFold


class CFG:

    train_path = Path('/kaggle/input/neurips-open-polymer-prediction-2025/train.csv')
    test_path = Path('/kaggle/input/neurips-open-polymer-prediction-2025/test.csv')
    subm_path = Path('/kaggle/input/neurips-open-polymer-prediction-2025/sample_submission.csv')
    
    colorscale = 'Redor'
    color = '#A2574F'

    batch_size = 8192
    early_stop = 300
    n_splits = 5

    ctb_params = {
        'loss_function': 'RMSE',
        'learning_rate': 0.03,
        'random_state': 42,
        'task_type': 'CPU',
        'reg_lambda': 4.0,
        'num_trees': 600,
        'depth': 2
    }

    lgb_params = {
        'objective': 'regression',
        'min_child_samples': 32,
        'num_iterations': 600,
        'learning_rate': 0.03,
        'extra_trees': True,
        'reg_lambda': 4.0,
        'reg_alpha': 0.1,
        'num_leaves': 64,
        'metric': 'rmse',
        'max_depth': 2,
        'device': 'cpu',
        'max_bin': 128,
        'verbose': -1,
        'seed': 42
    }


class FE:

    def __init__(self, batch_size):
        self._batch_size = batch_size

    def _load_data(self, path):

        return pl.read_csv(path, batch_size=self._batch_size)

    def _cast_datatypes(self, df):

        df = df.with_columns(pl.col('SMILES').cast(pl.String))
        df = df.with_columns(pl.col('id').cast(pl.Int64))

        for col in df.columns:
            if col not in ['id', 'SMILES']:
                df = df.with_columns(pl.col(col).cast(pl.Float32))  

        return df

    def info(self, df):
        
        print(f'\nShape of dataframe: {df.shape}') 
        
        mem = df.memory_usage().sum() / 1024**2
        print('Memory usage: {:.2f} MB\n'.format(mem))

        display(df.head())

    def apply_fe(self, path):

        df = self._load_data(path)                   
        df = self._cast_datatypes(df)        
        df = df.to_pandas()
        self.info(df)
        
        cat_cols = [col for col in df.columns if df[col].dtype == pl.String]

        return df, cat_cols


fe = FE(CFG.batch_size)


train_data, cat_cols = fe.apply_fe(CFG.train_path)


test_data, _ = fe.apply_fe(CFG.test_path)


MINMAX_DICT =  {
    'Tg': [-148.0297376, 472.25],
    'FFV': [0.2269924, 0.77709707],
    'Tc': [0.0465, 0.524],
    'Density': [0.748691234, 1.840998909],
    'Rg': [9.7283551, 34.672905605],
}

NULL_FOR_SUBMISSION = -9999


def scaling_error(labels, preds, property):
    
    error = np.abs(labels - preds)
    min_val, max_val = MINMAX_DICT[property]
    label_range = max_val - min_val
    
    return np.mean(error / label_range)


def get_property_weights(labels):
    
    property_weight = []
    
    for property in MINMAX_DICT.keys():
        valid_num = np.sum(labels[property] != NULL_FOR_SUBMISSION)
        property_weight.append(valid_num)
        
    property_weight = np.array(property_weight)
    property_weight = np.sqrt(1 / property_weight)
    
    return (property_weight / np.sum(property_weight)) * len(property_weight)


def new_score(
    solution: pd.DataFrame,
    submission: pd.DataFrame,
    row_id_column_name: str,
    properties: Optional[Union[str, List[str]]] = None
) -> float:
    
    if properties is None:
        props = list(MINMAX_DICT.keys())
        
    else:
        props = [properties] if isinstance(properties, str) else properties

    if len(props) == 1:
        p = props[0]
        mask = solution[p] != NULL_FOR_SUBMISSION
        
        return float(scaling_error(
            solution.loc[mask, p],
            submission.loc[mask, p],
            p
        ))

    weights = get_property_weights(solution[props])
    maes = []
    for p in props:
        mask = solution[p] != NULL_FOR_SUBMISSION
        maes.append(
            scaling_error(
                solution.loc[mask, p],
                submission.loc[mask, p],
                p
            )
        )
        
    if not maes:
        raise RuntimeError('No labels')
        
    return float(np.average(maes, weights=weights))


class EDA:
    
    def __init__(self, colorscale, color, data):
        self._colorscale = colorscale
        self._color = color  
        self.data = data
        
    def _plot_cv(self, scores, title, metric='wMAE'):
        
        fold_scores = [round(score, 3) for score in scores]
        mean_score = round(np.mean(scores), 3)

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x = list(range(1, len(fold_scores) + 1)),
            y = fold_scores,
            mode = 'markers', 
            name = 'Fold Scores',
            marker = dict(size = 27, color=self._color, symbol='diamond'),
            text = [f'{score:.3f}' for score in fold_scores],
            hovertemplate = 'Fold %{x}: %{text}<extra></extra>',
            hoverlabel = dict(font=dict(size=18))  
        ))

        fig.add_trace(go.Scatter(
            x = [1, len(fold_scores)],
            y = [mean_score, mean_score],
            mode = 'lines',
            name = f'Mean: {mean_score:.3f}',
            line = dict(dash = 'dash', color = '#B22222'),
            hoverinfo = 'none'
        ))
        
        fig.update_layout(
            title = f'{title} | Cross-validation Mean {metric} Score: {mean_score}',
            xaxis_title = 'Fold',
            yaxis_title = f'{metric} Score',
            plot_bgcolor = 'rgba(247, 230, 202, 1)',  
            paper_bgcolor = 'rgba(247, 230, 202, 1)',
            font = dict(color=self._color), 
            xaxis = dict(
                gridcolor = 'grey',
                tickmode = 'linear',
                tick0 = 1,
                dtick = 1,
                range = [0.5, len(fold_scores) + 0.5],
                zerolinecolor = 'grey'
            ),
            yaxis = dict(
                gridcolor = 'grey',
                zerolinecolor = 'grey'
            )
        )
        
        fig.show()


class MD:
    
    def __init__(self, 
                 colorscale, 
                 color,
                 train_data, 
                 test_data, 
                 cat_cols, 
                 lgb_params, 
                 ctb_params, 
                 early_stop, 
                 n_splits):
        
        self.eda = EDA(colorscale, color, train_data)
        
        self.train_data = train_data
        self.test_data = test_data
        self.cat_cols = cat_cols
        
        self._length = len(self.train_data) # Length of the original train data   
        
        self._lgb_params = lgb_params
        self._ctb_params = ctb_params
        self._early_stop = early_stop
        self._n_splits = n_splits

    def _prepare_cv(self, df):
        
        oof_preds = np.zeros(len(df))
            
        cv = KFold(n_splits=self._n_splits, shuffle=True, random_state=42)

        return cv, oof_preds
        
    def train_model(self, target, title):
        
        df = self.train_data.dropna(subset=[target]).reset_index(drop=True)
        
        X = df.drop(['id', 'Tg', 'FFV', 'Tc', 'Density', 'Rg'], axis=1)
        y = df[target]
        
        models, fold_scores = [], []
        
        cv, oof_preds = self._prepare_cv(X)
    
        for fold, (train_index, valid_index) in enumerate(cv.split(X, y)):
                
            X_train = X.iloc[train_index]
            X_valid = X.iloc[valid_index]
                
            y_train = y.iloc[train_index]
            y_valid = y.iloc[valid_index]
    
            if title.startswith('LightGBM'):
                        
                model = lgb.LGBMRegressor(**self._lgb_params)
                        
                model.fit(
                    X_train, 
                    y_train,  
                    eval_set=[(X_valid, y_valid)],
                    eval_metric='rmse',
                    callbacks=[lgb.early_stopping(self._early_stop, verbose=0), lgb.log_evaluation(0)]
                )
                        
            elif title.startswith('CatBoost'):
                        
                model = CatBoostRegressor(**self._ctb_params, verbose=0, cat_features=self.cat_cols)
                        
                model.fit(
                    X_train,
                    y_train,
                    eval_set=(X_valid, y_valid),
                    early_stopping_rounds=self._early_stop, 
                    verbose=0
                )               
                    
            models.append(model)
            preds = model.predict(X_valid)
            oof_preds[valid_index] = preds

            y_true_fold = df.loc[valid_index, ['id', target]].reset_index(drop=True)
            y_pred_fold = pd.DataFrame({'id': df.loc[valid_index, 'id'].values, target: preds})
            
            fold_score = new_score(
                y_true_fold,
                y_pred_fold,
                row_id_column_name='id',
                properties=target
            )
            
            fold_scores.append(fold_score)
        
        self.eda._plot_cv(fold_scores, f'{title} on Target: {target}')
        
        return models, oof_preds

    def infer_model(self, models):
        
        data = self.test_data.drop(['id'], axis=1)

        return np.mean([model.predict(data) for model in models], axis=0)

    def train_and_infer_model(self, target, title):
        
        for col in self.cat_cols:
            self.train_data[col] = self.train_data[col].astype('category')
            self.test_data[col] = self.test_data[col].astype('category')

        models, oof_preds = self.train_model(target, title)
        preds = self.infer_model(models)

        return oof_preds, preds

    def inference(self, ensemble_oof_preds, ensemble_preds):
        
        props = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
        n_train = len(self.train_data)
        n_test  = len(self.test_data)
    
        full_oof = []
    
        for arr, prop in zip(ensemble_oof_preds, props):
            mask = self.train_data[prop].notna().values
            vec  = np.full(n_train, np.nan, dtype=float)
            vec[mask] = arr
            full_oof.append(vec)
    
        oof_matrix = np.vstack(full_oof).T
    
        oof_df = pd.DataFrame(oof_matrix, columns=props)
        oof_df['id'] = self.train_data['id'].values
        oof_df = oof_df[['id'] + props]
    
        solution_df = self.train_data[['id'] + props]
        oof_score = score(solution_df, oof_df, row_id_column_name='id')
    
        print(f'Ensemble OOF weighted MAE (wMAE): {oof_score:.3f}\n')
    
        preds_matrix = np.vstack(ensemble_preds).T
    
        preds_df = pd.DataFrame(preds_matrix, columns=props)
        preds_df['id'] = self.test_data['id'].values
        preds_df = preds_df[['id'] + props]
        preds_df.to_csv('submission.csv', index=False)

        display(preds_df.head())


md = MD(
    CFG.colorscale, 
    CFG.color, 
    train_data,
    test_data, 
    cat_cols, 
    CFG.lgb_params, 
    CFG.ctb_params, 
    CFG.early_stop, 
    CFG.n_splits
)


# Aggregated predictions (to store preds for each target)
agg_oof_preds = []
agg_preds = []


lgb_tg_oof_preds, lgb_tg_preds = md.train_and_infer_model('Tg', 'LightGBM')


agg_oof_preds.append(lgb_tg_oof_preds)
agg_preds.append(lgb_tg_preds)


lgb_ffv_oof_preds, lgb_ffv_preds = md.train_and_infer_model('FFV', 'LightGBM')


agg_oof_preds.append(lgb_ffv_oof_preds)
agg_preds.append(lgb_ffv_preds)


lgb_tc_oof_preds, lgb_tc_preds = md.train_and_infer_model('Tc', 'LightGBM')


agg_oof_preds.append(lgb_tc_oof_preds)
agg_preds.append(lgb_tc_preds)


lgb_density_oof_preds, lgb_density_preds = md.train_and_infer_model('Density', 'LightGBM')


agg_oof_preds.append(lgb_density_oof_preds)
agg_preds.append(lgb_density_preds)


lgb_rg_oof_preds, lgb_rg_preds = md.train_and_infer_model('Rg', 'LightGBM')


agg_oof_preds.append(lgb_rg_oof_preds)
agg_preds.append(lgb_rg_preds)


ctb_tg_oof_preds, ctb_tg_preds = md.train_and_infer_model('Tg', 'CatBoost')


agg_oof_preds.append(ctb_tg_oof_preds)
agg_preds.append(ctb_tg_preds)


ctb_ffv_oof_preds, ctb_ffv_preds = md.train_and_infer_model('FFV', 'CatBoost')


agg_oof_preds.append(ctb_ffv_oof_preds)
agg_preds.append(ctb_ffv_preds)


ctb_tc_oof_preds, ctb_tc_preds = md.train_and_infer_model('Tc', 'CatBoost')


agg_oof_preds.append(ctb_tc_oof_preds)
agg_preds.append(ctb_tc_preds)


ctb_density_oof_preds, ctb_density_preds = md.train_and_infer_model('Density', 'CatBoost')


agg_oof_preds.append(ctb_density_oof_preds)
agg_preds.append(ctb_density_preds)


ctb_rg_oof_preds, ctb_rg_preds = md.train_and_infer_model('Rg', 'CatBoost')


agg_oof_preds.append(ctb_rg_oof_preds)
agg_preds.append(ctb_rg_preds)


lgb_oof_preds = agg_oof_preds[0:5]
ctb_oof_preds = agg_oof_preds[5:10]

lgb_preds = agg_preds[0:5]
ctb_preds = agg_preds[5:10]

ensemble_oof_preds = [np.mean([ctb_arr, lgb_arr], axis=0) for ctb_arr, lgb_arr in zip(ctb_oof_preds, lgb_oof_preds)]
ensemble_preds = [np.mean([ctb_arr, lgb_arr], axis=0) for ctb_arr, lgb_arr in zip(ctb_preds, lgb_preds)]


md.inference(ensemble_oof_preds, ensemble_preds)

