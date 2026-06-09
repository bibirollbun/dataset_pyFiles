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
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


class CFG:
    
    train_path = Path('/kaggle/input/playground-series-s5e8/train.csv')
    test_path = Path('/kaggle/input/playground-series-s5e8/test.csv')
    subm_path = Path('/kaggle/input/playground-series-s5e8/sample_submission.csv')
    
    colorscale = 'Redor'
    color = '#A2574F'
    
    early_stop = 300
    n_splits = 5
    
    weights = [0.28, 0.36, 0.36]
    
    lgb_params = {
        'min_child_samples': 32,
        'num_iterations': 6000,
        'learning_rate': 0.03,
        'objective': 'binary',
        'extra_trees': True,
        'reg_lambda': 8.0,
        'reg_alpha': 0.1,
        'num_leaves': 64,
        'max_depth': 8,
        'device': 'cpu',
        'max_bin': 128,
        'verbose': -1,
        'seed': 42
    }

    ctb_params = {
        'loss_function': 'Logloss',
        'grow_policy': 'Depthwise',
        'min_child_samples': 4,
        'learning_rate': 0.03,
        'random_state': 42,
        'task_type': 'CPU',
        'num_trees': 6000,
        'reg_lambda': 8.0,        
        'depth': 8
    }

    xgb_params = {
        'objective': 'reg:logistic',
        'enable_categorical': True,
        'max_cat_to_onehot': 8,
        'min_child_weight': 64,
        'learning_rate': 0.03,
        'n_estimators': 6000,
        'max_leaves': 64,
        'subsample': 0.8,
        'device': 'cpu',
        'verbosity': 0,
        'max_depth': 8,
        'lambda': 8.0,
        'alpha': 0.1,
        'seed': 42,
    }


class FE:

    def __init__(self):
        self._batch_size = 65536

    def _load_data(self, path):

        return pl.read_csv(path, batch_size=self._batch_size).drop('id')
        
    def _cast_datatypes(self, df):
        
        cat_cols = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']
        
        for col in df.columns:
            if col in cat_cols:
                df = df.with_columns(pl.col(col).cast(pl.String))
            
            else:
                df = df.with_columns(pl.col(col).cast(pl.Int32))
                
        return df, cat_cols
    
    def _zscore(self, df, by):        
        
        df = df.with_columns([
            pl.col(col)
            .sub(pl.col(col).mean().over(by))
            .truediv(pl.col(col).std().over(by))
            .cast(pl.Float32).alias(f'{col}_Zscore_{by}')
            for col in ['balance', 'duration']
        ])
        
        return df
    
    def _prob(self, df, by):
        
        df = df.with_columns([
            pl.col(col).mean().over(by).cast(pl.Float32).alias(f'{col}_Prob_{by}')
            for col in ['default', 'housing', 'loan']
        ])
        
        return df
    
    def _aggregate_data(self, df):
        
        df = self._zscore(df, 'age')
        df = self._zscore(df, 'job')
        df = self._zscore(df, 'marital')
        df = self._zscore(df, 'education')
        df = self._zscore(df, 'contact')
        df = self._zscore(df, 'poutcome')
        
        df = self._prob(df, 'age')
        df = self._prob(df, 'job')
        df = self._prob(df, 'marital')
        df = self._prob(df, 'education')
        df = self._prob(df, 'contact')
        df = self._prob(df, 'poutcome')
        
        return df

    def info(self, df):
        
        print(f'\nShape of dataframe: {df.shape}') 
        
        mem = df.memory_usage().sum() / 1024**2
        print('Memory usage: {:.2f} MB\n'.format(mem))
        
        display(df.head())
        
    def apply_fe(self, path):
        
        df = self._load_data(path)                  
        df, cat_cols = self._cast_datatypes(df)
        df = self._aggregate_data(df)
        df = df.to_pandas()
        self.info(df)
        
        return df, cat_cols


fe = FE()


train_data, cat_cols = fe.apply_fe(CFG.train_path)


test_data, _ = fe.apply_fe(CFG.test_path)


class EDA:
    
    def __init__(self, colorscale, color, data):
        self._colorscale = colorscale
        self._color = color  
        self.data = data
        
    def _template(self, fig, title):
        
        fig.update_layout(
            title=title,
            title_x=0.5, 
            plot_bgcolor='rgba(247, 230, 202, 1)',  
            paper_bgcolor='rgba(247, 230, 202, 1)', 
            font=dict(color=self._color),
            margin=dict(l=72, r=72, t=72, b=72), 
            height=720
        )
        
        return fig
        
    def distribution_plot(self, col):
        
        fig = px.histogram(
            self.data,
            x=col,
            nbins=100,
            color_discrete_sequence=[self._color]
        )
        
        fig.update_layout(
            xaxis_title='Values',
            yaxis_title='Count',
            bargap=0.1,
            xaxis=dict(gridcolor='grey'),
            yaxis=dict(gridcolor='grey', zerolinecolor='grey')
        )
        
        fig.update_traces(hovertemplate='Value: %{x:.2f}<br>Count: %{y:,}')
        
        fig = self._template(fig, f'{col}')
        fig.show()
    
    def bar_chart(self, col):
        
        value_counts = self.data[col].value_counts().reset_index()
        value_counts.columns = [col, 'count']
        
        fig = px.bar(
            value_counts,
            y=col,
            x='count',
            orientation='h',
            color='count',
            color_continuous_scale=self._colorscale,
        )
        
        fig.update_layout(
            xaxis_title='Count',
            yaxis_title='',
            xaxis=dict(gridcolor='grey'),
            yaxis=dict(gridcolor='grey', zerolinecolor='grey')
        )
        
        fig.update_traces(
            hovertemplate=(
                f'<b>{col}:</b> %{{y}}<br>'
                '<b>Count:</b> %{x:,}<br>'
            ),
            hoverlabel=dict(
                font=dict(color=self._color),
                bgcolor='rgba(247, 230, 202, 1)'
            )
        )
        
        fig = self._template(fig, f'{col}')
        fig.show()
        
    def _plot_cv(self, scores, title, metric='AUC'):
        
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


eda = EDA(CFG.colorscale, CFG.color, train_data)


eda.distribution_plot('job')


eda.bar_chart('marital')


eda.bar_chart('education')


eda.bar_chart('default')


eda.bar_chart('y')


class MD:
    
    def __init__(
        self, 
        colorscale, 
        color, 
        train_data, 
        test_data, 
        cat_cols,
        lgb_params, 
        ctb_params, 
        xgb_params, 
        early_stop, 
        n_splits
    ):
        
        self.eda = EDA(colorscale, color, train_data)
        
        self.train_data = train_data
        self.test_data = test_data      
        self.cat_cols = cat_cols

        self._length = len(self.train_data)
        
        self._lgb_params = lgb_params
        self._ctb_params = ctb_params
        self._xgb_params = xgb_params
        self._early_stop = early_stop
        self._n_splits = n_splits

    def _prepare_cv(self):
        
        oof_preds = np.zeros(self._length)
        
        cv = StratifiedKFold(n_splits=self._n_splits, shuffle=True, random_state=42)
        
        return cv, oof_preds
        
    def validate_model(self, preds, title):
        
        auc = roc_auc_score(self.train_data.y, preds)
        print(f'Overall AUC score for {title}: {auc:.5f}')
        
    def train_model(self, title):
        
        X = self.train_data.drop('y', axis=1)
        y = self.train_data['y']
        
        models, fold_scores = [], []
            
        cv, oof_preds = self._prepare_cv()
    
        for fold, (train_index, valid_index) in enumerate(cv.split(X, y)):
            
            X_train = X.iloc[train_index]
            X_valid = X.iloc[valid_index]
                
            y_train = y.iloc[train_index]
            y_valid = y.iloc[valid_index]
    
            if title.startswith('LightGBM'):
                
                model = lgb.LGBMClassifier(**self._lgb_params)
                        
                model.fit(
                    X_train, 
                    y_train,  
                    eval_set=[(X_valid, y_valid)],
                    eval_metric='rmse',
                    callbacks=[lgb.early_stopping(self._early_stop, verbose=0), lgb.log_evaluation(0)]
                )
                        
            elif title.startswith('CatBoost'):
                        
                model = CatBoostClassifier(**self._ctb_params, verbose=0, cat_features=self.cat_cols)
                
                model.fit(
                    X_train,
                    y_train,
                    eval_set=(X_valid, y_valid),
                    early_stopping_rounds=self._early_stop, 
                    verbose=0
                )    
    
            elif title.startswith('XGBoost'):
                
                model = XGBClassifier(**self._xgb_params) 
                
                model.fit(
                    X_train, 
                    y_train,
                    eval_set=[(X_valid, y_valid)],
                    early_stopping_rounds=self._early_stop,
                    verbose=False,
                )                  
            
            models.append(model)
            
            valid_preds = model.predict_proba(X_valid)[:, 1]
            oof_preds[valid_index] = valid_preds
            
            fold_score = roc_auc_score(y_valid, valid_preds)
            fold_scores.append(fold_score)
        
        self.eda._plot_cv(fold_scores, title)
        self.validate_model(oof_preds, title)
        
        return models, oof_preds
        
    def infer_model(self, models):
        
        return np.mean([model.predict_proba(self.test_data)[:, 1] for model in models], axis=0)
        
    def train_and_infer_model(self, title):
        
        for col in self.cat_cols:
            for data in [self.train_data, self.test_data]:
                data[col] = data[col].astype('category')
        
        models, oof_preds = self.train_model(title)
        preds = self.infer_model(models)
        
        return oof_preds, preds


md = MD(
    CFG.colorscale, 
    CFG.color, 
    train_data, 
    test_data, 
    cat_cols,
    CFG.lgb_params,
    CFG.ctb_params, 
    CFG.xgb_params, 
    CFG.early_stop,
    CFG.n_splits
)


lgb_oof_preds, lgb_preds = md.train_and_infer_model('LightGBM')


ctb_oof_preds, ctb_preds = md.train_and_infer_model('CatBoost')


xgb_oof_preds, xgb_preds = md.train_and_infer_model('XGBoost')


ensemble_oof_preds = np.dot(CFG.weights, [lgb_oof_preds, ctb_oof_preds, xgb_oof_preds])


ensemble_preds = np.dot(CFG.weights, [lgb_preds, ctb_preds, xgb_preds])


md.validate_model(ensemble_oof_preds, 'Ensemble')


subm_data = pd.read_csv(CFG.subm_path)
subm_data['y'] = ensemble_preds


subm_data.to_csv('submission.csv', index=False)
display(subm_data.head())

