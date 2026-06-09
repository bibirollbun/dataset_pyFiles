import re
import warnings
from pathlib import Path
warnings.filterwarnings('ignore')


import numpy as np
import polars as pl
import pandas as pd
import plotly.colors as pc
import plotly.express as px
from scipy.sparse import hstack
import plotly.graph_objects as go


import plotly.io as pio
pio.renderers.default = 'iframe'


import lightgbm as lgb
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer


class CFG:
    
    train_path = Path('/kaggle/input/jigsaw-agile-community-rules/train.csv')
    test_path = Path('/kaggle/input/jigsaw-agile-community-rules/test.csv')
    subm_path = Path('/kaggle/input/jigsaw-agile-community-rules/sample_submission.csv')
    
    color = '#A2574F'
    
    early_stop = 100
    n_splits = 5
    
    lgb_params = {
        'min_child_samples': 32,
        'num_iterations': 800,
        'learning_rate': 0.03,
        'objective': 'binary',
        'extra_trees': True,
        'reg_lambda': 4.0,
        'reg_alpha': 0.1,
        'num_leaves': 32,
        'max_depth': 4,
        'device': 'cpu',
        'max_bin': 64,
        'verbose': -1,
        'seed': 42
    }


class FE:
    
    def __init__(self):
        self._batch_size = 65536
        self._whitespace_re = re.compile(r'\s+')        
        self._lemmatizer = WordNetLemmatizer()
        
        self._word_ngrams_range = (1, 3)
        self._char_ngrams_range = (4, 6)
        self._char_wb_ngrams_range = (3, 4)
        
        self._word_max_features = 1024
        self._char_max_features = 512
        self._char_wb_max_features = 512
        
        self._min_df = 2
        self._max_df = 0.95
        self._lowercase = True
        self._sublinear_tf = True
        self._stop_words = 'english'
        
        self._word_vectorizer = None
        self._char_vectorizer = None
        self._char_wb_vectorizer = None
        
        self._word_features = []
        self._char_features = []
        self._char_wb_features = []
    
    def _load_data(self, path):
        
        return pl.read_csv(path, batch_size=self._batch_size)
    
    def _define_corpus(self, row):
        
        return (
            f'body: {row.body}\n'
            f'rule: {row.rule}\n'
            f'subreddit: {row.subreddit}\n'
            f'positive_example_1: {row.positive_example_1}\n'
            f'positive_example_2: {row.positive_example_2}\n'
            f'negative_example_1: {row.negative_example_1}\n'
            f'negative_example_2: {row.negative_example_2}'
        )
    
    def _merge_corpora(self, df):
        
        df['Corpus'] = df.apply(self._define_corpus, axis=1)
        df['Corpus'] = df['Corpus'].str.replace(self._whitespace_re, ' ', regex=True).str.strip()
        
        return df[['row_id', 'Corpus', 'rule_violation']] if 'rule_violation' in df.columns else df[['row_id', 'Corpus']]
    
    def _lemma_tokenizer(self, text):
        
        tokens = word_tokenize(text)
        
        return [self._lemmatizer.lemmatize(t) for t in tokens]
    
    def _to_tfidf(self, df):
        
        corpus = df['Corpus'].fillna('')
        
        if self._word_vectorizer is None:            
            self._word_vectorizer = TfidfVectorizer(
                analyzer='word',
                tokenizer=self._lemma_tokenizer,
                ngram_range=self._word_ngrams_range,
                max_features=self._word_max_features,
                min_df=self._min_df,
                max_df=self._max_df,
                lowercase=self._lowercase,
                stop_words=self._stop_words,
                sublinear_tf=self._sublinear_tf
            )
            
            X_word = self._word_vectorizer.fit_transform(corpus)
            self._word_features = [f'tfidf_word_{i}' for i in range(X_word.shape[1])]
            
            self._char_vectorizer = TfidfVectorizer(
                analyzer='char',
                tokenizer=self._lemma_tokenizer,
                ngram_range=self._char_ngrams_range,
                max_features=self._char_max_features,
                min_df=self._min_df,
                max_df=self._max_df,
                lowercase=self._lowercase,
                stop_words=self._stop_words,
                sublinear_tf=self._sublinear_tf
            )
            
            X_char = self._char_vectorizer.fit_transform(corpus)
            self._char_features = [f'tfidf_char_{i}' for i in range(X_char.shape[1])]
            
            self._char_wb_vectorizer = TfidfVectorizer(
                analyzer='char_wb',
                tokenizer=self._lemma_tokenizer,
                ngram_range=self._char_wb_ngrams_range,
                max_features=self._char_wb_max_features,
                min_df=self._min_df,
                max_df=self._max_df,
                lowercase=self._lowercase,
                stop_words=self._stop_words,
                sublinear_tf=self._sublinear_tf
            )
            
            X_char_wb = self._char_wb_vectorizer.fit_transform(corpus)
            self._char_wb_features = [f'tfidf_char_wb_{i}' for i in range(X_char_wb.shape[1])]
            
        else:
            X_word = self._word_vectorizer.transform(corpus)
            X_char = self._char_vectorizer.transform(corpus)
            X_char_wb = self._char_wb_vectorizer.transform(corpus)
        
        X = hstack([X_word, X_char, X_char_wb], format='csr')
        
        features = self._word_features + self._char_features + self._char_wb_features
        data = pd.DataFrame.sparse.from_spmatrix(X, index=df.index, columns=features)
        
        return pd.concat([df.drop('Corpus', axis=1), data], axis=1)
    
    def info(self, df):
        
        print(f'\nShape of dataframe: {df.shape}') 
        
        mem = df.memory_usage().sum() / 1024**2
        print('Memory usage: {:.2f} MB\n'.format(mem))
    
    def apply_fe(self, path):
        
        df = self._load_data(path)     
        df = pd.DataFrame(df.to_dicts())
        df = self._merge_corpora(df)
        df = self._to_tfidf(df)
        self.info(df)
        
        return df


fe = FE()


train_data = fe.apply_fe(CFG.train_path)


test_data = fe.apply_fe(CFG.test_path)


class EDA:
    
    def __init__(self, color, data):
        self._color = color  
        self.data = data
        
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


class MD:
    
    def __init__(
        self,
        color,
        train_data,
        test_data,
        lgb_params,
        early_stop,
        n_splits
    ):
        
        self.eda = EDA(color, train_data)
        
        self.train_data = train_data
        self.test_data = test_data   
        
        self._length = len(self.train_data)
        
        self._lgb_params = lgb_params        
        self._early_stop = early_stop
        self._n_splits = n_splits
    
    def _prepare_cv(self):
        
        oof_preds = np.zeros(self._length)
        
        cv = StratifiedKFold(n_splits=self._n_splits, shuffle=True, random_state=42)
        
        return cv, oof_preds
    
    def validate_model(self, y_true, y_pred, title):
        
        score = roc_auc_score(y_true, y_pred)
        print(f'AUC score for {title}: {score:.5f}')
        
    def _train_model(self, title):
        
        X = self.train_data.drop(['row_id', 'rule_violation'], axis=1)
        y = self.train_data['rule_violation']
        
        models, fold_scores = [], []
            
        cv, oof_preds = self._prepare_cv()
    
        for fold, (train_index, valid_index) in enumerate(cv.split(X, y)):
                
            X_train = X.iloc[train_index]
            X_valid = X.iloc[valid_index]
                
            y_train = y.iloc[train_index]
            y_valid = y.iloc[valid_index]
            
            model = lgb.LGBMClassifier(**self._lgb_params)                        
            model.fit(
                X_train, 
                y_train,  
                eval_set=[(X_valid, y_valid)],
                eval_metric='binary',
                callbacks=[lgb.early_stopping(self._early_stop, verbose=0), lgb.log_evaluation(0)]
            )           
                    
            models.append(model)
            
            valid_preds = model.predict_proba(X_valid)[:, 1]  
            oof_preds[valid_index] = valid_preds
            
            fold_score = roc_auc_score(y_valid, valid_preds)
            fold_scores.append(fold_score)
        
        self.eda._plot_cv(fold_scores, title)
        self.validate_model(y, oof_preds, title)
        
        return models, oof_preds
    
    def _infer_model(self, models):
        
        data = self.test_data.drop('row_id', axis=1)
        
        return np.mean([model.predict_proba(data)[:, 1] for model in models], axis=0)
    
    def train_and_infer_model(self, title):
        
        models, oof_preds = self._train_model(title)
        preds = self._infer_model(models)
        
        return oof_preds, preds


md = MD(
    CFG.color,
    train_data,
    test_data,
    CFG.lgb_params,
    CFG.early_stop,
    CFG.n_splits
)


oof_preds, preds = md.train_and_infer_model('LightGBM')


subm_data = pd.read_csv(CFG.subm_path)
subm_data['rule_violation'] = preds


subm_data.to_csv('submission.csv', index=False)
display(subm_data.head())

