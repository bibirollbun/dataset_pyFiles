# Import the necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
import warnings

warnings.filterwarnings('ignore')


# Configuration class, to help reduce redundancy in upcoming code
class CFG:
    train_path = '/kaggle/input/playground-series-s5e4/train.csv'
    test_path = '/kaggle/input/playground-series-s5e4/test.csv'
    sub_path = '/kaggle/input/playground-series-s5e4/sample_submission.csv'
    original_path = '/kaggle/input/podcast-listening-time-prediction-dataset/podcast_dataset.csv'
    target = 'Listening_Time_minutes'
    idx = 'id'
    n_splits = 5
    seed = 99


# Read the train, test and original data csv files into pandas DataFrames
train = pd.read_csv(CFG.train_path, index_col=CFG.idx)
test = pd.read_csv(CFG.test_path, index_col=CFG.idx)
original = pd.read_csv(CFG.original_path)


# Display a sample of 10 data points from train
train.sample(10)


# Display a sample of 10 data points from test
test.sample(10)


# Display a sample of 10 data points from original
original.sample(10)


# ;et's have a look at the shapes of the three datasets
print(f'Shape of train data: {train.shape}')
print(f'Shape of original data: {original.shape}')
print(f'Shape of test data: {test.shape}')


train.head()


# Check out the data types
display(train.dtypes.reset_index().rename(
    columns={'index': 'column', 0: 'dtype'}
))
print()
display(test.dtypes.reset_index().rename(
    columns={'index': 'column', 0: 'dtype'}
))
print()
display(original.dtypes.reset_index().rename(
    columns={'index': 'column', 0: 'dtype'}
))


def preprocess(df):
    df_ = df.copy()
    df_['Episode_Number'] = df_['Episode_Title'].str.split(' ').str.get(1).astype(float).astype(int)
    df_ = df_.drop('Episode_Title', axis=1)
    
    return df_

train = preprocess(train)
test = preprocess(test)
original = preprocess(original)


features = [c for c in train.columns if c not in ['Podcast_Name', CFG.target]]
num_features = [f for f in features if train[f].dtype != 'object']
cat_features = [f for f in features if f not in num_features]


class AdversarialValidation:
    def __init__(self, train, test, original, features, cat_features, num_features, target, params=None, paradigm='train_v_test', seed=99):
        self.train = train.copy()
        self.test = test.copy()
        self.original = original.copy()
        self.features = features
        self.cat_features = cat_features
        self.target = target
        self.seed = seed
        self.params = params or {
            'learning_rate': 0.05, 
            'max_depth': 4, 
            'subsample': 0.9,
            'colsample_bytree': 0.9,
            'objective': 'binary:logistic',
            'n_estimators': 100, 
            'gamma': 1, 
            'min_child_weight': 4,
            'verbosity': 0, 
            'enable_categorical': True,
            'eval_metric': 'logloss', 
            'early_stopping_rounds': 10,
            'random_state': seed 
        }
        self.paradigm = 0  if paradigm == 'train_v_test' else 1

        if self.paradigm == 0:
            self.df1, self.df2 = self.train.copy(), self.test.copy()
        else:
            self.df1 = pd.concat([self.train, self.test], axis=0).sample(frac=1.0, random_state=self.seed)
            self.df2 = self.original.copy().drop(target, axis=1)

    def run(self):
        self.df1 = self.df1.drop(self.target, axis=1, errors='ignore')
        self.df1['cat_'] = 0
        self.df2['cat_'] = 1

        df = pd.concat([self.df1, self.df2], axis=0).sample(frac=1.0, random_state=self.seed)
        df_num = df[num_features+['cat_']]
        df_cat = df[self.cat_features].apply(lambda x: pd.factorize(x)[0])
        df = pd.concat([df_cat, df_num], axis=1)

        X = df.drop(columns=['cat_'])
        y = df['cat_']

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.seed)
        fold_scores = []

        for fold, (trn_idx, val_idx) in enumerate(skf.split(X, y), 1):
            X_train, X_val = X.iloc[trn_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[trn_idx], y.iloc[val_idx]

            model = XGBClassifier(**self.params)
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False
            )

            oof_preds = model.predict_proba(X_val)[:, 1]

            fold_score = roc_auc_score(y_val, oof_preds)
            fold_scores.append(fold_score)
            print(f'Fold {fold}: ROC-AUC score = {fold_score:4f}')

        print(f'\nAverage ROC-AUC score: {np.mean(fold_scores):.4f}Â±{np.std(fold_scores):.4f}')

        return fold_scores
        


av = AdversarialValidation(
    train=train, 
    test=test,
    original=original,
    features=features,
    cat_features=cat_features,
    num_features=num_features,
    target=CFG.target,
    paradigm='train_v_test',
    seed=CFG.seed
)
_ = av.run()


av = AdversarialValidation(
    train=train, 
    test=test,
    original=original,
    features=features,
    cat_features=cat_features,
    num_features=num_features,
    target=CFG.target,
    paradigm='comp_v_original',
    seed=CFG.seed
)
_ = av.run()




