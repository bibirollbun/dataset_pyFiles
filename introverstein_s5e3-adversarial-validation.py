# Let's import the libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier


class CFG:
    train_path = '/kaggle/input/playground-series-s5e3/train.csv'
    test_path = '/kaggle/input/playground-series-s5e3/test.csv'
    original_path = '/kaggle/input/rainfall-prediction-using-machine-learning/Rainfall.csv'
    target = 'rainfall'
    idx = 'id'
    seed = 6543


# Read the data
train = pd.read_csv(CFG.train_path, index_col=CFG.idx)
original = pd.read_csv(CFG.original_path)
test = pd.read_csv(CFG.test_path, index_col=CFG.idx)


# Check out the datatypes
display(train.dtypes.reset_index().rename(
    columns={'index': 'column', 0: 'dtype'}
))
print()
display(test.dtypes.reset_index().rename(
    columns={'index': 'columns', 0: 'dtype'}
))


# Let's assign categorical and numerical features
features = [f for f in train.columns if f != CFG.target]
cat_features = ['day']
num_features = [f for f in features if f not in cat_features]


# Sanity check
len(cat_features + num_features) == len(features), train.shape[1] - 1 == len(features)


class AdversarialValidation:
    def __init__(self, train, test, original, features, cat_features, target, params=None, paradigm='train_v_test', seed=55):
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

        self.paradigm = 0 if paradigm == 'train_v_test' else 1

        if self.paradigm == 0:
            self.df1, self.df2 = self.train.copy(), self.test.copy()
        else:
            self.df1 = pd.concat([self.train, self.test], axis=0).sample(frac=1.0, random_state=self.seed)
            self.df2 = self.original.copy().drop(target, axis=1)

    def run(self):
        self.df1 = self.df1.drop(columns=[self.target], errors='ignore')
        self.df1['cat_'] = 0
        self.df2['cat_'] = 1

        df = pd.concat([self.df1, self.df2], axis=0).sample(frac=1.0, random_state=self.seed)

        num_features = [f for f in self.features if f not in self.cat_features]
        df_num = df[num_features+['cat_']]
        df_cat = df[self.cat_features].apply(lambda x: pd.factorize(x)[0])
        df = pd.concat([df_cat, df_num], axis=1)

        X = df.drop(columns=['cat_'])
        y = df['cat_']

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.seed)
        scores = []

        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            model = XGBClassifier(**self.params)
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False
            )

            oof_preds = model.predict_proba(X_val)[:, 1]

            score = roc_auc_score(y_val, oof_preds)
            scores.append(score)
            print(f'Fold {fold}: ROC-AUC score = {score:.5f}')

        print(f'\nAverage ROC-AUC score: {np.mean(scores):.5f} Â± {np.std(scores):.5f}')
        
        return scores


av = AdversarialValidation(
    train, 
    test,
    original,
    features,
    cat_features,
    CFG.target,
    paradigm='train_v_test'
)
_ = av.run()


av = AdversarialValidation(
    train, 
    test,
    original,
    features,
    cat_features,
    CFG.target,
    paradigm='comp_v_original'
)
_ = av.run()




