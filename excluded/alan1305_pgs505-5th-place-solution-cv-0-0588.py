%%time
import pandas as pd
import polars as pl
import numpy as np
import gc
import warnings
warnings.filterwarnings(action='ignore')

TARGET = 'Calories'

source = '/kaggle/input/pgs505-public/'


def load_data(name, columns=None):
    X = pl.scan_csv(f'{source}{name}_predictions.csv')
    if columns is None:
        X = X.select(pl.all().cast(pl.Float32)).collect()
    else:
        X = X.select(pl.col(col).cast(pl.Float32) for col in columns).collect()
    y_train = pd.read_csv(f'{source}y_true.csv')[TARGET].values.flatten()
    
    train_size = y_train.size
    X_train = X[:train_size]
    X_test = X[train_size:]
    gc.collect()
    return X_train, y_train, X_test


columns = None
name = 'final'
X_train, y_train, X_test = load_data(name, columns)
print('Number of models:', X_train.shape[1])


# Simple demo of each type of OOFs

# The column name refers to `{model}_{features_selected_by_model}__{hyperparameters}`
# For NN, they are all 4-layers MLP with (neurons, activation, dropout rate) settings
# Note that the features are not provided and therefore the result is not reproducible under same HP
[X_train.columns[i] for i in [0,1,3,7,15,18,28]]


import cvxpy
import numpy as np
import pickle

class HillClimber:
    def __init__(self, 
                 min_weight=0.
                ):
        self.min_weight = min_weight
        
    def fit(self, X, y, models: list[str]=None):
        '''
        X should be out of fold predictions
        y should be true target
        '''
        if models is None:
            if hasattr(X, 'columns'):
                self.models = np.array(list(X.columns))
            else:
                self.models = np.arange(X.shape[1])
        else:
            self.models = np.array(models)
        X = np.asarray(X)
        y = np.asarray(y).flatten()

        self.weights_ = self._hill_climb(X, y)
        
        self.models_ = self.models[self.weights_>0]
        self.model_weights_ = self.weights_[self.weights_>0]
        return self
        
            
    def _hill_climb(self, X, y):
        W = cvxpy.Variable(X.shape[1])
        # Objective
        objective = cvxpy.Minimize(cvxpy.sum_squares(y - X @ W))

        # Constraint: sum of W equals 1 and W>=min_weight
        constraints = [
            cvxpy.sum(W) == 1,
            W >= self.min_weight,
        ]
        # Problem definition and solve
        prob = cvxpy.Problem(objective, constraints)
        prob.solve()
        weights = W.value
        if weights is None:
            raise RuntimeError('No solution is found, check your inputs')
        # Clip negative weights due to precision issues
        return np.maximum(self.min_weight, weights)

    def predict(self, X):
        X = np.asarray(X)
        mask = np.where(self.weights_>0)[0]
        return X[:, mask]@self.weights_[mask]

    def save(self, name):
        '''
        This method saves the result as a dict containing 3 keys
        
        weights: all weights, including models(OOFs) that are not chosen
        model_weights: weights of the selected models(OOFs) only
        models: selected models(OOFs)
        '''
        result = {}
        result['weights'] = self.weights_
        result['model_weights'] = self.model_weights_
        result['models'] = self.models_
        with open(f'{name}.pkl', 'wb') as f:
            pickle.dump(result, f)


%%time
model = HillClimber(min_weight=0.)
model.fit(X_train, np.log1p(y_train))


print('Number of selected models: %d' %model.models_.size)


def rmse(true, pred):
    return np.sqrt(np.mean(np.square(true-pred)))
oof = model.predict(X_train)
test_pred = model.predict(X_test)
print('OOF Score:', rmse(np.log1p(y_train), oof))


sub = pd.read_csv('/kaggle/input/playground-series-s5e5/sample_submission.csv')
sub[TARGET] = np.clip(np.expm1(test_pred), y_train.min(), y_train.max())
sub.to_csv('submission.csv', index=False)


sub[TARGET].describe()


sub[TARGET].hist(bins=100)


df = {}
df['weight'] = model.weights_
df['CV'] = [rmse(np.log1p(y_train), np.asarray(X_train[model]) ) for model in X_train.columns]
df['model'] = [x.split('_')[0] for x in X_train.columns]

df = pd.DataFrame(df)


# weights are not highly correlated with CV scores.
# Diversity is more important than solo CV score.
df[['weight', 'CV']].corr()


# Better CV does not mean larger weight
df.sort_values('CV', ascending=True).head(10)


# The most contribution comes from NN!
df.sort_values('weight', ascending=False).head(10)


# Model counts
df['model'][df['weight']>0].value_counts()


# NN contributes the most!
df.groupby('model')['weight'].sum().sort_values(ascending=False)

