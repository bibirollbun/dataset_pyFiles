import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np

class MathFeatureCreator(BaseEstimator, TransformerMixin):
    def __init__(self, variables=None, operations=None):
        self.variables = variables
        self.operations = operations or ['add','sub','mul', 'div']
    
    def fit(self, X, y=None):
        return self  # Nothing to fit
    
    def transform(self, X):
        X = X.copy()
        new_features = []  # This will hold the new columns to concat at once

        for i in range(len(self.variables)):
            for j in range(i + 1, len(self.variables)):
                var1 = self.variables[i]
                var2 = self.variables[j]
                
                if 'add' in self.operations:
                    new_features.append(X[var1] + X[var2])
                if 'sub' in self.operations:
                    new_features.append(X[var1] - X[var2])
                if 'mul' in self.operations:
                    new_features.append(X[var1] * X[var2])
                if 'div' in self.operations:
                    new_features.append(X[var1] / X[var2].replace(0, np.nan))  # Handle div by 0
                

        # Concatenate all new features at once
        new_features_df = pd.concat(new_features, axis=1)
        
        # Rename columns
        new_feature_names = [
            f'{self.variables[i]}_{op}_{self.variables[j]}'
            for i in range(len(self.variables))
            for j in range(i + 1, len(self.variables))
            for op in self.operations
        ]
        new_features_df.columns = new_feature_names
        
        # Concatenate the original dataframe with the new features
        X = pd.concat([X, new_features_df], axis=1)
        
        return X



Train=pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv')
Test=pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')


def features(df):
    
    # Encode sex as binary (male = 1, female = 0)
    df['Sex'] = df['Sex'].apply(lambda x: 1 if x == 'male' else 0)
    return df
Train=features(Train)
Test=features(Test)


from sklearn.model_selection import KFold
from xgboost import XGBRegressor
FOLDS = 5
kf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)


FEATURES = [col for col in Train.columns if col != 'Calories']
trf_features = [feat for feat in FEATURES if feat != 'id']


trf_features


math_f=MathFeatureCreator(variables=trf_features)


train=math_f.fit_transform(Train)
test=math_f.fit_transform(Test)


FEATURES = [col for col in train.columns if col != 'Calories']


FEATURES


import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_log_error
from sklearn.model_selection import KFold

# Constants
FOLDS = 5
oof_xgb = np.zeros(len(train))
pred_xgb = np.zeros(len(test))
oof_dfs = []  # collect per-fold validation predictions here

# Custom RMSLE eval function for XGBoost
def rmsle_xgb(y_pred, dtrain):
    y_true = dtrain.get_label()
    rmsle = np.sqrt(mean_squared_log_error(y_true, np.maximum(0, y_pred)))
    return 'RMSLE', rmsle

kf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)

for i, (train_index, val_index) in enumerate(kf.split(train)):
    print("#" * 25)
    print(f"### Fold {i+1}")
    print("#" * 25)

    X_train = train.loc[train_index, FEATURES]
    y_train_fold = train.loc[train_index, 'Calories']
    X_val = train.loc[val_index, FEATURES]
    y_val = train.loc[val_index, 'Calories']
    X_test = test[FEATURES]

    dtrain = xgb.DMatrix(X_train, label=y_train_fold)
    dval = xgb.DMatrix(X_val, label=y_val)
    dtest = xgb.DMatrix(X_test)

    params = {
        'objective': 'reg:squarederror',
        'learning_rate': 0.01,
        'max_depth': 10,
        'device': 'cuda',  # Enable GPU
        'subsample': 0.8,
        'colsample_bytree': 0.6,
        
        
    }

    model_xgb = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=10000,
        evals=[(dval, 'validation')],
        early_stopping_rounds=25,
        custom_metric=rmsle_xgb,
        verbose_eval=2000
    )

    # Validation predictions
    val_preds = model_xgb.predict(dval)
    oof_xgb[val_index] = val_preds

    # Test predictions
    pred_xgb += model_xgb.predict(dtest)

    # Save index and predictions for this fold
    fold_df = pd.DataFrame({
        'index': val_index,
        'oof_pred': val_preds
    })
    oof_dfs.append(fold_df)

# Average test predictions
pred_xgb /= FOLDS

# Concatenate OOF predictions and sort
oof_df = pd.concat(oof_dfs).sort_values(by='index').reset_index(drop=True)

# Add true target values
oof_df['true'] = train['Calories'].values

# Final RMSLE
final_rmsle = np.sqrt(mean_squared_log_error(oof_df['true'], np.maximum(0, oof_df['oof_pred'])))
print(f"\nOverall RMSLE: {final_rmsle:.5f}")



oof_df.to_csv("xgb_oof_0.06025.csv", index=False)


sub = pd.read_csv("/kaggle/input/playground-series-s5e5/sample_submission.csv")
sub.Calories = pred_xgb
sub.to_csv("xgb_pred_0.06025.csv",index=False)
print("Sub shape:",sub.shape)
sub.head()




