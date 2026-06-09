# install the package
!pip install autogluon


import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
SEED = 42
TRAIN_PATH = '/kaggle/input/playground-series-s5e4/train.csv'
TEST_PATH = '/kaggle/input/playground-series-s5e4/test.csv'
SUB_PATH = '/kaggle/input/playground-series-s5e4/sample_submission.csv'

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)
sub_df = pd.read_csv(SUB_PATH)
train_df.head()


train_df = train_df.drop(columns=['id'])
test_id = test_df['id']
test_df = test_df.drop(columns=['id'])
train_df.info()


test_df.info()


from sklearn.model_selection import train_test_split

train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=SEED)
train_df.shape, val_df.shape, test_df.shape


from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from category_encoders import CatBoostEncoder
from sklearn.base import BaseEstimator, TransformerMixin

train_y = train_df['Listening_Time_minutes']
train_X = train_df.drop(columns=['Listening_Time_minutes'])
val_X = val_df.drop(columns=['Listening_Time_minutes'])
val_y = val_df['Listening_Time_minutes']
numeric_features = train_X.select_dtypes(include=['int64', 'float64']).columns.tolist()
catgorical_features = train_X.select_dtypes(include=['object']).columns.tolist()


class FeatureEngineeringTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
    
        
        X['Is_Weekend'] = X['Publication_Day'].apply(lambda x: 1 if x in ['Saturday', 'Sunday'] else 0)
        X['Daypart'] = X['Publication_Time'].map({'Morning': 0, 'Afternoon': 1, 'Evening': 2, 'Night': 3})    
        X['Host_Guest_Popularity_Ratio'] = X['Host_Popularity_percentage'] / (X['Guest_Popularity_percentage'] + 1e-5) 
        X['Ads_Per_Minute'] = X['Number_of_Ads'] / (X['Episode_Length_minutes'] + 1e-5)
        X['Popularity_Score'] = (X['Host_Popularity_percentage'] + X['Guest_Popularity_percentage']) / 2  
        X['Long_Episode'] = (X['Episode_Length_minutes'] > 75).astype(int)
        X['Highly_Popular_Host'] = (X['Host_Popularity_percentage'] > 75).astype(int)
        X['Highly_Popular_Guest'] = (X['Guest_Popularity_percentage'] > 75).astype(int)
        X['Host_Guest_Popularity_Diff'] = X['Host_Popularity_percentage'] - X['Guest_Popularity_percentage']
        X['Host_Guest_Popularity_Sum'] = X['Host_Popularity_percentage'] + X['Guest_Popularity_percentage']
        X['Ad_Impact'] = X['Number_of_Ads'] * X['Episode_Length_minutes']
        X['Episode_Length_Bin'] = pd.cut(X['Episode_Length_minutes'],
                                  bins=[-1, 187500, 375000, 562500, np.inf],
                                  labels=[0, 1, 2, 3])  
        X['Episode_Length_Bin'] = X['Episode_Length_Bin'].astype(int)
        X['High_Ad_Load'] = (X['Number_of_Ads'] > 2).astype(int)
        
        return X
    


train_X[numeric_features] = train_X[numeric_features].fillna(train_X[numeric_features].median())
val_X[numeric_features] = val_X[numeric_features].fillna(val_X[numeric_features].median())
test_df[numeric_features] = test_df[numeric_features].fillna(test_df[numeric_features].median())


fe_transformer = FeatureEngineeringTransformer()
train_X_with_features = fe_transformer.fit_transform(train_X)
val_X_with_features = fe_transformer.transform(val_X)
test_X_with_features = fe_transformer.transform(test_df)

all_numeric_features = train_X_with_features.select_dtypes(include=['int64', 'float64']).columns.tolist()
all_categorical_features = train_X_with_features.select_dtypes(include=['object']).columns.tolist()


preprocessor = ColumnTransformer(transformers=[
    ('num', 'passthrough', all_numeric_features),
    ('cat', OrdinalEncoder(), all_categorical_features),
    
])
train_X_final = preprocessor.fit_transform(train_X_with_features)
val_X_final = preprocessor.transform(val_X_with_features)
test_X_final = preprocessor.transform(test_X_with_features)

feature_names = all_numeric_features + all_categorical_features

train_X_final = pd.DataFrame(train_X_final, columns=feature_names)
val_X_final = pd.DataFrame(val_X_final, columns=feature_names)
test_X_final = pd.DataFrame(test_X_final, columns=feature_names)

train_X_final.head()


train_X_final.info()


from autogluon.tabular import TabularDataset, TabularPredictor
from autogluon.features.generators import AutoMLPipelineFeatureGenerator
train_data = train_X_final

train_data['target'] = train_y.values
train_data = TabularDataset(train_data)
test_data = TabularDataset(test_X_final)


predictor = TabularPredictor(label='target').fit(
    train_data=train_data,
    presets='medium_quality',
    time_limit=1800
)


# get the best model string
predictor.model_best


import optuna
import xgboost as xgb
from sklearn.metrics import mean_squared_error
import numpy as np
import warnings
warnings.filterwarnings('always')

val_data = TabularDataset(val_X_final)
val_data['target'] = val_y.values

TRIALS = 100
autogluon_preds = predictor.predict(val_data)

def objective(trial):
    
    params = {
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    'booster': trial.suggest_categorical('booster', ['gbtree', 'dart', 'gblinear']),

    'lambda': trial.suggest_float('lambda', 1e-3, 10.0, log=True),
    'alpha': trial.suggest_float('alpha', 1e-3, 10.0, log=True),

    'max_depth': trial.suggest_int('max_depth', 5, 12),

    'eta': trial.suggest_float('eta', 0.005, 0.2, log=True),
    'gamma': trial.suggest_float('gamma', 1e-5, 5.0, log=True),
    'grow_policy': trial.suggest_categorical('grow_policy', ['depthwise', 'lossguide']),

    'tree_method': trial.suggest_categorical('tree_method', ['auto', 'exact', 'approx', 'hist']),

    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
    'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.6, 1.0),
    'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),

    'xgb_weight': trial.suggest_float('xgb_weight', 0.05, 0.95),

    'num_boost_round': trial.suggest_int('num_boost_round', 100, 1500)
    }
    

    dtrain = xgb.DMatrix(train_X_final.values, label=train_y.values)
    dval = xgb.DMatrix(val_X_final.values, label=val_y.values)
    
    model = xgb.train(
        {k: v for k, v in params.items() if k != 'xgb_weight'},  
        dtrain,
        verbose_eval=False
    )
    
    xgb_preds = model.predict(dval)

    xgb_weight = params['xgb_weight']
    autogluon_weight = 1 - xgb_weight
    
    ensemble_preds = (xgb_weight * xgb_preds) + (autogluon_weight * autogluon_preds.values)

    rmse = np.sqrt(mean_squared_error(val_y.values, ensemble_preds))
    
    return rmse


study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=TRIALS)  


print("Best trial:")
trial = study.best_trial
print(f"  Value (RMSE): {trial.value:.4f}")
print("  Params: ")
for key, value in trial.params.items():
    print(f"    {key}: {value}")


best_params = {k: v for k, v in study.best_params.items() if k != 'xgb_weight'}
best_params['objective'] = 'reg:squarederror'
best_params['eval_metric'] = 'rmse'

xgb_weight = study.best_params['xgb_weight']
autogluon_weight = 1 - xgb_weight
print(f"Ensemble weights: XGBoost={xgb_weight:.4f}, AutoGluon={autogluon_weight:.4f}")


dtrain = xgb.DMatrix(train_X_final.values, label=train_y.values)
dval = xgb.DMatrix(val_X_final.values, label=val_y.values)

final_model = xgb.train(
    best_params,
    dtrain
)


xgb_val_preds = final_model.predict(dval)
ensemble_val_preds = (xgb_weight * xgb_val_preds) + (autogluon_weight * autogluon_preds.values)
ensemble_val_rmse = np.sqrt(mean_squared_error(val_y.values, ensemble_val_preds))
print(f"Ensemble validation RMSE: {ensemble_val_rmse:.4f}")


dtest = xgb.DMatrix(test_X_final.values)
xgb_test_preds = final_model.predict(dtest)
autogluon_test_preds = predictor.predict(test_data)
ensemble_test_preds = (xgb_weight * xgb_test_preds) + (autogluon_weight * autogluon_test_preds.values)

ensemble_test_preds.shape


sub_df['Listening_Time_minutes'] = ensemble_test_preds
sub_df.to_csv('submission.csv', index=False)
sub_df.shape

