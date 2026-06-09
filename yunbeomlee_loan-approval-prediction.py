!pip install catboost lightgbm xgboost seaborn cmaes optuna shap torch tqdm ipywidgets


import sys

import matplotlib.pyplot as plt
import pandas as pd
from joblib import dump, load
from sklearn.metrics import roc_auc_score
from tqdm.auto import tqdm


train = pd.read_csv(f'/kaggle/input/playground-series-s4e10/train.csv', index_col = 0)
test = pd.read_csv(f'/kaggle/input/playground-series-s4e10/test.csv', index_col = 0)


train.shape


train.info()


train.describe()


train.isnull().sum().sort_values


target_counts = train['loan_status'].value_counts(normalize=True) * 100
target_counts


import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize =(6,4))
sns.countplot(x='loan_status', data = train)
plt.show()


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import OrdinalEncoder

numeric_df = train.select_dtypes(include=['number'])
categorical_df = train.select_dtypes(include=['object'])

encoder = OrdinalEncoder()
categorical_encoded = pd.DataFrame(encoder.fit_transform(categorical_df), columns=categorical_df.columns)

full_df = pd.concat([numeric_df, categorical_encoded], axis=1)

correlation_matrix = full_df.corr()

plt.figure(figsize=(14, 12))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix (Including Encoded Categorical Features)")
plt.show()


train = pd.read_csv(f'/kaggle/input/playground-series-s4e10/train.csv', index_col = 0)
train['original'] = False
test = pd.read_csv(f'/kaggle/input/playground-series-s4e10/test.csv', index_col = 0)
test['original'] = False
original = pd.read_csv(f'/kaggle/input/loan-approval-prediction/credit_risk_dataset.csv')
original['original'] = True

target = 'loan_status'

train


import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.base import BaseEstimator, RegressorMixin

class EnsembleRegressor(BaseEstimator, RegressorMixin):
    def __init__(self):
        self.rf = RandomForestRegressor(n_estimators=300, random_state=0, n_jobs=-1)
        self.catboost = CatBoostRegressor(iterations=300, random_state=0, verbose=False, thread_count=-1)
        self.lightgbm = LGBMRegressor(n_estimators=300, random_state=0, n_jobs=-1, verbosity=-1)
        self.xgboost = XGBRegressor(n_estimators=300, random_state=0, n_jobs=-1)
        self.models = [self.rf, self.catboost, self.lightgbm, self.xgboost]

    def fit(self, X, y):
        for model in self.models:
            model.fit(X, y)
        return self

    def predict(self, X):
        predictions = []
        for model in self.models:
            predictions.append(model.predict(X))
        return np.mean(predictions, axis=0)


columns_to_impute = [col for col in original.columns if col not in ['loan_status', 'original']]

numeric_columns = original[columns_to_impute].select_dtypes(include=['int64', 'float64']).columns
categorical_columns = original[columns_to_impute].select_dtypes(include=['object']).columns

df_imputed = original[columns_to_impute].copy()

category_mappings = {}
for col in categorical_columns:
    category_mappings[col] = {v: k for k, v in enumerate(df_imputed[col].unique())}
    df_imputed[col] = df_imputed[col].map(category_mappings[col])

imputer = IterativeImputer(estimator=EnsembleRegressor(), max_iter=10, random_state=0)
imputed_data = imputer.fit_transform(df_imputed)

df_result = pd.DataFrame(imputed_data, columns=df_imputed.columns)

for col in categorical_columns:
    reverse_mapping = {v: k for k, v in category_mappings[col].items()}
    df_result[col] = df_result[col].round().map(reverse_mapping)

df_result['loan_status'] = original['loan_status']
df_result['original'] = original['original']

original = df_result.copy()

original.info()


train = pd.concat([train] + [original], axis=0)
train


import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures

def create_features(train, test):
    for df in [train, test]:
        df['income_to_age'] = df['person_income'] / df['person_age']
        df['loan_to_income'] = df['loan_amnt'] / df['person_income']
        df['rate_to_loan'] = df['loan_int_rate'] / df['loan_amnt']
        df['age_squared'] = df['person_age'] ** 2
        df['log_income'] = np.log1p(df['person_income'])
        df['age_credit_history_interaction'] = df['person_age'] * df['cb_person_cred_hist_length']
        df['age_category'] = pd.cut(df['person_age'], bins=[0, 25, 35, 45, 55, 100], labels=['Very Young', 'Young', 'Middle', 'Senior', 'Elder'])
        df['income_category'] = pd.qcut(df['person_income'], q=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
        df['high_loan_to_income'] = (df['loan_percent_income'] > 0.5).astype(int)
        df['intent_grade_interaction'] = df['loan_intent'].astype(str) + '_' + df['loan_grade'].astype(str)
        df['loan_to_employment'] = df['loan_amnt'] / (df['person_emp_length'] + 1)
        df['is_new_credit_user'] = (df['cb_person_cred_hist_length'] < 2).astype(int)
        df['home_ownership_intent'] = df['person_home_ownership'].astype(str) + '_' + df['loan_intent'].astype(str)
        df['rate_to_grade'] = df.groupby('loan_grade')['loan_int_rate'].transform('mean')
        df['high_interest_rate'] = (df['loan_int_rate'] > df['loan_int_rate'].mean()).astype(int)
        df['risk_score'] = df['loan_percent_income'] * df['loan_int_rate'] * (5 - df['loan_grade'].map({'A':5, 'B':4, 'C':3, 'D':2, 'E':1, 'F':0, 'G':0}))
        df['age_to_credit_history'] = df['person_age'] / (df['cb_person_cred_hist_length'] + 1)
        df['income_home_mismatch'] = ((df['person_income'] > df['person_income'].quantile(0.8)) & (df['person_home_ownership'] == 'RENT')).astype(int)
        df['default_grade_interaction'] = df['cb_person_default_on_file'].astype(str) + '_' + df['loan_grade'].astype(str)
        df['normalized_loan_amount'] = df.groupby('loan_intent')['loan_amnt'].transform(lambda x: (x - x.mean()) / x.std())
        df['income_to_loan'] = df['person_income'] / df['loan_amnt']
        df['age_cubed'] = df['person_age'] ** 3
        df['log_loan_amnt'] = np.log1p(df['loan_amnt'])
        df['age_interest_interaction'] = df['person_age'] * df['loan_int_rate']
        df['loan_amount_category'] = pd.qcut(df['loan_amnt'], q=5, labels=['Very Small', 'Small', 'Medium', 'Large', 'Very Large'])
        df['credit_history_to_age'] = df['cb_person_cred_hist_length'] / df['person_age']
        df['high_loan_amount'] = (df['loan_amnt'] > df['loan_amnt'].quantile(0.75)).astype(int)
        df['home_ownership_loan_interaction'] = df['person_home_ownership'].astype(str) + '_' + df['loan_amount_category'].astype(str)
        df['rate_to_credit_history'] = df['loan_int_rate'] / (df['cb_person_cred_hist_length'] + 1)
        df['intent_home_match'] = ((df['loan_intent'] == 'HOMEIMPROVEMENT') & (df['person_home_ownership'] == 'OWN')).astype(int)
        df['creditworthiness_score'] = (df['person_income'] / (df['loan_amnt'] * df['loan_int_rate'])) * (df['cb_person_cred_hist_length'] + 1)
        df['age_to_employment'] = df['person_age'] / (df['person_emp_length'] + 1)
        df['age_income_mismatch'] = ((df['person_age'] < 30) & (df['person_income'] > df['person_income'].quantile(0.9))).astype(int)
        df['default_rate_interaction'] = df['cb_person_default_on_file'].astype(str) + '_' + pd.cut(df['loan_int_rate'], bins=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High']).astype(str)
        df['normalized_income'] = df.groupby('age_category')['person_income'].transform(lambda x: (x - x.mean()) / x.std())
        df['rate_to_age'] = df['loan_int_rate'] / df['person_age']
        df['high_risk_flag'] = ((df['loan_percent_income'] > 0.4) &
                                (df['loan_int_rate'] > df['loan_int_rate'].mean()) &
                                (df['cb_person_default_on_file'] == 'Y')).astype(int)

        num_features = ['person_age', 'person_income', 'person_emp_length', 'loan_amnt', 'loan_int_rate', 'loan_percent_income']
        poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)
        df_poly_input = df[num_features].fillna(0)
        # Input X contains NaN 에러가 나와 결측치 있는 행을 제거
        poly_features = poly.fit_transform(df_poly_input)

        try:
            poly_features_names = poly.get_feature_names_out(num_features)
        except AttributeError:
            poly_features_names = poly.get_feature_names(num_features)

        for i, name in enumerate(poly_features_names[len(num_features):]):
            df[f'poly_{name}'] = poly_features[:, len(num_features) + i]

        df['age_sin'] = np.sin(2 * np.pi * df['person_age'] / 100)
        df['age_cos'] = np.cos(2 * np.pi * df['person_age'] / 100)
        df['stability_score'] = (df['person_emp_length'] * df['person_income']) / (df['loan_amnt'] * (df['cb_person_cred_hist_length'] + 1))

    return train, test

train, test = create_features(train, test)


import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


target_col = train['loan_status'].reset_index(drop=True)

train = train.drop('loan_status', axis=1)

combined_data = pd.concat([train, test], axis=0, ignore_index=True)

categorical_features = combined_data.select_dtypes(include=['object']).columns
numerical_features = combined_data.select_dtypes(include=['int64', 'float64']).columns

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), categorical_features)
    ])

preprocessed_data = preprocessor.fit_transform(combined_data)

onehot_encoder = preprocessor.named_transformers_['cat']
cat_feature_names = onehot_encoder.get_feature_names_out(categorical_features)

feature_names = list(numerical_features) + list(cat_feature_names)

preprocessed_df = pd.DataFrame(preprocessed_data, columns=feature_names)

for col in combined_data.columns:
    preprocessed_df[f'original_{col}'] = combined_data[col].values

preprocessed_df = preprocessed_df.reset_index(drop=True)

train_preprocessed = preprocessed_df.iloc[:len(train)]
test_preprocessed = preprocessed_df.iloc[len(train):]

train_preprocessed = pd.concat([train_preprocessed, target_col], axis=1)

print("Shape of preprocessed train data:", train_preprocessed.shape)
print("Shape of preprocessed test data:", test_preprocessed.shape)

train = train_preprocessed.copy()
test = test_preprocessed.copy()


def convert_data_types(train, test, target_column):
    train = train.copy()
    test = test.copy()

    target = train[target_column].copy()

    train = train.drop(columns=[target_column])

    combined = pd.concat([train, test], keys=['train', 'test'])

    new_columns = []
    for column in combined.columns:
        if combined[column].dtype == 'object' or combined[column].dtype.name == 'category':
            try:
                new_column = f'{column}_numeric'
                combined[new_column] = pd.to_numeric(combined[column], errors='coerce')
                new_columns.append(new_column)
            except ValueError:
                pass  

        elif np.issubdtype(combined[column].dtype, np.number):
            new_column = f'{column}_category'
            if combined[column].dtype == float:
                combined[new_column] = combined[column].round().fillna(-999999).astype(int).astype('category')
                combined[new_column] = combined[new_column].cat.add_categories('NaN')
                combined.loc[combined[new_column] == -999999, new_column] = 'NaN'
            else:
                combined[new_column] = combined[column].astype('category')
            new_columns.append(new_column)

    new_train = combined.loc['train'].copy()
    new_test = combined.loc['test'].copy()

    new_train[target_column] = target

    return new_train, new_test

train, test = convert_data_types(train, test, target)


import pandas as pd
import numpy as np

def convert_data_types(train, test, target_column):
    train = train.copy()
    test = test.copy()

    target = train[target_column].copy()

    train = train.drop(columns=[target_column])

    combined = pd.concat([train, test], keys=['train', 'test'])

    def process_column(df, col):
        if df[col].dtype == 'object':
            df[col] = df[col].fillna('unk').astype('category')
        elif df[col].dtype.name == 'category':
            if df[col].isnull().any():
                df[col] = df[col].cat.add_categories('unk').fillna('unk')
        return df

    for column in combined.columns:
        combined = process_column(combined, column)

    new_train = combined.loc['train'].copy()
    new_test = combined.loc['test'].copy()

    new_train[target_column] = target

    return new_train, new_test

train, test = convert_data_types(train, test, target)


categorical_features = train.select_dtypes(include=['object', 'category']).columns.tolist()
categorical_features


import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping
from optuna.samplers import TPESampler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from typing import Dict, List, Tuple, Union


class Model_gbdt:
    def __init__(self, train: pd.DataFrame, test: pd.DataFrame, target: str, categorical_feats: List[str], base_params=None):
        if base_params is None:
            base_params = {}
        self.train = train
        self.test = test
        self.model_dict: Dict[str, LGBMClassifier] = {}
        self.test_predict_list: List[np.ndarray] = []
        self.categorical_feats = categorical_feats
        self.target = target
        self.base_params = base_params

    def objective(self, trial: optuna.Trial) -> float:
        params = {
            "max_depth": trial.suggest_int("max_depth", 2, 10),
            "num_leaves": trial.suggest_int("num_leaves", 2, 256),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 50, 1500),
            "min_child_samples": trial.suggest_int("min_child_samples", 1, 100),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.1, 1.0),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "lambda_l1": trial.suggest_float("lambda_l1", 0, 1),
            "lambda_l2": trial.suggest_float("lambda_l2", 0, 1),
            **self.base_params
        }
        scores, _, _ = self.fit(params)
        return np.mean(scores)

    def fit(self, params: Dict[str, Union[int, float, str, bool, List[str]]]) -> Tuple[List[float], List[np.ndarray], np.ndarray]:
        label_columns = [self.target]
        train_cols = [col for col in self.train.columns.to_list() if col not in label_columns]
        scores = []
        mskf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        oof_valid_preds = np.zeros((self.train[train_cols].shape[0], 1))
        for fold, (train_idx, valid_idx) in enumerate(mskf.split(self.train[train_cols], self.train[label_columns])):
            X_train, y_train = self.train[train_cols].iloc[train_idx], self.train[label_columns].iloc[train_idx].values.ravel()
            X_valid, y_valid = self.train[train_cols].iloc[valid_idx], self.train[label_columns].iloc[valid_idx].values.ravel()

            model = LGBMClassifier(**params)
            model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], callbacks=[early_stopping(250)])

            valid_preds = model.predict_proba(X_valid)[:, 1]
            oof_valid_preds[valid_idx] = valid_preds.reshape(-1, 1)

            test_predict = model.predict_proba(self.test[train_cols])[:, 1]
            self.test_predict_list.append(test_predict)

            self.model_dict[f'fold_{fold}'] = model

        oof_score = roc_auc_score(self.train[label_columns], oof_valid_preds)
        scores.append(oof_score)
        print(f'The average ROC AUC score is {np.mean(scores)}')

        return scores, self.test_predict_list, oof_valid_preds

    def optimize(self, n_trials: int = 100) -> Dict[str, Union[int, float, str, bool]]:
        study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=42))
        study.optimize(self.objective, timeout=n_trials, show_progress_bar=True)

        print("Best trial:")
        trial = study.best_trial
        print(" Value:", trial.value)
        print(" Params:")
        for key, value in trial.params.items():
            print(f" {key}: {value}")

        return study.best_params



import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping
from optuna.samplers import TPESampler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from typing import Dict, List, Tuple, Union

class Model_goss:
    def __init__(self, train: pd.DataFrame, test: pd.DataFrame, target: str, categorical_feats: List[str],
                 base_params=None):
        self.train = train
        self.test = test
        self.model_dict: Dict[str, LGBMClassifier] = {}
        self.test_predict_list: List[np.ndarray] = []
        self.categorical_feats = categorical_feats
        self.target = target
        if base_params is None:
            base_params = {}
        self.base_params = base_params

    def objective(self, trial: optuna.Trial) -> float:
        params = {
            "max_depth": trial.suggest_int("max_depth", 2, 10),
            "num_leaves": trial.suggest_int("num_leaves", 2, 256),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 50, 1500),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.1, 1.0),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            **self.base_params
        }
        scores, _, _ = self.fit(params)
        return np.mean(scores)

    def fit(self, params: Dict[str, Union[int, float, str, bool, List[str]]]) -> Tuple[List[float], List[np.ndarray], np.ndarray]:
        label_columns = [self.target]
        train_cols = [col for col in self.train.columns.to_list() if col not in label_columns]
        scores = []
        mskf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        oof_valid_preds = np.zeros((self.train[train_cols].shape[0], 1))
        for fold, (train_idx, valid_idx) in enumerate(mskf.split(self.train[train_cols], self.train[label_columns])):
            X_train, y_train = self.train[train_cols].iloc[train_idx], self.train[label_columns].iloc[train_idx].values.ravel()
            X_valid, y_valid = self.train[train_cols].iloc[valid_idx], self.train[label_columns].iloc[valid_idx].values.ravel()

            model = LGBMClassifier(**params)
            model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], callbacks=[early_stopping(250)])

            valid_preds = model.predict_proba(X_valid)[:, 1]
            oof_valid_preds[valid_idx] = valid_preds.reshape(-1, 1)

            test_predict = model.predict_proba(self.test[train_cols])[:, 1]
            self.test_predict_list.append(test_predict)

            
            self.model_dict[f'fold_{fold}'] = model

        oof_score = roc_auc_score(self.train[label_columns], oof_valid_preds)
        scores.append(oof_score)
        print(f'The average ROC AUC score is {np.mean(scores)}')

        return scores, self.test_predict_list, oof_valid_preds

    def optimize(self, n_trials: int = 100) -> Dict[str, Union[int, float, str, bool]]:
        study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=42))
        study.optimize(self.objective, timeout=n_trials, show_progress_bar=True)

        print("Best trial:")
        trial = study.best_trial
        print(" Value:", trial.value)
        print(" Params:")
        for key, value in trial.params.items():
            print(f" {key}: {value}")

        return study.best_params


import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostClassifier
from optuna.samplers import TPESampler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from typing import Dict, List, Tuple, Union


class Model_loss:
    def __init__(self, train: pd.DataFrame, test: pd.DataFrame, target: str, categorical_feats: List[str], base_params=None):
        self.train = train
        self.test = test
        self.model_dict: Dict[str, CatBoostClassifier] = {}
        self.test_predict_list: List[np.ndarray] = []
        self.target = target
        self.categorical_feats = categorical_feats
        if base_params is None:
            base_params = {}
        self.base_params = base_params

    def objective(self, trial: optuna.Trial) -> float:
        # boosting_type = trial.suggest_categorical('boosting_type', ['Ordered', 'Plain'])
        # grow_policy = 'Lossguide'
        # all_score_functions = ['Cosine', 'L2']
        # score_function = trial.suggest_categorical('score_function', all_score_functions)
        #
        # if boosting_type == 'Ordered' and score_function in ['LOOL2', 'SolarL2', 'L2', 'NewtonL2']:
        #     raise optuna.exceptions.TrialPruned()
        #
        # if boosting_type == 'Ordered' and grow_policy in ['Lossguide', 'Depthwise']:
        #     raise optuna.exceptions.TrialPruned()

        params = {
            'iterations': trial.suggest_int('iterations', 50, 1500),
            'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
            **self.base_params
        }

        

        scores, _, _ = self.fit(params)
        return np.mean(scores)

    def fit(self, params: Dict[str, Union[int, float, str]]) -> Tuple[List[float], List[np.ndarray], np.ndarray]:
        target_columns = [self.target]
        train_cols = [col for col in self.train.columns.to_list() if col not in target_columns]
        scores = []

        kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        oof_valid_preds = np.zeros((self.train[train_cols].shape[0], 1))

        for fold, (train_idx, valid_idx) in enumerate(kf.split(self.train[train_cols], self.train[target_columns])):
            X_train, y_train = self.train[train_cols].iloc[train_idx], self.train[target_columns].iloc[train_idx]
            X_valid, y_valid = self.train[train_cols].iloc[valid_idx], self.train[target_columns].iloc[valid_idx]

            model = CatBoostClassifier(**params)
            model.fit(X_train, y_train, eval_set=(X_valid, y_valid), early_stopping_rounds=250, verbose=100,
                      cat_features=self.categorical_feats)

            valid_preds = model.predict_proba(X_valid)[:, 1]

            oof_valid_preds[valid_idx] = valid_preds.reshape(-1, 1)

            test_predict = model.predict_proba(self.test[train_cols])[:, 1]

            self.test_predict_list.append(test_predict)

            self.model_dict[f'fold_{fold}'] = model

        oof_score = roc_auc_score(self.train[target_columns], oof_valid_preds)
        scores.append(oof_score)

        print(f'The average ROC AUC score is {np.mean(scores)}')
        return scores, self.test_predict_list, oof_valid_preds

    def optimize(self, n_trials: int = 100) -> Dict[str, Union[int, float, str]]:
        study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=42))
        study.optimize(self.objective, timeout=n_trials, show_progress_bar=True)

        print("Best trial:")
        trial = study.best_trial
        print("  Value:", trial.value)
        print("  Params:")
        for key, value in trial.params.items():
            print(f"    {key}: {value}")

        return study.best_params


import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostClassifier
from optuna.samplers import TPESampler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from typing import Dict, List, Tuple, Union


class Model_sym:
    def __init__(self, train: pd.DataFrame, test: pd.DataFrame, target: str, categorical_feats: List[str], base_params=None):
        self.train = train
        self.test = test
        self.model_dict: Dict[str, CatBoostClassifier] = {}
        self.test_predict_list: List[np.ndarray] = []
        self.target = target
        self.categorical_feats = categorical_feats
        if base_params is None:
            base_params = {}
        self.base_params = base_params

    def objective(self, trial: optuna.Trial) -> float:
        # boosting_type = trial.suggest_categorical('boosting_type', ['Ordered', 'Plain'])
        # grow_policy = 'SymmetricTree'
        # all_score_functions = ['Cosine', 'L2']
        # score_function = trial.suggest_categorical('score_function', all_score_functions)
        #
        # if boosting_type == 'Ordered' and score_function in ['LOOL2', 'SolarL2', 'L2', 'NewtonL2']:
        #     raise optuna.exceptions.TrialPruned()
        #
        # if boosting_type == 'Ordered' and grow_policy in ['Lossguide', 'Depthwise']:
        #     raise optuna.exceptions.TrialPruned()

        params = {
            'iterations': trial.suggest_int('iterations', 50, 1500),
            'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
            
            **self.base_params
        }
        scores, _, _ = self.fit(params)
        return np.mean(scores)

    def fit(self, params: Dict[str, Union[int, float, str]]) -> Tuple[List[float], List[np.ndarray], np.ndarray]:
        target_columns = [self.target]
        train_cols = [col for col in self.train.columns.to_list() if col not in target_columns]
        scores = []

        kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        oof_valid_preds = np.zeros((self.train[train_cols].shape[0], 1))

        for fold, (train_idx, valid_idx) in enumerate(kf.split(self.train[train_cols], self.train[target_columns])):
            X_train, y_train = self.train[train_cols].iloc[train_idx], self.train[target_columns].iloc[train_idx]
            X_valid, y_valid = self.train[train_cols].iloc[valid_idx], self.train[target_columns].iloc[valid_idx]

            model = CatBoostClassifier(**params)
            model.fit(X_train, y_train, eval_set=(X_valid, y_valid), early_stopping_rounds=250, verbose=100,
                      cat_features=self.categorical_feats)

            valid_preds = model.predict_proba(X_valid)[:, 1]

            oof_valid_preds[valid_idx] = valid_preds.reshape(-1, 1)

            test_predict = model.predict_proba(self.test[train_cols])[:, 1]

            self.test_predict_list.append(test_predict)

            self.model_dict[f'fold_{fold}'] = model

        oof_score = roc_auc_score(self.train[target_columns], oof_valid_preds)
        scores.append(oof_score)

        print(f'The average ROC AUC score is {np.mean(scores)}')
        return scores, self.test_predict_list, oof_valid_preds

    def optimize(self, n_trials: int = 100) -> Dict[str, Union[int, float, str]]:
        study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=42))
        study.optimize(self.objective, timeout=n_trials, show_progress_bar=True)

        print("Best trial:")
        trial = study.best_trial
        print("  Value:", trial.value)
        print("  Params:")
        for key, value in trial.params.items():
            print(f"    {key}: {value}")

        return study.best_params


import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostClassifier
from optuna.samplers import TPESampler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from typing import Dict, List, Tuple, Union


class Model_depth:
    def __init__(self, train: pd.DataFrame, test: pd.DataFrame, target: str, categorical_feats: List[str], base_params=None):
        self.train = train
        self.test = test
        self.model_dict: Dict[str, CatBoostClassifier] = {}
        self.test_predict_list: List[np.ndarray] = []
        self.target = target
        self.categorical_feats = categorical_feats
        if base_params is None:
            base_params = {}
        self.base_params = base_params

    def objective(self, trial: optuna.Trial) -> float:

        params = {
            'iterations': trial.suggest_int('iterations', 50, 1500),
            'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
            # 'depth': trial.suggest_int('depth', 2, 10),
            # 'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10, log=True),
            # 'boosting_type': boosting_type,
            # 'border_count': trial.suggest_int('border_count', 32, 255),
            # 'min_child_samples': trial.suggest_int('min_child_samples', 1, 100),
            # 'bootstrap_type': trial.suggest_categorical('bootstrap_type', ['Bayesian', 'Bernoulli', 'No', 'MVS']),
            **self.base_params
        }

        # if params['grow_policy'] == 'Lossguide':
        #     params['max_leaves'] = trial.suggest_int('max_leaves', 2, 64)
        #
        # if params['bootstrap_type'] in ['Bernoulli', 'Poisson']:
        #     params['subsample'] = trial.suggest_float('subsample', 0.1, 1.0)
        #
        # if params['bootstrap_type'] == 'Bayesian':
        #     params['bagging_temperature'] = trial.suggest_float('bagging_temperature', 0, 10)

        scores, _, _ = self.fit(params)
        return np.mean(scores)

    def fit(self, params: Dict[str, Union[int, float, str]]) -> Tuple[List[float], List[np.ndarray], np.ndarray]:
        target_columns = [self.target]
        train_cols = [col for col in self.train.columns.to_list() if col not in target_columns]
        scores = []

        kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        oof_valid_preds = np.zeros((self.train[train_cols].shape[0], 1))

        for fold, (train_idx, valid_idx) in enumerate(kf.split(self.train[train_cols], self.train[target_columns])):
            X_train, y_train = self.train[train_cols].iloc[train_idx], self.train[target_columns].iloc[train_idx]
            X_valid, y_valid = self.train[train_cols].iloc[valid_idx], self.train[target_columns].iloc[valid_idx]

            model = CatBoostClassifier(**params)
            model.fit(X_train, y_train, eval_set=(X_valid, y_valid), early_stopping_rounds=250, verbose=100,
                      cat_features=self.categorical_feats)

            valid_preds = model.predict_proba(X_valid)[:, 1]

            oof_valid_preds[valid_idx] = valid_preds.reshape(-1, 1)

            test_predict = model.predict_proba(self.test[train_cols])[:, 1]

            self.test_predict_list.append(test_predict)

            self.model_dict[f'fold_{fold}'] = model

        oof_score = roc_auc_score(self.train[target_columns], oof_valid_preds)
        scores.append(oof_score)

        print(f'The average ROC AUC score is {np.mean(scores)}')
        return scores, self.test_predict_list, oof_valid_preds

    def optimize(self, n_trials: int = 100) -> Dict[str, Union[int, float, str]]:
        study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=42))
        study.optimize(self.objective, timeout=n_trials, show_progress_bar=True)

        print("Best trial:")
        trial = study.best_trial
        print("  Value:", trial.value)
        print("  Params:")
        for key, value in trial.params.items():
            print(f"    {key}: {value}")

        return study.best_params


import numpy as np
import optuna
import pandas as pd
from optuna.samplers import TPESampler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from typing import Dict, List, Tuple, Union
from xgboost import XGBClassifier


class Model_gbtree:
    def __init__(self, train: pd.DataFrame, test: pd.DataFrame, target: str, base_params=None):
        self.train = train
        self.test = test
        self.model_dict: Dict[str, XGBClassifier] = {}
        self.test_predict_list: List[np.ndarray] = []
        self.target = target
        if base_params is None:
            base_params = {}
        self.base_params = base_params

    def objective(self, trial: optuna.Trial) -> float:
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 50, 1500),
            **self.base_params
        }
        scores, _, _ = self.fit(params)
        return np.mean(scores)

    def fit(self, params: Dict[str, Union[int, float, str, bool]]) -> Tuple[List[float], List[np.ndarray], np.ndarray]:
        label_columns = [self.target]
        train_cols = [col for col in self.train.columns.to_list() if col not in label_columns]
        scores = []
        mskf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        oof_valid_preds = np.zeros((self.train[train_cols].shape[0], 1))
        for fold, (train_idx, valid_idx) in enumerate(mskf.split(self.train[train_cols], self.train[label_columns])):
            X_train, y_train = self.train[train_cols].iloc[train_idx], self.train[label_columns].iloc[train_idx]
            X_valid, y_valid = self.train[train_cols].iloc[valid_idx], self.train[label_columns].iloc[valid_idx]
            model = XGBClassifier(**params)
            model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)
            valid_preds = model.predict_proba(X_valid)[:, 1]
            oof_valid_preds[valid_idx] = valid_preds.reshape(-1, 1)
            test_predict = model.predict_proba(self.test[train_cols])[:, 1]
            self.test_predict_list.append(test_predict)
            self.model_dict[f'fold_{fold}'] = model


        oof_score = roc_auc_score(self.train[label_columns], oof_valid_preds)
        scores.append(oof_score)
        print(f'The average ROC AUC score is {np.mean(scores)}')
        return scores, self.test_predict_list, oof_valid_preds

    def optimize(self, n_trials: int = 100) -> Dict[str, Union[int, float, str, bool]]:
        study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=42))
        study.optimize(self.objective, timeout=n_trials, show_progress_bar=True)
        print("Best trial:")
        trial = study.best_trial
        print(" Value:", trial.value)
        print(" Params:")
        for key, value in trial.params.items():
            print(f" {key}: {value}")
        return study.best_params


from typing import Any, List, Optional, Tuple

def get_model_configs(device: str, categorical_feats: List[str], num_gpus: int) -> List[Tuple[Any, dict, bool, str]]:
    base_config = {
        'cpu': {
            'lgb': {"random_state": 42, "n_jobs": -1, "metric": "auc", 'cat_features': categorical_feats, 'verbosity': -1},
            'cat': {'task_type': 'CPU', 'devices': '0', 'random_seed': 42, 'eval_metric': 'AUC', 'thread_count': -1},
            'xgb': {"random_state": 42, "n_jobs": -1, "objective": "binary:logistic", "eval_metric": "auc", 'enable_categorical': True, 'early_stopping_rounds': 250}
        },
        'gpu': {
            'lgb': {"random_state": 42, "n_jobs": -1, "metric": "auc", 'cat_features': categorical_feats, 'verbosity': -1, 'device': 'gpu'},
            'cat': {'task_type': 'GPU', 'devices': ':'.join(map(str, range(num_gpus))), 'random_seed': 42, 'eval_metric': 'AUC', 'thread_count': -1},
            'xgb': {"random_state": 42, "n_jobs": -1, "objective": "binary:logistic", "eval_metric": "auc", 'enable_categorical': True, 'early_stopping_rounds': 250}
        }
    }

    return [
        (Model_goss, {**base_config[device]['lgb'], "boosting_type": 'goss'}, True, 'lightgbm_goss'),
        (Model_gbdt, {**base_config[device]['lgb'], "boosting_type": 'gbdt'}, True, 'lightgbm_gbdt'),
        (Model_loss, {**base_config[device]['cat'], 'grow_policy': 'Lossguide'}, True, 'catboost_lossguide'),
        (Model_sym, {**base_config[device]['cat'], 'grow_policy': 'SymmetricTree'}, True, 'catboost_symmetric'),
        (Model_depth, {**base_config[device]['cat'], 'grow_policy': 'Depthwise'}, True, 'catboost_depthwise'),
        (Model_gbtree, {**base_config[device]['xgb'], "booster": "gbtree"}, False, 'xgboost')
    ]

def model_optimization(train: Any, test: Any, target: str, categorical_feats: List[str], time_per_model: int, device: str) -> None:
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    print(f"Number of available GPUs: {num_gpus}")

    models = get_model_configs(device, categorical_feats, num_gpus)
    n_imp_features = {model_type: load(os.path.join(base_path, 'features', f'n_imp_features_{model_type.lower()}.joblib')) for _, _, _, model_type in models}

    os.makedirs(os.path.join(base_path, 'preds'), exist_ok=True)
    os.makedirs(os.path.join(base_path, 'params'), exist_ok=True)

    for i, (model_cls, base_params, use_cat_feats, model_type) in enumerate(models):
        print(f"Model {i} ({model_type}) optimization started.")
        selected_features = n_imp_features[model_type]
        train_selected = train[selected_features + [target]]
        test_selected = test[selected_features]
        cat_feats_selected = list(set(categorical_feats) & set(selected_features)) if use_cat_feats else None

        if use_cat_feats:
            opt_model = model_cls(train_selected, test_selected, target, cat_feats_selected, base_params)
        else:
            opt_model = model_cls(train_selected, test_selected, target, base_params)

        best_params = opt_model.optimize(n_trials=time_per_model)
        dump(best_params, os.path.join(base_path, 'params', f'best_params_{i}.joblib'))

        if use_cat_feats:
            train_model = model_cls(train_selected, test_selected, target, cat_feats_selected)
        else:
            train_model = model_cls(train_selected, test_selected, target)

        print(base_params)
        print(best_params)

        best_params.update(base_params)

        scores, preds, oof_valid_preds = train_model.fit(best_params)
        score = np.mean(scores)

        for item, filename in zip([score, preds, oof_valid_preds, train_model],
                                  [f'score{i}.joblib', f'preds_{i}.joblib', f'oof_valid_preds_{i}.joblib', f'model_{i}.joblib']):
            dump(item, os.path.join(base_path, 'preds', filename))

        print(f"Model {i} ({model_type}) optimization completed. Mean ROC AUC score: {score}")
        print(f"Number of selected features: {len(selected_features)}")
        print(f"Selected features: {selected_features}")


import os
from functools import partial

base_path = '/kaggle/input/'

def blending(train: Any, target: str, n_trials: int) -> Tuple[np.ndarray, List[Any], np.ndarray]:
    indices = range(6)

    oof_valid_preds = [load(os.path.join(base_path, 'preds-loan', f'oof_valid_preds_{i}.joblib')) for i in indices]
    preds = [load(os.path.join(base_path, 'preds-loan', f'preds_{i}.joblib')) for i in indices]

    ow = OptunaWeights(random_state=42, n_trials=n_trials)
    ow.fit(train[target], y_preds=oof_valid_preds)

    selected_predictions = [pred for pred, selected in zip(preds, ow.selected_preds) if selected]
    final_predictions = np.sum([weight * np.mean(pred, axis=0) for weight, pred in zip(ow.weights, selected_predictions)], axis=0)

    return final_predictions, np.array(ow.weights)



class OptunaWeights:
    def __init__(self, random_state: int, n_trials: int = 5000):
        self.study: Optional[optuna.Study] = None
        self.weights: Optional[List[float]] = None
        self.random_state = random_state
        self.n_trials = n_trials
        self.selected_preds: Optional[List[bool]] = None

    def _objective(self, trial: optuna.Trial, y_true: np.ndarray, y_preds: List[np.ndarray]) -> float:
        weights = [trial.suggest_float(f"weight{n}", 0, 1) for n in range(len(y_preds))]
        # selected_preds = [trial.suggest_categorical(f"select_pred{n}", [True, False]) for n in range(len(y_preds))]
        selected_preds = [True for n in range(len(y_preds))]

        selected_weights = [w for w, s in zip(weights, selected_preds) if s]
        weight_sum = sum(selected_weights)
        if weight_sum == 0:
            return 0.0
        norm_weights = [w / weight_sum for w in selected_weights]
        selected_y_preds = [pred for pred, s in zip(y_preds, selected_preds) if s]
        weighted_pred = np.average(np.array(selected_y_preds), axis=0, weights=norm_weights)
        return roc_auc_score(y_true, weighted_pred)

    def fit(self, y_true: np.ndarray, y_preds: List[np.ndarray]) -> None:
        optuna.logging.set_verbosity(optuna.logging.ERROR)
        self.study = optuna.create_study(
            #sampler=optuna.samplers.CmaEsSampler(seed=self.random_state),
            sampler=optuna.samplers.TPESampler(seed=self.random_state),
            pruner=optuna.pruners.HyperbandPruner(),
            #study_name="OptunaWeights",
            direction='maximize'
        )
        objective_partial = partial(self._objective, y_true=y_true, y_preds=y_preds)
        self.study.optimize(objective_partial, n_trials=self.n_trials, show_progress_bar=True)

        weights = [self.study.best_params[f"weight{n}"] for n in range(len(y_preds))]
        # self.selected_preds = [self.study.best_params[f"select_pred{n}"] for n in range(len(y_preds))]
        self.selected_preds = [True for n in range(len(y_preds))]
        selected_weights = [w for w, s in zip(weights, self.selected_preds) if s]
        weight_sum = sum(selected_weights)
        if weight_sum == 0:
            raise ValueError("All weights are zero. Unable to normalize.")
        self.weights = [w / weight_sum for w in selected_weights]


final_pred, final_weights = blending(train, target, 2200)

