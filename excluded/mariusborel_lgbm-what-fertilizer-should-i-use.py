import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.pipeline import make_pipeline, Pipeline
from sklearn .metrics import roc_auc_score, roc_curve
from sklearn .metrics import  roc_auc_score, roc_curve
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import (train_test_split, GridSearchCV, KFold, RepeatedKFold,
                                     RepeatedStratifiedKFold, RandomizedSearchCV, cross_val_score,
                                     StratifiedKFold, TimeSeriesSplit as TSS)

import sklearn
from sklearn.preprocessing import (MaxAbsScaler, MinMaxScaler, Normalizer, minmax_scale, 
                                   PowerTransformer, QuantileTransformer, LabelEncoder,
                                   RobustScaler, StandardScaler, FunctionTransformer,
                                   LabelEncoder, OneHotEncoder, OrdinalEncoder)
from sklearn.feature_selection import mutual_info_classif, SelectKBest, RFE

import xgboost as xgb
from xgboost import XGBClassifier, plot_importance, cv
from sklearn.compose import make_column_transformer
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier, Pool

import shap

import optuna

import plotly.io as pio
pio.renderers.default = 'notebook'
import plotly.express as px

import warnings
warnings.filterwarnings('ignore')

# verify the versions of my tools
print(f'pandas version: {pd.__version__}')
print(f'numpy version: {np.__version__}')
print(f'seaborn version: {sns.__version__}')
print(f'sklearn version: {sklearn.__version__}')
# print(f'optuna version : {optuna.__version__}')


k = None
def apk(actual, predicted, k=k):
    """
    Computes the average precision at k.

    This function computes the average prescision at k between two lists of
    items.

    Parameters
    ----------
    actual : list
             A list of elements that are to be predicted (order doesn't matter)
    predicted : list
                A list of predicted elements (order does matter)
    k : int, optional
        The maximum number of predicted elements

    Returns
    -------
    score : double
            The average precision at k over the input lists

    """
    if len(predicted)>k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i,p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)

    if not actual:
        return 0.0

    return score / min(len(actual), k)

def mapk(actual, predicted, k=k):
    """
    Computes the mean average precision at k.

    This function computes the mean average prescision at k between two lists
    of lists of items.

    Parameters
    ----------
    actual : list
             A list of lists of elements that are to be predicted 
             (order doesn't matter in the lists)
    predicted : list
                A list of lists of predicted elements
                (order matters in the lists)
    k : int, optional
        The maximum number of predicted elements

    Returns
    -------
    score : double
            The mean average precision at k over the input lists

    """
    return np.mean([apk(a,p,k) for a,p in zip(actual, predicted)])


def apk(actual, predicted, k=10):
    if not actual:
        return 0.0

    predicted = predicted[:k]
    score = 0.0
    num_hits = 0

    seen = set()
    actual_set = set(actual)

    for i, p in enumerate(predicted):
        if p in actual_set and p not in seen:
            num_hits += 1
            score += num_hits / (i + 1)
            seen.add(p)

    return score / min(len(actual), k)


def mapk(actual, predicted, k=10):
    return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])


def map3(y_true,y_pred):
    m = (y_true.reshape((-1,1)) == y_pred)
    return np.mean(np.where(m.any(axis=1),m.argmax(axis=1)+1,np.inf)**(-1))


def mapk_single_label(y_true, y_pred, k=3):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)[:, :k]
    matches = (y_true.reshape(-1, 1) == y_pred)
    ranks = np.where(matches.any(axis=1), matches.argmax(axis=1) + 1, np.inf)
    return np.mean(ranks ** -1)



train_raw = pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv', index_col='id')
test_raw = pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv', index_col='id')
# orig_raw = pd.read_csv('/kaggle/input/calories-burnt-prediction/calories.csv').drop(columns=['User_ID'])
sample_submission = pd.read_csv('/kaggle/input/playground-series-s5e6/sample_submission.csv')

target = 'Fertilizer Name'

train_raw.head(3)


print(f'The train dataset has {train_raw.shape} and the test set has {test_raw.shape} shape.')


display(train_raw.info())
print('\nThe columns comprise of three object and six int64 dtypes')


num_col = test_raw.select_dtypes('int64').columns.tolist()
num_col


cat_col = test_raw.select_dtypes('object').columns.tolist()
cat_col


print('There are {} unique soil types'.format(train_raw['Soil Type'].nunique()))
print('and {} unique crop types'.format(train_raw['Crop Type'].nunique()))


train_raw.describe().T


train_raw.describe(exclude='number').T


plt.figure(figsize=(10, 6))
for c, col in enumerate(cat_col, start=1):
    plt.subplot(1,2,c)
    if col == 'Soil Type':
        explode_ = [0.05, 0.08, 0.1, 0.12, 0.15]
    else:
        explode_ = [0.05, 0.05, 0.05, 0.1, 0.1, 0.1, 0.15, 0.15, 0.15, 0.2, 0.2]
    train_raw[col].value_counts().plot.pie( autopct='%1.1f%%', 
                                           explode=explode_,
                                           shadow=True,
                                           title=f'{col} proportions',
                                           cmap='Accent')
    plt.ylabel('')
plt.tight_layout()


plt.figure(figsize=(10, 4))
for c, col in enumerate(cat_col, start=1):
    plt.subplot(1,2,c)
    ax = train_raw[col].value_counts().plot.barh(color='Peru', title=f'{col} counts')
    for count in ax.containers:
        ax.bar_label(count, color='Peru')
plt.tight_layout()


display(train_raw[target].value_counts(), train_raw[target].value_counts(normalize=True))
print(f'\nDAP and Urea are the least represented classes in the training dataset')


print(f'There are {train_raw[target].nunique()} unique targets')


plt.figure(figsize=(13, 5))
plt.subplot(121)
ax = train_raw[target].value_counts().plot.barh(color='Peru', title=f'Target counts')
for count in ax.containers:
    ax.bar_label(count, color='Peru')
plt.subplot(122)
# plt.figure(figsize=(8, 6))
train_raw[target].value_counts().plot.pie( autopct='%1.1f%%', 
                                          explode=[0.04, 0.04, .08, .08, 0.12, 0.12, .04],
                                          shadow=True,
                                          radius=1.2, startangle=90)
plt.ylabel('')
plt.show()


plt.figure(figsize=(12,6))
for c, col in enumerate(num_col, start=1):
    plt.subplot(2,3,c)
    sns.violinplot(train_raw, x=col, y='Soil Type', palette='Accent')
    if c not in [1, 4]:
        plt.ylabel('')
plt.tight_layout()


plt.figure(figsize=(12,6))
for c, col in enumerate(num_col, start=1):
    plt.subplot(2,3,c)
    sns.kdeplot(train_raw, x=col, hue='Soil Type', palette='Accent')
    if c not in [1, 4]:
        plt.ylabel('')
plt.tight_layout()


plt.figure(figsize=(12,8))
for c, col in enumerate(num_col, start=1):
    plt.subplot(2,3,c)
    sns.violinplot(train_raw, x=col, y='Crop Type', palette='Accent')
    if c not in [1, 4]:
        plt.ylabel('')
plt.tight_layout()


plt.figure(figsize=(12,8))
for c, col in enumerate(num_col, start=1):
    plt.subplot(2,3,c)
    sns.kdeplot(train_raw, x=col, hue='Crop Type', palette='Accent')
    if c not in [1, 4]:
        plt.ylabel('')
plt.tight_layout()


plt.figure(figsize=(12,7))
for c, col in enumerate(num_col, start=1):
    plt.subplot(2,3,c)
    sns.violinplot(train_raw, x=col, y=target, palette='Accent')
    if c not in [1, 4]:
        plt.ylabel('')
plt.tight_layout()


plt.figure(figsize=(12,7))
for c, col in enumerate(num_col, start=1):
    plt.subplot(2,3,c)
    sns.kdeplot(train_raw, x=col, hue=target, palette='Accent')
    if c not in [1, 4]:
        plt.ylabel('')
plt.tight_layout()


corr = train_raw.corr(numeric_only=True)
sns.heatmap(corr, annot=True, fmt='.3f', cbar=False, cmap='Accent_r')
plt.show()

print('There are no pairs of highly correlated features')


# Separate the data and target
def DataTargetPrep(df):
    data_ = df.copy()
    # data_ = featEng(data_)
    try:
        target_ = data_.pop(target)
    except:
        pass
    try:
        return data_, target_
    except:
        return data_


# Should the external dataset be included?
include_orig_data = True
if include_orig_data:
    train_set = train_raw
    # train_set = pd.concat([train_raw, orig_raw], ignore_index=True)
else:
    train_set = train_raw

# Prep the train data and target
train_prep = train_set.copy()
X_train_prep, y_train = DataTargetPrep(train_prep)

# Prep the test data
X_test_prep = DataTargetPrep(test_raw)


# def the target encoder
Encoder = LabelEncoder()

# Encode the target
y = y_train.copy()
y_enc = Encoder.fit_transform(y_train)

y_enc


# Separate the train set from the validation and test sets
X_tr, X_va, y_tr, y_va = train_test_split(X_train_prep, y_enc, 
                                                test_size=0.2, 
                                                random_state=81545)

# Chech the size of the sets
[d.shape for d in [X_tr, y_tr, X_va, y_va]]


# Should we engeneer the features?
create_new_features = False

# Define function for features engeneering
class Feature_Eng(BaseEstimator, TransformerMixin):
    def fit(self, df, y=None):
        return self
    
    def transform(self, df):
        df = df.copy()
        if create_new_features:
            # Proportion
            df['N_/_all'] = df['Nitrogen']/(df['Nitrogen'] + df['Potassium'] + df['Phosphorous'])
            df['K_/_all'] = df['Potassium']/(df['Nitrogen'] + df['Potassium'] + df['Phosphorous'])
            df['P_/_all'] = df['Phosphorous']/(df['Nitrogen'] + df['Potassium'] + df['Phosphorous'])
            
            # Division
            df['T_/_H'] = df['Humidity']/df['Temparature']
            df['T_/_M'] = df['Moisture']/df['Temparature']
            df['M_/_H'] = df['Moisture']/df['Humidity']
            
            # Multiplication
            df['T_x_H'] = df['Humidity']*df['Temparature']
            df['T_x_M'] = df['Moisture']*df['Temparature']
            df['M_x_H'] = df['Moisture']*df['Humidity']

            # df['Moisture_/_Humidity'] = df['Moisture']/df['Humidity']
            
            df['Soil_Crop'] = df['Soil Type'] + '_' + df['Crop Type']
            
        else:
            df = df
        
        return df


#### handling of cat features
ohe = OneHotEncoder()
le = LabelEncoder()

train_eg = train_raw['Soil Type'].copy()
train_eg = le.fit_transform(train_eg)

train_eg


# Instanciate the model
lgb_model = LGBMClassifier(
     n_estimators= 1214,
     learning_rate= 0.06408094783107429,
     # num_leaves= 169,
     max_depth =10,
     min_child_samples= 19,
     subsample= 0.6420340301820501,
     colsample_bytree= 0.43403799235854973,
     reg_alpha=6.294093849568123,
     reg_lambda= 5.5559072866866455,
     random_state=42,
     verbosity =-1
)


lgb_clf = make_pipeline(Feature_Eng(), ohe, lgb_model)
# Fit the model
lgb_clf.fit(X_tr, y_tr)


# import numpy as np
# import pandas as pd
# import logging

# logging.basicConfig(level=logging.ERROR, format="%(asctime)s - %(levelname)s - %(message)s")
# logger = logging.getLogger(__name__)

# def predict_eval(data, target, k, final=False):
#     """Evaluate model predictions with enhanced error handling and validation."""

#     # Validate input types
#     if not isinstance(data, pd.DataFrame):
#         raise TypeError("Expected `data` to be a pandas DataFrame.")
#     if not isinstance(target, np.ndarray):
#         raise TypeError("Expected `target` to be a NumPy array.")

#     try:
#         if not final:
#             # Predict probabilities without retraining
#             y_hat = lgb_clf.predict_proba(data)
#         else:
#             # Retrain model before predicting
#             lgb_clf.fit(X_train_prep, y_enc)
#             y_hat = lgb_clf.predict_proba(data)

#         # Sort predictions by probability
#         sorted_y_hat_ids = np.argsort(-y_hat)
#         top_k_ids = sorted_y_hat_ids[:, :k][:, ::-1]
#         top_k_ids_reshaped = top_k_ids.reshape(-1, 1)

#         # Reverse encoding
#         sorted_names_reshaped = Encoder.inverse_transform(top_k_ids_reshaped)
#         sorted_names = sorted_names_reshaped.reshape(top_k_ids.shape)

#         # Compute MAP@k score
#         try:
#             map_k = mapk(Encoder.inverse_transform(target.reshape(-1, 1)), sorted_names, k=k)
#             logger.info(f'MAP@{k} score: {map_k:.8f}')
#         except Exception as e:
#             logger.error(f"Failed to compute MAP@{k}: {e}")

#         # Build predictions dataframe
#         try:
#             actual_values = Encoder.inverse_transform(target) if target is not None else None
#             preds_df = pd.DataFrame({
#                 'id': data.index,
#                 f'First {k} Fertilizer': [' | '.join(preds) for preds in sorted_names],
#                 'Actual': actual_values if actual_values is not None else "Unknown"
#             })
#         except Exception as e:
#             logger.error(f"Error while creating predictions dataframe: {e}")
#             preds_df = pd.DataFrame({
#                 'id': data.index,
#                 f'First {k} Fertilizer': [' '.join(preds) for preds in sorted_names]
#             })

#         # Display sample predictions
#         display(preds_df.head(10))
#         return preds_df

#     except Exception as e:
#         logger.critical(f"Unexpected error in `predict_eval`: {e}")
#         return None



# def predict_eval(data, target, k, final=False):
#     if not final:
#         # predict_proba on va set
#         y_hat = lgb_clf.predict_proba(data)
#     else:
#         lgb_clf.fit(X_train_prep, y_enc)
#         # predict_proba on va set
#         y_hat = lgb_clf.predict_proba(data)
#     # Sort the predictions
#     sorted_y_hat_ids = np.argsort(-y_hat)
#     # Pick the three indices with the highest probabilities
#     top_k_ids = sorted_y_hat_ids[:,:k][:, ::-1]
#     # Reshape to 2D array
#     top_k_ids_reshaped =  top_k_ids.reshape(-1, 1)
#     # Reverse the encoding
#     sorted_names_reshaped = Encoder.inverse_transform(top_k_ids_reshaped)
#     # Reshape the array to its initial shape
#     sorted_names = sorted_names_reshaped.reshape(top_k_ids.shape)
#     # View the predictions
#     try:
#         # Score the model
#         map_k = mapk(Encoder.inverse_transform(target.reshape(-1, 1)), sorted_names, k=k)
#         print('\nmap_{}_score: {:.8}'.format(k, map_k))
#         # Build validation prediction dataframe
#     except:
#         pass
#     try:
#         preds_df = pd.DataFrame({
#             'id': data.index,
#             'Fertilizer Name': [' '.join(preds) for preds in sorted_names],
#             'Actual': Encoder.inverse_transform(target)
#         })
#     except:
#         preds_df = pd.DataFrame({
#             'id': data.index,
#             'Fertilizer Name': [' '.join(preds) for preds in sorted_names],
#         })
#     # Display some of the predictions
#     display(preds_df.head(10))
#     return preds_df


def predict_eval(data, target, k, final=False):
    if not final:
        # predict_proba on va set
        y_hat = lgb_clf.predict_proba(data)
    else:
        lgb_clf.fit(X_train_prep, y_enc)
        # predict_proba on va set
        y_hat = lgb_clf.predict_proba(data)
    # Sort the predictions
    sorted_y_hat_ids = np.argsort(-y_hat)
    # Pick the three indices with the highest probabilities
    top_k_ids = sorted_y_hat_ids[:,:k][:, ::-1]
    # Reshape to 2D array
    top_k_ids_reshaped =  top_k_ids.reshape(-1, 1)
    # Reverse the encoding
    sorted_names_reshaped = Encoder.inverse_transform(top_k_ids_reshaped)
    # Reshape the array to its initial shape
    sorted_names = sorted_names_reshaped.reshape(top_k_ids.shape)
    # View the predictions

    try:
        map_k = mapk(Encoder.inverse_transform(target.reshape(-1, 1)), sorted_names, k=k)
        print(f'\nmap_{k}_score: {map_k:.8f}')
        print(f'\nmap_3_score: {map_3:.8f}')
    except Exception as e:
        print(f"Error while calculating MAP@{k}: {e}")

    try:
        actual_values = Encoder.inverse_transform(target) if target is not None else None
        preds_df = pd.DataFrame({
            'id': data.index,
            'Fertilizer Name': [' | '.join(preds) for preds in sorted_names],
            'Actual': actual_values if actual_values is not None else "Unknown"
        })
    except Exception as e:
        print(f"Error while creating predictions dataframe: {e}")
        preds_df = pd.DataFrame({
            'id': data.index,
            'Fertilizer Name': [' '.join(preds) for preds in sorted_names]
        })

    # Display some of the predictions
    display(preds_df.head(10))
    return preds_df


va_preds_df = predict_eval(X_va, y_va, 3)


va_preds_df.head(50)


submission = predict_eval(X_test_prep, _, 3, True)


submission.to_csv('submission.csv', index=False)
print('The file is ready for submission')

