import pandas as pd
import numpy as np
import seaborn as sns
import warnings

from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import LabelEncoder

from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder


def plot_compare(train, test, col, kind='numeric'):
    plt.figure(figsize=(12, 4))
    
    if kind == 'numeric':
        plt.subplot(1, 2, 1)
        sns.histplot(train[col], kde=True, color='blue')
        plt.title(f'Train - {col}')
        
        plt.subplot(1, 2, 2)
        sns.histplot(test[col], kde=True, color='orange')
        plt.title(f'Test - {col}')
        
    elif kind == 'categorical':
        plt.subplot(1, 2, 1)
        train[col].value_counts(normalize=True).plot(kind='bar', color='blue')
        plt.title(f'Train - {col}')
        
        plt.subplot(1, 2, 2)
        test[col].value_counts(normalize=True).plot(kind='bar', color='orange')
        plt.title(f'Test - {col}')
        
    plt.tight_layout()
    plt.show()

def val_loss_function(actual, predicted, k=3):
    score = 0.0
    for a, p in zip(actual, predicted):
        if a in p:
            index = p.index(a)
            score += 1.0 / (index + 1)
    return score / len(actual)

def cross_val_predict(model, X_train, y_train, X_test, val_fn, n_splits=20, random_state=42, prob_threshold=0.1):
    print(f"Model: {model.__class__.__name__}")

    num_classes = len(np.unique(y_train))
    oof_preds = np.zeros((X_train.shape[0], num_classes))
    test_preds_sum = np.zeros((X_test.shape[0], num_classes))

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    val_score = 0.0

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
        print(f"\nFold {fold + 1}")

        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

        model.fit(X_tr, y_tr)
        val_proba = model.predict_proba(X_val)
        oof_preds[val_idx] = val_proba

        top3_val = np.argsort(val_proba, axis=1)[:, -3:][:, ::-1]
        fold_map3 = val_fn(y_val, top3_val.tolist())
        print(f"MAP@3 for fold {fold + 1}: {fold_map3:.5f}")
        val_score += fold_map3 / n_splits

        test_preds_sum += model.predict_proba(X_test) / n_splits

    top3_oof = np.argsort(oof_preds, axis=1)[:, -3:][:, ::-1]

    topk_test = []
    for row in test_preds_sum:
        indices = np.argsort(row)[::-1]  # sortowanie malejąco
        filtered = [idx for idx in indices if row[idx] >= prob_threshold]
        topk_test.append(filtered[:3])  # maksymalnie 3 etykiety

    print(f"\nAverage MAP@3 validation score: {val_score:.5f}")

    return top3_oof, topk_test, val_score


train = pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv')


train


train['Fertilizer Name'].value_counts()


target = 'Fertilizer Name'

numeric_cols = train.select_dtypes(include='number').columns
categorical_cols = train.select_dtypes(include='object').columns.drop(target)


for col in list(numeric_cols):
    plot_compare(train, test, col, kind='numeric')

for col in list(categorical_cols):
    # plot_compare(train, test, col, kind='categorical')
    print(col)


def preprocess_data(df, encoder=None, fit=True):
    df = df.drop(columns=['id'], errors='ignore')

    categorical_cols = ['Crop Type', 'Soil Type']
    numerical_cols = [col for col in df.columns if col not in categorical_cols]

    if fit:
        encoder = OneHotEncoder(drop='first', handle_unknown='ignore', sparse=False)
        X_cat_encoded = encoder.fit_transform(df[categorical_cols])
    else:
        X_cat_encoded = encoder.transform(df[categorical_cols])

    X_cat_df = pd.DataFrame(X_cat_encoded, columns=encoder.get_feature_names_out(categorical_cols), index=df.index)
    
    df_processed = pd.concat([df[numerical_cols], X_cat_df], axis=1)

    df['N_P_ratio'] = df['Nitrogen'] / (df['Phosphorous'] + 1e-5)
    df['K_total'] = df['Potassium'] + df['Phosphorous'] + df['Nitrogen']
    df['Moisture_Temp_ratio'] = df['Moisture'] / (df['Temparature'] + 1e-5)

    

    return df_processed, encoder


train, encoder = preprocess_data(train, fit=True)
test, _ = preprocess_data(test, encoder=encoder, fit=False)


num_cols = train.columns.drop(target)
num_cols


X_train = train[num_cols]
X_test = test[num_cols]
y_train = train[target]


X_train.head(2)


y_train


fertilizer_to_id = {
    '14-35-14': 0,
    '10-26-26': 1,
    '17-17-17': 2,
    '28-28':    3,
    '20-20':    4,
    'DAP':      5,
    'Urea':     6
}

y_train_encoded = y_train.map(fertilizer_to_id)


y_train_encoded


models = [
    XGBClassifier(
    tree_method='gpu_hist',
    predictor='gpu_predictor',
    gpu_id=0),
    
    # CatBoostClassifier(verbose=0, 
    #                    task_type='GPU',
    #                     devices='0',)
]

results = {}

for model in models:
    oof, test, score = cross_val_predict(model, X_train, y_train_encoded, 
                                         X_test, val_loss_function)
    results[model.__class__.__name__] = {
        "oof": oof,
        "test": test,
        "score": score
    }
    print(f"Final validation score for {model.__class__.__name__}: {score}\n")


y_pred = results['XGBClassifier']['test']


fertilizer_to_id = {
    '14-35-14': 0,
    '10-26-26': 1,
    '17-17-17': 2,
    '28-28':    3,
    '20-20':    4,
    'DAP':      5,
    'Urea':     6
}

id_to_fertilizer = {v: k for k, v in fertilizer_to_id.items()}

y_pred_labels = [' '.join([id_to_fertilizer[i] for i in row]) for row in y_pred]


sub = pd.read_csv('/kaggle/input/playground-series-s5e6/sample_submission.csv')
sub[target] = y_pred_labels
sub.to_csv('submission.csv', index = False)
sub

