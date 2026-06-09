import pandas as pd
import numpy as np
import lightgbm as lgb
import seaborn as sns
import os
import matplotlib.pyplot as plt


from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.datasets import make_classification
from sklearn.feature_selection import mutual_info_classif



pd.set_option('display.max_rows', 200)


for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

train = pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv')
original = pd.read_csv('/kaggle/input/fertilizers-original-dataset/Fertilizer Prediction.csv')
prediction_fertilizer = pd.read_csv('/kaggle/input/d/irakozekelly/fertilizer-prediction/Fertilizer Prediction.csv')
train.head()


original.head()


prediction_fertilizer.head()


def preprocessing(df):
    df = df.copy()
    df.columns = df.columns.str.lower().str.replace(' ', '_')

    if 'id' in df.columns:
        df.drop(['id'], axis=1, inplace=True)

    label_encoders = {}
    cat_cols = df.select_dtypes(include='object').columns

    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

    return df, label_encoders


X_train = pd.concat([original, train, prediction_fertilizer], axis=0).reset_index(drop=True)
X_test, label_encoders_test = preprocessing(test)
X_train, label_encoders = preprocessing(X_train)


def calculate_mutual_info(data, discrete_features='auto', random_state=42, plot=True):
    X = data.drop('fertilizer_name', axis=1)
    y = data['fertilizer_name']

    # calculate mutual information
    mi = mutual_info_classif(X, y, discrete_features=discrete_features, random_state=random_state)
    mi_series = pd.Series(mi, index=X.columns).sort_values(ascending=False)

     # Ğ¡ÑƒĞ¼Ğ¼Ğ° Ğ¸ Ñ�Ñ€ĞµĞ´Ğ½ĞµĞµ
    mi_sum = mi_series.sum()
    mi_mean = mi_series.mean()

    print(f"Summary of Mutual Information: {mi_sum:.6f}")
    print(f"Average mutual information per feature: {mi_mean:.6f}")

    if plot:
        plt.figure(figsize=(10, 6))
        sns.barplot(x=mi_series.values, y=mi_series.index)
        plt.title('Mutual Information between Features and Target')
        plt.xlabel('Mutual Information')
        plt.ylabel('Features')
        plt.tight_layout()
        plt.show()

    return mi_series


calculate_mutual_info(X_train)


X_train['humidity'].unique()


def create_full_feature_set(df):
    df_new = df.copy()

    numeric_features = ['temparature', 'humidity', 'moisture', 'nitrogen', 'phosphorous', 'potassium']
    
    # â–¶ï¸� Binned features (categorical)
    for feature in numeric_features:
        df_new[f'{feature}_binned'] = df_new[feature].astype(str)

    # â–¶ï¸� Interaction features
    df_new['NPK_ratio'] = df_new['nitrogen'] / (df_new['phosphorous'] + df_new['potassium'] + 1e-5)
    df_new['PK_ratio'] = df_new['phosphorous'] / (df_new['potassium'] + 1e-5)
    df_new['temp_humidity_interaction'] = df_new['temparature'] * df_new['humidity']
    df_new['moisture_npk_sum'] = (df_new['moisture'] + df_new['nitrogen'] + df_new['phosphorous'] + df_new['potassium'] ) 

    # â–¶ï¸� NPK ratios relative to minimum
    min_npk = df_new[['nitrogen', 'phosphorous', 'potassium']].min(axis=1).replace(0, 1)
    df_new['N_ratio'] = df_new['nitrogen'] / min_npk
    df_new['P_ratio'] = df_new['phosphorous'] / min_npk
    df_new['K_ratio'] = df_new['potassium'] / min_npk
    
    # â–¶ï¸� Soil Moisture Index (SMI)
    df_new['SMI'] = (df_new['humidity']**3) / ((df_new['temparature']**2) + 1e-7)

    # Polynomial Feature Engineering 
    df_new['nitrogen_squred'] = df_new['nitrogen'] ** 2
    df_new['phosphorous_squred'] = df_new['phosphorous'] ** 2
    df_new['potassium_squred'] = df_new['potassium'] ** 2
    df_new['temparature_squred'] = df_new['temparature'] ** 2
    df_new['humidity_squred'] = df_new['humidity'] ** 2
    df_new['moisture_squred'] = df_new['moisture'] ** 2

    # New categorical with combine soil_type and crop_type
    df_new['soil_type_crop_type'] = df_new['soil_type'].astype(str) + "_" + df_new['crop_type'].astype(str)
    
    return df_new


X_test =  create_full_feature_set(X_test)
X_train =  create_full_feature_set(X_train)


X = X_train.drop('fertilizer_name', axis=1)
y = X_train['fertilizer_name']


X_test.columns


X.columns


X.shape, X_test.shape, y.shape


num_classes = 7

def map3_metric(y_true, y_pred_proba):
    top_3 = np.argsort(y_pred_proba, axis=1)[:, -3:][:, ::-1]
    return np.mean([
        1 / (np.where(top == true)[0][0] + 1) if true in top else 0
        for top, true in zip(top_3, y_true)
    ])


base_lgb_parameters  = {
    'objective': 'multiclass',
    'num_class': num_classes,
    'metric': 'multi_logloss',
    'boosting_type': 'gbdt',
    'num_leaves': 64,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'max_depth': 8,
    'min_data_in_leaf': 20,
    'lambda_l1': 0.1,
    'lambda_l2': 0.1,
    'verbosity': -1,
    'random_state': 42,
    'early_stopping_rounds': 100
    }

def train_lgb_models(X, y, X_test, n_splits=5, random_state=42):

    try:
        import pynvml
        pynvml.nvmlInit()
        gpu_available = True
    except Exception:
        gpu_available = False

    # ĞšĞ¾Ğ¿Ğ¸Ñ€ÑƒĞµĞ¼ Ğ¿Ğ°Ñ€Ğ°Ğ¼ĞµÑ‚Ñ€Ñ‹ Ğ¸ Ğ´Ğ¾Ğ±Ğ°Ğ²Ğ»Ñ�ĞµĞ¼ ÑƒÑ�Ñ‚Ñ€Ğ¾Ğ¹Ñ�Ñ‚Ğ²Ğ¾
    lgb_parameters = base_lgb_parameters .copy()
    if gpu_available:
        lgb_parameters['device'] = 'gpu'
        print("âœ… GPU detected. Using GPU for training.")
    else:
        lgb_parameters['device'] = 'cpu'
        print("âš ï¸�  GPU not available. Falling back to CPU.")
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    categorical = [col for col in X.columns if X[col].dtype == 'object']

    for col in categorical:
        X[col] = X[col].astype("category")
        X_test[col] = X_test[col].astype("category")

    map3_scores = []
    models = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f" Fold {fold + 1}")

        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=categorical)
        val_data = lgb.Dataset(X_val, label=y_val, categorical_feature=categorical)

        model = lgb.train(
            lgb_parameters,
            train_data,
            num_boost_round=8000,
            valid_sets=[val_data],
            callbacks=[
                lgb.early_stopping(stopping_rounds=100),
                lgb.log_evaluation(period=1000)
            ])

        val_proba = model.predict(X_val)
        fold_score = map3_metric(y_val, val_proba)
        map3_scores.append(fold_score)
        print(f" MAP@3 Fold {fold + 1}: {fold_score:.5f}")

        models.append(model)

    print(f"\n Average MAP@3 across folds: {np.mean(map3_scores):.5f}")
    
    return models


models = train_lgb_models(X, y, X_test, n_splits=5, random_state=42)


def create_submission(models, X_test, ids, label_encoder, filename="submission.csv", top_n=3, separator=" "):
   
    n_classes = models[0].params['num_class']
    test_preds = np.zeros((X_test.shape[0], n_classes))

    # Averaging predictions from all models
    for model in models:
        preds = model.predict(X_test, num_iteration=model.best_iteration)
        test_preds += preds
    test_preds /= len(models)

    # Top-N indices
    top_n_idx = np.argsort(test_preds, axis=1)[:, -top_n:][:, ::-1]

    # Back to titles
    top_n_labels = label_encoder.inverse_transform(top_n_idx.flatten()).reshape(top_n_idx.shape)

    # Formation of prediction strings
    prediction_strings = [separator.join(row) for row in top_n_labels]

    # Final DataFrame
    submission = pd.DataFrame({'id': ids, 'Fertilizer Name': prediction_strings})
    submission.to_csv(filename, index=False)
    print(f"âœ… Submission file '{filename}' created successfully.")
    
    return submission


label_encoder = label_encoders['fertilizer_name']
submission = create_submission(models, X_test, test['id'], label_encoder=label_encoder)


def plot_feature_importance(models, top_n=30, importance_type='gain'):

    # We collect importances from all models
    all_importances = []

    for i, model in enumerate(models):
        imp_df = pd.DataFrame({
            'feature': model.feature_name(),
            'importance': model.feature_importance(importance_type=importance_type),
            'model': f'fold_{i + 1}'
        })
        all_importances.append(imp_df)

    # Let's unite
    full_importance_df = pd.concat(all_importances, axis=0)

    # Aggregate: average importance across all models
    mean_importance = (
        full_importance_df
        .groupby('feature')['importance']
        .mean()
        .reset_index()
        .sort_values(by='importance', ascending=False)
    )

    plt.figure(figsize=(12, 8))
    sns.barplot(
        data=mean_importance.head(top_n),
        y='feature',
        x='importance',
        palette='coolwarm'
    )
    plt.title(f'Top {top_n} Feature Importances ({importance_type}) averaged over folds')
    plt.xlabel('Mean Importance')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.show()

    return mean_importance


plot_feature_importance(models)

