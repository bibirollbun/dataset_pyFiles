import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import optuna
import xgboost
import matplotlib.pyplot as plt
import seaborn as sns


pd.set_option('display.max_columns', None)
pd.set_option('display.width',500)
warnings.filterwarnings("ignore")


print(xgboost.__version__)


train_df = pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv')

train_id = train_df['id']
train_df.drop('id', axis=1, inplace=True)


train_df.head()



test_df.head()


numerical_col = [col for col in train_df.columns if train_df[col].dtype == "int64"]
categorical_cols = [col for col in train_df.columns if col not in numerical_col]


plt.figure(figsize=(14, len(numerical_col) * 3))

for i, col in enumerate(numerical_col):
    # Histogram
    plt.subplot(len(numerical_col), 2, 2*i + 1)
    sns.histplot(train_df[col], kde=True, color="skyblue")
    plt.title(f"{col} Histogram")
    
    # Boxplot
    plt.subplot(len(numerical_col), 2, 2*i + 2)
    sns.boxplot(x=train_df[col], color="skyblue")
    plt.title(f"{col} Boxplot")

plt.tight_layout()
plt.suptitle("NUMERICAL VARIABLES: Histogram + Boxplot", fontsize=16, y=1.02)
plt.show()


plt.figure(figsize=(10, len(categorical_cols) * 3))

for i, col in enumerate(categorical_cols):
    plt.subplot(len(categorical_cols), 1, i + 1)
    sns.countplot(x=train_df[col], palette="Set2")
    plt.title(f"{col} Countplot")
    plt.xticks(rotation=45)

plt.tight_layout()
plt.suptitle("CATEGORICAL VARIABLES", fontsize=16, y=1.02)
plt.show()


train_df['New_Temparature_Cat'] = pd.qcut(train_df['Temparature'],q=3,labels=["cold", "mean", "hot"]).astype('object')

train_df['New_Humidity_Cat'] = pd.qcut(train_df['Humidity'],q=3,labels=["dry", "normal", "humid"]).astype('object')


cat_col = ['Moisture','Nitrogen','Potassium','Phosphorous']
for col in cat_col:
    train_df[f'New_{col}_Cat'] = pd.qcut(train_df[col], q=3, labels=["low", "medium", "high"]).astype('object')



categorical_columns = [col for col in train_df.columns if train_df[col].dtype == 'O']
categorical_columns.remove('Fertilizer Name')

ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
ohe_array = ohe.fit_transform(train_df[categorical_columns])

encoded_cols = ohe.get_feature_names_out(categorical_columns)
encoded_df = pd.DataFrame(ohe_array, columns=encoded_cols, index=train_df.index)

train_df.drop(categorical_columns,axis=1, inplace=True)
train_df = pd.concat([train_df,encoded_df], axis=1)


le = LabelEncoder()
train_df['Fertilizer Name'] =  le.fit_transform(train_df['Fertilizer Name'])



train_df


def data_preprocessing(data):
    print('--------------------- Data Preprocessing Starting --------------------')
    numerical_col = [col for col in data.columns if data[col].dtype == "int64"]

    def outliers_thresholds(dataframe, variable, q1=0.10, q3=0.90):
        quartile1 = dataframe[variable].quantile(q1)
        quartile3 = dataframe[variable].quantile(q3)
        iqr = quartile3 - quartile1
        up_limit = quartile3 + 1.5 * iqr
        low_limit = quartile1 - 1.5 * iqr
        return low_limit, up_limit

    def replace_with_threshold(dataframe, variable):
        low_limit, up_limit = outliers_thresholds(dataframe, variable)
        dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
        dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

    def check_outliers(dataframe, variable):
        low_limit, up_limit = outliers_thresholds(dataframe, variable)
        if dataframe[(dataframe[variable] < low_limit) | (dataframe[variable] > up_limit)].any(axis=None):
            return True
        else:
            return False


    for col in numerical_col:
        if check_outliers(data, col):
            replace_with_threshold(data,col)
    
    ################### Feature Engineering ##########################
    data['New_Temparature_Cat'] = pd.qcut(data['Temparature'], q=3, labels=["cold", "mean", "hot"]).astype(
        'object')

    data['New_Humidity_Cat'] = pd.qcut(data['Humidity'], q=3, labels=["dry", "normal", "humid"]).astype(
        'object')

    cat_col = ['Moisture', 'Nitrogen', 'Potassium', 'Phosphorous']
    for col in cat_col:
        data[f'New_{col}_Cat'] = pd.qcut(data[col], q=3, labels=["low", "medium", "high"]).astype('object')

    ####################### Encoding ######################################################
    categorical_columns = [col for col in data.columns if data[col].dtype == 'O']

    
    ohe_array = ohe.transform(data[categorical_columns])

    encoded_cols = ohe.get_feature_names_out(categorical_columns)
    encoded_df = pd.DataFrame(ohe_array, columns=encoded_cols, index=data.index)

    data.drop(categorical_columns, axis=1, inplace=True)
    data = pd.concat([data, encoded_df], axis=1)
    print('--------------------- Data Preprocessing End --------------------')
    return data



test_id = test_df['id']
test_df.drop('id', axis=1, inplace=True)

test_df = data_preprocessing(test_df)
test_df.head()


X = train_df.drop('Fertilizer Name', axis=1)
y = train_df['Fertilizer Name']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.20, random_state=7)


models = {
    'XGBoost': XGBClassifier(random_state=7, verbosity=0),
    'CatBoost': CatBoostClassifier(random_state=7, verbose=False),
    'LightGBM':LGBMClassifier(random_state=7,verbose=-1)
}

for name, model in models.items():
    print(f'------------{name}----------------')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    print(classification_report(y_val, y_pred))
    print('############################################')



def mapk(y_true, y_pred, k=3):
    score = 0.0
    for true_label, pred_labels in zip(y_true, y_pred):
        try:
            rank = pred_labels.index(true_label)
            if rank < k:
                score += 1.0 / (rank + 1)
        except ValueError:
            continue  # true_label not in top-k predictions
    return score / len(y_true)


for name, model in models.items():
    print(f"------------- {name} -------------")
    model.fit(X_train, y_train)

    y_proba = model.predict_proba(X_val)
    top_3_preds = np.argsort(y_proba, axis=1)[:, -3:][:, ::-1]

    # MAP@3 
    score = mapk(y_val.tolist(), top_3_preds.tolist(), k=3)

    print(f"MAP@3 score: {score:.4f}")
    print("###########################################")


def objective(trial):
    params = {
        'max_depth': trial.suggest_int('max_depth', 8, 16),
        'n_estimators': trial.suggest_int('n_estimators', 1500, 5000),
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.05, log=True),
        'subsample': trial.suggest_float('subsample', 0.4, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 0.7),
        'gamma': trial.suggest_float('gamma', 0.09, 0.7),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.1, 0.5),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.05, 0.5),
        'max_delta_step': trial.suggest_int('max_delta_step', 2, 11),
        'objective': 'multi:softprob',
        'random_state': 7,
        'n_jobs': -1,
        'enable_categorical': True,
        'tree_method': 'gpu_hist',
        'predictor': 'gpu_predictor',
        'eval_metric': 'mlogloss',
    }

    xgb_model = XGBClassifier(**params)

    xgb_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=100,
        verbose=False
    )

    # Pruner'a bilgi ver
    evals_result = xgb_model.evals_result()
    last_mlogloss = evals_result['validation_0']['mlogloss'][-1]
    trial.report(last_mlogloss, step=0)

    if trial.should_prune():
        raise optuna.TrialPruned()

    # MAP@3 hesabÄ±
    y_proba = xgb_model.predict_proba(X_val)
    top_3_preds = np.argsort(y_proba, axis=1)[:, -3:][:, ::-1]
    score = mapk(y_val.tolist(), top_3_preds.tolist(), k=3)

    return score

#pruner = optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=20)
#study = optuna.create_study(direction="maximize", pruner=pruner)
#optuna.logging.set_verbosity(optuna.logging.INFO)
#study.optimize(objective, n_trials=25, catch=(Exception,))


#print("Best trial:")
#print(study.best_trial)


best_params = {
        'max_depth':11,
        'colsample_bytree':0.3508548763606836,
        'subsample':0.6142056550341887,
        'n_estimators':3413,
        'learning_rate':0.009318260040590403,
        'gamma':0.452760148679334,
        'max_delta_step':6,
        'reg_alpha':0.13339412011885346,
        'reg_lambda':0.46614123300729193,
        'objective':'multi:softprob',
        'random_state':7,
        'enable_categorical':True,
        'n_jobs':-1,
        'eval_metric':'mlogloss'
}

final_model = XGBClassifier(**best_params)
final_model.fit(X, y)

print('Model fitting completed')


test_proba = final_model.predict_proba(test_df)
top_3_preds = np.argsort(test_proba, axis=1)[:, -3:][:, ::-1]
top_3_labels = le.inverse_transform(top_3_preds.ravel()).reshape(top_3_preds.shape)

print('The target variable was estimated using the test data set.')


submission = pd.DataFrame({
   'id': test_id,  # test setindeki Ã¶rnek id'leri
   'predictions': [' '.join(row) for row in top_3_labels]
})

submission.columns = ['id','Fertilizer Name' ]
submission.to_csv('submission.csv', index=False)




