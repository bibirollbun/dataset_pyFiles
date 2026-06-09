import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, AdaBoostClassifier, RandomForestClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, roc_curve, auc, roc_auc_score
import optuna

import warnings
warnings.filterwarnings("ignore")

sns.set_theme(context='notebook')


train = pd.read_csv(
    '/kaggle/input/playground-series-s5e3/train.csv', index_col='id')

test = pd.read_csv(
    '/kaggle/input/playground-series-s5e3/test.csv', index_col='id')


train.info()


CAT_COLS = ['rainfall']

NUM_COLS = ['pressure', 'maxtemp', 'temparature', 'mintemp', 'dewpoint',
            'humidity', 'cloud', 'sunshine', 'winddirection', 'windspeed']


train.duplicated().sum()


test.duplicated().sum()


train.isna().sum()


test.isna().sum()


test.fillna(test.mean(), inplace=True)


rainfall_counts = train['rainfall'].value_counts()

plt.figure(figsize=(8, 8))
plt.pie(rainfall_counts, labels=rainfall_counts.index, autopct='%1.1f%%', startangle=140)
plt.title('Rainfall')
plt.show()


grouped_data = train.groupby('rainfall')[NUM_COLS].sum()
grouped_data.T.plot(kind='barh', stacked=True, figsize=(12, 8))

plt.xlabel('Rainfall')
plt.ylabel('Numerical Columns')
plt.legend(title='Rainfall', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()


for col in NUM_COLS:
    plt.figure(figsize=(20, 5))

    plt.subplot(1, 3, 1)
    train[col].plot.hist(bins=20)
    plt.title(f"Histogram of {col}")

    plt.subplot(1, 3, 2)
    stats.probplot(train[col], dist="norm", plot=plt)
    plt.title(f"QQ plot of {col}")

    plt.subplot(1, 3, 3)
    sns.boxenplot(x=train[col])
    plt.title(f"Boxen plot of {col}")

    plt.tight_layout()
    plt.show()


train.drop(columns=['day'], inplace=True)
test.drop(columns=['day'], inplace=True)


scaler = MinMaxScaler()
train[NUM_COLS] = scaler.fit_transform(train[NUM_COLS])
test[NUM_COLS] = scaler.transform(test[NUM_COLS])


predictors = train.drop(columns=['rainfall'])
target = train['rainfall']

train_predictors, eval_predictors, train_target, eval_target = train_test_split(
    predictors, target, test_size=0.2, random_state=42)


models = {
    'KNN': KNeighborsClassifier(),
    'DT': DecisionTreeClassifier(),
    'HGB': HistGradientBoostingClassifier(),
    'ADA': AdaBoostClassifier(),
    'RF': RandomForestClassifier(),
    'XGB': XGBClassifier(),
    'CB': CatBoostClassifier(verbose=0),
    'LGBM': LGBMClassifier(verbose=0)
}


for model in models:
    print('Training', model)
    models[model].fit(train_predictors, train_target)


model_accuracies = {}

for model_name, model in models.items():

    predictions = model.predict(eval_predictors)
    accuracy = accuracy_score(eval_target, predictions)
    model_accuracies[model_name] = accuracy

sorted_model_accuracies = sorted(
    model_accuracies.items(), key=lambda x: x[1], reverse=True)

for model_name, accuracy in sorted_model_accuracies:
    print(f"Model: {model_name}, Accuracy: {accuracy:.4f}")


model_aucs = {}

for model_name, model in models.items():

    y_pred_proba = model.predict_proba(eval_predictors)[:, 1]
    auc_score = roc_auc_score(eval_target, y_pred_proba)
    model_aucs[model_name] = auc_score

sorted_model_aucs = sorted(
    model_aucs.items(), key=lambda x: x[1], reverse=True)

for model_name, auc_score in sorted_model_aucs:
    print(f"Model: {model_name}, AUC: {auc_score:.4f}")


plt.figure(figsize=(20, 10))

for i, (model_name, model) in enumerate(models.items(), 1):

    y_pred_proba = model.predict_proba(eval_predictors)[:, 1]
    fpr, tpr, _ = roc_curve(eval_target, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    plt.subplot(len(models) // 4 + 1, 4, i)
    plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve for {model_name}')
    plt.legend(loc="lower right")

plt.tight_layout()
plt.show()


def objective(trial):
    param = {
        'iterations': trial.suggest_int('iterations', 100, 1000),
        'depth': trial.suggest_int('depth', 1, 10),
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-3, 1e-1),
        'l2_leaf_reg': trial.suggest_loguniform('l2_leaf_reg', 1e-3, 10),
        'border_count': trial.suggest_int('border_count', 1, 255),
        'random_strength': trial.suggest_loguniform('random_strength', 1e-3, 10),
        'bagging_temperature': trial.suggest_loguniform('bagging_temperature', 1e-3, 10),
        'od_type': trial.suggest_categorical('od_type', ['IncToDec', 'Iter']),
        'od_wait': trial.suggest_int('od_wait', 10, 50)
    }

    model = CatBoostClassifier(**param, verbose=0)
    model.fit(train_predictors, train_target, eval_set=(eval_predictors, eval_target), early_stopping_rounds=100, verbose=0)
    predictions = model.predict(eval_predictors)
    accuracy = accuracy_score(eval_target, predictions)
    return accuracy

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)

print('Number of finished trials:', len(study.trials))
print('Best trial:', study.best_trial.params)


best_params = study.best_params
final_model = CatBoostClassifier(**best_params, verbose=0)
final_model.fit(predictors, target)


test['rainfall'] = final_model.predict_proba(test)[:, 1]
test.to_csv('s5e3-cb-submission.csv', columns=['rainfall'], index=True)

