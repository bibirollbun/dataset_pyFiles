import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")



train = pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv")
sub = pd.read_csv("/kaggle/input/playground-series-s5e7/sample_submission.csv")


train.head()


train.info()


train.describe([0.05, 0.25, 0.75, 0.95])


print(train['Stage_fear'].unique())
print(train['Drained_after_socializing'].unique())
print(train['Personality'].unique())



# making thooose yes to 1  and no to 0 
train['Stage_fear'] = train['Stage_fear'].map({'Yes': 1, 'No': 0})
train['Drained_after_socializing'] = train['Drained_after_socializing'].map({'Yes': 1, 'No': 0})

test['Stage_fear'] = test['Stage_fear'].map({'Yes': 1, 'No': 0})
test['Drained_after_socializing'] = test['Drained_after_socializing'].map({'Yes': 1, 'No': 0})



# checking if there is any missing value 
print(train.isnull().sum())
print("--------------")
print(test.isnull().sum())



#checking if there is any duplicated data
train.duplicated().sum()



# Stage_fear and Drained_after_socializing, i take empty data as NO so i filled them with 0 
train['Stage_fear'] = train['Stage_fear'].map({'Yes': 1, 'No': 0})
train['Drained_after_socializing'] = train['Drained_after_socializing'].map({'Yes': 1, 'No': 0})

test['Stage_fear'] = test['Stage_fear'].map({'Yes': 1, 'No': 0})
test['Drained_after_socializing'] = test['Drained_after_socializing'].map({'Yes': 1, 'No': 0})

train['Stage_fear'].fillna(0, inplace=True)
train['Drained_after_socializing'].fillna(0, inplace=True)
test['Stage_fear'].fillna(0, inplace=True)
test['Drained_after_socializing'].fillna(0, inplace=True)


# and for the numeric ones, i fill them with target disturbition  median
for col in ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside', 'Friends_circle_size', 'Post_frequency']:
    train.loc[train['Personality'] == 'Introvert', col] = train.loc[train['Personality'] == 'Introvert', col].fillna(
        train.loc[train['Personality'] == 'Introvert', col].median()
    )
    train.loc[train['Personality'] == 'Extrovert', col] = train.loc[train['Personality'] == 'Extrovert', col].fillna(
        train.loc[train['Personality'] == 'Extrovert', col].median()
    )
    # test doesnt have target so i filled it with train's median
    test[col].fillna(train[col].median(), inplace=True)



num_cols = ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside', 
            'Friends_circle_size', 'Post_frequency']

for col in num_cols:
    plt.figure(figsize=(7,4))
    sns.kdeplot(data=train, x=col, hue='Personality', fill=True, alpha=0.4)
    plt.title(f'{col} distribution by Personality')
    plt.xlabel(col)
    plt.ylabel('Density')
    plt.show()


def plot_target(df: pd.DataFrame, col: str, title: str, pie_colors:list) -> None:
    fig, ax = plt.subplots(1,2,figsize=(15, 6), width_ratios=[2,1])

    textprops={'fontsize': 12, 'weight': 'bold',"color": "black"}
    ax[0].pie(df[col].value_counts().to_list(),
            colors=pie_colors,
            labels=df[col].value_counts().index.to_list(),
            autopct='%1.f%%', 
            explode=([.05]*df[col].nunique()),
            pctdistance=0.5,
            wedgeprops={'linewidth' : 1, 'edgecolor' : 'black'}, 
            textprops=textprops)

    sns.countplot(x = col, data=df, palette = "pastel6", order=df[col].value_counts().to_dict().keys())
    for p, count in enumerate(df[col].value_counts().to_dict().values(),0):
        ax[1].text(p-0.11, count+np.sqrt(count)+1000, count, color='black', fontsize=13)
    plt.setp(ax[1].get_xticklabels(), fontweight="bold")
    plt.yticks([])
    plt.box(False)
    fig.suptitle(x=0.56, t=f'â–º {title} Distribution â—„', fontsize=18, fontweight='bold')
    plt.tight_layout()
    plt.show()
    


plot_target(train, 
            col="Personality", 
            title="Target analysis", 
            pie_colors=["#abc9ea","#98daa7","#f3aba8","#d3c3f7","#f3f3af","#c0ebe9"])


#making this target column to numeric
train['Personality'] = train['Personality'].map({'Introvert': 0, 'Extrovert': 1})



X = train.drop(columns=['id', 'Personality'])
y = train['Personality']


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.1, random_state = 42)


from lightgbm import LGBMClassifier
import optuna

def objective_lgb(trial):
    """Define the objective function"""

    params = {
        'metric': trial.suggest_categorical('metric', ['auc']),
        'max_depth': trial.suggest_int('max_depth', 1, 10),
        'min_child_samples': trial.suggest_int('min_child_samples', 1, 15),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.05),
        'n_estimators': trial.suggest_int('n_estimators', 300, 700),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'subsample': trial.suggest_float('subsample', 0.1, 0.9),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.01, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.01, 1.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.01, 1.0),
        "seed" : trial.suggest_categorical('seed', [42]),
        'device': trial.suggest_categorical('device', ['cpu']),
        'verbose': -1
    }


    model_lgb = LGBMClassifier(**params)
    model_lgb.fit(X_train, y_train)
    y_pred = model_lgb.predict_proba(X_test)[:,1]
    return roc_auc_score(y_test,y_pred)


study_lgb = optuna.create_study(direction='maximize')
study_lgb.optimize(objective_lgb, n_trials=50)

print("Best trial:")
print(study.best_trial.params)
print("Best ROC-AUC:", study.best_value)



lgb = LGBMClassifier(**study_lgb.best_params)
lgb.fit(X_train, y_train)
y_pred = lgb.predict_proba(X_test)[:,1]
print('Accuracy: ', roc_auc_score(y_test, y_pred))


from sklearn.metrics import ConfusionMatrixDisplay

ConfusionMatrixDisplay.from_estimator(lgb,X_test, y_test,display_labels=("False", "True"),cmap="RdPu");


from xgboost import XGBClassifier
import optuna
def objective_xg(trial):
    """Define the objective function"""

    params = {
        'booster': trial.suggest_categorical('booster', ['gbtree']),
        'max_depth': trial.suggest_int('max_depth', 1, 10),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.05),
        'n_estimators': trial.suggest_int('n_estimators', 300, 1000),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma': trial.suggest_loguniform('gamma', 1e-8, 1.0),
        'subsample': trial.suggest_loguniform('subsample', 0.3, 0.9),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.01, 1.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.01, 1.0),
        "seed" : trial.suggest_categorical('seed', [42]),
        'tree_method': trial.suggest_categorical('tree_method', ['auto']),
        'eval_metric': trial.suggest_categorical('eval_metric', ['auc']),
    }
    model_xgb = XGBClassifier(**params)
    model_xgb.fit(X_train, y_train)
    y_pred = model_xgb.predict_proba(X_test)[:,1]
    return roc_auc_score(y_test,y_pred)


study_xgb = optuna.create_study(direction='maximize')
optuna.logging.set_verbosity(optuna.logging.WARNING)
study_xgb.optimize(objective_xg, n_trials=50,show_progress_bar=True)


# Print the best parameters
print('Best parameters', study_xgb.best_params)


xgb = XGBClassifier(**study_xgb.best_params)
xgb.fit(X_train, y_train)
y_pred = xgb.predict_proba(X_test)[:,1]

print('Accuracy: ', roc_auc_score(y_test, y_pred))


from sklearn.metrics import ConfusionMatrixDisplay

ConfusionMatrixDisplay.from_estimator(xgb,X_test, y_test,display_labels=("False", "True"),cmap="RdPu");


from sklearn.ensemble import VotingClassifier
voting = VotingClassifier(estimators=[
                                      ('lgbm', lgb), 
                                      ('xgb', xgb)], voting='soft')
voting.fit(X_train,y_train)
voting_pred = voting.predict_proba(X_test)[:,1]

print('Accuracy: ', roc_auc_score(y_test, voting_pred))


from sklearn.metrics import ConfusionMatrixDisplay

ConfusionMatrixDisplay.from_estimator(voting,X_test, y_test,display_labels=("False", "True"),cmap="RdPu");


sub["Personality"] = voting.predict_proba(test_features)[:, 1]



sub["Personality"] = (voting.predict_proba(test_features)[:, 1] >= 0.5).astype(int)



sub['Personality'] = sub['Personality'].map({0: 'Introvert', 1: 'Extrovert'})



sub.to_csv('submission.csv',index=False)
sub

