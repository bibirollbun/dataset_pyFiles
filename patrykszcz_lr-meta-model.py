import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay,auc,roc_curve
from sklearn.metrics import classification_report
import warnings
import xgboost as xgb
from sklearn.ensemble import GradientBoostingClassifier
import lightgbm as lgb

warnings.filterwarnings("ignore", category=FutureWarning)


train = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv',index_col = 'id')
test = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv',index_col = 'id')



train['Stage_fear'] = train['Stage_fear'].map({'No':1,'Yes':2})
test['Stage_fear'] = test['Stage_fear'].map({'No':1,'Yes':2})

train['Drained_after_socializing'] = train['Drained_after_socializing'].map({'No':1, "Yes":2})
test['Drained_after_socializing'] = test['Drained_after_socializing'].map({'No':1, "Yes":2})

train = train.fillna(-1)
test = test.fillna(-1)



cat_columns = ['Stage_fear','Drained_after_socializing']
num_columns = ['Time_spent_Alone','Social_event_attendance','Going_outside','Friends_circle_size','Post_frequency']


train['Personality'] = train['Personality'].map({'Extrovert':0,'Introvert':1})


n_features = len(train.columns)-1
cols = train.drop('Personality',axis=1).columns
fig, axs = plt.subplots(nrows = n_features,ncols=2,figsize = (12, 4 * n_features),dpi = 100)

for i, col in enumerate(cols):
    ## HISTOGRAMS
    sns.histplot(data=train, x=col, hue='Personality', kde=True, ax=axs[i,0], multiple='dodge',palette='seismic')
    axs[i,0].set_title(f'HISTOGRAM_{col}')
    axs[i,0].grid(True, linestyle='--', alpha = 0.5)
    ## VIOLINPLOT
    sns.violinplot(data = train,x='Personality', y = col, ax=axs[i,1],palette='seismic')
    axs[i,1].set_title(f'Personality_vs_{col}')
    axs[i,1].grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()


g = sns.pairplot(data=train, hue='Personality')
g._legend.remove()
plt.tight_layout()
plt.show()


fig = plt.figure(figsize = (8,8),dpi=100)
sns.heatmap(train.corr(),annot=True)
plt.title(f'Correletions')
plt.tight_layout()
plt.show()


X_pca = train.drop('Personality', axis =1 ) 
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X_pca)
## EXPLAINED VARIANCE AFTER PCA
pca.explained_variance_ratio_.sum()


df_pca = pd.DataFrame(X_pca,columns= ['PCA1','PCA2','PCA3'],index=train.index) ## Remember to define the index after dropping duplicates
df_pca = pd.concat([df_pca,train['Personality']],axis=1)
df_pca.head()


fig = plt.figure(figsize=(10,9))
ax = fig.add_subplot(111,projection='3d')

scatter = ax.scatter(
    df_pca['PCA1'],
    df_pca['PCA2'],
    df_pca['PCA3'],
    c = df_pca['Personality'],
    cmap='rainbow',
    marker='o',
    alpha=0.6,
)
ax.set_title(f'3D PLOT')
ax.set_xlabel('PCA1')
ax.set_ylabel('PCA2')
ax.set_zlabel('PCA3')
plt.colorbar(scatter)
plt.show()


X = train.drop('Personality', axis=1)
y = train['Personality']


oof_1 = np.zeros(len(X))
test_preds_1 = np.zeros(len(test))  

oof_2 = np.zeros(len(X))
test_preds_2 = np.zeros(len(test))

oof_3 = np.zeros(len(X))
test_preds_3 = np.zeros(len(test))

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

params_1 = {
    'max_depth': 12,
    'colsample_bytree': 0.5590480329278282, 
    'subsample': 0.4572883708952284,
    'learning_rate': 0.009106526547027212,
    'gamma': 0.7861025486849484, 
    'max_delta_step': 3,
    'reg_alpha': 0.36003155961495115,
    'reg_lambda': 0.8184527950570306,
    'random_state': 42,
    'tree_method': 'hist',
    'use_label_encoder': False,
    'eval_metric': 'logloss',
    'verbosity': 0,
    'device': 'cuda',
    'enable_categorical': True
}
params_2 = {
    'max_depth': 17,
    'colsample_bytree': 0.4676229983116439,
    'subsample': 0.6291279139685357,
    'learning_rate': 0.04066897855866283,
    'gamma': 0.5687531379507675,
    'max_delta_step': 2,
    'reg_alpha': 1.7497269071672237,
    'reg_lambda': 1.720417488093518,
    'random_state': 42,
    'tree_method': 'hist',
    'use_label_encoder': False,
    'eval_metric': 'logloss',
    'verbosity': 0,
    'device': 'cuda',
    'enable_categorical': True
}

params_3 = {
    'max_depth': 15, 
    'colsample_bytree': 0.5229756674615937,
    'subsample': 0.41369064730226945,
    'learning_rate': 0.07946121695724359,
    'gamma': 0.9033772095580684,
    'max_delta_step': 1, 
    'reg_alpha': 1.3455207125071622,
    'reg_lambda': 0.6412838269619918,
    'random_state': 42,
    'tree_method': 'hist',
    'use_label_encoder': False,
    'eval_metric': 'logloss',
    'verbosity': 0,
    'device': 'cuda',
    'enable_categorical': True
}

for i, (train_idx, valid_idx) in enumerate(skf.split(X, y)):
    x_train, x_valid = X.iloc[train_idx], X.iloc[valid_idx]
    y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

    ## model 1
    model_1 = XGBClassifier(**params_1, n_estimators=6_000, early_stopping_rounds=50)
    model_1.fit(x_train, y_train, eval_set=[(x_valid, y_valid)], verbose=False)

    preds_valid = model_1.predict(x_valid)
    oof_1[valid_idx] = preds_valid

    acc_score = accuracy_score(y_valid, preds_valid)

    preds_test = model_1.predict_proba(test)[:, 1]
    test_preds_1 += (preds_test >= 0.5).astype(int) / skf.n_splits
    print(f"âœ… MODEL 1 FOLD {i+1}: ACC Score: {acc_score:.5f}")

    ## model 2
    model_2 = XGBClassifier(**params_2, n_estimators=6_000, early_stopping_rounds=50)
    model_2.fit(x_train, y_train, eval_set=[(x_valid, y_valid)], verbose=False)

    preds_valid = model_2.predict(x_valid)
    oof_2[valid_idx] = preds_valid

    acc_score = accuracy_score(y_valid, preds_valid)

    preds_test = model_2.predict_proba(test)[:, 1]
    test_preds_2 += (preds_test >= 0.5).astype(int) / skf.n_splits
    print(f"âœ… MODEL 2 FOLD {i+1}: ACC Score: {acc_score:.5f}")

    ## model 3
    model_3 = XGBClassifier(**params_3, n_estimators=6_000, early_stopping_rounds=50)
    model_3.fit(x_train, y_train, eval_set=[(x_valid, y_valid)], verbose=False)

    preds_valid = model_3.predict(x_valid)
    oof_3[valid_idx] = preds_valid

    acc_score = accuracy_score(y_valid, preds_valid)

    preds_test = model_3.predict_proba(test)[:, 1]
    test_preds_3 += (preds_test >= 0.5).astype(int) / skf.n_splits
    print(f"âœ… MODEL 3 FOLD {i+1}: ACC Score: {acc_score:.5f}")

final_acc = accuracy_score(oof_1, y)
print(f"âœ… Final ACC Score model_1: {final_acc:.5f}")
final_acc = accuracy_score(oof_2, y)
print(f"âœ… Final ACC Score model_2: {final_acc:.5f}")
final_acc = accuracy_score(oof_3, y)
print(f"âœ… Final ACC Score model_3: {final_acc:.5f}")


skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

oof_4 = np.zeros(len(X))
test_preds_4 = np.zeros(len(test))

params_4 = {
    'n_estimators': 716, 
    'learning_rate': 0.011737479687009764,
    'max_depth': 12, 
    'min_samples_split': 4,
    'min_samples_leaf': 10, 
    'subsample': 0.6383811282105497,
    'max_features': 'sqrt'
}

for fold, (train_idx, valid_idx) in enumerate(skf.split(X, y)):
    print(f"##### FOLD {fold+1} #####")
    x_train, x_valid = X.iloc[train_idx], X.iloc[valid_idx]
    y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

    model = GradientBoostingClassifier(**params_4)
    model.fit(x_train, y_train)

    # Prognozy klasy na zbiorze walidacyjnym (standardowy threshold 0.5)
    preds_valid = model.predict(x_valid)
    oof_4[valid_idx] = preds_valid
    acc = accuracy_score(y_valid, preds_valid)

    # Prognozy probabilistyczne na teÅ›cie, a potem binarizacja standardowym progiem 0.5
    test_proba = model.predict_proba(test)[:, 1]
    test_preds_4 += (test_proba >= 0.5).astype(int) / skf.n_splits

    print(f"âœ… ACC: {acc:.5f} ")

final_acc = accuracy_score(y, oof_4)
print(f"Final ACC score model_4: {final_acc:.5f}")


oof_5 = np.zeros(len(X))
test_preds_5 = np.zeros(len(test))  

oof_6 = np.zeros(len(X))
test_preds_6 = np.zeros(len(test))

skf = StratifiedKFold(n_splits=8, shuffle=True, random_state=42)

params_5 = {
    'learning_rate': 0.024186746031817606,
    'num_leaves': 253,
    'max_depth': 5,
    'min_data_in_leaf': 40,
    'feature_fraction': 0.8647314718281799,
    'bagging_fraction': 0.512095016710228,
    'bagging_freq': 10,
    'lambda_l1': 4.133422080729172,
    'lambda_l2': 2.969191943015198,
    'objective': 'binary',
    'metric': 'binary_logloss',
    'boosting_type': 'gbdt',
    'verbosity': -1,
    'n_jobs': -1,
    'device': 'gpu', 
    'early_stopping_rounds':100,
    'verbose': -1,
    'seed': 42,
}

params_6= {
    'learning_rate': 0.014788729006025281,
    'num_leaves': 134,
    'max_depth': 14,
    'min_data_in_leaf': 23,
    'feature_fraction': 0.9433874081612675,
    'bagging_fraction': 0.6067614275625031,
    'bagging_freq': 4,
    'lambda_l1': 4.774862920590915,
    'lambda_l2': 4.736430955111038,
    'random_state': 42,
    'objective': 'binary',
    'metric': 'binary_logloss',
    'boosting_type': 'gbdt',
    'verbosity': -1,
    'n_jobs': -1,
    'device': 'gpu', 
    'early_stopping_rounds':100,
    'verbose': -1,
    'seed': 42,
}

for i, (train_idx, valid_idx) in enumerate(skf.split(X, y)):
    x_train, x_valid = X.iloc[train_idx], X.iloc[valid_idx]
    y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

    ## model 5
    model_5 = lgb.LGBMClassifier(**params_5, n_estimators=6000)
    model_5.fit(
        x_train, y_train,
        eval_set=[(x_valid, y_valid)],
        eval_metric='logloss',
        )
    preds_valid = model_5.predict(x_valid)
    oof_5[valid_idx] = preds_valid
    acc_score = accuracy_score(y_valid, preds_valid)
    preds_test = model_5.predict_proba(test)[:, 1]
    test_preds_5 += (preds_test >= 0.5).astype(int) / skf.n_splits
    print(f"âœ… MODEL 5 FOLD {i+1}: ACC Score: {acc_score:.5f}")

    ## model 6
    model_6 = lgb.LGBMClassifier(**params_6, n_estimators=6000)
    model_6.fit(
        x_train, y_train,
        eval_set=[(x_valid, y_valid)],
        eval_metric='logloss',
        )
    preds_valid = model_6.predict(x_valid)
    oof_6[valid_idx] = preds_valid
    acc_score = accuracy_score(y_valid, preds_valid)
    preds_test = model_6.predict_proba(test)[:, 1]
    test_preds_6 += (preds_test >= 0.5).astype(int) / skf.n_splits
    print(f"âœ… MODEL 6 FOLD {i+1}: ACC Score: {acc_score:.5f}")


final_acc = accuracy_score(y, oof_5)
print(f"âœ… Final ACC Score model_5: {final_acc:.5f}")
final_acc = accuracy_score(y, oof_6)
print(f"âœ… Final ACC Score model_6: {final_acc:.5f}")


def plot_cm(y_true, y_pred, ax=None):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Extrovert', 'Introvert'])
    disp.plot(ax=ax,colorbar=False)

def auc_roc_plot(y_true,y_pred, ax = None):
    fpr, tpr ,_ = roc_curve(y_true,y_pred)
    roc_auc = auc(fpr,tpr)
    ax.plot(fpr, tpr, color='blue', lw=2, label=f'ROC Curve (AUC = {roc_auc:.2f})')
    ax.plot([0, 1], [0, 1], color='grey', linestyle='--')  
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate (FPR)')
    ax.set_ylabel('True Positive Rate (TPR)')
    ax.set_title('ROC Curve')
    ax.legend(loc='lower right')
    ax.grid()


fig, axs = plt.subplots(nrows=6, ncols=2, figsize=(10, 20), dpi=100)
plot_cm(oof_1, y, ax=axs[0,0])
axs[0,0].set_title('Confusion matrix')
auc_roc_plot(oof_1, y, ax=axs[0,1])
axs[0,1].set_title('ROC AUC')

plot_cm(oof_2, y, ax=axs[1,0])
axs[1,0].set_title('Confusion matrix')
auc_roc_plot(oof_2, y, ax=axs[1,1])
axs[1,1].set_title('ROC AUC')

plot_cm(oof_3, y, ax=axs[2,0])
axs[2,0].set_title('Confusion matrix')
auc_roc_plot(oof_3, y, ax=axs[2,1])
axs[2,1].set_title('ROC AUC')

plot_cm(oof_4, y, ax=axs[3,0])
axs[3,0].set_title('Confusion matrix')
auc_roc_plot(oof_4, y, ax=axs[3,1])
axs[3,1].set_title('ROC AUC')

plot_cm(oof_5, y, ax=axs[4,0])
axs[4,0].set_title('Confusion matrix')
auc_roc_plot(oof_5, y, ax=axs[4,1])
axs[4,1].set_title('ROC AUC')

plot_cm(oof_6, y, ax=axs[5,0])
axs[5,0].set_title('Confusion matrix')
auc_roc_plot(oof_6, y, ax=axs[5,1])
axs[5,1].set_title('ROC AUC')

plt.tight_layout()
plt.show()


X_meta = np.column_stack([oof_1,oof_4,oof_5,])
x_test = np.column_stack([test_preds_1,test_preds_4,test_preds_5])


from sklearn.metrics import accuracy_score
import optuna
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
import numpy as np

def objective(trial):
    C = trial.suggest_float('C', 1e-3, 10.0)
    tol = trial.suggest_float('tol', 1e-5, 1e-1)
    penalty = trial.suggest_categorical('penalty', ['l1', 'l2'])
    solver = trial.suggest_categorical('solver', ['saga', 'lbfgs', 'newton-cg', 'newton-cholesky'])
    max_iter = trial.suggest_int('max_iter', 1000, 3001)

    valid_combinations = {
        'saga': ['l1', 'l2'],
        'lbfgs': ['l2'],
        'newton-cg': ['l2'],
        'newton-cholesky': ['l2']
    }

    if penalty not in valid_combinations[solver]:
        raise optuna.exceptions.TrialPruned()

    if penalty == 'elasticnet':
        raise optuna.exceptions.TrialPruned()

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof = np.zeros(len(X_meta))

    for i, (train_idx, valid_idx) in enumerate(skf.split(X_meta, y)):
        print('#' * 15, f" FOLD {i+1} ", '#' * 15)
        x_train, x_valid = X_meta[train_idx], X_meta[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

        try:
            model_params = {
                'C': C,
                'tol': tol,
                'penalty': penalty,
                'solver': solver,
                'max_iter': max_iter,
                'fit_intercept': True
            }

            model = LogisticRegression(**model_params)
            model.fit(x_train, y_train)

            preds = model.predict(x_valid)  # domyÅ›lne progowanie 0.5

            oof[valid_idx] = preds

            acc = accuracy_score(y_valid, preds)
            print(f"âœ… FOLD {i+1}: Accuracy: {acc:.5f}")

        except Exception as e:
            print(f"âš ï¸� Exception on fold {i+1}: {e}")
            return 0.0

    overall_acc = accuracy_score(y, oof)
    print(f"âœ… Overall Accuracy: {overall_acc:.5f}")
    return overall_acc

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=30, show_progress_bar=True)

print("Best trial:")
print(study.best_trial.params)


skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

oof_stack = np.zeros(len(X_meta))
test_preds_stack = []

print("\nStarting 5-Fold LR training...")

for i, (train_idx, valid_idx) in enumerate(skf.split(X_meta, y)):
    lr_model = LogisticRegression(
        **study.best_trial.params,
        fit_intercept=True
    )

    x_train, x_valid = X_meta[train_idx], X_meta[valid_idx]
    y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

    lr_model.fit(x_train, y_train)

    preds_valid_proba = lr_model.predict_proba(x_valid)[:, 1]
    preds_valid = (preds_valid_proba >= 0.5).astype(int)
    oof_stack[valid_idx] = preds_valid

    preds_test_proba = lr_model.predict_proba(x_test)[:, 1]
    test_preds_stack.append(preds_test_proba)

    acc = accuracy_score(y_valid, preds_valid)
    print(f"âœ… FOLD {i+1}: Accuracy: {acc:.5f}")
final_acc = accuracy_score(y, oof_stack)

print(f"\nâœ… Final OOF Accuracy: {final_acc:.5f}")


test_preds_final = (test_preds_stack[2] >= 0.5).astype(int)


test_preds_final  = pd.Series(test_preds_final).map({0: 'Extrovert', 1: 'Introvert'})
test_preds_final


submission = pd.read_csv('/kaggle/input/playground-series-s5e7/sample_submission.csv')
submission['Personality'] = test_preds_final
submission.to_csv('submission.csv',index = False)


submission.head()

