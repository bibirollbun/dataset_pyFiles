import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, classification_report
from scipy.stats import randint, uniform
import warnings
import os
warnings.filterwarnings('ignore')

sns.set_style('whitegrid')

os.chdir('/kaggle/working')


df_train = pd.read_csv('/kaggle/input/playground-series-s5e8/train.csv')
df_test = pd.read_csv('/kaggle/input/playground-series-s5e8/test.csv')
df_train.head()


print('Checking for missing values (NaNs) in training data:')
print(df_train.isnull().sum().sum())


df_train.hist(figsize=(12, 10), bins=30, color='skyblue', edgecolor='black')
plt.suptitle('Distributions of Original Numerical Features', fontsize=18)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()


plt.figure(figsize=(12, 10))
numerical_df = df_train.select_dtypes(include=np.number)
correlation_matrix = numerical_df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix of Numerical Features')
plt.show()


print('--- Preparing Data for Modeling ---')
data = df_train.drop(['id'], axis=1)
X = data.drop('y', axis=1)
y = data['y']
numerical_features = X.select_dtypes(include=np.number).columns.tolist()
categorical_features = X.select_dtypes(include='object').columns.tolist()
preprocessor = ColumnTransformer(
    transformers=[('num', StandardScaler(), numerical_features),
                  ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)])
X_processed = preprocessor.fit_transform(X)
X_train, X_val, y_train, y_val = train_test_split(
    X_processed, y, test_size=0.2, random_state=42, stratify=y
)
print('--- Data Ready ---')


# scale_pos_weight = y_train.value_counts()[0] / y_train.value_counts()[1]
# models_to_tune = {
#     'RandomForest': (RandomForestClassifier(random_state=42, class_weight='balanced'), {'n_estimators': [100,300,500,700]}, 'device': 'gpu', 'gpu_platform_id': 0, 'gpu_device_id': 0,),
#     'CatBoost': (CatBoostClassifier(random_state=42, auto_class_weights='Balanced', verbose=0), {'iterations': randint(100, 200)},'learning_rate': 0.005, 'max_depth': 4, 'random_state': SEED,
#             'tree_method': 'gpu_hist',      # <--- GPU-accelerated algorithm
#         'predictor': 'gpu_predictor',  ),
#     'LightGBM': (LGBMClassifier(random_state=42, class_weight='balanced'), {'n_estimators': [100,300,500,700]}),
#     'XGBoost': (XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss', scale_pos_weight=scale_pos_weight), {'n_estimators': [100,300,500,700]}),
#     'ANN': (MLPClassifier(random_state=42, max_iter=1000, early_stopping=True, hidden_layer_sizes=(100,50)), {'alpha': uniform(0.0001, 0.05)})
# }
# best_estimators = {}
# final_results = {}
# print('--- Tuning and Evaluating Individual Models ---')
# for name, (model, param_dist) in models_to_tune.items():
#     print(f'Tuning {name}...', end='')
#     random_search = RandomizedSearchCV(estimator=model, param_distributions=param_dist, n_iter=4, scoring='accuracy', cv=3, n_jobs=-1, verbose=0, random_state=42)
#     random_search.fit(X_train, y_train)
#     best_estimators[name] = random_search.best_estimator_
#     y_pred = best_estimators[name].predict(X_val)
#     accuracy = accuracy_score(y_val, y_pred)
#     final_results[name] = accuracy
#     print(f' Done. Accuracy on internal validation: {accuracy:.4f}')
#     print(f' {name} best parameter: {random_search.best_params_}')

# print('--- Evaluating The Ultimate Ensemble Model ---')
# print('Training the VotingClassifier...', end='')
# voting_clf = VotingClassifier(estimators=list(best_estimators.items()), voting='soft')
# voting_clf.fit(X_train, y_train)
# y_pred_voting = voting_clf.predict(X_val)
# accuracy_voting = accuracy_score(y_val, y_pred_voting)
# final_results['VotingClassifier'] = accuracy_voting
# print(f' Done. Accuracy on internal validation: {accuracy_voting:.4f}')





# cv = StratifiedKFold(n_splits=10, random_state=42, shuffle=True)
# for name, model in best_estimators.items():
#     print(f'Valdite {name}...', end='')
#     score = cross_validate(model, X_train, y_train, cv=cv, scoring=scoring)
    
#     avg_scores = {m: score[f'test_{m}'].mean() for m in scoring}
#     print(f"{name} : " + " | ".join([f"{m}: {avg_scores[m]:.5f}" for m in scoring]))


%%time

import pandas as pd 
import numpy as np

!git clone https://github.com/muhammadabdullah0303/AbdML

import sys
sys.path.append('/kaggle/working/repository')

from AbdML.main import AbdBase
SEED = 0


%%time

train = pd.read_csv('/kaggle/input/playground-series-s5e8/train.csv')
sample = pd.read_csv("/kaggle/input/playground-series-s5e8/sample_submission.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e8/test.csv")

train = train.drop('id', axis=1)
test = test.drop('id', axis=1)

%time

def NEW_FE(df):
    
    df['balance_log'] = np.log1p(df['balance'].clip(lower=0))
    df['job_edu'] = df['job'].astype(str) + "_" + df['education'].astype(str)
    df['contacted_before'] = (df['pdays'] != -1).astype(int)

    return df

train = NEW_FE(train)
test = NEW_FE(test)

cat_c = ['job','marital', "education", 'contact', 'poutcome','month','default','housing','loan','job_edu']

def update(df):

    for col in cat_c:
        df[col] = df[col].astype('category')
    return df

train = update(train)
test = update(test)

train.head()


%%time

from sklearn.metrics import roc_auc_score

def ROC_AUC(y_true, y_pred_proba):
    return roc_auc_score(y_true, y_pred_proba)

cat_c = ['job','marital', "education", 'contact', 'poutcome','month','default','housing','loan','job_edu']

encode_c = {'cat_c': cat_c}

base = AbdBase(train_data=train, test_data=test, target_column='y',gpu=True, prob=True, test_prob=True,
                 problem_type="classification", metric="custom", seed=SEED,ohe_fe=encode_c,ordinal_encoder=False,
                 n_splits=5,early_stop=True,num_classes=2,cat_features=False,custom_metric=ROC_AUC,
                 fold_type='SKF')


%%time

ParamsXgb = {'max_depth': 13, 'learning_rate': 0.01036808915308291, 'min_child_weight': 7, 'subsample': 0.4406011562109482,
             'colsample_bytree': 0.8033679369123714, 'gamma': 2.4652180617514747, 'reg_alpha': 2.1421895943084053,
             'reg_lambda': 1.5758614095439158, 'n_estimators': 50000} 

results_XGB_1 = base.Train_ML(ParamsXgb,'XGB',e_stop=150)



ParamsLGBM = {              # GPU 下建议减小 max_bin（默认 255）
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'auc',
    'learning_rate': 0.03,
    'max_depth': 8,
    'num_leaves': 31,
    'min_child_samples': 20,
    'reg_alpha': 1.0,
    'reg_lambda': 1.0,
    'colsample_bytree': 0.7,
    'subsample': 0.8,
    'subsample_freq': 1,
    'n_jobs': -1,
    'verbosity': -1
}
results_LGBM_1 = base.Train_ML(ParamsLGBM,'LGBM',e_stop=150)


%%time

def save_outputs(base_file_name, oof, pred):
    oof_df = pd.DataFrame(oof)
    pred_df = pd.DataFrame(pred)

    oof_df.to_csv(f"{base_file_name}_OOF.csv", index=False)
    pred_df.to_csv(f"{base_file_name}_PREDS.csv", index=False)

save_outputs('XGB',results_XGB_1[0], results_XGB_1[1])
save_outputs('LGBM',results_LGBM_1[0], results_LGBM_1[1])

# def unload(base_file_name):
#     oof = pd.read_csv(f"/kaggle/working/{base_file_name}_OOF.csv")
#     preds = pd.read_csv(f"/kaggle/working/{base_file_name}_PREDS.csv")
#     return oof,preds

# L_oof, L_pred = unload("XGB_0.9688")
ensemble_oof = results_XGB_1[0] * 0.5 + results_LGBM_1[0] * 0.5
ensemble_pred = results_XGB_1[1] * 0.5 + results_LGBM_1[1] * 0.5

# ensemble_oof = results_XGB_1[0] 
# ensemble_pred = results_XGB_1[1]

print(f"Ensemble Score is: {ROC_AUC(base.y_train, ensemble_oof)}")

# sample['y'] = ensemble_pred
# sample.to_csv('submission.csv', index=False)
# sample.head() 


# print('--- Loading validation.csv for final testing ---')

# # Separate features and target from the validation set
# X_final_validate = df_test.drop(['id'], axis=1)

# # IMPORTANT: Use the *same* preprocessor that was fitted on the training data
# X_final_validate_processed = preprocessor.transform(X_final_validate)

# # Make predictions with the best model
# final_predictions = voting_clf.predict(X_final_validate_processed)


# submission file

submission = pd.DataFrame({
    'id': df_test['id'],
    'y': ensemble_pred
})

submission.to_csv('submission.csv', index=False)
submission.head()

