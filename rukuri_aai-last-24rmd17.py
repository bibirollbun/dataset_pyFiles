import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
import optuna



train=pd.read_csv('/kaggle/input/playground-series-s3e24/train.csv')
test=pd.read_csv('/kaggle/input/playground-series-s3e24/test.csv')


train.head()


test.head()


train.info()


test.info()


# # トレーニングデータとテストデータの分布を可視化
# # Plot histograms (train and test)

# #feat_train = data_all.columns.drop(['id']).tolist()
# feat_test = test.columns.drop(['id']).tolist()

# for feat in feat_test:
#     plt.figure(figsize=(12,3))
#     ax1 = plt.subplot(1,2,1)
#     train[feat].plot(kind='hist', bins=50, color='blue')
#     plt.title(feat + ' / train')
#     ax2 = plt.subplot(1,2,2, sharex=ax1)
#     test[feat].plot(kind='hist', bins=50, color='green')
#     plt.title(feat + ' / test')
#     plt.show()


# # トレーニングデータおよびテストデータの分布を可視化
# # Visualize the distributions of data in train and test

# for feat in feat_test:
#     plt.figure(figsize=(12,4))
#     ax1 = plt.subplot(1,2,1)
#     sns.boxplot(data=train, x='smoking', y=feat)
#     plt.title('smoking vs ' + feat + ' / train')
#     x1 = plt.subplot(1,2,2)
#     sns.boxplot(data=test, y=feat)
#     plt.title(feat + ' / test')
#     plt.show()


# 外れ値のデータを削除
# Remove outliers

train = train.drop(train[train['triglyceride'] > 700].index)
train = train.drop(train[train['HDL'] > 350].index)
#train = train.drop(train[train['LDL'] > 1500].index)

train


# # Heatmap(train)

# corr = train.drop(columns=['id']).corr().round(1)
# plt.figure(figsize=(20,10))
# sns.heatmap(corr, vmin=-1, vmax=1, center=0, square=False, annot=True, cmap='coolwarm')
# plt.show()


# from sklearn.metrics import roc_auc_score

# for col in train.columns:
#     if col not in ["smoking", "id"]:
#         auc = roc_auc_score(train["smoking"], train[col])
#         print(f"{col}: AUC = {auc:.4f}")



#新特徴量生成
def do_feature_eng(dataset):
    dataset = dataset.copy()


    # ignore_list = ['id', 'smoking']
    # features = [feat for feat in dataset.columns if feat not in ignore_list]

    # for idx1, col_one in enumerate(features):
    #     for idx2, col_two in enumerate(features):
    #         if idx1 < idx2:
    #             dataset[col_one +'_to_'+ col_two] = dataset[col_one] / dataset[col_two]

    dataset['Gtp'] = np.log(dataset['Gtp']+1)

    # Calculate BMI
    dataset['BMI'] = dataset['weight(kg)'] / ((dataset['height(cm)'] / 100) ** 2)

    # Calculate waist to height ratio
    dataset['waist_height_ratio'] = dataset['waist(cm)'] / dataset['height(cm)']

    # Calculate average eyesight
    dataset['avg_eyesight'] = (dataset['eyesight(left)'] + dataset['eyesight(right)']) / 2

    # Calculate average hearing
    dataset['avg_hearing'] = (dataset['hearing(left)'] + dataset['hearing(right)']) / 2

    # Categorize blood pressure into ranges
    dataset['blood_pressure_category'] = pd.cut(dataset['systolic'], bins=[0, 120, 140, np.inf], labels=[0, 1, 2])
                                                                                                 
    # Calculate cholesterol ratio
    dataset['cholesterol_ratio'] = dataset['HDL'] / dataset['LDL']

    dataset['avg_eyesight'] = np.abs(dataset['eyesight(left)'] - dataset['eyesight(right)'])

    dataset['avg_eyesight'] = np.abs(dataset['hearing(left)'] - dataset['hearing(right)'])

    dataset["BSA"] = 0.007184 * dataset['height(cm)']**0.725 * dataset['weight(kg)']**0.425 
    
    # dataset["hemoglobin^2"] = dataset["hemoglobin"] * dataset["hemoglobin"]

    # dataset["hemoglobin*height"] = dataset["hemoglobin"] * dataset["height(cm)"]

    # dataset["hemoglobin*weight"] = dataset["hemoglobin"] * dataset["weight(kg)"]

    # dataset["weight*height"] = dataset["height(cm)"] * dataset["weight(kg)"]

    # dataset["hemoglobin*Gtp"] = dataset["hemoglobin"] * dataset["Gtp"]

    # dataset["hemoglobin*serum_creatinine"] = dataset["hemoglobin"] * dataset["serum creatinine"]
    
    dataset["clipped eyesight(left)"] = np.where(dataset["eyesight(left)"] > 1.5, 1.5, dataset["eyesight(left)"])
    dataset["clipped eyesight(right)"] = np.where(dataset["eyesight(right)"] > 1.5, 1.5, dataset["eyesight(right)"])

    dataset["MAP"] = 1 / 3 * dataset["systolic"] + 2 / 3 * dataset["relaxation"]

    dataset["De Ritis ratio"] = dataset["ALT"] / dataset["AST"]

    dataset["Clipped LDL"] = np.where(dataset["LDL"].abs() > 250, 250, dataset["LDL"])

    # dataset["LDL HDL Total"] = dataset["HDL"] + dataset["Clipped LDL"]
    # dataset["LDL HDL diff"] = dataset["HDL"] - dataset["Clipped LDL"]


    return dataset


train = do_feature_eng(train)
test = do_feature_eng(test)

train


target_col = 'smoking'
id_col = 'id'
all_data = pd.concat([train.drop(columns=[target_col]), test], axis=0)
all_data = all_data.reset_index(drop=True)
all_data.replace([np.inf, -np.inf], np.nan, inplace=True)


X = all_data.iloc[:len(train)].drop(columns=[id_col])
X_test = all_data.iloc[len(train):].drop(columns=[id_col])
y = train[target_col]

# X = train[["Gtp"]]
# y = train["smoking"]
# X_test = test["Gtp"]

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.3, random_state=71, stratify=y)



# def objective(trial):
#     param = {
#         'booster': 'gbtree',
#         'objective': 'binary:logistic',
#         'eval_metric': 'auc',
#         'use_label_encoder': False,
#         'lambda': trial.suggest_float('lambda', 1e-3, 10.0, log=True),
#         'alpha': trial.suggest_float('alpha', 1e-3, 10.0, log=True),
#         'colsample_bytree': trial.suggest_categorical('colsample_bytree', [0.5, 0.7, 0.9, 1.0]),
#         'subsample': trial.suggest_categorical('subsample', [0.5, 0.7, 0.9, 1.0]),
#         'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
#         'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
#         'max_depth': trial.suggest_int('max_depth', 3, 10),
#         'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
#     }

#     model = XGBClassifier(**param, random_state = 71, enable_categorical=True)
#     model.fit(X_train, y_train)
    
#     preds = model.predict_proba(X_valid)[:, 1]
#     accuracy = roc_auc_score(y_valid, y_pred)
#     return accuracy


# study = optuna.create_study(direction='maximize')
# study.optimize(objective, n_trials=10)

# print('Number of finished trials:', len(study.trials))
# print('Best trial:', study.best_trial.params)


# # Train final model with best parameters
# best_params = study.best_trial.params
# best_model = XGBClassifier(**best_params, enable_categorical='True', random_state=71)
# best_model.fit(X_train, y_train)

# # Make predictions
# y_pred = best_model.predict_proba(X_valid)[:, 1]

# # Evaluate the model
# accuracy = roc_auc_score(y_valid, y_pred)
# print(f'Accuracy: {accuracy}')


# model = XGBClassifier(random_state=71, use_label_encoder=False, eval_metric='auc', enable_categorical=True)
# model.fit(X_train, y_train)


# # バリデーションスコア確認
# y_pred = model.predict_proba(X_valid)[:, 1]
# auc = roc_auc_score(y_valid, y_pred)
# print(f'Validation ROC AUC: {auc:.5f}')


from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

n_splits = 10

skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=71)
auc_scores = []
best_model = None
best_auc = 0.0


params = {'learning_rate': 0.019093223225293453, 
            'colsample_bytree': 0.21329015151846925,
            'colsample_bylevel': 0.9148369225084079,
            'subsample': 0.8831564960046078,
            'reg_alpha': 1.1496763786731952e-05, 
            'reg_lambda': 7.512814356733987e-07, 
            'max_depth': 12, 
            'n_estimators': 1950,
            'min_child_weight': 21,
          'eval_metric': 'auc',
          'booster': 'gbtree',
          'n_jobs': -1,
          'verbosity': 0}

# Wrap the loop with tqdm for a progress bar
for train_index, test_index in tqdm(skf.split(X, y), total=n_splits):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    model = XGBClassifier(**params,enable_categorical=True, random_state=71)

    model.fit(X_train, y_train)

    y_pred_proba = model.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, y_pred_proba)
    auc_scores.append(auc)

    if auc > best_auc:
        best_auc = auc
        best_model = model


print(best_auc)
print(auc_scores)
X_test = all_data.iloc[len(train):].drop(columns=[id_col])


# # Featrure importances
# fti = model.feature_importances_   
# for i, feat in enumerate(X_train.columns):
#     print('\t{0:20s} : {1:>.6f}'.format(feat, fti[i]))


# feature_importances = model.feature_importances_
# feature_names = X.columns 
# feature_importance_dict = dict(zip(feature_names, feature_importances))
# sorted_feature_importance = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=False)
# sorted_feature_names, sorted_importance_scores = zip(*sorted_feature_importance)
# plt.figure(figsize=(8, 5))
# plt.barh(sorted_feature_names, sorted_importance_scores)
# plt.xlabel("Feature Importance")
# plt.ylabel("Feature Name")
# plt.title("Feature Importance")
# plt.show()


# y_test=best_model.predict_proba(X_test)[:, 1]
y_test=model.predict_proba(X_test)[:, 1]
submission=pd.DataFrame({'id':test['id'],'defects':y_test})
submission.to_csv('submission.csv',index=False)
submission.head()




