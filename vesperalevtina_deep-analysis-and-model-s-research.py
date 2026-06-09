import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split


df = pd.read_csv('/kaggle/input/playground-series-s3e24/train.csv', index_col='id')


raw_df = df.copy()


raw_X_train = raw_df.drop(columns = ['smoking'])
raw_y_train = raw_df['smoking']

X_raw_train, X_raw_test, y_raw_train, y_raw_test = train_test_split(raw_X_train, raw_y_train, test_size = 0.2, random_state = 42)


df_test = pd.read_csv("/kaggle/input/playground-series-s3e24/test.csv", index_col='id')
raw_df_test = df_test.copy()


raw_model = LogisticRegression(solver='liblinear', random_state=42)


raw_model.fit(X_raw_train, y_raw_train)
raw_y_train_pred = raw_model.predict(X_raw_test)


raw_accuracy = accuracy_score(y_raw_test, raw_y_train_pred)
raw_precision = precision_score(y_raw_test, raw_y_train_pred)
raw_recall = recall_score(y_raw_test, raw_y_train_pred)
raw_f1 = f1_score(y_raw_test, raw_y_train_pred)
raw_roc_auc = roc_auc_score(y_raw_test, raw_model.predict_proba(X_raw_test)[:, 1])

print(f'Accuracy for raw DataFrame: {raw_accuracy}')
print(f'Precision for raw DataFrame: {raw_precision}')
print(f'Recall for raw DataFrame: {raw_recall}')
print(f'F1 Score for raw DataFrame: {raw_f1}')
print(f'ROC AUC for raw DataFrame: {raw_roc_auc}')


print(df.info())


print(df.describe()) 


print(df.columns) 


def duplicates_nan(df, is_train=True):
    print(f'Number of features with missing values: {df.isnull().any().sum()}')
    print(f'Total number of missing values: {df.isnull().values.sum()}')
    
    zero_rows = df[(df == 0).all(axis=1)]
    print(f'Number of all-zero rows: {zero_rows.shape[0]}')
    
    duplicates = df[df.duplicated()]
    print(f'Duplicated rows:\n{duplicates}')
    
    df = df[~(df == 0).all(axis=1)]

    if df.isnull().values.sum() > 0:
        df = df.dropna(how='any')

    if df.duplicated().sum() > 0:
        df = df.drop_duplicates(keep='first')
    
    return df

df = duplicates_nan(df)


print(df.shape)


X_train =df.drop(columns = ['smoking'])
y_train = df['smoking']


X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size = 0.2, random_state = 42, stratify=y_train)


X_train = X_train.reset_index(drop=True)
X_test = X_test.reset_index(drop=True)


plt.figure(figsize=(18,8))
sns.heatmap(df.corr(), annot=True, fmt=".2f")
plt.savefig('corr.jpg', bbox_inches='tight', pad_inches=0.1, dpi=600)
None





from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif

bestfeature = SelectKBest(score_func=mutual_info_classif)
bestfeature.fit(X_train, y_train)
sc = bestfeature.scores_
impt_feature = pd.DataFrame({"Feature":X_train.columns,"Score":sc})
impt_feature = impt_feature.sort_values(by="Score",ascending=False)
print(impt_feature)


plt.figure(figsize=(8, 4))
plt.bar(impt_feature["Feature"], impt_feature["Score"], color="skyblue")

plt.xticks(rotation=90)
plt.xlabel("Features")
plt.ylabel("Feature Importance (Score)")
plt.title("Feature Importance by SelectKBest")

plt.show()






X_train['BMI'] = X_train['weight(kg)'] / ((X_train['height(cm)'] / 100) **2)
X_train['waist_height_ratio'] = X_train['waist(cm)'] / X_train['height(cm)']
X_train['chol_ratio'] = X_train['LDL'] / X_train['HDL']


#We will also add these features to the DataFrame, 
#which will later be cleaned from outliers. 
#It will be determined later that the outlier removal did not improve the metrics, 
#but this step has been preserved in the research history.


# df_cleaned_from_outliers = df.copy() 

# df_cleaned_from_outliers['BMI'] = df_cleaned_from_outliers['weight(kg)'] / ((df_cleaned_from_outliers['height(cm)'] / 100) **2)
# df_cleaned_from_outliers['waist_height_ratio'] = df_cleaned_from_outliers['waist(cm)'] / df_cleaned_from_outliers['height(cm)']
# df_cleaned_from_outliers['chol_ratio'] = df_cleaned_from_outliers['LDL'] / df_cleaned_from_outliers['HDL']


print(X_train[['BMI', 'waist_height_ratio', 'chol_ratio']].describe())


X_train.head()



X_train.hist(bins=30, figsize=(15, 10))
plt.tight_layout() 
None


columns_un_values = ['hearing(left)', 'hearing(right)', 'Urine protein', 'dental caries']
def check_unique(dataframe):
    for column in columns_un_values:
        print(dataframe[column].value_counts())
     
print(check_unique(X_train))
X_train


columns_recoding = ['hearing(left)', 'hearing(right)', 'Urine protein']
def recoding(dataframe, is_train=True):
    for column in columns_recoding:
        dataframe[column] = dataframe[column] - 1
    return dataframe
X_train = recoding(X_train)
    


columns_un_values = ['hearing(left)', 'hearing(right)', 'Urine protein', 'dental caries']
def check_unique(dataframe):
    for column in columns_un_values:
        print(dataframe[column].value_counts())
     
print(check_unique(X_train))
X_train


X_train.boxplot(figsize=(15, 10), vert=False)
plt.tight_layout()


# columns_with_outliers = ['age', 'height(cm)', 'weight(kg)', 'waist(cm)', 'eyesight(left)',
#        'eyesight(right)', 'systolic',
#        'relaxation', 'fasting blood sugar', 'Cholesterol', 'triglyceride',
#        'HDL', 'LDL', 'hemoglobin', 'serum creatinine', 'AST',
#        'ALT', 'Gtp', 'BMI', 'waist_height_ratio', 'chol_ratio']  

# df_cleaned_from_outliers = df_cleaned_from_outliers.copy()  

# for column in columns_with_outliers:
#     Q1 = df_cleaned_from_outliers.loc[:, column].quantile(0.25)  
#     Q3 = df_cleaned_from_outliers.loc[:, column].quantile(0.75)  
#     IQR = Q3 - Q1  
#     lower_bound = Q1 - 1.5 * IQR  
#     upper_bound = Q3 + 1.5 * IQR  
    
#     df_cleaned_from_outliers.loc[:, f'{column}_IQR_Outlier'] = (df_cleaned_from_outliers[column] < lower_bound) | (df_cleaned_from_outliers[column] > upper_bound)
    
#     outliers_number = df_cleaned_from_outliers[f'{column}_IQR_Outlier'].value_counts().get(True, 0)
#     print(f'Outlier values in column {column}: {outliers_number}')

#     df_cleaned_from_outliers = df_cleaned_from_outliers[~df_cleaned_from_outliers[f'{column}_IQR_Outlier']]  
    
#     df_cleaned_from_outliers.drop(columns=[f'{column}_IQR_Outlier'], inplace=True)

# print(f"Remaining rows in df_cleaned_from_outliers after removing outliers: {df_cleaned_from_outliers.shape[0]}")






print(X_train.columns)


columns_log = ['Cholesterol','triglyceride', 'LDL', 'HDL', 'AST', 'ALT', 'Gtp']
def log_func(dataframe):
    dataframe[columns_log] = dataframe[columns_log].map(lambda x: x if x > 0 else 1) 
    dataframe[columns_log] = np.log(dataframe[columns_log])
    dataframe.hist(bins=30, figsize=(15, 10))
    plt.tight_layout()
    None
    plt.savefig('g.png', dpi=600)
    return dataframe

print(log_func(X_train))


columns_scale = ['age', 'height(cm)', 'weight(kg)', 'waist(cm)', 'eyesight(left)',
       'eyesight(right)', 'systolic',
       'relaxation', 'fasting blood sugar', 'Cholesterol', 'triglyceride',
       'HDL', 'LDL', 'hemoglobin', 'serum creatinine', 'AST',
       'ALT', 'Gtp', 'chol_ratio', 'BMI']



scaler = StandardScaler()
X_train[columns_scale] = scaler.fit_transform(X_train[columns_scale])
X_train


X_test['BMI'] = X_test['weight(kg)'] / ((X_test['height(cm)'] / 100) **2)
X_test['waist_height_ratio'] = X_test['waist(cm)'] / X_test['height(cm)']
X_test['chol_ratio'] = X_test['LDL'] / X_test['HDL']
X_test = duplicates_nan(X_test)
X_test = recoding(X_test)
X_test = log_func(X_test)
X_test[columns_scale] = scaler.transform(X_test[columns_scale])
X_test








model = LogisticRegression(solver='liblinear', random_state=42)


model.fit(X_train, y_train)
y_train_pred = model.predict(X_test)


accuracy_prepross = accuracy_score(y_test, y_train_pred)
precision_prepross = precision_score(y_test, y_train_pred)
recall_prepross = recall_score(y_test, y_train_pred)
f1_prepross = f1_score(y_test, y_train_pred)
roc_auc_prepross = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

print(f'Accuracy: {accuracy_prepross}')
print(f'Precision: {precision_prepross}')
print(f'Recall: {recall_prepross}')
print(f'F1 Score: {f1_prepross}')
print(f'ROC AUC: {roc_auc_prepross}')








target_counts = y_train.value_counts()

plt.figure(figsize=(8, 8))

explode = (0, 0.1)

plt.pie(
    target_counts, 
    labels=['Non-smoker', 'Smoker'], 
    autopct='%1.1f%%', 
    startangle=100, 
    colors=['#00B47F', 'lightcoral'], 
    explode=explode,
    wedgeprops={'edgecolor': 'black', 'linewidth': 0.1},
    shadow=False,
    textprops={'fontsize': 16}
)

plt.title('Class Distribution', fontsize=16, fontweight='bold')
plt.axis('equal')
plt.savefig('classes.jpg', bbox_inches='tight', pad_inches=0.1, dpi=600)
plt.show()




from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)


columns_polynome = ['hemoglobin', 'height(cm)', 'weight(kg)', 'Gtp', 'Cholesterol', 'BMI', 'waist_height_ratio',
       'chol_ratio', 'systolic',
       'relaxation', 'age', 'AST',
       'ALT', 'dental caries', 'triglyceride', 'HDL', 'LDL']


from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline


model2 = make_pipeline(PolynomialFeatures(degree=2), LogisticRegression(solver='lbfgs', max_iter=10000, C=0.1, class_weight='balanced'))
model2.fit(X_resampled[columns_polynome], y_resampled)
y_train_pred_2 = model2.predict(X_resampled[columns_polynome])
accuracy_2 = accuracy_score(y_resampled, y_train_pred_2)
precision_2 = precision_score(y_resampled, y_train_pred_2)
recall_2 = recall_score(y_resampled, y_train_pred_2)
f1_2 = f1_score(y_resampled, y_train_pred_2)
roc_auc_2 = roc_auc_score(y_resampled, model2.predict_proba(X_resampled[columns_polynome])[:, 1])


print(f'Accuracy: {accuracy_2}')
print(f'Precision: {precision_2}')
print(f'Recall: {recall_2}')
print(f'F1 Score: {f1_2}')
print(f'ROC AUC: {roc_auc_2}')


y_test_pred3 = model2.predict(X_test[columns_polynome])
accuracy_3 = accuracy_score(y_test, y_test_pred3)
precision_3 = precision_score(y_test, y_test_pred3)
recall_3 = recall_score(y_test, y_test_pred3)
f1_3 = f1_score(y_test, y_test_pred3)
roc_auc_3 = roc_auc_score(y_test, model2.predict_proba(X_test[columns_polynome])[:, 1])


print(f'Accuracy: {accuracy_3}')
print(f'Precision: {precision_3}')
print(f'Recall: {recall_3}')
print(f'F1 Score: {f1_3}')
print(f'ROC AUC: {roc_auc_3}')





from sklearn.model_selection import cross_val_score
from sklearn import tree
import optuna
def objective(trial):
    max_depth = trial.suggest_int('max_depth', 2, 6)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 5, 60)
    max_leaf_nodes = trial.suggest_int('max_leaf_nodes', 10, 60)
    criterion = trial.suggest_categorical('criterion', ['gini', 'entropy'])

    model = tree.DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_leaf_nodes=max_leaf_nodes,
        criterion=criterion
    )
    
    f1 = cross_val_score(model, X_train, y_train, scoring='f1_weighted', cv=5).mean()
    return f1

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)

print(f'Best parameters: {study.best_params}')

best_params = study.best_params
best_model2 = tree.DecisionTreeClassifier(
    max_depth=best_params['max_depth'],
    min_samples_split=best_params['min_samples_split'],
    min_samples_leaf=best_params['min_samples_leaf'],
    max_leaf_nodes=best_params['max_leaf_nodes'],
    criterion=best_params['criterion']
)


best_model2.fit(X_train, y_train)

y_train_pred_opt = best_model2.predict(X_train)


print(f"F1 on training dataset: {f1_score(y_train, y_train_pred_opt, average='weighted')}")


plt.figure(figsize=(24, 24))
tree.plot_tree(best_model2, class_names=['diabet positive', 'diabet negativ'], filled=True)
plt.tight_layout()
None


accuracy_train_opt = accuracy_score(y_train, y_train_pred_opt)
precision_train_opt = precision_score(y_train, y_train_pred_opt)
recall_train_opt = recall_score(y_train, y_train_pred_opt)
roc_auc_train_opt = roc_auc_score(y_train, best_model2.predict_proba(X_train)[:, 1])

print(f'Accuracy for the training set: {accuracy_train_opt}')
print(f'Precision for the training set: {precision_train_opt}')
print(f'Recall for the training set: {recall_train_opt}')
print(f'ROC AUC for the training set: {roc_auc_train_opt}')






import optuna
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

y_train = y_train.ravel()
y_test = y_test.ravel()

def objective(trial):
    param = {
        'objective': 'binary:logistic',
        'random_state': 42,
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 1),
        'n_estimators': trial.suggest_int('n_estimators', 50, 1000),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 7),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.6, 1.0),
        'colsample_bynode': trial.suggest_float('colsample_bynode', 0.6, 1.0),
        'max_delta_step': trial.suggest_int('max_delta_step', 0, 50),
        'scale_pos_weight': len(y_train[y_train == 0]) / len(y_train[y_train == 1])  # handle class imbalance
    }
    model4 = XGBClassifier(**param)
    model4.fit(X_train, y_train)
    y_pred4 = model4.predict(X_test)
    return f1_score(y_test, y_pred4) 


study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)

print("Best parameters:", study.best_params)
print("Best F1-score during training:", study.best_value)

best_model = XGBClassifier(**study.best_params)
best_model.fit(X_train, y_train)

y_test_pred = best_model.predict(X_test)

f1_best = f1_score(y_test, y_test_pred)
accuracy_best = accuracy_score(y_test, y_test_pred)

print(f'F1 Score of best model: {f1_best}')
print(f'Accuracy of best model: {accuracy_best}')



accuracy_test_xgb = accuracy_score(y_test, y_test_pred)
precision_test_xgb = precision_score(y_test, y_test_pred)
recall_test_xgb = recall_score(y_test, y_test_pred)
roc_auc_test_xgb = roc_auc_score(y_test, best_model.predict_proba(X_test)[:, 1])

print(f'Accuracy for test set: {accuracy_test_xgb}')
print(f'Precision for test set: {precision_test_xgb}')
print(f'Recall for test set: {recall_test_xgb}')
print(f'F1 Score for test set: {f1_best}')
print(f'ROC AUC for test set: {roc_auc_test_xgb}')







from sklearn.ensemble import VotingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

xgb_model = XGBClassifier(random_state=42)
lgbm_model = LGBMClassifier(random_state=42)
catboost_model = CatBoostClassifier(random_state=42, verbose=0)

xgb_model.fit(X_train, y_train)
lgbm_model.fit(X_train, y_train)
catboost_model.fit(X_train, y_train)

y_pred_xgb = xgb_model.predict(X_test)
y_pred_lgbm = lgbm_model.predict(X_test)
y_pred_catboost = catboost_model.predict(X_test)

y_pred_proba_xgb = xgb_model.predict_proba(X_test)[:, 1]
y_pred_proba_lgbm = lgbm_model.predict_proba(X_test)[:, 1]
y_pred_proba_catboost = catboost_model.predict_proba(X_test)[:, 1]

accuracy_xgb = accuracy_score(y_test, y_pred_xgb)
f1_xgb = f1_score(y_test, y_pred_xgb)
precision_xgb = precision_score(y_test, y_pred_xgb)
recall_xgb = recall_score(y_test, y_pred_xgb)
roc_auc_xgb = roc_auc_score(y_test, y_pred_proba_xgb)

accuracy_lgbm = accuracy_score(y_test, y_pred_lgbm)
f1_lgbm = f1_score(y_test, y_pred_lgbm)
precision_lgbm = precision_score(y_test, y_pred_lgbm)
recall_lgbm = recall_score(y_test, y_pred_lgbm)
roc_auc_lgbm = roc_auc_score(y_test, y_pred_proba_lgbm)

accuracy_catboost = accuracy_score(y_test, y_pred_catboost)
f1_catboost = f1_score(y_test, y_pred_catboost)
precision_catboost = precision_score(y_test, y_pred_catboost)
recall_catboost = recall_score(y_test, y_pred_catboost)
roc_auc_catboost = roc_auc_score(y_test, y_pred_proba_catboost)

voting_model = VotingClassifier(estimators=[
    ('xgb', xgb_model),
    ('lgbm', lgbm_model),
    ('catboost', catboost_model)
], voting='soft')


voting_model.fit(X_train, y_train)


y_pred_vote = voting_model.predict(X_test)
y_pred_proba_vote = voting_model.predict_proba(X_test)[:, 1]


accuracy_vote = accuracy_score(y_test, y_pred_vote)
f1_vote = f1_score(y_test, y_pred_vote)
precision_vote = precision_score(y_test, y_pred_vote)
recall_vote = recall_score(y_test, y_pred_vote)
roc_auc_vote = roc_auc_score(y_test, y_pred_proba_vote)


print(f"XGBoost - Accuracy: {accuracy_xgb}, F1-Score: {f1_xgb}, Precision: {precision_xgb}, Recall: {recall_xgb}, ROC AUC: {roc_auc_xgb}")
print(f"LightGBM - Accuracy: {accuracy_lgbm}, F1-Score: {f1_lgbm}, Precision: {precision_lgbm}, Recall: {recall_lgbm}, ROC AUC: {roc_auc_lgbm}")
print(f"CatBoost - Accuracy: {accuracy_catboost}, F1-Score: {f1_catboost}, Precision: {precision_catboost}, Recall: {recall_catboost}, ROC AUC: {roc_auc_catboost}")
print(f"Voting Classifier - Accuracy: {accuracy_vote}, F1-Score: {f1_vote}, Precision: {precision_vote}, Recall: {recall_vote}, ROC AUC: {roc_auc_vote}")


print(f'Accuracy for test set: {accuracy_vote}')
print(f'Precision for test set: {precision_vote}')
print(f'Recall for test set: {recall_vote}')
print(f'F1 Score for test set: {f1_vote}')
print(f'ROC AUC for test set: {roc_auc_vote}')






import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.ensemble import VotingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

polymod = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = polymod.fit_transform(X_train)
X_test_poly = polymod.transform(X_test)

poly_features = polymod.get_feature_names_out(X_train.columns)

X_train_poly_df = pd.DataFrame(X_train_poly, columns=poly_features)
X_test_poly_df = pd.DataFrame(X_test_poly, columns=poly_features)

xgb_model1 = XGBClassifier(random_state=42, scale_pos_weight=(len(y_train) - sum(y_train)) / sum(y_train))
lgbm_model1 = LGBMClassifier(random_state=42, class_weight='balanced')
catboost_model1 = CatBoostClassifier(random_state=42, verbose=0, class_weights=[1, (len(y_train) - sum(y_train)) / sum(y_train)])

xgb_model1.fit(X_train_poly_df, y_train)
lgbm_model1.fit(X_train_poly_df, y_train)
catboost_model1.fit(X_train_poly_df, y_train)

y_pred_xgb1 = xgb_model1.predict(X_test_poly_df)
y_pred_proba_xgb1 = xgb_model1.predict_proba(X_test_poly_df)[:, 1]

y_pred_lgbm1 = lgbm_model1.predict(X_test_poly_df)
y_pred_proba_lgbm1 = lgbm_model1.predict_proba(X_test_poly_df)[:, 1]

y_pred_catboost1 = catboost_model1.predict(X_test_poly_df)
y_pred_proba_catboost1 = catboost_model1.predict_proba(X_test_poly_df)[:, 1]

accuracy_xgb1 = accuracy_score(y_test, y_pred_xgb1)
f1_xgb1 = f1_score(y_test, y_pred_xgb1)
precision_xgb1 = precision_score(y_test, y_pred_xgb1)
recall_xgb1 = recall_score(y_test, y_pred_xgb1)

accuracy_lgbm1 = accuracy_score(y_test, y_pred_lgbm1)
f1_lgbm1 = f1_score(y_test, y_pred_lgbm1)
precision_lgbm1 = precision_score(y_test, y_pred_lgbm1)
recall_lgbm1 = recall_score(y_test, y_pred_lgbm1)

accuracy_catboost1 = accuracy_score(y_test, y_pred_catboost1)
f1_catboost1 = f1_score(y_test, y_pred_catboost1)
precision_catboost1 = precision_score(y_test, y_pred_catboost1)
recall_catboost1 = recall_score(y_test, y_pred_catboost1)

voting_model1 = VotingClassifier(estimators=[
    ('xgb', xgb_model1),
    ('lgbm', lgbm_model1),
    ('catboost', catboost_model1)
], voting='soft')

voting_model1.fit(X_train_poly_df, y_train)

y_pred_vote1 = voting_model1.predict(X_test_poly_df)
y_pred_proba_vote1 = voting_model1.predict_proba(X_test_poly_df)[:, 1]

accuracy_vote1 = accuracy_score(y_test, y_pred_vote1)
f1_vote1 = f1_score(y_test, y_pred_vote1)
precision_vote1 = precision_score(y_test, y_pred_vote1)
recall_vote1 = recall_score(y_test, y_pred_vote1)
roc_auc_vote1 = roc_auc_score(y_test, y_pred_proba_vote1)


print(f"XGBoost - Accuracy: {accuracy_xgb1}, F1-Score: {f1_xgb1}, Precision: {precision_xgb1}, Recall: {recall_xgb1}")
print(f"LightGBM - Accuracy: {accuracy_lgbm1}, F1-Score: {f1_lgbm1}, Precision: {precision_lgbm1}, Recall: {recall_lgbm1}")
print(f"CatBoost - Accuracy: {accuracy_catboost1}, F1-Score: {f1_catboost1}, Precision: {precision_catboost1}, Recall: {recall_catboost1}")
print(f"Voting Classifier - Accuracy: {accuracy_vote1}, F1-Score: {f1_vote1}, Precision: {precision_vote1}, Recall: {recall_vote1}, ROC AUC: {roc_auc_vote1}")



print(f'Accuracy on test set: {accuracy_vote1}')
print(f'Precision on test set: {precision_vote1}')
print(f'Recall on test set: {recall_vote1}')
print(f'F1 Score on test set: {f1_vote1}')
print(f'ROC AUC on test set: {roc_auc_vote1}')






xgb_model2 = XGBClassifier(
    random_state=42, 
    scale_pos_weight=(len(y_train) - sum(y_train)) / sum(y_train),
    eval_metric='aucpr',
    n_estimators=1000,
)


lgbm_model2 = LGBMClassifier(
    random_state=42, 
    class_weight='balanced',
    metric='f1'
)


catboost_model2 = CatBoostClassifier(
    random_state=42, 
    verbose=0, 
    class_weights=[1, (len(y_train) - sum(y_train)) / sum(y_train)], 
    loss_function='Logloss', 
    eval_metric='F1'
)


xgb_model2.fit(X_train, y_train)
lgbm_model2.fit(X_train, y_train)
catboost_model2.fit(X_train, y_train)


y_pred_xgb2 = xgb_model2.predict(X_test)
y_pred_proba_xgb2 = xgb_model2.predict_proba(X_test)[:, 1]

y_pred_lgbm2 = lgbm_model2.predict(X_test)
y_pred_proba_lgbm2 = lgbm_model2.predict_proba(X_test)[:, 1]

y_pred_catboost2 = catboost_model2.predict(X_test)
y_pred_proba_catboost2 = catboost_model2.predict_proba(X_test)[:, 1]

accuracy_xgb2 = accuracy_score(y_test, y_pred_xgb2)
f1_xgb2 = f1_score(y_test, y_pred_xgb2)
precision_xgb2 = precision_score(y_test, y_pred_xgb2)
recall_xgb2 = recall_score(y_test, y_pred_xgb2)

accuracy_lgbm2 = accuracy_score(y_test, y_pred_lgbm2)
f1_lgbm2 = f1_score(y_test, y_pred_lgbm2)
precision_lgbm2 = precision_score(y_test, y_pred_lgbm2)
recall_lgbm2 = recall_score(y_test, y_pred_lgbm2)

accuracy_catboost2 = accuracy_score(y_test, y_pred_catboost2)
f1_catboost2 = f1_score(y_test, y_pred_catboost2)
precision_catboost2 = precision_score(y_test, y_pred_catboost2)
recall_catboost2 = recall_score(y_test, y_pred_catboost2)


voting_model2 = VotingClassifier(estimators=[
    ('xgb', xgb_model2),
    ('lgbm', lgbm_model2),
    ('catboost', catboost_model2)
], voting='soft')  

voting_model2.fit(X_train, y_train)

y_pred_vote2 = voting_model2.predict(X_test)
y_pred_proba_vote2 = voting_model2.predict_proba(X_test)[:, 1]


accuracy_vote2 = accuracy_score(y_test, y_pred_vote2)
f1_vote2 = f1_score(y_test, y_pred_vote2)
precision_vote2 = precision_score(y_test, y_pred_vote2)
recall_vote2 = recall_score(y_test, y_pred_vote2)
roc_auc_vote2 = roc_auc_score(y_test, y_pred_proba_vote2)


print(f"XGBoost - Accuracy: {accuracy_xgb2}, F1-Score: {f1_xgb2}, Precision: {precision_xgb2}, Recall: {recall_xgb2}")
print(f"LightGBM - Accuracy: {accuracy_lgbm2}, F1-Score: {f1_lgbm2}, Precision: {precision_lgbm2}, Recall: {recall_lgbm2}")
print(f"CatBoost - Accuracy: {accuracy_catboost2}, F1-Score: {f1_catboost2}, Precision: {precision_catboost2}, Recall: {recall_catboost2}")
print(f"Voting Classifier - Accuracy: {accuracy_vote2}, F1-Score: {f1_vote2}, Precision: {precision_vote2}, Recall: {recall_vote2}, ROC AUC: {roc_auc_vote2}")


print(f'Accuracy on test set: {accuracy_vote2}')
print(f'Precision on test set: {precision_vote2}')
print(f'Recall on test set: {recall_vote2}')
print(f'F1 Score on test set: {f1_vote2}')
print(f'ROC AUC on test set: {roc_auc_vote2}')





xgb_model3 = XGBClassifier(
    random_state=42, 
    scale_pos_weight=(len(y_train) - sum(y_train)) / sum(y_train),
    eval_metric='aucpr',
    n_estimators=1000,
    max_depth = 3, 
    learning_rate = 0.286556051096985, 
    colsample_bytree = 0.6780794857495402,
    min_child_weight = 7,
    subsample = 0.8607954635659916, 
    colsample_bylevel = 0.7442931921834796, 
    colsample_bynode = 0.8916954166069389, 
    max_delta_step = 42
)


lgbm_model3 = LGBMClassifier(
    random_state=42, 
    class_weight='balanced',
    metric='f1'
)


catboost_model3 = CatBoostClassifier(
    random_state=42, 
    verbose=0, 
    class_weights=[1, (len(y_train) - sum(y_train)) / sum(y_train)], 
    loss_function='Logloss', 
    eval_metric='F1'
)


xgb_model3.fit(X_train, y_train)
lgbm_model3.fit(X_train, y_train)
catboost_model3.fit(X_train, y_train)


y_pred_xgb3 = xgb_model3.predict(X_test)
y_pred_proba_xgb3 = xgb_model3.predict_proba(X_test)[:, 1]

y_pred_lgbm3 = lgbm_model3.predict(X_test)
y_pred_proba_lgbm3 = lgbm_model3.predict_proba(X_test)[:, 1]

y_pred_catboost3 = catboost_model3.predict(X_test)
y_pred_proba_catboost3 = catboost_model3.predict_proba(X_test)[:, 1]

accuracy_xgb3 = accuracy_score(y_test, y_pred_xgb3)
f1_xgb3 = f1_score(y_test, y_pred_xgb3)
precision_xgb3 = precision_score(y_test, y_pred_xgb3)
recall_xgb3 = recall_score(y_test, y_pred_xgb3)

accuracy_lgbm3 = accuracy_score(y_test, y_pred_lgbm3)
f1_lgbm3 = f1_score(y_test, y_pred_lgbm3)
precision_lgbm3 = precision_score(y_test, y_pred_lgbm3)
recall_lgbm3 = recall_score(y_test, y_pred_lgbm3)

accuracy_catboost3 = accuracy_score(y_test, y_pred_catboost3)
f1_catboost3 = f1_score(y_test, y_pred_catboost3)
precision_catboost3 = precision_score(y_test, y_pred_catboost3)
recall_catboost3 = recall_score(y_test, y_pred_catboost3)


voting_model3 = VotingClassifier(estimators=[
    ('xgb', xgb_model3),
    ('lgbm', lgbm_model3),
    ('catboost', catboost_model3)
], voting='soft')  

voting_model3.fit(X_train, y_train)

y_pred_vote3 = voting_model3.predict(X_test)
y_pred_proba_vote3 = voting_model3.predict_proba(X_test)[:, 1]


accuracy_vote3 = accuracy_score(y_test, y_pred_vote3)
f1_vote3 = f1_score(y_test, y_pred_vote3)
precision_vote3 = precision_score(y_test, y_pred_vote3)
recall_vote3 = recall_score(y_test, y_pred_vote3)
roc_auc_vote3 = roc_auc_score(y_test, y_pred_proba_vote3)


print(f"XGBoost - Accuracy: {accuracy_xgb3}, F1-Score: {f1_xgb3}, Precision: {precision_xgb3}, Recall: {recall_xgb3}")
print(f"LightGBM - Accuracy: {accuracy_lgbm3}, F1-Score: {f1_lgbm3}, Precision: {precision_lgbm3}, Recall: {recall_lgbm3}")
print(f"CatBoost - Accuracy: {accuracy_catboost3}, F1-Score: {f1_catboost3}, Precision: {precision_catboost3}, Recall: {recall_catboost3}")
print(f"Voting Classifier - Accuracy: {accuracy_vote3}, F1-Score: {f1_vote3}, Precision: {precision_vote3}, Recall: {recall_vote3}, ROC AUC: {roc_auc_vote3}")








print(f'Accuracy: {accuracy_prepross}')
print(f'Precision: {precision_prepross}')
print(f'Recall: {recall_prepross}')
print(f'F1 Score: {f1_prepross}')
print(f'ROC AUC: {roc_auc_prepross}')


print(f'Accuracy on test set: {accuracy_vote3}')
print(f'Precision on test set: {precision_vote3}')
print(f'Recall on test set: {recall_vote3}')
print(f'F1 Score on test set: {f1_vote3}')
print(f'ROC AUC on test set: {roc_auc_vote3}')





import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC']
raw_values = [raw_accuracy, raw_precision, raw_recall, raw_f1, raw_roc_auc]
processed_values = [accuracy_prepross, precision_prepross, recall_prepross, f1_prepross, roc_auc_prepross]
polynomial_values = [accuracy_3, precision_3, recall_3, f1_3, roc_auc_3]
voting_values = [accuracy_vote3, precision_vote3, recall_vote3, f1_vote3, roc_auc_vote3]
print("Baseline Model (Raw):")
for metric, value in zip(metrics, raw_values):
    print(f'{metric}: {value}')

print("\nPreprocessed Feature Model:")
for metric, value in zip(metrics, processed_values):
    print(f'{metric}: {value}')

print("\nPolynomial Model:")
for metric, value in zip(metrics, polynomial_values):
    print(f'{metric}: {value}')

print("\nVoting Model with Boosting:")
for metric, value in zip(metrics, voting_values):
    print(f'{metric}: {value}')
x = np.arange(len(metrics))

width = 0.2


colors = sns.light_palette('seagreen', n_colors=5)


fig, ax = plt.subplots(figsize=(12, 8))


bars1 = ax.bar(x - 1.5 * width, raw_values, width, label='Baseline Model (Raw data)', color=colors[0])
bars2 = ax.bar(x - 0.5 * width, processed_values, width, label='LR with preprocessing', color=colors[1])
bars3 = ax.bar(x + 0.5 * width, polynomial_values, width, label='Polynomial model', color='#c5e7d1')
bars4 = ax.bar(x + 1.5 * width, voting_values, width, label='Voting Model with Boosting', color='#00B47F')


ax.set_xlabel('Metrics', fontsize=12)
ax.set_ylabel('Score', fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels(metrics, fontsize=11)


ax.legend(loc='lower left', bbox_to_anchor=(0.01, 0.01), frameon=True, fontsize=10)


for bars in [bars1, bars2, bars3, bars4]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height + 0.01, f'{height:.2f}', 
                ha='center', va='bottom', fontsize=10)


plt.savefig('compar.jpg', bbox_inches='tight', pad_inches=0.1, dpi=600)


plt.show()



import matplotlib.pyplot as plt
import numpy as np

models = ['XGBoost', 'LightGBM', 'CatBoost', 'Voting Classifier']
accuracy = [accuracy_xgb3, accuracy_lgbm3, accuracy_catboost3, accuracy_vote3]
f1 = [f1_xgb3, f1_lgbm3, f1_catboost3, f1_vote3]
precision = [precision_xgb3, precision_lgbm3, precision_catboost3, precision_vote3]
recall = [recall_xgb3, recall_lgbm3, recall_catboost3, recall_vote3]


metrics = ['Accuracy', 'F1-Score', 'Precision', 'Recall']
metric_values = np.array([accuracy, f1, precision, recall])


plt.figure(figsize=(12, 8))  


for i, model in enumerate(models):
    plt.plot(metrics, metric_values[:, i], label=model, linestyle='-', marker='o', linewidth=2, markersize=8)


plt.xlabel('Metrics', fontsize=14)
plt.ylabel('Scores', fontsize=14)
plt.title('Comparison of Boosting Metrics in VotingClassifier', fontsize=16)


plt.ylim(0.7, 0.875)


plt.legend(title='Models', fontsize=12, title_fontsize=14)


plt.xticks(fontsize=12)

plt.yticks(fontsize=12)

plt.savefig('boost.jpg', bbox_inches='tight', pad_inches=0.1, dpi=600)
plt.show()



plt.figure(figsize=(10, 6))

plt.hist(y_pred_proba_vote3[y_test == 0], bins=20, alpha=0.6, label="Class 0 (Non-smoker)", color="#00B47F")
plt.hist(y_pred_proba_vote3[y_test == 1], bins=20, alpha=0.6, label="Class 1 (Smoker)", color="salmon")

plt.title("Probability Distribution of Predictions by Class", fontsize=14, fontweight="bold")
plt.xlabel("Predicted Probability", fontsize=12)
plt.ylabel("Frequency", fontsize=12)
plt.legend(fontsize=12)
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.savefig('dictrib.jpg', bbox_inches='tight', pad_inches=0.1, dpi=600)
plt.show()






import joblib

MODEL_FILE = "voting_model3.pkl"

joblib.dump(voting_model3, MODEL_FILE)

print(f"Model saved to {MODEL_FILE}")








raw_df_test





raw_df_test['BMI'] = raw_df_test['weight(kg)'] / ((raw_df_test['height(cm)'] / 100) **2)
raw_df_test['waist_height_ratio'] = raw_df_test['waist(cm)'] / raw_df_test['height(cm)']
raw_df_test['chol_ratio'] = raw_df_test['LDL'] / raw_df_test['HDL']
raw_df_test = duplicates_nan(raw_df_test)
raw_df_test = recoding(raw_df_test)
prep_df_test = log_func(raw_df_test)
prep_df_test[columns_scale] = scaler.transform(prep_df_test[columns_scale])
prep_df_test











y_pred_vote_test = voting_model3.predict(prep_df_test)

submission = pd.DataFrame({
    'id': prep_df_test.index,
    'smoking': y_pred_vote_test
})





submission.to_csv('submission.csv', index=False)

