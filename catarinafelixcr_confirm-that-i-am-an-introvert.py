import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')

from sklearn.impute import KNNImputer
from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler
from sklearn.model_selection import cross_val_score, cross_val_predict, StratifiedKFold, GridSearchCV
from sklearn.metrics import make_scorer, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, StackingClassifier
from sklearn.svm import SVC
import lightgbm as lgb


df_train = pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv")
df_test = pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv")
sample_submission_df = pd.read_csv('/kaggle/input/playground-series-s5e7/sample_submission.csv')


df_train.head(5)


df_test.head()


df_train.info()


num_duplicates = df_train.duplicated().sum()
print(num_duplicates) # good


missing_train = df_train.isnull().sum()
print(missing_train[missing_train > 0])


missing_test = df_test.isnull().sum()
print(missing_test[missing_test > 0])


total_rows_train = len(df_train)
missing_percent_train = (df_train.isnull().sum() / total_rows_train) * 100
missing_percent_train = missing_percent_train.round(2)

print("Percentage of missing values in Training Data:")
print(missing_percent_train[missing_percent_train > 0].sort_values(ascending=False))


df_test.select_dtypes(include=['object']).columns


df_train_processed = df_train.copy()
df_test_processed = df_test.copy()
test_ids = df_test_processed['id']

# 1st we must transform the text data into a numerical format,
#as the KNN algorithm requires numerical input
categorical_cols = df_train_processed.select_dtypes(include=['object']).drop("Personality", axis=1).columns
encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1) 

df_train_processed[categorical_cols] = encoder.fit_transform(df_train_processed[categorical_cols])
df_test_processed[categorical_cols] = encoder.transform(df_test_processed[categorical_cols])

# 2nd normalize
features = []
for col in df_train_processed.columns:
    if col not in ['id', "Personality"]:
        features.append(col)

scaler = MinMaxScaler()
df_train_processed[features] = scaler.fit_transform(df_train_processed[features])
df_test_processed[features] = scaler.transform(df_test_processed[features])


# 3th KNN
k = 5
knn_imputer = KNNImputer(n_neighbors=k)

df_train_processed[features] = knn_imputer.fit_transform(df_train_processed[features])
df_test_processed[features] = knn_imputer.transform(df_test_processed[features])


df_train = df_train_processed.copy()
df_test = df_test_processed.copy()


print(f"Missing values in imputed training data: {df_train.isnull().sum().sum()}")
print(f"Missing values in imputed test data: {df_test.isnull().sum().sum()}")


df_train.info() 


df_train.head()


df_train.describe()


plt.figure(figsize=(8, 5))
sns.countplot(data=df_train, x='Personality')
plt.title('Personalitys')
plt.xticks([0, 1], ['Introvert', 'Extrovert'])
plt.ylabel('Count')
plt.show()

print(df_train['Personality'].value_counts(normalize=True))


df_train['Personality'] = df_train['Personality'].apply(lambda x: 1 if x == 'Extrovert' else 0)

plt.figure(figsize=(12, 10))
correlation_matrix = df_train.drop('id', axis=1).corr() # Excluir o ID
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Heatmap of the Correlation Matrix for All Features')
plt.show()


X = df_train.drop(['id', 'Personality'], axis=1)
y = df_train['Personality']

print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")


model_results_cv = {}

def cross_validate(model, X, y, n_splits=5, model_name="Model"):
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    accuracy_scores = cross_val_score(model, X, y, cv=skf, scoring='accuracy')
    mean_accuracy = accuracy_scores.mean()
    std_accuracy = accuracy_scores.std()
    
    y_pred_cv = cross_val_predict(model, X, y, cv=skf)
    
    model_results_cv[model_name] = {'Mean Accuracy': mean_accuracy, 'Std Deviation': std_accuracy}
    
    print(f"Mean Accuracy: {mean_accuracy:.4f} (+/- {std_accuracy:.4f})")
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y, y_pred_cv)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Introvert (0)', 'Extrovert (1)'])
    
    disp.plot(cmap='Blues')
    plt.title(f'Confusion Matrix for {model_name}')
    plt.show()


param_grid = [{'penalty': ['l1'],
               'solver': ['liblinear', 'saga'],
               'C': [0.01, 0.1, 1, 10, 100]},
              
              {'penalty': ['l2'],
               'solver': ['liblinear', 'lbfgs', 'saga'], 
               'C': [0.01, 0.1, 1, 10, 100]}
             ]


lr_for_tuning = LogisticRegression(random_state=42, max_iter=5000)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


grid_search = GridSearchCV(
    estimator=lr_for_tuning,
    param_grid=param_grid,
    cv=skf,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)


grid_search.fit(X, y)


print(f"Best Hyperparameters Found: {grid_search.best_params_}")
print(f"Best Cross-Validated Accuracy: {grid_search.best_score_:.4f}")

best_lr_model = grid_search.best_estimator_  # 0.9692


cross_validate(best_lr_model, X, y, model_name="Logist Regression")


param_grid_xgb = {'max_depth': [3, 5, 7],
                  'n_estimators': [100, 200, 300],
                  'learning_rate': [0.01, 0.1],
                  'subsample': [0.8, 1.0],
                  'colsample_bytree': [0.8, 1.0]}


counts = y.value_counts()
counts


count_majority = counts[1]
count_minority = counts[0]
scale_pos_weight_value = count_minority / count_majority

xgb_for_tuning = xgb.XGBClassifier(
    objective='binary:logistic',
    scale_pos_weight=scale_pos_weight_value,
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss'
)


grid_search_xgb = GridSearchCV(
    estimator=xgb_for_tuning,
    param_grid=param_grid_xgb,
    cv=skf,
    scoring='accuracy',
    n_jobs=-1,
    verbose=2
)


grid_search_xgb.fit(X, y)


print(f"Best Hyperparameters Found: {grid_search_xgb.best_params_}")
print(f"Best Cross-Validated Accuracy: {grid_search_xgb.best_score_:.4f}")
best_xgb_model = grid_search_xgb.best_estimator_ # 0.9692


cross_validate(best_xgb_model, X, y, model_name="XBoost")


param_grid_rf = {
    'n_estimators': [100, 200, 300],         # Número de árvores na floresta
    'max_depth': [10, 20, None],            # Profundidade máxima de cada árvore (None = sem limite)
    'min_samples_leaf': [1, 2, 4],          # Número mínimo de amostras num nó folha (para regularização)
    'min_samples_split': [2, 5, 10]         # Número mínimo de amostras para dividir um nó
}


rf_for_tuning = RandomForestClassifier(random_state=42, class_weight='balanced')

grid_search_rf = GridSearchCV(
    estimator=rf_for_tuning,
    param_grid=param_grid_rf,
    cv=skf,
    scoring='accuracy',
    n_jobs=-1,
    verbose=2
)


grid_search_rf.fit(X, y)


print(f"Melhores Hiperparâmetros Encontrados: {grid_search_rf.best_params_}")
print(f"Melhor Acurácia em Validação Cruzada: {grid_search_rf.best_score_:.4f}")
best_rf_model = grid_search_rf.best_estimator_


cross_validate(best_rf_model, X, y, model_name="XBoost")


param_grid_lgbm = {
    'n_estimators': [100, 200, 400],
    'learning_rate': [0.01, 0.05, 0.1],
    'num_leaves': [20, 31, 40],        # Principal parâmetro para controlar a complexidade
    'max_depth': [-1, 10, 20]           # -1 significa sem limite
}


lgbm_for_tuning = lgb.LGBMClassifier(random_state=42, class_weight='balanced')

grid_search_lgbm = GridSearchCV(
    estimator=lgbm_for_tuning,
    param_grid=param_grid_lgbm,
    cv=skf,
    scoring='accuracy',
    n_jobs=-1,
    verbose=2
)


grid_search_lgbm.fit(X, y)


print(f"Melhores Hiperparâmetros Encontrados: {grid_search_lgbm.best_params_}")
print(f"Melhor Acurácia em Validação Cruzada: {grid_search_lgbm.best_score_:.4f}")
best_lgbm_model = grid_search_lgbm.best_estimator_  # 0.9689 e LENTO!!!


cross_validate(best_lgbm_model, X, y, model_name="LightGBM")


param_grid_svc = {'C': [0.1, 1, 10, 100],               
                  'gamma': ['scale', 'auto', 0.1, 0.01], 
                  'kernel': ['rbf']}


svc_for_tuning = SVC(random_state=42, class_weight='balanced', probability=True)

grid_search_svc = GridSearchCV(
    estimator=svc_for_tuning,
    param_grid=param_grid_svc,
    cv=skf,
    scoring='accuracy',
    n_jobs=-1,
    verbose=2
)


grid_search_svc.fit(X, y)


print(f"Melhores Hiperparâmetros Encontrados: {grid_search_svc.best_params_}")
print(f"Melhor Acurácia em Validação Cruzada: {grid_search_svc.best_score_:.4f}")
best_svc_model = grid_search_svc.best_estimator_  # 0.9690 not best cause its very slow


cross_validate(best_svc_model, X, y, model_name="SVM")


voting_clf = VotingClassifier(
    estimators=[
        ('lgbm', best_lgbm_model), 
        ('xgb', best_xgb_model), 
        ('rf', best_rf_model)
    ],
    voting='soft'
)


cross_validate(voting_clf, X, y, model_name="Soft Voting Ensemble")  # 0.9693 


voting_clf.fit(X, y)

X_test = df_test[X.columns] 
final_predictions = voting_clf.predict(X_test)


final_predictions = pd.DataFrame({'id': test_ids, 'Personality': final_predictions})
final_predictions.head()


final_predictions['Personality'] = final_predictions['Personality'].replace({1:'Extrovert',0: 'Introvert'})


final_predictions['Personality'].value_counts()


final_predictions.to_csv('submission.csv', index=False)


estimators = [
    ('lgbm', best_lgbm_model),
    ('xgb', best_xgb_model),
    ('rf', best_rf_model)
]


meta_model = LogisticRegression()


stack_clf = StackingClassifier(
    estimators=estimators, 
    final_estimator=meta_model, 
    cv=skf, 
    passthrough=True 
)


cross_validate(stack_clf, X, y, model_name="Stacking Model") # 0.9693




