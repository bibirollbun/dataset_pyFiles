## importing important libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import pickle as pk
import warnings
warnings.filterwarnings('ignore')
%matplotlib inline


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier,AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix,precision_score,recall_score,f1_score,roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import roc_auc_score,roc_curve
from lightgbm import LGBMClassifier


train = pd.read_csv(r'/kaggle/input/playground-series-s5e3/train.csv')
train.head()


test = pd.read_csv(r'/kaggle/input/playground-series-s5e3/test.csv')
test.head()


train.shape


test.shape


train.isnull().sum()


train.columns


### checking all the categories
train['day'].value_counts().unique()


train['pressure'].value_counts().unique()


train['maxtemp'].value_counts().unique()


train['temparature'].value_counts().unique()


train['mintemp'].value_counts().unique()


train['dewpoint'].value_counts().unique()


train['rainfall'].value_counts()


## check Missing Values 
### these are the features with nan value
features_with_nan = [features for features in train.columns if train[features].isnull().sum() >= 1]
for feature in features_with_nan:
    print(feature,np.round(train[feature].isnull().mean()*100,5), '% missing values')


train.info()


test.columns


test.isnull().sum()


test['winddirection'] = test['winddirection'].fillna(test['winddirection'].median())


test.isnull().sum()


scaler = StandardScaler()
scaler


columns_to_scale = ['pressure', 'maxtemp', 'temparature', 'mintemp',
                    'dewpoint', 'humidity', 'cloud', 'sunshine', 
                    'winddirection', 'windspeed']


train[columns_to_scale]=scaler.fit_transform(train[columns_to_scale])


train.head()


x =  train.drop(columns=['id','rainfall'])
y =  train['rainfall']


x_train,x_cv,y_train,y_cv = train_test_split(x,y,test_size=0.2,random_state=42)


models = {
    'Logistic_Reg' : LogisticRegression(),
    # 'SVC' : SVC(),
    'DT' : DecisionTreeClassifier(),
    'RF' : RandomForestClassifier(),
    'XGB': XGBClassifier(),
    "Gradient Boost":GradientBoostingClassifier(),
    "AdaBoost " : AdaBoostClassifier(),
    'LGB': LGBMClassifier(),
}
for i in range(len(list(models))):
    model =  list(models.values())[i]
    model.fit(x_train,y_train) # Train Model
    
    #Make Prediction
    y_train_pred = model.predict(x_train)
    y_cv_pred = model.predict(x_cv)
    y_prob = model.predict_proba(x_cv)[:, 1]

    ## Training set performance
    model_train_accuracy = accuracy_score(y_train,y_train_pred) # Calculate train data accuracy score
    model_train_f1 = f1_score(y_train,y_train_pred,average='weighted') # Calculate f1
    model_train_precision = precision_score(y_train,y_train_pred) # calculate precision
    model_train_recall = recall_score(y_train,y_train_pred) # calculate recall
    model_train_roc_auc_score = roc_auc_score(y_train,y_train_pred) # calculate roc_auc_score

    ## Test set performance
    model_test_accuracy = accuracy_score(y_cv,y_cv_pred) # Calculate  accuracy score
    model_test_f1 = f1_score(y_cv,y_cv_pred,average='weighted') # Calculate f1
    model_test_precision = precision_score(y_cv,y_cv_pred) # calculate precision
    model_test_recall = recall_score(y_cv,y_cv_pred) # calculate recall
    model_test_roc_auc_score = roc_auc_score(y_cv,y_cv_pred) # calculate roc_auc_score

    #overall performance 
    model_confusion_matrix =  confusion_matrix(y_cv,y_cv_pred)
    report = classification_report(y_cv,y_cv_pred)

    model_name =list(models.keys())[i]

    print(model_name)

    print('Model performance for Training set')
    print("- Accuracy: {:.4f}".format(model_train_accuracy))
    print('- F1 score: {:.4f}'.format(model_train_f1))
    
    print('- Precision: {:.4f}'.format(model_train_precision))
    print('- Recall: {:.4f}'.format(model_train_recall))
    print('- Roc Auc Score: {:.4f}'.format(model_train_roc_auc_score))

    
    
    print('----------------------------------')
    
    print('Model performance for Test set')
    print('- Accuracy: {:.4f}'.format(model_test_accuracy))
    print('- F1 score: {:.4f}'.format(model_test_f1))
    print('- Precision: {:.4f}'.format(model_test_precision))
    print('- Recall: {:.4f}'.format(model_test_recall))
    print('- Roc Auc Score: {:.4f}'.format(model_test_roc_auc_score))

    print('-'*25)
    print(f" confusion matrix {model_name}")
    print(model_confusion_matrix)
    print(f" classificatin report {model_name}")
    print(report)
    plt.figure(figsize=(8, 6))
    sns.heatmap(model_confusion_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(f'{model_name}_matrix.png')
    plt.tight_layout()
    plt.show()

    # Plot ROC curve
    plt.figure(figsize=(8, 6))
    fpr, tpr, _ = roc_curve(y_cv, y_prob)
    plt.plot(fpr, tpr, label=f'{model_name} (AUC = {model_test_roc_auc_score:.4f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name}')
    plt.legend()
    plt.savefig(f'{model_name}_curve.png')
    plt.tight_layout()
    plt.show()

    
    print('='*35)
    print('\n')
        

        


param_grid_logistic = {
    "penalty": ["l1", "l2", "elasticnet", None],  # Regularization type
    "C": [0.01, 0.1, 1, 10, 100],  # Regularization strength
    "solver": ["liblinear", "saga"],  # Solvers that support l1 & l2
    "max_iter": [100, 500, 1000, 10000]  # Iteration limits
}

# svc_params = {
#     'C':[100,10,1.0,0.1,0.01],
#     "max_iter": [100, 500, 1000, 10000]
# }
param_dt = {
    'criterion':['gini', 'entropy', 'log_loss'],
    'splitter':['best', 'random'],
    'max_depth':[1,2,3,4,5],
    'max_features':['auto','sqrt','log2']
}

rf_params = {"max_depth": [5, 8, 15, None, 10],
             "max_features": [5, 7, "auto", 8],
             "min_samples_split": [2, 8, 15, 20],
             "n_estimators": [100, 200, 500, 1000]}

xgboost_params = {
    "learning_rate": [0.1, 0.01],
    "max_depth": [5, 8, 12, 20, 30],
    "n_estimators": [100, 200, 300],
    "colsample_bytree": [0.5, 0.8, 1, 0.3, 0.4]
}
adaboost_params = {"n_estimators":[50,60,70,80,90],
    "algorithm":['SAMME']}
gradient_params={"loss": ['log_loss','deviance','exponential'],
             "criterion": ['friedman_mse','squared_error','mse'],
             "min_samples_split": [2, 8, 15, 20],
             "n_estimators": [100, 200, 500],
              "max_depth": [5, 8, 15, None, 10]}

lightgbm_params = {
        'classifier__n_estimators': [100, 200],
        'classifier__learning_rate': [0.05, 0.1],
        'classifier__max_depth': [3, 5],
        'classifier__subsample': [0.8, 1.0]
    }


# Models list for Hyperparameter tuning
randomcv_models = [("LOG",LogisticRegression(),param_grid_logistic),
                   
                   ("DT",DecisionTreeClassifier(),param_dt),
                    ("RF", RandomForestClassifier(), rf_params),
                    ("XGB",XGBClassifier(),xgboost_params),
                    ("GB",GradientBoostingClassifier(),gradient_params),
                    ("LightGBM",LGBMClassifier(),lightgbm_params),
                   ("AdaBoost",AdaBoostClassifier(),adaboost_params)
                   
                   ]


randomcv_models


model_param = {}
for name, model, params in randomcv_models:
    random = RandomizedSearchCV(estimator=model,
                                   param_distributions=params,
                                   n_iter=100,
                                   cv=3,
                                   verbose=2,
                                   n_jobs=-1)
    random.fit(x_train, y_train)
    model_param[name] = random.best_params_

for model_name in model_param:
    print(f"---------------- Best Params for {model_name} -------------------")
    print(model_param[model_name])


model_param


## for logistic regression parameters 
solver_lg = model_param['LOG']['solver']
penalty_lg = model_param['LOG']['penalty']
max_iter_lg = model_param['LOG']['max_iter']
C_lg = model_param['LOG']['C']


## for Decision Tree parameters
splitter_DT = model_param['DT']['splitter']
max_features_DT = model_param['DT']['max_features']
max_depth_DT = model_param['DT']['max_depth']
criteria_DT = model_param['DT']['criterion']

## for random_forest  parameters
estimate_rf = model_param['RF']['n_estimators']
min_sample_split_rf = model_param['RF']['min_samples_split']
max_feature_rf  = model_param['RF']['max_features']
max_depth_rf = model_param['RF']['max_depth']

## for xgboost parameters
estimate_xg = model_param['XGB']['n_estimators']
max_depth_xg = model_param['XGB']['max_depth']
learning_rate_xg = model_param['XGB']['learning_rate']
colsample_bytree_xg = model_param['XGB']['colsample_bytree']

## for  gradient boost parameters
estimate_gradient = model_param['GB']['n_estimators']
min_samples_gradient = model_param['GB']['min_samples_split']
max_depth_gradient = model_param['GB']['max_depth']
loss_gradient = model_param['GB']['loss']
critera =  model_param['GB']['criterion']

## for lightgbm parameters
subsample_light = model_param['LightGBM']['classifier__subsample']
learning_rate_light = model_param['LightGBM']['classifier__learning_rate']
max_depth_light = model_param['LightGBM']['classifier__max_depth']
estimate_light = model_param['LightGBM']['classifier__n_estimators']

## for Adaboost parameters
estimate_ada = model_param['AdaBoost']['n_estimators']
algorithm_ada = model_param['AdaBoost']['algorithm']





models={
    'Logistic_Reg' : LogisticRegression(penalty=penalty_lg,solver=solver_lg,max_iter=max_iter_lg,C=C_lg),

    'DT' : DecisionTreeClassifier(splitter=splitter_DT,max_features=max_depth_DT,
                                max_depth=max_depth_DT,criterion=criteria_DT),
    
    "Random Forest":RandomForestClassifier(n_estimators=estimate_rf,min_samples_split=min_sample_split_rf,
                                          max_features=max_feature_rf,max_depth=max_depth_rf),
    
    "XGBoost":XGBClassifier(learning_rate=learning_rate_xg,max_depth=max_depth_xg,n_estimators=estimate_xg
                            ,colsample_bytree=colsample_bytree_xg),
    
    "GradientBoost":GradientBoostingClassifier(n_estimators=estimate_gradient,
                                               min_samples_split=min_samples_gradient,
                                               max_depth=max_depth_gradient,
                                               loss=loss_gradient,criterion=critera),
    
    "LightGBM":LGBMClassifier(n_estimators=estimate_light,subsample=subsample_light,
                                  max_depth=max_depth_light,learning_rate=learning_rate_light),
                                  
    "AdaBoost":AdaBoostClassifier(n_estimators=estimate_ada,algorithm=algorithm_ada)
    
    
}

accuracy_score_train,auc_score_train,accuracy_score_test,auc_score_test = {},{},{},{}
for i in range(len(list(models))):
    model =  list(models.values())[i]
    model.fit(x_train,y_train) # Train Model

       
    #Make Prediction
    y_train_pred = model.predict(x_train)
    y_cv_pred = model.predict(x_cv)
    y_prob = model.predict_proba(x_cv)[:, 1]

    ## Training set performance
    model_train_accuracy = accuracy_score(y_train,y_train_pred) # Calculate train data accuracy score
    model_train_f1 = f1_score(y_train,y_train_pred,average='weighted') # Calculate f1
    model_train_precision = precision_score(y_train,y_train_pred) # calculate precision
    model_train_recall = recall_score(y_train,y_train_pred) # calculate recall
    model_train_roc_auc_score = roc_auc_score(y_train,y_train_pred) # calculate roc_auc_score
    accuracy_score_train[model] = model_train_accuracy
    auc_score_train[model] = model_train_roc_auc_score

    ## Test set performance
    model_test_accuracy = accuracy_score(y_cv,y_cv_pred) # Calculate  accuracy score
    model_test_f1 = f1_score(y_cv,y_cv_pred,average='weighted') # Calculate f1
    model_test_precision = precision_score(y_cv,y_cv_pred) # calculate precision
    model_test_recall = recall_score(y_cv,y_cv_pred) # calculate recall
    model_test_roc_auc_score = roc_auc_score(y_cv,y_cv_pred) # calculate roc_auc_score
    accuracy_score_test[model] = model_test_accuracy
    auc_score_test[model] = model_test_roc_auc_score

    #overall performance 
    model_confusion_matrix =  confusion_matrix(y_cv,y_cv_pred)
    report = classification_report(y_cv,y_cv_pred)

    model_name =list(models.keys())[i]

    print(model_name)

    print('Model performance for Training set')
    print("- Accuracy: {:.4f}".format(model_train_accuracy))
    print('- F1 score: {:.4f}'.format(model_train_f1))
    
    print('- Precision: {:.4f}'.format(model_train_precision))
    print('- Recall: {:.4f}'.format(model_train_recall))
    print('- Roc Auc Score: {:.4f}'.format(model_train_roc_auc_score))

    
    
    print('----------------------------------')
    
    print('Model performance for Test set')
    print('- Accuracy: {:.4f}'.format(model_test_accuracy))
    print('- F1 score: {:.4f}'.format(model_test_f1))
    print('- Precision: {:.4f}'.format(model_test_precision))
    print('- Recall: {:.4f}'.format(model_test_recall))
    print('- Roc Auc Score: {:.4f}'.format(model_test_roc_auc_score))

    print('-'*25)
    print(f" confusion matrix {model_name}")
    print(model_confusion_matrix)
    print(f" classificatin report {model_name}")
    print(report)
    plt.figure(figsize=(8, 6))
    sns.heatmap(model_confusion_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(f'{model_name}_matrix.png')
    plt.tight_layout()
    plt.show()

    # Plot ROC curve
    plt.figure(figsize=(8, 6))
    fpr, tpr, _ = roc_curve(y_cv, y_prob)
    plt.plot(fpr, tpr, label=f'{model_name} (AUC = {model_test_roc_auc_score:.4f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name}')
    plt.legend()
    plt.savefig(f'{model_name}_curve.png')
    plt.tight_layout()
    plt.show()

    
    print('='*35)
    print('\n')


## comparing models
plt.figure(figsize=(12, 6))
model_names = ["Logistic Regression","Decision Tree","Random Forest","XGBoost","GradientBoost","lightGBM","AdaBoost"]
accuracies_train = [ele for ele in accuracy_score_train.values()]
accuracies_test = [ele for ele in accuracy_score_test.values()]
roc_aucs_train = [ele for ele in auc_score_train.values()]
roc_aucs_test = [ele for ele in auc_score_test.values()]
x = np.arange(len(model_names))
width = 0.1

plt.bar(x - width/4, accuracies_train, width, label='Accuracy_train')
plt.bar(x - width/2,accuracies_test,width,label='Accuracy_test')
plt.bar(x + width/2,roc_aucs_test,width,label='ROC AUC test')
plt.bar(x + width/4, roc_aucs_train, width, label='ROC AUC train')

plt.xlabel('Model')
plt.ylabel('Score')
plt.title('Model Comparison')
plt.xticks(x, model_names, rotation=45)
plt.ylim(0.7, 1.0)
plt.legend()
plt.tight_layout()
plt.savefig("modl_comparison.png")
plt.show()


test.head()


test.isnull().sum()


columns_to_scale_test = ['pressure', 'maxtemp', 'temparature', 'mintemp',
                    'dewpoint', 'humidity', 'cloud', 'sunshine', 
                    'winddirection', 'windspeed']


test[columns_to_scale_test]=scaler.transform(test[columns_to_scale_test])


test.head()


x_test = test.drop(columns=['id'])


tuned_model = models['LightGBM']
tuned_model


final_predictions = tuned_model.predict(x_test)
final_prob = tuned_model.predict_proba(x_test)[:,1]
print(final_predictions)
print("final_prob",final_prob)


## creating submission dataframe
submission =  pd.DataFrame(
    {
        'id':test['id'],
        'Rainfall':final_prob
    }
)

submission.head()


submission['Rainfall'].value_counts()


## plotting prediction distribution
plt.figure(figsize=(12,8))
sns.histplot(final_prob,bins=50,kde=True)
plt.axvline(0.5, color='red', linestyle='--')
plt.title('Distribution of Prediction Probabilities')
plt.xlabel('Probability of Rainfall')
plt.ylabel('Count')
plt.tight_layout()
plt.show()


## saving submission file
submission.to_csv("submission.csv",index = False)
print("✅ Submission file 'submission.csv' created successfully!")
print("\nFinal model :")
print(tuned_model)


submission = pd.read_csv(r'/kaggle/working/submission.csv')
submission.head(10)




