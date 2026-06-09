import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
from sklearn.model_selection import train_test_split,StratifiedKFold,GridSearchCV
from sklearn.feature_selection import SelectPercentile, chi2, mutual_info_classif
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,classification_report,confusion_matrix,RocCurveDisplay
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.neural_network import MLPClassifier


data = pd.read_csv('../input/playground-series-s4e6/train.csv')
data.head()


data.info()


pd.set_option('display.max_columns', None)
data.describe(include='int64')


data = data.drop(['id','Course'],axis=1)


data.head()


special_category_feature = data.iloc[:,[1,4,6,7,8,9,10]].columns


data.describe(include='float64')


numeric_feature = data.select_dtypes('float64').columns
category_feature = data.select_dtypes('int64').columns
category_feature = category_feature.drop(special_category_feature)


fig , ax = plt.subplots(4,3,figsize=(12,8))
ax = ax.flatten()

for index,value in enumerate(ax):
    if index < len(numeric_feature):
        sns.histplot(data,x=numeric_feature[index],
                     hue='Target',ax = value, kde=True,binwidth=1)
        if index > 0:
            value.legend_.remove()
    else:
        value.axis('off')
    
plt.tight_layout()


sns.histplot(data,x='Age at enrollment',hue='Target',kde = True,binwidth = 1)
plt.gca().set_xlim(15,50)


fig , ax = plt.subplots(3,3,figsize=(16,8))
ax = ax.flatten()

for index,value in enumerate(ax):
    if index < len(category_feature[index]):
        sns.countplot(data,x=category_feature[index],
                     hue='Target',ax = value)
        if index > 0:
            value.legend_.remove()
    else:
        value.axis('off')
    
plt.tight_layout()


fig , ax = plt.subplots(4,3,figsize=(16,8))
ax = ax.flatten()

for index,value in enumerate(ax):
    if index < len(special_category_feature):
        sns.histplot(data,x=special_category_feature[index],
                     hue='Target',ax = value,binwidth=3)
        if index > 0:
            value.legend_.remove()
        value.set_xlim(0,50)
    else:
        value.axis('off')
    
plt.tight_layout()


label_to_index = {
    'Graduate': 2,
    'Dropout': 0,
    'Enrolled': 1,
}

data['Target'] = data['Target'].map(label_to_index)


x_train, x_test, y_train, y_test = train_test_split(data.drop('Target',axis=1),data['Target'],
                                                    test_size = 0.1,
                                                    random_state=42,
                                                    stratify=data['Target'])
x_train, x_valid, y_train, y_valid = train_test_split(x_train,y_train,
                                                      test_size=0.1,
                                                      random_state=42,
                                                      stratify=y_train)
print(x_train.shape,y_train.shape)
print(x_valid.shape, y_valid.shape)
print(x_test.shape, y_test.shape)


select_feature = SelectPercentile(score_func=mutual_info_classif,percentile=100)
select_feature.fit(x_train,y_train)


score_data = pd.DataFrame({
    'Feature_name': data.drop('Target',axis=1).columns,
    'Score': select_feature.scores_
})
score_data = score_data.sort_values(by='Score', ascending=False).reset_index(drop=True)
print(score_data)


feature = score_data[:17]
feature


x_train_new = x_train[feature['Feature_name']]
x_test_new = x_test[feature['Feature_name']]


transformer = ColumnTransformer(transformers=[
    ('Scaler',StandardScaler(),list(x_train_new.iloc[:,[0,1,2,3,4,5,7,8,10,11,12,13,15]].columns)),
],remainder='passthrough')


x_train_new = transformer.fit_transform(x_train_new)
x_test_new = transformer.transform(x_test_new)


scoring = 'accuracy'
cv = StratifiedKFold(n_splits=5)

def fit_model_cv(model,x_train,y_train,param):
    model_cv = GridSearchCV(model,
                            param,
                            cv = cv,
                            scoring = scoring,
                            verbose = 3
                            )
    model_cv.fit(x_train,y_train)
    print(model.__class__.__name__)
    print(model_cv.best_params_)
    model = model_cv.best_estimator_
    return model

dict_str_label = {
    2: 'Graduate',
    0: 'Dropout',
    1: 'Enrolled'
}

def evaluation_all(model,x_test,y_test):
    y_pred = model.predict(x_test)
    if y_pred.ndim > 1: 
        y_pred = np.argmax(y_pred, axis=-1)
    if y_test.ndim > 1:
        y_test = np.argmax(y_test, axis=-1)
    acc = accuracy_score(y_test, y_pred)
    pre = precision_score(y_test, y_pred, average='weighted')
    re = recall_score(y_test, y_pred, average='weighted')
    f = f1_score(y_test, y_pred, average='weighted')
    print(model)
    print('Accuracy: {}'.format(acc))
    print('Precision: {}'.format(pre))
    print('Recall: {}'.format(re))
    print('F1: {}'.format(f))
    print('---------------------------------------')
    report = classification_report(y_test, y_pred, target_names = dict_str_label.values())
    print(report)
    return [acc , pre , re , f]


param_xgb = {
    'n_estimators' : [100,200],
    'max_depth' : [5,7]
}


xgb = fit_model_cv(XGBClassifier(random_state=42),
                                 x_train_new,y_train,
                                 param_xgb)


nn = MLPClassifier(hidden_layer_sizes=(512,),
                 activation='relu',
                 batch_size = 128,
                 verbose = True,
                 tol=1e-10,
                 early_stopping=True)
nn.fit(x_train_new,y_train)


print('XGB')
score_xgb = evaluation_all(xgb,x_test_new,y_test)
print('MLP')
score_nn = evaluation_all(nn,x_test_new,y_test)


pd.set_option('display.float_format', '{:.4f}'.format)

score = pd.DataFrame({
    'XGB' : score_xgb,
    'MLP' : score_nn
})
score.index = ['Accuracy' , 'Precision' , 'Recall' , 'F1']

print(score)


score.plot(kind='bar', figsize=(8, 6))
plt.title('Model Comparison (XGBoost vs MLP)')
plt.ylabel('Score')
plt.xlabel('Metrics')
plt.xticks(rotation=0)
plt.legend(loc='lower right')
plt.show()


def plot_roc_curve(model_name,model,x_test,y_test,class_name):
    y_test_binarized = label_binarize(y_test, classes = class_name) 
    n_classes = y_test_binarized.shape[1]
    
    y_pred_proba = model.predict_proba(x_test_new)
    fig, ax = plt.subplots(figsize=(6, 4))
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_test_binarized[:, i], y_pred_proba[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, lw=2, label=f'Class {i+1} (AUC = {roc_auc:.2f})')
    
    ax.plot([0, 1], [0, 1], color='gray', linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_title(f'{model_name}')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.legend(loc='lower right')

plot_roc_curve('XGB',xgb,x_test_new,y_test,[0,1,2])


plot_roc_curve('MLP',nn,x_test_new,y_test,[0,1,2])


def plot_importance_features(model):
    importance_feature = model.feature_importances_
    importance = pd.DataFrame({
        'Feature' : feature['Feature_name'],
        'Importance' : importance_feature
    })
    importance = importance.sort_values(by = 'Importance' , ascending = False)
    result = sns.barplot(importance , y = 'Feature' , x = 'Importance' , palette='dark')
    plt.title('{}'.format(model.__class__.__name__))
    plt.tight_layout()
    plt.show()

plot_importance_features(xgb)


data_test = pd.read_csv('../input/playground-series-s4e6/test.csv')
data_test.shape


data_test_new = data_test[feature['Feature_name']]
data_test_new.shape


data_test_new = transformer.transform(data_test)
data_test_new.shape


y_pred = xgb.predict(data_test_new)
y_pred = pd.DataFrame({
    'id' : data_test.id,
    'Target' : y_pred
}
)
y_pred['Target'] = y_pred['Target'].map(dict_str_label)
y_pred


y_pred.to_csv('submission.csv',index=False)

