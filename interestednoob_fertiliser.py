# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import pandas
from pandas.plotting import scatter_matrix
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import top_k_accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_curve
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import f1_score
from sklearn.metrics import jaccard_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.metrics import median_absolute_error
from sklearn.metrics import mean_squared_log_error
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import Ridge
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer #transform different types
from sklearn.datasets import fetch_openml
import numpy
from numpy import sqrt
from numpy import sum
from numpy import square
import seaborn
import matplotlib
import matplotlib.pyplot as plt
import statsmodels
import keras
import cv2
import time
from skimage import data, io, filters, color, draw, exposure, transform, metrics, measure
#io is for input output(io.imreadio.imshow imsave) color for changing color format draw for drawing basic shit exposure for changing gamma exposure transform for rotating scaling etc. Metrics for structural similarity comparison. Measure for ransac.
import os
from struct import pack
import shap


## Descriptive
df_train = pandas.read_csv("/kaggle/input/playground-series-s5e6/train.csv")
df_submission = pandas.read_csv("/kaggle/input/playground-series-s5e6/test.csv")


df_train.head()


df_train.tail()


df_train.describe()


df_train.groupby(["Fertilizer Name"]).size()


df_train['Moisture_cat'] = pandas.cut(df_train['Moisture'], bins=[25,35,45,55,65], labels=["Low", "Medium", "High", "Very High"], right=False)
pandas.pivot_table(df_train, values=['Nitrogen','Potassium','Phosphorous'],
                       columns=['Moisture_cat'], aggfunc="mean")


df_train['Temparature_cat'] = pandas.cut(df_train['Temparature'], bins=[25,29,33,38], labels=["Low", "Medium", "High"], right=False)
pandas.pivot_table(df_train, values=['Nitrogen','Potassium','Phosphorous'],
                       columns=['Temparature_cat'], aggfunc="mean")


df_train['Humidity_cat'] = pandas.cut(df_train['Humidity'], bins=[50,55.5,61,66.5,72], labels=["Low", "Medium", "High","Very High"], right=False)
pandas.pivot_table(df_train, values=['Nitrogen','Potassium','Phosphorous'],
                       columns=['Humidity_cat'], aggfunc="mean")


pandas.pivot_table(df_train, index = ['Temparature_cat'], values=['Nitrogen','Potassium','Phosphorous'],
                       columns=['Humidity_cat'], aggfunc="mean")


pandas.pivot_table(df_train, values=['Nitrogen','Potassium','Phosphorous','Temparature', 'Humidity', 'Moisture'],
                       columns=['Crop Type'], aggfunc="mean")


pandas.pivot_table(df_train, values=['Nitrogen','Potassium','Phosphorous', 'Temparature', 'Humidity', 'Moisture'],
                       columns=['Soil Type'], aggfunc="mean")


## Correlation
temp = df_train[['Temparature','Humidity','Moisture','Nitrogen','Potassium','Phosphorous']]
correlation_matrix = temp.corr()
print(correlation_matrix)


seaborn.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Heatmap')
plt.show()


## Transformations and encoding
drop_enc = OneHotEncoder(drop='first').fit(df_train[['Soil Type', 'Crop Type']])
drop_enc.categories_


drop_enc = OneHotEncoder(drop='first').fit(df_train[['Soil Type','Crop Type']])
temp = drop_enc.transform(df_train[['Soil Type','Crop Type']]).toarray()
temp = pandas.DataFrame(temp).rename(columns={0: 'Clayey', 1:'Loamy', 2:'Red', 3:'Sandy', 4: 'Cotton', 5:'Ground Nuts', 6:'Maize', 7:'Millets', 8:'Oil seeds', 9:'Paddy', 10:'Pulses', 11:'Sugarcane', 12:'Tobacco', 13:'Wheat'})


scaler = StandardScaler().fit(df_train[['Temparature', 'Moisture', 'Humidity']])
temp2 = scaler.transform(df_train[['Temparature','Moisture','Humidity']])
temp2 = pandas.DataFrame(temp2).rename(columns={0:'Temparature_std',1:'Moisture_std',2:'Humidity_std'})


le = LabelEncoder().fit(df_train[['Fertilizer Name']])
temp3 = le.transform(df_train[['Fertilizer Name']])
temp3 = pandas.DataFrame(temp3).rename(columns={0:'Fertilizer Code'})


temp


temp2


temp3


df_train


train_final = df_train.join(temp).join(temp2)
train_final = train_final.drop(['id', 'Crop Type','Soil Type'], axis = 1)


train_final


## Split
X = train_final.loc[1:10000, train_final.columns != 'Fertilizer Name'].values
y = train_final.loc[1:10000,'Fertilizer Name'].values
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.30, stratify=y)


train_final.loc[1:1000, train_final.columns != 'Fertilizer Name']


def mapk(trues, preds, k=3):
    total_score = 0.0
    for true, pred in zip(trues, preds):
        try:
            rank = pred.index(true) + 1
            total_score += 1.0 / rank
        except ValueError:
            pass
    return total_score / len(trues)


top3_preds = numpy.argsort(y_preds_df, axis=1)[:, -3:][:, ::-1]
map3 = mapk(y_df.tolist(), top3_preds.tolist(), k=3)
acc3 = top_k_accuracy_score(y_df, y_preds_df, k)

print(f"\nOOF Top-3 Accuracy: {top3_acc:.4f}")
print(f"OOF MAP@3: {map3:.4f}")


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda = LinearDiscriminantAnalysis()
lda.fit(X_train, y_train)
print(lda.score(X_test, y_test))

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
qda = QuadraticDiscriminantAnalysis()
qda.fit(X_train, y_train)
print(qda.score(X_test, y_test))


rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)

y_pred = rfc.predict(X_train)

#explainer = shap.TreeExplainer(rfc)
#explanation = explainer(X_train)

#shap_values = explanation.values


explainer = shap.KernelExplainer(rfc.predict_proba, X_train)
shap_values = explainer.shap_values(X_test)


explainer = shap.TreeExplainer(rfc)
shap_values = explainer.shap_values(X_test)


shap.force_plot(explainer.expected_value[0], shap_values[0], X_test)


shap.plots.waterfall(shap_values[0], show=False)
plt.title("PartitionExplainer for instance 0 with the KernelExplainer")
plt.show()


shap.plots.bar(shap_values)


shap.plots.force(explainer[0])


svc = SVC(probability=True)
sgdc = SGDClassifier(loss='log_loss')
rfc = RandomForestClassifier()
gnaive = GaussianNB()
knn = KNeighborsClassifier()
dtc = DecisionTreeClassifier()


models = [('SVC', svc), ('SGD', sgdc), ('Random Forest', rfc), ('Gaussian NB', gnaive), ('K-Nearest neighbours', knn),('Decision Tree',dtc)]


for model_name, model in models:
    kfold = model_selection.StratifiedKFold(n_splits=5) # stratified for classing
	
	# execute cross val to est skill of ML model (cross_val_score neater)
    cv_results1 = model_selection.cross_val_score(model, X_train, y_train, cv=kfold, scoring = 'accuracy')
    cv_results2 = model_selection.cross_val_score(model, X_train, y_train, cv=kfold, scoring = 'roc_auc_ovo')
    
	# print results
    msg1 = "%s: mean accuracy: %f (SD: %f)" % (model_name, cv_results1.mean(), cv_results1.std())
    msg2 = "%s: AUC: %f (SD: %f)" % (model_name, cv_results2.mean(), cv_results2.std())
    print(msg1, "\n", msg2)


from sklearn.linear_model import Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.neural_network import MLPClassifier

pcpt = Perceptron()
pac = PassiveAggressiveClassifier()
rfc = RandomForestClassifier()
etc = ExtraTreesClassifier()
knn = KNeighborsClassifier()
mlpc = MLPClassifier()


models = [('Perceptron', pcpt), ('PAC', pac), ('Random Forest', rfc), ('Extra Trees Class Ensemble', etc), ('K-Nearest neighbours', knn),('MLP class',mlpc)]


for model_name, model in models:
    kfold = model_selection.StratifiedKFold(n_splits=5) # stratified for classing
	
	# execute cross val to est skill of ML model (cross_val_score neater)
    cv_results1 = model_selection.cross_val_score(model, X_train, y_train, cv=kfold, scoring = 'accuracy')
    cv_results2 = model_selection.cross_val_score(model, X_train, y_train, cv=kfold, scoring = 'roc_auc_ovr')
    
	# print results
    msg1 = "%s: mean accuracy: %f (SD: %f)" % (model_name, cv_results1.mean(), cv_results1.std())
    msg2 = "%s: AUC: %f (SD: %f)" % (model_name, cv_results2.mean(), cv_results2.std())
    print(msg1, "\n", msg2)


from sklearn.ensemble import BaggingClassifier
from sklearn.tree import ExtraTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier, HistGradientBoostingClassifier, GradientBoostingClassifier, AdaBoostClassifier


ec = ExtraTreeClassifier()
bg = BaggingClassifier(ec)
rfc = RandomForestClassifier()
etc = ExtraTreesClassifier()
hgbc =  HistGradientBoostingClassifier()
gbc = GradientBoostingClassifier() 
abc = AdaBoostClassifier()


models = [('Extra Tree DT Class', bg), ('Random Forest', rfc), ('Extra Trees Class Ensemble', etc),('Hist gb Class',hgbc),('Gb Class', gbc),('Ada Boost class',abc)]


for model_name, model in models:
    kfold = model_selection.StratifiedKFold(n_splits=5) # stratified for classing
	
	# execute cross val to est skill of ML model (cross_val_score neater)
    cv_results1 = model_selection.cross_val_score(model, X_train, y_train, cv=kfold, scoring = 'accuracy')
    cv_results2 = model_selection.cross_val_score(model, X_train, y_train, cv=kfold, scoring = 'roc_auc_ovr')
    
	# print results
    msg1 = "%s: mean accuracy: %f (SD: %f)" % (model_name, cv_results1.mean(), cv_results1.std())
    msg2 = "%s: AUC: %f (SD: %f)" % (model_name, cv_results2.mean(), cv_results2.std())
    print(msg1, "\n", msg2)


from sklearn.tree import ExtraTreeClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, VotingClassifier, StackingClassifier, HistGradientBoostingClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier

X = train_final.loc[1:10000, train_final.columns != 'Fertilizer Name'].values
y = train_final.loc[1:10000,'Fertilizer Name'].values
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.30)

etc = ExtraTreesClassifier()
hgbc =  HistGradientBoostingClassifier()
rfc = RandomForestClassifier()
etdt = ExtraTreeClassifier()
bgec = BaggingClassifier(etdt)
mlpc = MLPClassifier(max_iter=400)
gbc = GradientBoostingClassifier()

models = [('Extra Trees Class Ensemble', etc),('Random Forest Class', rfc),('GB Class', gbc),('Hist GB Class',hgbc),('Multi-layer Perceptron', mlpc),('Bagged Extra Decision Trees Class', bgec)]


for model_name, model in models:
    ave_roc, ave_acc = [],[]
    for i in range(0,30):
        kfold = model_selection.StratifiedKFold(n_splits=5) # stratified for classing
	
    	# execute cross val to est skill of ML model (cross_val_score neater)
        start = time.time()
        cv_results1 = model_selection.cross_val_score(model, X_train, y_train, cv=kfold, scoring = 'accuracy')
        cv_results2 = model_selection.cross_val_score(model, X_train, y_train, cv=kfold, scoring = 'roc_auc_ovr')
        end = time.time()
        duration =  end-start
        
        ave_roc.append(cv_results2.mean())
        ave_acc.append(cv_results1.mean())

    # print results
    import statistics
    msg1 = "%s: mean accuracy: %f (SD: %f)" % (model_name, statistics.mean(ave_acc),statistics.stdev(ave_acc) )
    msg2 = "%s: ROC AUC: %f (SD: %f)" % (model_name, statistics.mean(ave_roc), statistics.stdev(ave_roc) )
    print(msg1, "\n", msg2,"\n Time ~", duration)


### ETC for balanced (between training tim#e and std dev and auc roc and accuracy)
### Hist GBC for perfection and quick training
### Bagged Extra Decisioin Trees for fastest training


from sklearn.tree import ExtraTreeClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, VotingClassifier, StackingClassifier, HistGradientBoostingClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import Perceptron, PassiveAggressiveClassifier, SGDClassifier, RidgeClassifier

X = train_final.loc[1:36000, train_final.columns != 'Fertilizer Name'].values
y = train_final.loc[1:36000,'Fertilizer Name'].values
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.30, stratify=y)
kfold = model_selection.StratifiedKFold(n_splits=5)

etc = ExtraTreesClassifier()
hgbc = HistGradientBoostingClassifier()
sgdc = SGDClassifier()
rc = RidgeClassifier()
pac = PassiveAggressiveClassifier()
bgrc = BaggingClassifier(rc)
bpac = BaggingClassifier(pac)
bsgd = BaggingClassifier(sgdc)
betc = BaggingClassifier(etc)
bhgb = BaggingClassifier(hgbc)

vlm = VotingClassifier(estimators=[('etc',betc),('histoGB',bhgb),('pac', bpac), ('sgd', bsgd), ('ridge', bgrc)], voting='hard')
#vem = VotingClassifier(estimators=[('etc',betc),('histoGB',bhgb)], voting='hard')
vml1 = VotingClassifier(estimators=[('histoGB',bhgb),('sgd',bsgd)], voting='soft')
sml1 = StackingClassifier(estimators=[('lm',vlm)], final_estimator=vml1,cv=kfold)
sml2 = StackingClassifier(estimators=[('Sml1',sml1)], final_estimator=betc)

start = time.time()
sml2.fit(X_train, y_train)
end = time.time()
y_pred = sml2.predict(X_test)
duration =  end-start

print("\nTraining Run Time: %f" % (duration))


print("Accuracy %f" % (accuracy_score(y_test, y_pred)))


y_pred = sml2.predict_proba(X_test)
print("ROC AUC %f" % (roc_auc_score(y_test, y_pred,multi_class='ovr')))


#sml2.fit(X_train, y_train)
y_pred = sml1.predict_proba(X_test)
top3_preds = numpy.argsort(y_pred)[:, -3:][:, ::-1]
map3 = mapk(y_test.tolist(), top3_preds.tolist(), k=3)
acc3 = top_k_accuracy_score(y_test, y_pred, k=3)

print("\nOOF Top-3 Accuracy: %f" % (acc3))
print("OOF MAP@3: %f" % (map3))


from sklearn.tree import ExtraTreeClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, VotingClassifier, StackingClassifier, HistGradientBoostingClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import Perceptron, PassiveAggressiveClassifier, SGDClassifier, RidgeClassifier

X = train_final.loc[1:36000, train_final.columns != 'Fertilizer Name'].values
y = train_final.loc[1:36000,'Fertilizer Name'].values
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.30, stratify=y)
kfold = model_selection.StratifiedKFold(n_splits=10)

etc = ExtraTreesClassifier()
hgbc = HistGradientBoostingClassifier()
sgdc = SGDClassifier()
mlpc = MLPClassifier()
bsgd = BaggingClassifier(sgdc)
betc = BaggingClassifier(etc)
bhgb = BaggingClassifier(hgbc)

sml1 = StackingClassifier(estimators=[('histoGB',bhgb),('sgd',bsgd),('mlp',mlpc),('etc',betc)], final_estimator=betc,cv=kfold, n_jobs=-1)

start = time.time()
sml1.fit(X_train, y_train)
end = time.time()
y_pred = sml1.predict(X_test)
duration =  end-start

print("\nTraining Run Time: %f" % (duration))


print("Accuracy %f ROC AUC %f" % (accuracy_score(y_test, y_pred), roc_auc_score(y_test, sml1.predict_proba(X_test), multi_class='ovr')))


from sklearn.tree import ExtraTreeClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, VotingClassifier, StackingClassifier, HistGradientBoostingClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import Perceptron, PassiveAggressiveClassifier, SGDClassifier, RidgeClassifier

X = train_final.loc[1:10000, train_final.columns != 'Fertilizer Name'].values
y = train_final.loc[1:10000,'Fertilizer Name'].values
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.30, stratify=y)
kfold = model_selection.StratifiedKFold(n_splits=5)

etc = ExtraTreesClassifier()
hgbc = HistGradientBoostingClassifier()
sgdc = SGDClassifier()
mlpc = MLPClassifier()
bsgd = BaggingClassifier(sgdc)
betc = BaggingClassifier(etc)
bhgb = BaggingClassifier(hgbc)

vm = VotingClassifier(estimators=[('histoGB',bhgb),('sgd',bsgd),('mlp', mlpc),('etc',betc)], voting='soft', n_jobs=-1)

start = time.time()
vm.fit(X_train, y_train)
end = time.time()
y_pred = vm.predict(X_test)
duration =  end-start

print("\nTraining Run Time: %f" % (duration))


print("Accuracy %f ROC AUC %f" % (accuracy_score(y_test, y_pred), roc_auc_score(y_test, vm.predict_proba(X_test), multi_class='ovr')))


from sklearn.tree import ExtraTreeClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, VotingClassifier, StackingClassifier, HistGradientBoostingClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier

X = train_final.loc[:, train_final.columns != 'Fertilizer Name'].values
y = train_final.loc[:,'Fertilizer Name'].values
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.30)

param_grid = {
    'bootstrap': [True, False],
    'max_features':[0.1, 0.5, 1],
    'n_estimators':[10,20,50]
}

edtc = ExtraTreeClassifier()
model = BaggingClassifier(estimator=edtc,n_jobs=6)

kfold = model_selection.StratifiedKFold(n_splits=5) # stratified for binary y
start = time.time()
# Define the grid search we want to run. Run it with six cpus in parallel.
gs_cv = model_selection.GridSearchCV(model, param_grid, cv=kfold, n_jobs=6, verbose=100)

# Run the grid search - on only the training data
gs_cv.fit(X_train, y_train)

# Print the parameters that gave us the best result
print(gs_cv.best_params_)
end = time.time()
print(end - start)


from sklearn.tree import ExtraTreeClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, VotingClassifier, StackingClassifier, HistGradientBoostingClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier

X = train_final.loc[:, train_final.columns != 'Fertilizer Name'].values
y = train_final.loc[:,'Fertilizer Name'].values
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.30)

param_grid = {
    'criterion': ['gini', 'entropy', 'log_loss'],
    'max_depth': [None,10,20],
    'max_features':['sqrt', 'log2', None],
    'n_estimators':[100,200]
}

model = ExtraTreesClassifier(n_jobs=6)

kfold = model_selection.StratifiedKFold(n_splits=5) # stratified for binary y
start = time.time()
# Define the grid search we want to run. Run it with six cpus in parallel.
gs_cv = model_selection.GridSearchCV(model, param_grid, cv=kfold, n_jobs=6, verbose=100)

# Run the grid search - on only the training data
gs_cv.fit(X_train, y_train)

# Print the parameters that gave us the best result
print(gs_cv.best_params_)
end = time.time()
print(end - start)


from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier, HistGradientBoostingClassifier, GradientBoostingClassifier, AdaBoostClassifier

X = train_final.loc[:, train_final.columns != 'Fertilizer Name'].values
y = train_final.loc[:,'Fertilizer Name'].values
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.30)

param_grid = {
    'learning_rate': [0.1, 0.5, 1],
    'max_iter': [100,200],
    'max_depth': [None,30,20],
    'scoring':['roc_auc_ovr','f1','loss'],
    'tol': [1e-8,1e-9],
    'l2_regularization':[1,0]
}

model = HistGradientBoostingClassifier()

kfold = model_selection.StratifiedKFold(n_splits=5) # stratified for binary y
start = time.time()
# Define the grid search we want to run. Run it with six cpus in parallel.
gs_cv = model_selection.GridSearchCV(model, param_grid, cv=kfold, n_jobs=6, verbose=100)

# Run the grid search - on only the training data
gs_cv.fit(X_train, y_train)

# Print the parameters that gave us the best result
print(gs_cv.best_params_)
end = time.time()
print(end - start)


import joblib
from sklearn.tree import ExtraTreeClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, VotingClassifier, StackingClassifier, HistGradientBoostingClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier

X = train_final.loc[:, train_final.columns != 'Fertilizer Name'].values
y = train_final.loc[:,'Fertilizer Name'].values
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.30)

edtc = ExtraTreeClassifier()
models = [("HistGBC",HistGradientBoostingClassifier(l2_regularization=1, learning_rate=0.1, max_depth=None, max_iter=100, scoring='roc_auc_ovr', tol= 1e-08)), ("BaggingEnsemble",BaggingClassifier(estimator=edtc, n_jobs=6, bootstrap= True, max_features= 0.5, n_estimators= 50)), ("ExtraTrees",ExtraTreesClassifier(criterion='gini', max_depth= None, max_features= 'sqrt', n_estimators= 100))]

for name, model in models:
    model.fit(X_train, y_train)
    fp = '/kaggle/working/%s.pkl' % (name)
    joblib.dump(model, fp)
    
    auc = roc_auc_score(y_test, model.predict_proba(X_test), multi_class='ovr',average='weighted')
    acc = accuracy_score(y_test, model.predict(X_test))
    print("%s Accuracy: %f ROC AUC: %f" % (name, acc, auc))


import joblib

models = [("VM",vm)]

for name, model in models:
    fp = '/kaggle/working/%s.pkl' % (name)
    joblib.dump(model, fp)



df_submission.head()


df_submission.tail()


drop_enc = OneHotEncoder(drop='first').fit(df_submission[['Soil Type','Crop Type']])
temp = drop_enc.transform(df_submission[['Soil Type','Crop Type']]).toarray()
temp = pandas.DataFrame(temp).rename(columns={0: 'Clayey', 1:'Loamy', 2:'Red', 3:'Sandy', 4: 'Cotton', 5:'Ground Nuts', 6:'Maize', 7:'Millets', 8:'Oil seeds', 9:'Paddy', 10:'Pulses', 11:'Sugarcane', 12:'Tobacco', 13:'Wheat'})


scaler = StandardScaler().fit(df_submission[['Temparature', 'Moisture', 'Humidity']])
temp2 = scaler.transform(df_submission[['Temparature','Moisture','Humidity']])
temp2 = pandas.DataFrame(temp2).rename(columns={0:'Temparature_std',1:'Moisture_std',2:'Humidity_std'})


submit_final = df_submission.join(temp).join(temp2)
submit_final = submit_final.drop(['id', 'Crop Type','Soil Type'], axis = 1)


submit_final


import joblib

VM1 = joblib.load('/kaggle/working/VM.pkl')

X_submit = submit_final.values
y_submit = VM1.predict(X_submit)


sub=pandas.DataFrame(y_submit).rename(columns={0: "Fertilizer Name"})
sub['id']= numpy.arange(750000,1000000,1)
sub.to_csv('/kaggle/working/submission.csv', index=False)


pandas.read_csv('/kaggle/working/submission.csv')

