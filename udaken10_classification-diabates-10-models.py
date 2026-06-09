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


import numpy as np
import pandas as pd
from pandas import read_csv
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import graphviz
from sklearn import model_selection
# from sklearn.preprocessing import 
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, cross_val_score, KFold, learning_curve, StratifiedKFold, train_test_split
from sklearn.metrics import confusion_matrix, make_scorer, accuracy_score
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier as MLPC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import warnings
warnings.filterwarnings("ignore")
%matplotlib inline



train_df = pd.read_csv('/kaggle/input/playground-series-s5e12/train.csv', index_col='id')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e12/test.csv', index_col='id')

sub = pd.read_csv('/kaggle/input/playground-series-s5e12/sample_submission.csv')

print(train_df.shape)
print(test_df.shape)
print(sub.shape)
print(train_df.head())


target = 'diagnosed_diabetes'
target_df = train_df[target]
target_df



train_df.columns.values


for col in train_df.columns:
    print(col)
    print(col, train_df[col].unique().shape)
    print(train_df[col].value_counts )
    print('===================================')





import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer # 変更点: Imputer -> SimpleImputer

from sklearn.preprocessing import LabelEncoder


all_data = pd.concat([train_df, test_df], axis = 0)


all_data_numeric = all_data.select_dtypes(include=[np.number])
len(all_data_numeric.columns)



all_data_categorical = all_data.select_dtypes(include=['object'])
all_data_numeric = all_data.select_dtypes(exclude=['object'])
len(all_data_numeric.columns)


for col in all_data_numeric.columns:
    print(col, all_data_numeric[col].unique())


categorical_numeric_cols = ['alcohol_consumption_per_week', 'family_history_diabetes',  'cardiovascular_history']


all_data_categorical.tail()


all_data_numeric.tail()


for col in all_data_numeric.columns:
    print(all_data_numeric[col].unique())
    
    print(all_data_numeric[col].value_counts())
    print('=================================')


all_data_categorical.columns, all_data_numeric[categorical_numeric_cols].columns


real_categorial_cols = ['gender', 'ethnicity', 'education_level', 'income_level',
        'smoking_status', 'employment_status','alcohol_consumption_per_week', 'family_history_diabetes',
        'cardiovascular_history']


read_categorical_all_data = all_data[real_categorial_cols].copy()
read_categorical_all_data


real_numeric_all_data_df = all_data_numeric.drop(categorical_numeric_cols, axis = 1).copy()
real_numeric_all_data_df.head()


real_categorical_all_data = all_data.drop(real_numeric_all_data_df.columns,  axis = 1).copy()
real_categorical_all_data


for col in real_categorical_all_data.columns:
    le = LabelEncoder()
    real_categorical_all_data[col] = le.fit_transform(real_categorical_all_data[col])

real_all_data = pd.concat([real_numeric_all_data_df, real_categorical_all_data], axis=1)
real_all_data.head()
    


train_data = real_all_data.iloc[:len(train_df)]
test_data = real_all_data.iloc[len(train_df):].drop('diagnosed_diabetes',axis = 1 )
train_data.shape, test_data.shape, train_df.shape, test_df.shape


train_data.shape, train_data.isnull().sum()


test_data.shape, test_data.isnull().sum()



# プロット用ヘルパー関数（元のノートブックの定義をベースに実用的に修正）
def plotHistogram(values, label, feature, title):
    sns.set_style("whitegrid")
    # hue（色分け）を指定してプロット
    # labelがNoneでない場合のみhueを設定
    if label and label in values.columns:
        plotOne = sns.FacetGrid(values, hue=label, aspect=2)
    else:
        plotOne = sns.FacetGrid(values, aspect=2)
    
    # distplotは推奨されなくなったためhistplotを使用（古い環境ならdistplotでも可）
    plotOne.map(sns.histplot, feature, kde=False) 
    
    if feature in values.columns:
        plotOne.set(xlim=(0, values[feature].max()))
    
    if label:
        plotOne.add_legend()
        
    plotOne.set_axis_labels(str(feature), 'Proportion')
    plotOne.fig.suptitle(title)
    plt.show()
X = train_data.drop('diagnosed_diabetes', axis=1)
y = target_df
# データの分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7, shuffle=True)

# 欠損値の補完 (Imputer -> SimpleImputer)
imputer = SimpleImputer(missing_values=0, strategy='median')
X_train2 = imputer.fit_transform(X_train)
X_test2 = imputer.transform(X_test)

# プロット用データの準備
# numpy arrayからDataFrameに戻す際にカラム名を保持する（わかりやすさのため）
X_train3 = pd.DataFrame(X_train2, columns=X.columns)

# 色分けのためにターゲット変数（Outcome）を追加
# これがないと plotHistogram の label 引数が機能しません
X_train3['Outcome'] = y_train.reset_index(drop=True)

# プロット
# カラム名で指定（4番目: Insulin, 3番目: SkinThickness）
target_col = 'Outcome'
plotHistogram(X_train3, target_col, X_train3.columns[4], 'Insulin vs Diagnosis (Blue = Healthy; Orange = Diabetes)')
plotHistogram(X_train3, target_col, X_train3.columns[3], 'SkinThickness vs Diagnosis (Blue = Healthy; Orange = Diabetes)')



# プロット用ヘルパー関数（元のノートブックの定義をベースに実用的に修正）
def plotHistogram(values, label, feature, title):
    sns.set_style("whitegrid")
    # hue（色分け）を指定してプロット
    # labelがNoneでない場合のみhueを設定
    if label and label in values.columns:
        plotOne = sns.FacetGrid(values, hue=label, aspect=2)
    else:
        plotOne = sns.FacetGrid(values, aspect=2)
    
    # distplotは推奨されなくなったためhistplotを使用（古い環境ならdistplotでも可）
    plotOne.map(sns.histplot, feature, kde=False) 
    
    if feature in values.columns:
        plotOne.set(xlim=(0, values[feature].max()))
    
    if label:
        plotOne.add_legend()
        
    plotOne.set_axis_labels(str(feature), 'Proportion')
    plotOne.fig.suptitle(title)
    plt.show()
X = train_data.drop('diagnosed_diabetes', axis=1)
y = target_df
# データの分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7, shuffle=True)

# 欠損値の補完 (Imputer -> SimpleImputer)
imputer = SimpleImputer(missing_values=0, strategy='median')
X_train2 = imputer.fit_transform(X_train)
X_test2 = imputer.transform(X_test)

# プロット用データの準備
# numpy arrayからDataFrameに戻す際にカラム名を保持する（わかりやすさのため）
X_train3 = pd.DataFrame(X_train2, columns=X.columns)

# 色分けのためにターゲット変数（Outcome）を追加
# これがないと plotHistogram の label 引数が機能しません
X_train3['Outcome'] = y_train.reset_index(drop=True)

# プロット
# カラム名で指定（4番目: Insulin, 3番目: SkinThickness）
target_col = 'Outcome'
plotHistogram(X_train3, target_col, X_train3.columns[4], 'Insulin vs Diagnosis (Blue = Healthy; Orange = Diabetes)')
plotHistogram(X_train3, target_col, X_train3.columns[3], 'SkinThickness vs Diagnosis (Blue = Healthy; Orange = Diabetes)')


X = train_data.drop('diagnosed_diabetes', axis=1)
y = target_df
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
imputer = SimpleImputer(missing_values=0,strategy='median')
X_train2 = imputer.fit_transform(X_train)
X_test2 = imputer.transform(X_test)
X_train3 = pd.DataFrame(X_train2)
plotHistogram(X_train3,None,4,'Insulin vs Diagnosis (Blue = Healthy; Orange = Diabetes)')
plotHistogram(X_train3,None,3,'SkinThickness vs Diagnosis (Blue = Healthy; Orange = Diabetes)')


X = train_data.iloc[:, :-1]
y = train_data.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
imputer = SimpleImputer(missing_values=0,strategy='median')
X_train2 = imputer.fit_transform(X_train)
X_test2 = imputer.transform(X_test)
X_train3 = pd.DataFrame(X_train2)
plotHistogram(X_train3,None,4,'Insulin vs Diagnosis (Blue = Healthy; Orange = Diabetes)')
plotHistogram(X_train3,None,3,'SkinThickness vs Diagnosis (Blue = Healthy; Orange = Diabetes)')


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Plots a learning curve. http://scikit-learn.org/stable/modules/learning_curve.html
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")
    plt.legend(loc="best")
    return plt

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def compareABunchOfDifferentModelsAccuracy(a, b, c, d):
    """
    compare performance of classifiers on X_train, X_test, Y_train, Y_test
    http://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html#sklearn.metrics.accuracy_score
    http://scikit-learn.org/stable/modules/model_evaluation.html#accuracy-score
    """
    print('\nCompare Multiple Classifiers: \n')
    print('K-Fold Cross-Validation Accuracy: \n')
    names = []
    models = []
    resultsAccuracy = []
    models.append(('LR', LogisticRegression()))
    models.append(('RF', RandomForestClassifier()))
    models.append(('KNN', KNeighborsClassifier()))
    models.append(('SVM', SVC()))
    models.append(('LSVM', LinearSVC()))
    models.append(('GNB', GaussianNB()))
    models.append(('DTC', DecisionTreeClassifier()))
    models.append(('GBC', GradientBoostingClassifier()))
    for name, model in models:
        model.fit(a, b)
        kfold = model_selection.KFold(n_splits=10, random_state=7, shuffle=True)
        accuracy_results = model_selection.cross_val_score(model, a,b, cv=kfold, scoring='accuracy')
        resultsAccuracy.append(accuracy_results)
        names.append(name)
        accuracyMessage = "%s: %f (%f)" % (name, accuracy_results.mean(), accuracy_results.std())
        print(accuracyMessage)
    # Boxplot
    fig = plt.figure()
    fig.suptitle('Algorithm Comparison: Accuracy')
    ax = fig.add_subplot(111)
    plt.boxplot(resultsAccuracy)
    ax.set_xticklabels(names)
    ax.set_ylabel('Cross-Validation: Accuracy Score')
    plt.show()

def defineModels():
    print('\nLR = LogisticRegression')
    print('RF = RandomForestClassifier')
    print('KNN = KNeighborsClassifier')
    print('SVM = Support Vector Machine SVC')
    print('LSVM = LinearSVC')
    print('GNB = GaussianNB')
    print('DTC = DecisionTreeClassifier')
    print('GBC = GradientBoostingClassifier \n\n')

names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process",
         "Decision Tree", "Random Forest", "MLPClassifier", "AdaBoost",
         "Naive Bayes", "QDA"]

classifiers = [
    KNeighborsClassifier(),
    SVC(kernel="linear"),
    SVC(kernel="rbf"),
    GaussianProcessClassifier(),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    MLPClassifier(),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis()
]

dict_characters = {0: 'Healthy', 1: 'Diabetes'}


compareABunchOfDifferentModelsAccuracy(X_train2, y_train, X_test2, y_test)
defineModels()
# iterate over classifiers; adapted from https://www.kaggle.com/hugues/basic-ml-best-of-10-classifiers
results = {}
for name, clf in zip(names, classifiers):
    scores = cross_val_score(clf, X_train2, y_train, cv=5)
    results[name] = scores
for name, scores in results.items():
    print("%20s | Accuracy: %0.2f%% (+/- %0.2f%%)" % (name, 100*scores.mean(), 100*scores.std() * 2))


def runDecisionTree(a, b, c, d):
    model = DecisionTreeClassifier()
    accuracy_scorer = make_scorer(accuracy_score)
    model.fit(a, b)
    kfold = model_selection.KFold(n_splits=10, random_state=7,shuffle=True)
    accuracy = model_selection.cross_val_score(model, a, b, cv=kfold, scoring='accuracy')
    mean = accuracy.mean()
    stdev = accuracy.std()
    prediction = model.predict(c)
    cnf_matrix = confusion_matrix(d, prediction)
    #plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,title='Normalized confusion matrix')
    plot_learning_curve(model, 'Learning Curve For DecisionTreeClassifier', a, b, (0.60,1.1), 10)
    #learning_curve(model, 'Learning Curve For DecisionTreeClassifier', a, b, (0.60,1.1), 10)
    plt.show()
    plot_confusion_matrix(cnf_matrix, classes=dict_characters,title='Confusion matrix')
    plt.show()
    print('DecisionTreeClassifier - Training set accuracy: %s (%s)' % (mean, stdev))
    return
runDecisionTree(X_train2, y_train, X_test2, y_test)
feature_names1 = X.columns.values



def plot_decision_tree1(a,b):
    dot_data = tree.export_graphviz(a, out_file=None,
                             feature_names=b,
                             class_names=['Healthy','Diabetes'],
                             filled=False, rounded=True,
                             special_characters=False)
    graph = graphviz.Source(dot_data)
    return graph
clf1 = tree.DecisionTreeClassifier(max_depth=3,min_samples_leaf=12)
clf1.fit(X_train2, y_train)
plot_decision_tree1(clf1,feature_names1)


feature_names = X.columns.values
clf1 = tree.DecisionTreeClassifier(max_depth=3,min_samples_leaf=12)
clf1.fit(X_train2, y_train)
print('Accuracy of DecisionTreeClassifier: {:.2f}'.format(clf1.score(X_test2, y_test)))
columns = X.columns
coefficients = clf1.feature_importances_.reshape(X.columns.shape[0], 1)
absCoefficients = abs(coefficients)
fullList = pd.concat((pd.DataFrame(columns, columns = ['Variable']), pd.DataFrame(absCoefficients, columns = ['absCoefficient'])), axis = 1).sort_values(by='absCoefficient', ascending = False)
print('DecisionTreeClassifier - Feature Importance:')
print('\n',fullList,'\n')

feature_names = X.columns.values
clf2 = RandomForestClassifier(max_depth=3,min_samples_leaf=12)
clf2.fit(X_train2, y_train)
print('Accuracy of RandomForestClassifier: {:.2f}'.format(clf2.score(X_test2, y_test)))
columns = X.columns
coefficients = clf2.feature_importances_.reshape(X.columns.shape[0], 1)
absCoefficients = abs(coefficients)
fullList = pd.concat((pd.DataFrame(columns, columns = ['Variable']), pd.DataFrame(absCoefficients, columns = ['absCoefficient'])), axis = 1).sort_values(by='absCoefficient', ascending = False)
print('RandomForestClassifier - Feature Importance:')
print('\n',fullList,'\n')

clf3 = XGBClassifier()
clf3.fit(X_train2, y_train)
print('Accuracy of XGBClassifier: {:.2f}'.format(clf3.score(X_test2, y_test)))
columns = X.columns
coefficients = clf3.feature_importances_.reshape(X.columns.shape[0], 1)
absCoefficients = abs(coefficients)
fullList = pd.concat((pd.DataFrame(columns, columns = ['Variable']), pd.DataFrame(absCoefficients, columns = ['absCoefficient'])), axis = 1).sort_values(by='absCoefficient', ascending = False)
print('XGBClassifier - Feature Importance:')
print('\n',fullList,'\n')



X_train3, X_test3, y_train3, y_test3 = train_test_split(X2, y2, test_size=0.2, random_state=1)
imputer = SimpleImputer(missing_values=0,strategy='median')
X_train3 = imputer.fit_transform(X_train3)
X_test3 = imputer.transform(X_test3)
clf4 = XGBClassifier()
clf4.fit(X_train3, y_train3)
print('Accuracy of XGBClassifier in Reduced Feature Space: {:.2f}'.format(clf4.score(X_test3, y_test3)))
columns = X2.columns
coefficients = clf4.feature_importances_.reshape(X2.columns.shape[0], 1)
absCoefficients = abs(coefficients)
fullList = pd.concat((pd.DataFrame(columns, columns = ['Variable']), pd.DataFrame(absCoefficients, columns = ['absCoefficient'])), axis = 1).sort_values(by='absCoefficient', ascending = False)
print('\nXGBClassifier - Feature Importance:')
print('\n',fullList,'\n')

clf3 = XGBClassifier()
clf3.fit(X_train2, y_train)
print('\n\nAccuracy of XGBClassifier in Full Feature Space: {:.2f}'.format(clf3.score(X_test2, y_test)))
columns = X.columns
coefficients = clf3.feature_importances_.reshape(X.columns.shape[0], 1)
absCoefficients = abs(coefficients)
fullList = pd.concat((pd.DataFrame(columns, columns = ['Variable']), pd.DataFrame(absCoefficients, columns = ['absCoefficient'])), axis = 1).sort_values(by='absCoefficient', ascending = False)
print('XGBClassifier - Feature Importance:')
print('\n',fullList,'\n')


# linearSVMを用いて、２値分類の予測モデルを構築する。誤差関数をroc accuracy scoreとする。
def runLinearSVM(a, b, c, d):
    model = LinearSVC()
    accuracy_scorer = make_scorer(accuracy_score)
    model.fit(a, b)
    kfold = model_selection.KFold(n_splits=10, random_state=7,shuffle=True)
    accuracy = model_selection.cross_val_score(model, a, b, cv=kfold, scoring='accuracy')
    mean = accuracy.mean()
    stdev = accuracy.std()
    prediction = model.predict(c)
    cnf_matrix = confusion_matrix(d, prediction)
    #plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,title='Normalized confusion matrix')
    plot_learning_curve(model, 'Learning Curve For LinearSVC', a, b, (0.60,1.1), 10)
    #learning_curve(model, 'Learning Curve For LinearSVC', a, b, (0.60,1.1), 10)
    plt.show()
    plot_confusion_matrix(cnf_matrix, classes=dict_characters,title='Confusion matrix')
    plt.show()
    print('LinearSVC - Training set accuracy: %s (%s)' % (mean, stdev))
    return
runLinearSVM(X_train2, y_train, X_test2, y_test)



# LinearSVMを用いて、test_dfからsubmision.csvを作成する

model = LinearSVC()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
submission = pd.DataFrame({
    "id": test_df["id"],
    "diagnosed_diabetes": predictions
})
submission.to_csv("submission.csv", index=False)

    

