import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from matplotlib.cm import rainbow
%matplotlib inline


class textColor:
   PURPLE = '\033[95m'
   CYAN = '\033[96m'
   DARKCYAN = '\033[36m'
   BLUE = '\033[94m'
   GREEN = '\033[92m'
   YELLOW = '\033[93m'
   RED = '\033[91m'
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'


import warnings
warnings.filterwarnings('ignore')


train=pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv',index_col='id')


train['day'].value_counts()


train.info()


train=train[train['mintemp']<train['maxtemp']]
train = train.fillna(train.mean())


from sklearn.feature_selection import VarianceThreshold

selector=VarianceThreshold(threshold=0)
selector.fit_transform(train)

len(train.columns[selector.get_support()])-len(train.columns)


import math

def histo(data,columns):

    BINS=50
    COLS=6
    ROWS=math.ceil(len(columns)/COLS)

    histplot_hyperparams={'kde':True, 'alpha':0.6,'stat' : 'percent','bins' : BINS}

    fig, ax = plt.subplots(ROWS, COLS, figsize=(22, 8))
    ax=ax.ravel()

    for i, column in enumerate(columns):
        plot_axes=[ax[i]]
        sns.histplot(data, x=column, ax=ax[i], **histplot_hyperparams)
        ax[i].set_title(f'{column} Distribution', fontsize=6)
        ax[i].set_xlabel(None, fontsize=6)  
        ax[i].set_ylabel(None, fontsize=6)

    handles, labels = ax[0].get_legend_handles_labels()
    # plt.legend(handles, labels, title=target_col)

    for i in range(i + 1, len(ax)):
        ax[i].axis('off')

    fig.suptitle(f'Numerical Features Distributions\n\n\n', ha='center', fontweight='bold', fontsize=12, y=0.93)
    plt.tight_layout()
    plt.show()

columns=list(train.drop(columns='rainfall').columns)

histo(train,columns)


def boxy(data,columns):

    COLS=6
    ROWS=math.ceil(len(columns)/COLS)

    plt.figure(figsize=(22, 8))
    for idx, feature in enumerate(columns, 1):
        plt.subplot(ROWS,COLS, idx)
        sns.boxplot( y=feature, data=data)
        plt.title(f'{feature}', fontsize=6)
    plt.tight_layout()
    plt.show()

boxy(train,columns)


X=train.drop(columns=['rainfall'])
Y=train['rainfall']

from sklearn.preprocessing import StandardScaler

def season(doy):
    spring = range(80, 172)
    summer = range(172, 264)
    fall = range(264, 355)
    
    if doy in spring:
        season = 0
    elif doy in summer:
        season = 1
    elif doy in fall:
        season = 2
    else:
        season = 3
    return season

def featureTransform(data):
    data['season']= data['day'].apply(lambda x : season(x))
    data["sunshine"] = np.log1p(data["sunshine"])
    data["dewpoint"] = np.sin(data["dewpoint"])
    data["humidity"] = np.sin(data["humidity"])
    data["cloud"] = np.sin(data["cloud"])
    data["windspeed"] = np.sin(data["windspeed"])
    data["humidity_pressure_interaction"] = np.sin(data["humidity"] * data["pressure"])
    data["temp_range"] = np.sin(data["maxtemp"] - data["mintemp"])

    scaler=StandardScaler()
    colms=list(data.drop(columns=['season']).columns)
    data[colms] = scaler.fit_transform(data[colms])
    return data.drop(columns=['maxtemp', 'mintemp','day'])


X = featureTransform(X)

X.head()


histo(X,list(X.columns))
boxy(X,list(X.columns))


from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier, VotingClassifier, HistGradientBoostingClassifier
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.dummy import DummyClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis, LinearDiscriminantAnalysis
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.calibration import CalibratedClassifierCV
import catboost
import xgboost
import lightgbm
from sklearn.metrics import accuracy_score, f1_score, roc_curve, auc
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split


models = {
    # 'CatBoost': CatBoostClassifier(verbose=0),
    # "Logistic Regression": LogisticRegression(),
    # "Random Forest": RandomForestClassifier(),
    # "HistGradientBoosting": HistGradientBoostingClassifier(),
    # "CatBoost": catboost.CatBoostClassifier(learning_rate=0.05, depth=6, iterations=1000, custom_metric=['AUC']),
    # "XGBoost": xgboost.XGBClassifier(learning_rate=0.05, max_depth=6, n_estimators=1000, objective='binary:logistic'),
    # "LightGBM": lightgbm.LGBMClassifier(learning_rate=0.05, max_depth=6, n_estimators=1000),

    # "Gradient Boosting": GradientBoostingClassifier(),
    # "AdaBoost": AdaBoostClassifier(),
    # "Bagging Classifier": BaggingClassifier(),
    # "Support Vector Classifier": SVC(probability=True),
    # "K-Nearest Neighbors": KNeighborsClassifier(),
    # "Naive Bayes (Gaussian)": GaussianNB(),
    # "Naive Bayes (Bernoulli)": BernoulliNB(),
    # "Decision Tree": DecisionTreeClassifier(),
    # "MLP Classifier": MLPClassifier(),
    # "Dummy Classifier": DummyClassifier(strategy='most_frequent'),
    # "Quadratic Discriminant Analysis": QuadraticDiscriminantAnalysis(),
    # "Linear Discriminant Analysis": LinearDiscriminantAnalysis(),
    # "Gaussian Process Classifier": GaussianProcessClassifier(),
    # "Extra Trees Classifier": ExtraTreesClassifier(),
    "Voting Classifier": VotingClassifier(estimators=[
        ('lr', LogisticRegression()), 
        ('rf', RandomForestClassifier()), 
        ('svc', SVC(probability=True))
    ], voting='soft'),

}

test=pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv',index_col='id')
test_ids = test.index
test = test.fillna(test.mean())
test = featureTransform(test)


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.30, random_state=42)

model_accuracy=[]
# Train and evaluate each model using Purged Cross-Validation
for model_name, model in models.items():
    print(f"Training {model_name}...")
    auc_scores = []
    
    model.fit(X_train, y_train)
    val_preds = model.predict(X_test)
    val_proba = model.predict_proba(X_test)[:, 1]
    accuracy = accuracy_score(y_test, val_preds)
    f1 = f1_score(y_test, val_preds)
    fpr, tpr, _ = roc_curve(y_test, val_proba)
    roc_auc = auc(fpr, tpr)
        
    print(f"{model_name} - Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}, AUC: {roc_auc:.4f}")
    auc_scores.append(roc_auc)


    print(f"Average AUC for {model_name}: {np.mean(auc_scores):.4f}")
    model_accuracy.append([model_name, np.mean(auc_scores)])
    model.fit(X, Y)  
    test_proba = model.predict_proba(test)[:, 1]
    
    submission = pd.DataFrame({
        'id': test_ids,
        'target': test_proba
    })
    
    submission_filename = f'/kaggle/working/submission.csv'
    submission.to_csv(submission_filename, index=False)
    print(f"Submission file for {model_name} saved as {submission_filename}\n")


model_accuracy
df = pd.DataFrame(model_accuracy, columns=['model_name', 'AUC'])
df.sort_values("AUC")

