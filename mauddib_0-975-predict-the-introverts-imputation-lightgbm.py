import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
tqdm.pandas()
import warnings
warnings.filterwarnings("ignore")
from scipy.sparse import hstack 
from textblob import TextBlob
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import accuracy_score
import lightgbm as lgb
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import langid
from sklearn.metrics import roc_auc_score as ras
import string

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


train = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')
train.head(2)


test.head(2)


print(len(test))
print(test.isna().sum())
print(train.isna().sum())


classdummies = pd.get_dummies(train.Stage_fear)

train = pd.concat([train, classdummies], axis=1)
train.rename(columns={'No': 'Stage_fear_No', 'Yes': 'Stage_fear_Yes'}, inplace=True)
train.head(2)


classdummies = pd.get_dummies(test.Stage_fear)

test = pd.concat([test, classdummies], axis=1)
test.rename(columns={'No': 'Stage_fear_No', 'Yes': 'Stage_fear_Yes'}, inplace=True)
test.head(2)


classdummies = pd.get_dummies(train.Drained_after_socializing)

train = pd.concat([train, classdummies], axis=1)
train.rename(columns={'No': 'Drained_after_socializing_No', 'Yes': 'Drained_after_socializing_Yes'}, inplace=True)
train.head(2)


classdummies = pd.get_dummies(test.Drained_after_socializing)

test = pd.concat([test, classdummies], axis=1)
test.rename(columns={'No': 'Drained_after_socializing_No', 'Yes': 'Drained_after_socializing_Yes'}, inplace=True)
test.head(2)


from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.metrics import mean_squared_error

cont_feats = [col for col in train[['Time_spent_Alone','Stage_fear_No','Stage_fear_Yes'
                                    ,'Social_event_attendance','Going_outside','Drained_after_socializing_No'
                                    ,'Drained_after_socializing_Yes','Friends_circle_size','Post_frequency']].columns]

# Iterative Imputation

iter_imputer = IterativeImputer(random_state=42)
iter_imputed = iter_imputer.fit_transform(train[['Time_spent_Alone','Stage_fear_No','Stage_fear_Yes'
                                    ,'Social_event_attendance','Going_outside','Drained_after_socializing_No'
                                    ,'Drained_after_socializing_Yes','Friends_circle_size','Post_frequency']])
_columns = ['Time_spent_Alone','Stage_fear_No','Stage_fear_Yes'
                                    ,'Social_event_attendance','Going_outside','Drained_after_socializing_No'
                                    ,'Drained_after_socializing_Yes','Friends_circle_size','Post_frequency']
df_iter_imputed = pd.DataFrame(iter_imputed,columns = _columns)
train['Time_spent_Alone'] = df_iter_imputed['Time_spent_Alone']
train['Stage_fear_No'] = df_iter_imputed['Stage_fear_No']
train['Stage_fear_Yes'] = df_iter_imputed['Stage_fear_Yes']
train['Social_event_attendance'] = df_iter_imputed['Social_event_attendance']
train['Going_outside'] = df_iter_imputed['Going_outside']
train['Drained_after_socializing_No'] = df_iter_imputed['Drained_after_socializing_No']
train['Drained_after_socializing_Yes'] = df_iter_imputed['Drained_after_socializing_Yes']
train['Friends_circle_size'] = df_iter_imputed['Friends_circle_size']
train['Post_frequency'] = df_iter_imputed['Post_frequency']
train.isnull().sum()[:]


cont_feats = [col for col in test[['Time_spent_Alone','Stage_fear_No','Stage_fear_Yes'
                                    ,'Social_event_attendance','Going_outside','Drained_after_socializing_No'
                                    ,'Drained_after_socializing_Yes','Friends_circle_size','Post_frequency']].columns]

# Iterative Imputation

iter_imputer = IterativeImputer(random_state=42)
iter_imputed = iter_imputer.fit_transform(test[['Time_spent_Alone','Stage_fear_No','Stage_fear_Yes'
                                    ,'Social_event_attendance','Going_outside','Drained_after_socializing_No'
                                    ,'Drained_after_socializing_Yes','Friends_circle_size','Post_frequency']])
_columns = ['Time_spent_Alone','Stage_fear_No','Stage_fear_Yes'
                                    ,'Social_event_attendance','Going_outside','Drained_after_socializing_No'
                                    ,'Drained_after_socializing_Yes','Friends_circle_size','Post_frequency']
df_iter_imputed = pd.DataFrame(iter_imputed,columns = _columns)
test['Time_spent_Alone'] = df_iter_imputed['Time_spent_Alone']
test['Stage_fear_No'] = df_iter_imputed['Stage_fear_No']
print('ihere')
test['Stage_fear_Yes'] = df_iter_imputed['Stage_fear_Yes']
test['Social_event_attendance'] = df_iter_imputed['Social_event_attendance']
test['Going_outside'] = df_iter_imputed['Going_outside']
test['Drained_after_socializing_No'] = df_iter_imputed['Drained_after_socializing_No']
test['Drained_after_socializing_Yes'] = df_iter_imputed['Drained_after_socializing_Yes']
test['Friends_circle_size'] = df_iter_imputed['Friends_circle_size']
test['Post_frequency'] = df_iter_imputed['Post_frequency']
test.isnull().sum()[:]


def corrdot(*args, **kwargs):
    corr_r = args[0].corr(args[1], 'pearson')
    corr_text = f"{corr_r:2.2f}".replace("0.", ".")
    ax = plt.gca()
    ax.set_axis_off()
    marker_size = abs(corr_r) * 10000
    ax.scatter([.5], [.5], marker_size, [corr_r], alpha=0.6, cmap="coolwarm",
               vmin=-1, vmax=1, transform=ax.transAxes)
    font_size = abs(corr_r) * 40 + 5
    ax.annotate(corr_text, [.5, .5,],  xycoords="axes fraction",
                ha='center', va='center', fontsize=font_size)


sns.set(style='white', font_scale=1.0)

g = sns.PairGrid(train[['Time_spent_Alone','Stage_fear','Social_event_attendance','Going_outside','Drained_after_socializing','Friends_circle_size','Post_frequency']], aspect=1.4, diag_sharey=False)
g.map_lower(sns.regplot, lowess=True, ci=False, line_kws={'color': 'black'})
g.map_diag(sns.distplot, kde_kws={'color': 'black'})
g.map_upper(corrdot)
plt.show()


sns.boxplot(x=train['Personality'],y=train['Time_spent_Alone'])


sns.boxplot(x=train['Personality'],y=train['Social_event_attendance'])


test.dtypes


import warnings
warnings.filterwarnings("ignore")
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


i = 1
model = make_pipeline(StandardScaler(), lgb.LGBMClassifier(verbose= -100, max_depth=2, learning_rate=0.05, n_estimators=500,objective='binary'))

scores=[]
while i <= 100:
    train['Personality_'] = np.where(train['Personality'] == 'Introvert',1,0)
    idx_train, idx_validation = train_test_split(train.index,test_size=0.2,stratify=train["Personality_"], random_state=i)
    X_train, y_train = train.loc[idx_train],  train.loc[idx_train, 'Personality_']
    X_val, y_val = train.loc[idx_validation],  train.loc[idx_validation, 'Personality_']
    
    
    # Create LightGBM Dataset for training and validation
    X_train_all_features = X_train[['Time_spent_Alone','Stage_fear_No','Stage_fear_Yes'
                                        ,'Social_event_attendance','Going_outside','Drained_after_socializing_No'
                                        ,'Drained_after_socializing_Yes','Friends_circle_size','Post_frequency']]
    X_val_all_features = X_val[['Time_spent_Alone','Stage_fear_No','Stage_fear_Yes'
                                        ,'Social_event_attendance','Going_outside','Drained_after_socializing_No'
                                        ,'Drained_after_socializing_Yes','Friends_circle_size','Post_frequency']]
    # model.fit(X_train_all_features, y_train)
    model.fit(X_train_all_features, y_train)
    y_pred = model.predict(X_val_all_features)
    score = accuracy_score(y_val, y_pred)
    scores.append(score)
    # just print out every tenth value, save space
    if i % 10 == 0:
        print(f'seed: {i}, Val accuracy: {score:.4f}')
    i = i + 1;


from sklearn.metrics import roc_auc_score
y_train_pred = model.predict(X_train_all_features)
y_val_pred = model.predict(X_val_all_features)
metric_train = roc_auc_score(y_train, y_train_pred)
metric_val = roc_auc_score(y_val, y_val_pred)
print(f"[auc] train:{metric_train},val:{metric_val}")


print(max(scores))
print(pd.Series(scores).idxmax())
print(scores[pd.Series(scores).idxmax()])



train['Personality_'] = np.where(train['Personality'] == 'Introvert',1,0)
idx_train, idx_validation = train_test_split(train.index,test_size=0.2,stratify=train["Personality_"], random_state=pd.Series(scores).idxmax())
X_train, y_train = train.loc[idx_train],  train.loc[idx_train, 'Personality_']
X_val, y_val = train.loc[idx_validation],  train.loc[idx_validation, 'Personality_']
model = make_pipeline(StandardScaler(), lgb.LGBMClassifier(verbose= -100, max_depth=2, learning_rate=0.05, n_estimators=500,objective='binary'))


# Create LightGBM Dataset for training and validation
X_train_all_features = X_train[['Time_spent_Alone','Stage_fear_No','Stage_fear_Yes'
                                    ,'Social_event_attendance','Going_outside','Drained_after_socializing_No'
                                    ,'Drained_after_socializing_Yes','Friends_circle_size','Post_frequency']]
X_val_all_features = X_val[['Time_spent_Alone','Stage_fear_No','Stage_fear_Yes'
                                    ,'Social_event_attendance','Going_outside','Drained_after_socializing_No'
                                    ,'Drained_after_socializing_Yes','Friends_circle_size','Post_frequency']]
model.fit(X_train_all_features, y_train)


from sklearn.metrics import roc_auc_score
y_train_pred = model.predict(X_train_all_features)
y_val_pred = model.predict(X_val_all_features)
metric_train = roc_auc_score(y_train, y_train_pred)
metric_val = roc_auc_score(y_val, y_val_pred)
print(f"[auc] train:{metric_train},val:{metric_val}")


print(len(y_train_pred))
print(len(X_val_all_features))


y_val_pred = model.predict(X_val_all_features)
cm_labels = np.unique(y_val)
cm_array = confusion_matrix(y_val.astype(int), y_val_pred)
cm_array_df = pd.DataFrame(cm_array, index=cm_labels, columns=cm_labels)


group_names = ['True Neg','False Pos','False Neg','True Pos']
group_counts = ["{0:0.0f}".format(value) for value in
                cm_array.flatten()]
group_percentages = ["{0:.2%}".format(value) for value in
                     cm_array.flatten()/np.sum(cm_array)]
labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in
          zip(group_names,group_counts,group_percentages)]
labels = np.asarray(labels).reshape(2,2)
sns.heatmap(cm_array, annot=labels, fmt='', cmap='Blues')


test['Personality'] = np.where(model.predict(test[['Time_spent_Alone','Stage_fear_No','Stage_fear_Yes'
                                    ,'Social_event_attendance','Going_outside','Drained_after_socializing_No'
                                    ,'Drained_after_socializing_Yes','Friends_circle_size','Post_frequency']]) == 1,'Introvert','Extrovert')

sub=test[["id","Personality"]]
sub.head(2)


sub.to_csv("/kaggle/working/submission.csv",index=False)

