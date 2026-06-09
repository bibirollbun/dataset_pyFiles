import pandas as pd

df = pd.read_csv('/kaggle/input/playground-series-s5e12/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e12/test.csv')
df.head()


df.shape, test.shape


df.drop(['id'], axis=1, inplace=True)
test_id = test['id']
test.drop(['id'], axis=1, inplace=True)
df.isna().sum().any(), (df.dtypes=='object').sum()


import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import plotly.io as pio
pio.renderers.default = "kaggle"
import seaborn as sns
import warnings 
warnings.filterwarnings('ignore')



columns =[]
for col in df.columns:
    if df[col].nunique()>10:
        columns.append(col)
    

r = len(columns)//3 +1
c = 3
colors = ["red", "blue", "green", "purple", "orange", "cyan", "magenta", "brown", "gray"]
fig,ax  = plt.subplots(r, c, figsize=(5*c, 3*r))
ax = ax.flatten()
for i,col in enumerate(columns):
    sns.histplot(df[col], bins=30, kde=True, ax=ax[i], color=colors[i%len(colors)], alpha=.8)
    ax[i].set_title(f'Distribution of {col}\n skewness= {df[col].skew()}')

for j in range(i+1, len(ax)):
    fig.delaxes(ax[j])

plt.suptitle("Histogram + KDE Distribution\n", fontsize=20)
plt.tight_layout()
plt.show()


from scipy import stats

df['physical_activity_minutes_per_week'], fitted_lambda = stats.boxcox(df['physical_activity_minutes_per_week'])
print("Lambda:", fitted_lambda)


fig, ax = plt.subplots(1, 2, figsize=(15, 4))
sns.histplot(df['physical_activity_minutes_per_week'], bins=30, kde=True,ax=ax[0], color='blue', alpha=1)
ax[0].set_title(f'Distribution after normalization\n Skewness={df["physical_activity_minutes_per_week"].skew()}')
sns.scatterplot(df['physical_activity_minutes_per_week'], color='blue', ax=ax[1], alpha=.7)
ax[1].set_title('Check an outliers')

plt.tight_layout()
plt.show()


"""sns.boxplot(x=df['physical_activity_minutes_per_week'], color='blue')
plt.tight_layout()
plt.show()"""


test['physical_activity_minutes_per_week'] = stats.boxcox(test['physical_activity_minutes_per_week'], lmbda=fitted_lambda)


df.select_dtypes('object').nunique()


for col in df.select_dtypes('object').columns:
    print(f'Unique values of {col} feature:\n', df[col].unique())
    print('-------------------------------------------------------------')


from plotly.subplots import make_subplots
import plotly.express as px
import plotly.graph_objects as go

columns = []
for col in df.columns:
    if 2<df[col].nunique()<9:
        columns.append(col)
        
fig = make_subplots(rows=1, cols=len(columns), subplot_titles=columns, specs=[[{'type':'domain'}]*len(columns)])
for i,col in enumerate(columns):
    fig.add_trace(
        go.Pie(labels=df[col].value_counts().index, values=df[col].value_counts().values, textinfo="label+percent", insidetextorientation='radial'),
        row=1, col=i+1
    )


fig.update_layout(height=400, width=400*len(columns), title_text="Pie Charts for Categorical Features", title_x=0, title_font_color='darkblue')
fig.show()


columns = []
for col in df.columns:
    if 9<=df[col].nunique()<=10:
        columns.append(col)
        
columns


counts = df['alcohol_consumption_per_week'].value_counts().sort_values(ascending=True)

fig = px.bar(x=counts.values, y=counts.index, orientation='h', text=counts.values,
            labels={'x':'Counts', 'y':'Categories fo alcohol_consumption_per_week'}, title='Distribution of Alcohol Consumption per Week')

fig.update_xaxes(type='log')
fig.update_layout(title_x=0.5, title_xanchor='center',title_font_size=20)
fig.update_traces(texttemplate='%{text:,}', textposition='outside')
fig.show()


columns = []
for col in df.columns:
    if df[col].nunique()<=2:
        columns.append(col)

# lnegth of (columns) = 4
fig, ax = plt.subplots(1,4, figsize=(16, 4))
for i,col in enumerate(columns):
    sns.countplot(data=df, x=col, ax=ax[i], palette='deep')
    ax[i].set_title(f'Count of {col}')
    ax[i].set_label({'x':f'{col}', 'y':'Count'})

fig.suptitle('Count plots of binary categorical features', fontsize=18)
plt.tight_layout()
plt.show()


from sklearn.model_selection import train_test_split
X,y = df.drop(['diagnosed_diabetes'], axis=1), df['diagnosed_diabetes']
X_train, X_valid, y_train, y_valid = train_test_split(X,y, test_size=.3, random_state=42)


emp_age = X_train.groupby('employment_status')['age'].mean()
global_mean_age = X_train['age'].mean()

X_train['emp_age'] = X_train['employment_status'].map(emp_age)
X_valid['emp_age'] = X_valid['employment_status'].map(emp_age)
test['emp_age'] = test['employment_status'].map(emp_age)

X_valid['emp_age'] = X_valid['emp_age'].fillna(global_mean_age)
test['emp_age'] = test['emp_age'].fillna(global_mean_age)


X_train.bmi.min(), X_train.bmi.max() 


bins = [0, 18.5, 25, 30, float('inf')]
labels = ['Underweight', 'Normal', 'Overweight', 'Obese']

X_train['bmi_category'] = pd.cut(X_train['bmi'], bins=bins, labels=labels, right=False)
X_valid['bmi_category'] = pd.cut(X_valid['bmi'], bins=bins, labels=labels, right=False)
test['bmi_category'] = pd.cut(test['bmi'], bins=bins, labels=labels, right=False)


cat_mean = X_train.groupby('bmi_category')['triglycerides'].mean()
global_mean = X_train['triglycerides'].mean()

X_train['bmi_category_triglycerides'] = X_train['bmi_category'].map(cat_mean)
X_valid['bmi_category_triglycerides'] = X_valid['bmi_category'].map(cat_mean)
test['bmi_category_triglycerides'] = test['bmi_category'].map(cat_mean)

#X_valid['bmi_category_triglycerides'] = X_valid['bmi_category_triglycerides'].fillna(global_mean)
#test['bmi_category_triglycerides'] = test['bmi_category_triglycerides'].fillna(global_mean)
X_valid.dropna(inplace=True)
test.dropna(inplace=True)


X_train['bmi_category'] = X_train['bmi_category'].replace({'Underweight':0, 'Normal':1, 'Overweight':2, 'Obese':3})
X_valid['bmi_category'] = X_valid['bmi_category'].replace({'Underweight':0, 'Normal':1, 'Overweight':2, 'Obese':3})
test['bmi_category']    = test['bmi_category'].replace({'Underweight':0, 'Normal':1, 'Overweight':2, 'Obese':3})


X_train['smoking_status'] = X_train['smoking_status'].replace({'Current':2, 'Former':1, 'Never':0})
X_valid['smoking_status'] = X_valid['smoking_status'].replace({'Current':2, 'Former':1, 'Never':0})
test['smoking_status'] = test['smoking_status'].replace({'Current':2, 'Former':1, 'Never':0})


X_train['risk_score'] = (2*X_train['smoking_status']+ 3*X_train['family_history_diabetes']+ 2*X_train['hypertension_history'])
X_valid['risk_score'] = (2*X_valid['smoking_status']+ 3*X_valid['family_history_diabetes']+ 2*X_valid['hypertension_history'])
test['risk_score']    = (2*test['smoking_status']+ 3*test['family_history_diabetes']+ 2*test['hypertension_history'])


X_train['family_hyper_interaction'] = (X_train['family_history_diabetes'] *X_train['hypertension_history'])
X_valid['family_hyper_interaction'] = (X_valid['family_history_diabetes'] *X_valid['hypertension_history'])
test['family_hyper_interaction']    = (test['family_history_diabetes'] *test['hypertension_history'])

# binary feature for boosting
X_train['high_risk'] = (X_train['risk_score'] >= 5).astype(int)
X_valid['high_risk'] = (X_valid['risk_score'] >= 5).astype(int)
test['high_risk']    = (test['risk_score'] >= 5).astype(int)


from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier

onehot_features = ['gender', 'ethnicity', 'employment_status']
ordinal_features = ['education_level', 'income_level']

ordinal_categories = [
    ['No formal', 'Highschool', 'Graduate', 'Postgraduate'],  # education_level
    ['Low', 'Lower-Middle', 'Middle', 'Upper-Middle', 'High']  # income_level
]


# Apply encoding
######################

# 1. One Hot Encoding
for column in onehot_features:
    oh = OneHotEncoder()
    train_new_features = oh.fit_transform(X_train[[column]]).toarray()
    valid_new_features = oh.transform(X_valid[[column]]).toarray()
    test_new_features = oh.transform(test[[column]]).toarray()
    for i,col in enumerate(oh.categories_[0]):
        X_train[f"{column}_{col}"] = train_new_features[:,i]
        X_valid[f"{column}_{col}"] = valid_new_features[:,i]
        test[f"{column}_{col}"] = test_new_features[:,i]

    X_train.drop([column], axis=1, inplace=True)
    X_valid.drop([column], axis=1, inplace=True)
    test.drop([column], axis=1, inplace=True)

# 2. Ordinal Encoding
for i,column in enumerate(ordinal_features):
    encoder = OrdinalEncoder(categories=[ordinal_categories[i]], handle_unknown="use_encoded_value", unknown_value=-1)
    X_train[column] = encoder.fit_transform(X_train[[column]])
    X_valid[column] = encoder.transform(X_valid[[column]])
    test[column] = encoder.transform(test[[column]])


X_train.shape, X_valid.shape


from sklearn.feature_selection import mutual_info_classif

def make_mi_sccores(X, y):
    mi_scores = mutual_info_classif(X,y)
    mi_scores = pd.Series(mi_scores, name='MI Scores', index=X.columns)
    mi_scores = mi_scores.sort_values(ascending=False)
    return mi_scores


mi_scores = make_mi_sccores(X_train, y_train)
mi_scores.head(), mi_scores.tail(7)


X_train = X_train.drop(['cardiovascular_history', 'employment_status_Unemployed', 'ethnicity_Other', 'gender_Other', 'employment_status_Student'], axis=1)
X_valid = X_valid.drop(['cardiovascular_history', 'employment_status_Unemployed', 'ethnicity_Other', 'gender_Other', 'employment_status_Student'], axis=1)
test    = test.drop(['cardiovascular_history', 'employment_status_Unemployed', 'ethnicity_Other', 'gender_Other', 'employment_status_Student'], axis=1)


def plot_scores(scores):
    scores = scores.sort_values(ascending=True)
    fig = px.bar(x=np.arange(len(scores)), y=scores.index, orientation='h', color=scores.values, color_continuous_scale='Reds',
                labels={'x':'MI Score', 'y':'Feature'})
    fig.update_layout(title='Mutual Information Scores')
    fig.show()

plot_scores(mi_scores)


y_train.value_counts()


from sklearn.model_selection import cross_val_score, StratifiedKFold, cross_validate
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, roc_auc_score, classification_report
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import RobustScaler, MinMaxScaler


base = DecisionTreeClassifier(max_depth=2, min_samples_leaf=20, class_weight='balanced')
model_1 = LGBMClassifier(n_estimators=1000, learning_rate=0.03, num_leaves=31,
                         max_depth=-1, subsample=0.8, colsample_bytree=0.8,
                         class_weight='balanced', random_state=42)

model_2 = HistGradientBoostingClassifier(learning_rate=0.03, max_depth=6, max_iter=300,
                                         min_samples_leaf=20, l2_regularization=0.1,
                                         max_bins=255,early_stopping=True)

model_3 = AdaBoostClassifier(estimator=base,n_estimators=300,
                            learning_rate=0.05,algorithm='SAMME.R')


model = VotingClassifier(estimators=[('lgb',model_1), ('hgb',model_2),
                         ('ada', model_3)], voting='soft', weights=[7, 4, 3]) 

cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
ResultCV = cross_validate(model, X=X_train, y=y_train, cv=cv,
                          scoring=['accuracy', 'precision', 'recall', 'f1', 'roc_auc'],
                         return_train_score=True)
ResultCV = pd.DataFrame(ResultCV)
ResultCV


print('CV Train Mean Accuracy:', ResultCV['train_accuracy'].mean())
print('CV Test Mean Accuracy :', ResultCV['test_accuracy'].mean())


# Calibration final model
from sklearn.calibration import CalibratedClassifierCV

cal_model = CalibratedClassifierCV(estimator=model,
                                   method='sigmoid',cv=3)

cal_model.fit(X_train, y_train)


from sklearn.metrics import classification_report

cal_model.fit(X_train, y_train)

y_pred = cal_model.predict(X_valid)
print(classification_report(y_valid, y_pred))


from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

y_proba = cal_model.predict_proba(X_valid)[:, 1]
fpr, tpr, threshold = roc_curve(y_valid, y_proba)
auc_score = roc_auc_score(y_valid, y_proba)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f'AUC = {auc_score:.3f}')
plt.plot([0, 1], [0, 1], linestyle='--')

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()


y_pred_proba = cal_model.predict_proba(test)
y_pred_proba.shape


submission_df = pd.DataFrame(test_id.values, columns=['id']) 
submission_df['diagnosed_diabetes'] = y_pred_proba[:,1].reshape(-1,1)
submission_df.to_csv("submission.csv", index=False)
print("✅ submission.csv saved!")
submission_df.head(10)

