import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
# Set renderer for Kaggle
pio.renderers.default = 'iframe_connected'
from plotly.subplots import make_subplots

import warnings
warnings.filterwarnings('ignore')


train = pd.read_csv('/kaggle/input/playground-series-s4e10/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s4e10/test.csv')
sample_submission = pd.read_csv('/kaggle/input/playground-series-s4e10/sample_submission.csv')


train.head(5)


# Fortunately there is no NaN values.
train.isna().sum()


print(len(train.columns))  # So there are total 12 categorical section I need to work on
train.columns


train.loan_status.value_counts()  


# The percentage distribution of the data 
train.loan_status.value_counts(normalize = True)*100



# information of the train df
train.info()


# Creating the histogram
fig = px.histogram(
    train, 
    x='person_age', 
    color='loan_status', 
    barmode='stack',  
    nbins=20,         
    color_discrete_sequence=px.colors.qualitative.Set2,
    title='Age Distribution by Loan Status'
)


fig.update_layout(
    xaxis_title='Person Age',
    yaxis_title='Count',
    bargap=0.1
)

fig.show()


# categorical_count_by_categorical_status
def count_by_status(df, categorical_column, categorical_status): 
    count_by_status = df.groupby(
        [categorical_column, categorical_status]).size().reset_index(name='count')
    return count_by_status

# age_by_loan
age_by_loan = count_by_status(train, 'person_age', 'loan_status')
print(age_by_loan.shape)
age_by_loan.head(10)


# Taking the top 10 of candidate

top_10 = age_by_loan.head(20)
plt.figure(figsize=(10,5))
sns.barplot(data = top_10, x='person_age', y='count', hue='loan_status', palette='Set2')
plt.title('Top 10 Person Age Counts by Loan Status')
plt.xlabel('Person Age')
plt.ylabel('Count')
plt.show()


# income histplot
plt.figure(figsize=(14, 5))
sns.histplot(data = train, x = 'person_income', bins = 100)
plt.title('distribution of person_income in train_df')
plt.xlabel('person_income')
plt.ylabel('counts')
plt.show()


# filtered income range
plt.figure(figsize=(14, 5))
filtered_df = train[(train['person_income'] >= 0) & (train['person_income'] <= 150000)]
sns.histplot(data=filtered_df, x='person_income', bins=100 , kde = True)
plt.title('distribution of person_income in train df')
plt.xlabel('person_income')
plt.ylabel('counts')
plt.show()


# categorical_sum
def categorical_sum(df, categorical_column): 
    categorical_sum = df.groupby([categorical_column]).size().reset_index(name='sum')
    return categorical_sum

# home_ownership_sum
home_ownership_sum = categorical_sum(train, 'person_home_ownership')
print(home_ownership_sum.shape)
home_ownership_sum.head()


# home_ownership_loan
home_ownership_loan = count_by_status(train, 'person_home_ownership', 'loan_status')
print(home_ownership_loan.shape)
home_ownership_loan


plt.figure(figsize=(12,5))
sns.barplot(data = home_ownership_loan, x = 'person_home_ownership', y = 'count', hue = 'loan_status')
plt.title('Top 10 Person Home Ownership Counts by Loan Status')
plt.yscale('log')  # <-- makes small values more visible
plt.xlabel('Person Age')
plt.ylabel('Count')
plt.show()


# Calculating total and accepted loan counts for each home ownership type
ratio_df = (
    home_ownership_loan.groupby('person_home_ownership')
    .apply(lambda x: x.loc[x['loan_status'] == 1, 'count'].sum() / x['count'].sum())
    .reset_index(name='acceptance_ratio')
)

ratio_df



plt.figure(figsize=(7, 4)) 

# Plot the pie chart
plt.pie(
    x = ratio_df['acceptance_ratio'],  
    labels = ratio_df['person_home_ownership'],
    autopct='%1.1f%%',  # format percentage
    startangle = 90,       
    wedgeprops={'edgecolor': 'black'}
)

plt.title('Visual Representation of Loan Acceptance Ratio')
plt.axis('equal')
plt.show()


train.person_emp_length.describe()


test.person_emp_length.describe() # fior test_df the max is ok.


def plot_histogram(df , name):
    plt.figure(figsize=(14, 5))  
    plt.subplot(2, 1, 1) 
    sns.histplot(data = df , x = 'person_emp_length', bins = 100 , kde = True)
    plt.title(f'distribution of person_emp_length in {name}')
    plt.xlabel('person_emp_length')
    plt.ylabel('count')
    plt.tight_layout() 
    plt.show()

# person_emp_length histplot
plot_histogram(train , 'train')
plot_histogram(test , 'test')


# emp_length range grouping 
bins = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 9.5, 20.5]
labels = ['Q1','Q2','Q3','Q4','Q5','Q6','Q7','Q8','Q9','Q10']

loan_emp_length = pd.DataFrame()
loan_emp_length['emp_length_group'] = pd.cut(train['person_emp_length'], bins = bins, labels = labels)

loan_emp_length['person_emp_length'] = filtered_df.person_emp_length
loan_emp_length['loan_status'] = filtered_df.loan_status

loan_emp_length = loan_emp_length.reset_index(drop = True)
print(loan_emp_length.shape)
loan_emp_length.head()


# emp_length_sum
emp_length_group_sum = categorical_sum(loan_emp_length, 'emp_length_group')
print(emp_length_group_sum.shape)
emp_length_group_sum.head()


# emp_length_loan
emp_length_loan = count_by_status(loan_emp_length, 'emp_length_group', 'loan_status')
print(emp_length_loan.shape)
emp_length_loan.head()


# aceeptance (1) ratio by emp_length
loan_1_emp_length_group = emp_length_loan[emp_length_loan['loan_status'] == 1].reset_index(drop=True)
loan_1_emp_length_group['acceptance_ratio'] = loan_1_emp_length_group['count'] / emp_length_group_sum['sum']
print(loan_1_emp_length_group.shape)
loan_1_emp_length_group.head()


# loan_1 ratio by emp_length_group 
plt.figure(figsize=(10, 2))
sns.barplot(x = 'emp_length_group', y = 'acceptance_ratio', data = loan_1_emp_length_group)
plt.title('loan_1 ratio by emp_length_group')
plt.ylabel('loan_1 ratio (%)')
plt.show()


train['loan_intent'].value_counts()


intent_by_age = count_by_status(train, 'loan_intent', 'person_age')
print(intent_by_age.shape)
intent_by_age.head()


# Filtering the dataframe for ages between 0 and 40
filtered_df = intent_by_age[
    (intent_by_age['person_age'] >= 0) & 
    (intent_by_age['person_age'] <= 40)
]

# Creating the interactive bar chart
fig = px.bar(
    filtered_df,
    x = 'person_age',
    y = 'count',
    color = 'loan_intent',  
    barmode = 'group',     
    title = 'Loan Intent by Person Age (0â€“40)',
    labels = {'person_age': 'Age', 'count': 'Count', 'loan_intent': 'Loan Intent'},
    color_discrete_sequence=px.colors.qualitative.Set1  # optional color palette
)


fig.update_layout(
    xaxis=dict(title='Person Age'),
    yaxis=dict(title='Loan Intent Count'),
    legend_title_text='Loan Intent',
    bargap=0.2
)

fig.show()


# Filtering the dataframe for ages between 0 and 40
filtered_df = intent_by_age[
    (intent_by_age['person_age'] >= 41) & 
    (intent_by_age['person_age'] <= 60)
]

# Creating the interactive bar chart
fig = px.bar(
    filtered_df,
    x = 'person_age',
    y = 'count',
    color = 'loan_intent',  
    barmode = 'group',     
    title = 'Loan Intent by Person Age (41 - 80)',
    labels = {'person_age': 'Age', 'count': 'Count', 'loan_intent': 'Loan Intent'},
    color_discrete_sequence=px.colors.qualitative.Set1  # optional color palette
)


fig.update_layout(
    xaxis=dict(title='Person Age'),
    yaxis=dict(title='Loan Intent Count'),
    legend_title_text='Loan Intent',
    bargap = 0.2
)

fig.show()


# loan_grade_sum
loan_grade_sum = categorical_sum(train , 'loan_grade')
print(loan_grade_sum.shape)
loan_grade_sum.head()


# loan_grade_loan
loan_grade_loan = count_by_status(train , 'loan_grade', 'loan_status')
print(loan_grade_loan.shape)
loan_grade_loan.head()


# loan_1 ratio by loan_grade
loan_1_loan_grade = loan_grade_loan[loan_grade_loan['loan_status'] == 1].reset_index(drop=True)
loan_1_loan_grade['acceptance_ratio'] = loan_1_loan_grade['count'] / loan_grade_sum['sum']
print(loan_1_loan_grade.shape)
loan_1_loan_grade.head()


# loan acceptance ratio by loan_grade 
plt.figure(figsize=(14, 2))
sns.barplot(x='loan_grade', y='acceptance_ratio', data=loan_1_loan_grade)
plt.title('loan_1 ratio by emp_length_group')
plt.ylabel('loan_1 ratio (%)')
plt.show()


train.loan_int_rate.describe()


# List of features
features = ['person_age', 'person_income', 'person_emp_length', 
            'loan_percent_income', 'cb_person_cred_hist_length']

corr = train[['loan_int_rate'] + features].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.show()


train['loan_percent_income'].describe()


# distribution
plt.figure(figsize=(7, 3))

plt.subplot(1, 2, 1)  
sns.histplot(data = train, x = "loan_percent_income", bins = 30, kde=True)
plt.title("dist. of loan_percent_income")

plt.subplot(1, 2, 2) 
train['log_loan_percent_income'] = np.log(train['loan_percent_income']+0.001)
sns.histplot(data = train, x = "log_loan_percent_income", bins = 30, kde = True)
plt.title("dist. of log transformation")

plt.tight_layout()  
plt.show()


train['cb_person_default_on_file'].value_counts()



# default_by_loan
default_by_loan = count_by_status(train, 'cb_person_default_on_file', 'loan_status')
print(default_by_loan.shape)
default_by_loan.head()


default_sum = categorical_sum(train, 'cb_person_default_on_file')

# loan_1 ratio by default
default_loan_1 = default_by_loan[default_by_loan['loan_status'] == 1].reset_index(drop=True)
default_loan_1['acceptance_ratio'] = default_loan_1['count'] / default_sum['sum']
print(default_loan_1.shape)
default_loan_1.head()

# It is easier to get a loan if you have a history of defaults.???


train['cb_person_cred_hist_length'].describe()


# cred_hist_length with loan_status
plt.figure(figsize=(14, 3))
sns.countplot(x = "cb_person_cred_hist_length", hue='loan_status', data = train)
plt.title("countplot of cb_person_cred_hist_length")
plt.show()


feature_train = train.copy()
feature_train.drop(columns = ['id'] , inplace=True)

numerical_columns = feature_train.select_dtypes(include = ['int64' , 'float64']).columns
numerical_columns = list(numerical_columns)

plt.figure(figsize=(15, 10)) 
corr = train[numerical_columns].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap of Numerical Features')
plt.show()


train = pd.read_csv('/kaggle/input/playground-series-s4e10/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s4e10/test.csv')
sample_submission = pd.read_csv('/kaggle/input/playground-series-s4e10/sample_submission.csv')


train.columns


test.columns


# filtered income range
filtered_train = train[(train['person_income'] >= 0) & 
                          (train['person_income'] <= 150000)] # 150K
print(filtered_train.shape)
print(filtered_train.shape[0]/train.shape[0])

filtered_train = filtered_train[(filtered_train['person_age'] >= 20) & 
                                (filtered_train['person_age'] <= 70)] #60
print(filtered_train.shape)
print(filtered_train.shape[0]/train.shape[0])

filtered_train = filtered_train[(filtered_train['person_emp_length'] >= -0.5) & 
                                (filtered_train['person_emp_length'] <= 20.5)] 
print(filtered_train.shape)
print(filtered_train.shape[0]/train.shape[0])


filtered_train.reset_index(drop = True, inplace = True)
print(filtered_train.shape)
filtered_train.tail()


# grouping in train
gen_feature = pd.DataFrame()
gen_feature['income_group'] = pd.qcut(filtered_train['person_income'], q = 10, 
                                      labels=['Q1','Q2','Q3','Q4','Q5','Q6','Q7','Q8','Q9','Q10'])

bins = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 9.5, 20.5]
labels = ['Q1','Q2','Q3','Q4','Q5','Q6','Q7','Q8','Q9','Q10']
gen_feature['emp_length_group'] = pd.cut(filtered_train['person_emp_length'], bins=bins, labels=labels)

print(gen_feature.shape)
gen_feature.head(2)


# grouping in test
gen_test = pd.DataFrame()
gen_test['income_group'] = pd.qcut(test['person_income'], q=10, 
                                      labels=['Q1','Q2','Q3','Q4','Q5','Q6','Q7','Q8','Q9','Q10'])

bins = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 9.5, 20.5]
labels = ['Q1','Q2','Q3','Q4','Q5','Q6','Q7','Q8','Q9','Q10']
gen_test['emp_length_group'] = pd.cut(test['person_emp_length'], bins=bins, labels=labels)

print(gen_test.shape)
gen_test.head(2)


filtered_train = pd.concat([filtered_train, gen_feature], axis=1)
test_df = pd.concat([test, gen_test], axis=1)
filtered_train.shape, test.shape


feature = filtered_train.drop(['id', 'loan_status'], axis=1) 
target = filtered_train[['loan_status']]
test = test.drop(['id'], axis=1) 
feature.shape, target.shape, test.shape


target.info()


from sklearn.preprocessing import LabelEncoder, OneHotEncoder

le = LabelEncoder()
target_le = le.fit_transform(target)
target_le


# numeric variables in feature

feature_num = feature.select_dtypes(include=['int', 'float']).columns
print(feature[feature_num].shape)
feature[feature_num].tail()


# numeric variables in test data

test_num = test.select_dtypes(include=['int', 'float']).columns
print(test[test_num].shape)
test[test_num].head()


# Standardization

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(feature[feature_num])

feature_num_st = scaler.transform(feature[feature_num])
feature_num_st = pd.DataFrame(feature_num_st, columns = feature[feature_num].columns)

test_num_st = scaler.transform(test[test_num])
test_num_st = pd.DataFrame(test_num_st, columns = test[test_num].columns)

print(feature_num_st.shape, test_num_st.shape)
feature_num_st.head()


# categoric variables in train data

feature_cat = feature.select_dtypes(include=['object']).columns
feature_cat = feature[feature_cat]
feature_cat.head()


for column in feature_cat.columns:
    print(f"{column} unique? {feature_cat[column].nunique(), feature_cat[column].unique()}")


# categoric variables in test data

test_cat = test.select_dtypes(include=['object']).columns
test_cat = test[test_cat]
test_cat.head()


for column in test_cat.columns:
    print(f"{column} unique? {test_cat[column].nunique(), test_cat[column].unique()}")


# one_hot_encoding for feature_cat

feature_cat_ohe = pd.get_dummies(feature_cat, columns = feature_cat.columns, dtype=int, 
                                 drop_first=True)
print(feature_cat_ohe.shape)
feature_cat_ohe.head(2)


# one_hot_encoding for test_cat

test_cat_ohe = pd.get_dummies(test_cat, columns = test_cat.columns, dtype=int, 
                              drop_first=True)
print(test_cat_ohe.shape)
test_cat_ohe.head(2)


# total transformed data

feature_tf = pd.concat([feature_num_st, feature_cat_ohe], axis=1) 
test_tf = pd.concat([test_num_st, test_cat_ohe], axis=1) 
print(feature_tf.shape, test_tf.shape)
feature_tf.head(2)


train_all = pd.concat([feature_tf, target], axis=1) 
print(train_all.shape)
train_all.head(2)


train_all.info()


train_all['loan_status_float'] = train_all['loan_status'].astype('float')
train_all_num = train_all.select_dtypes(include=['int', 'float']).columns
train_all_num


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc

X_train, X_val, y_train, y_val = train_test_split(feature_tf, target_le, test_size=0.2, random_state=2410)
X_train.shape, X_val.shape, y_train.shape, y_val.shape


# randomforest model
randomforest = RandomForestClassifier(random_state=2410, n_jobs=-1) #class_weight='balanced'
rf_model = randomforest.fit(X_train, y_train)

rf_train_proba = rf_model.predict_proba(X_train)[:, 1]
train_fpr, train_tpr, thresholds = roc_curve(y_train, rf_train_proba)
train_roc_auc = auc(train_fpr, train_tpr)

rf_val_proba = rf_model.predict_proba(X_val)[:, 1]
val_fpr, val_tpr, thresholds = roc_curve(y_val, rf_val_proba)
val_roc_auc = auc(val_fpr, val_tpr)

print(train_roc_auc, val_roc_auc)


# parameter optimization for randomforest model
from sklearn.model_selection import GridSearchCV

param_grid = {'n_estimators': [1000], 
              'max_depth': [19], 
              'min_samples_split': [15],
             }

grid_search = GridSearchCV(estimator=RandomForestClassifier(random_state=2410, n_jobs=-1), 
                           param_grid=param_grid, cv=5, scoring='roc_auc')

grid_search.fit(X_train, y_train)
print(grid_search.best_params_)
print("Best score: ", grid_search.best_score_)


# randomforest model with optimized parameter
randomforest = RandomForestClassifier(random_state=2410, n_jobs=-1, n_estimators=1000, 
                                      max_depth=19, min_samples_split=15)

rf_op_model = randomforest.fit(X_train, y_train)

rf_op_train_proba = rf_op_model.predict_proba(X_train)[:, 1]
train_fpr, train_tpr, thresholds = roc_curve(y_train, rf_op_train_proba)
train_roc_auc = auc(train_fpr, train_tpr)

rf_op_val_proba = rf_op_model.predict_proba(X_val)[:, 1]
val_fpr, val_tpr, thresholds = roc_curve(y_val, rf_op_val_proba)
val_roc_auc = auc(val_fpr, val_tpr)

print(train_roc_auc, val_roc_auc)


# plotting function for roc_curves 
def plot_roc_curves(y_val, y_val_proba, y_train, y_train_proba):

    fpr, tpr, thresholds = roc_curve(y_val, y_val_proba)
    val_roc_auc = auc(fpr, tpr)

    fpr_train, tpr_train, thresholds_train = roc_curve(y_train, y_train_proba)
    train_roc_auc = auc(fpr_train, tpr_train)

    plt.plot(fpr_train, tpr_train, label='ROC curve (train, area = %0.4f)' % train_roc_auc)
    plt.plot(fpr, tpr, label='ROC curve (area = %0.4f)' % val_roc_auc)
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")


# roc_curves for rf_model and rf_op_model
plt.figure(figsize=(10, 3))

plt.subplot(1, 2, 1)  
plt.title("ROC Curve for rf_model")
plot_roc_curves(y_val, rf_val_proba, y_train, rf_train_proba)

plt.subplot(1, 2, 2) 
plt.title("ROC Curve for rf_op_model")
plot_roc_curves(y_val, rf_op_val_proba, y_train, rf_op_train_proba)
plt.tight_layout()  
plt.show()


# model_light gradient boosting
import lightgbm as lgb

lgb_model = lgb.LGBMClassifier(random_state=2410, verbose=-1)
lgb_model.fit(X_train, y_train)

lgb_train_proba = lgb_model.predict_proba(X_train)[:, 1]
train_fpr, train_tpr, thresholds = roc_curve(y_train, lgb_train_proba)
train_roc_auc = auc(train_fpr, train_tpr)

lgb_val_proba = lgb_model.predict_proba(X_val)[:, 1]
val_fpr, val_tpr, thresholds = roc_curve(y_val, lgb_val_proba)
val_roc_auc = auc(val_fpr, val_tpr)

print(train_roc_auc, val_roc_auc)


# parameter optimization for lgb model
from sklearn.model_selection import RandomizedSearchCV

param_grid = {'n_estimators': [440],            
              'learning_rate': [0.03683007894727166],     
              'min_child_samples': [29],        
             }

model = lgb.LGBMClassifier(random_state=2410, verbose=-1, n_jobs=4)

Randomized_search = RandomizedSearchCV(model, param_grid, cv=5, scoring='roc_auc')
Randomized_search.fit(X_train, y_train)

print("The best hyperparameters are ", Randomized_search.best_params_)
print("Best score: ", Randomized_search.best_score_)
# Best score:  0.9577535382098512 


# lgb_model with optimized parameter
lgb_op_model = lgb.LGBMClassifier(random_state=2410, verbose=-1, n_jobs=4, 
                                  n_estimators = 440, learning_rate = 0.03683007894727166, 
                                  min_child_samples = 29)
lgb_op_model.fit(X_train, y_train)

lgb_op_train_proba = lgb_op_model.predict_proba(X_train)[:, 1]
train_fpr, train_tpr, thresholds = roc_curve(y_train, lgb_op_train_proba)
train_roc_auc = auc(train_fpr, train_tpr)

lgb_op_val_proba = lgb_op_model.predict_proba(X_val)[:, 1]
val_fpr, val_tpr, thresholds = roc_curve(y_val, lgb_op_val_proba)
val_roc_auc = auc(val_fpr, val_tpr)

print(train_roc_auc, val_roc_auc)


# roc_curves for lgb_model and lgb_op_model
plt.figure(figsize=(10, 3))

plt.subplot(1, 2, 1)  
plt.title("ROC Curve for lgb_model")
plot_roc_curves(y_val, lgb_val_proba, y_train, lgb_train_proba)

plt.subplot(1, 2, 2) 
plt.title("ROC Curve for lgb_op_model")
plot_roc_curves(y_val, lgb_op_val_proba, y_train, lgb_op_train_proba)
plt.tight_layout()  
plt.show()


# model catboost
from catboost import CatBoostClassifier

cat_model = CatBoostClassifier(random_state = 2410, verbose=0)
cat_model.fit(X_train, y_train)

cat_train_proba = cat_model.predict_proba(X_train)[:, 1]
train_fpr, train_tpr, thresholds = roc_curve(y_train, cat_train_proba)
train_roc_auc = auc(train_fpr, train_tpr)

cat_val_proba = cat_model.predict_proba(X_val)[:, 1]
val_fpr, val_tpr, thresholds = roc_curve(y_val, cat_val_proba)
val_roc_auc = auc(val_fpr, val_tpr)

print(train_roc_auc, val_roc_auc)


from sklearn.model_selection import GridSearchCV

param_grid = {'iterations': [1000], 
              'learning_rate': [0.25],
              'depth': [3],
              'l2_leaf_reg': [8], 
             }

model = CatBoostClassifier(random_state=2410, verbose=0)

grid_search = GridSearchCV(model, param_grid, cv=5, scoring='roc_auc')
grid_search.fit(X_train, y_train)

print("The best hyperparameters are ", grid_search.best_params_)
print("Best score: ", grid_search.best_score_)


# cat_model with optimized parameter

cat_op_model = CatBoostClassifier(random_state = 2410, verbose=0, 
                                  iterations = 1000, learning_rate = 0.25, 
                                  depth = 3, l2_leaf_reg = 8)
cat_op_model.fit(X_train, y_train)

cat_op_train_proba = cat_op_model.predict_proba(X_train)[:, 1]
train_fpr, train_tpr, thresholds = roc_curve(y_train, cat_op_train_proba)
train_roc_auc = auc(train_fpr, train_tpr)

cat_op_val_proba = cat_op_model.predict_proba(X_val)[:, 1]
val_fpr, val_tpr, thresholds = roc_curve(y_val, cat_op_val_proba)
val_roc_auc = auc(val_fpr, val_tpr)

print(train_roc_auc, val_roc_auc)


# roc_curves for cat_model and cat_op_model
plt.figure(figsize=(10, 3))

plt.subplot(1, 2, 1)  
plt.title("ROC Curve for cat_model")
plot_roc_curves(y_val, cat_val_proba, y_train, cat_train_proba)

plt.subplot(1, 2, 2) 
plt.title("ROC Curve for cat_op_model")
plot_roc_curves(y_val, cat_op_val_proba, y_train, cat_op_train_proba)
plt.tight_layout()  
plt.show()


from sklearn.ensemble import VotingClassifier

ensemble_model = VotingClassifier(
    estimators=[('rf', rf_op_model), ('lgb', lgb_op_model), ('cat', cat_op_model)],
    voting='soft'  # 'soft' uses probabilities
)



ensemble_model.fit(X_train, y_train)

ensemble_model_train_proba = ensemble_model.predict_proba(X_train)[:, 1]

train_fpr, train_tpr, thresholds = roc_curve(y_train, ensemble_model_train_proba)

train_roc_auc = auc(train_fpr, train_tpr)

ensemble_model_val_proba = ensemble_model.predict_proba(X_val)[:, 1]

val_fpr, val_tpr, thresholds = roc_curve(y_val, ensemble_model_val_proba)

val_roc_auc = auc(val_fpr, val_tpr)

print(train_roc_auc, val_roc_auc)


import numpy as np
from sklearn.linear_model import LogisticRegression

# ----- 1ï¸�âƒ£ Get base model predictions on validation data -----
train_rf = rf_op_model.predict_proba(X_train)[:, 1]
train_lgb = lgb_op_model.predict_proba(X_train)[:, 1]
train_cat = cat_op_model.predict_proba(X_train)[:, 1]

val_rf = rf_op_model.predict_proba(X_val)[:, 1]
val_lgb = lgb_op_model.predict_proba(X_val)[:, 1]
val_cat = cat_op_model.predict_proba(X_val)[:, 1]

# ----- 2ï¸�âƒ£ Stack predictions horizontally -----
stack_train = np.column_stack((train_rf, train_lgb, train_cat))
stack_val = np.column_stack((val_rf, val_lgb, val_cat))

# ----- 3ï¸�âƒ£ Train meta-model -----
meta_model = LogisticRegression()
meta_model.fit(stack_train, y_train)

# ----- 4ï¸�âƒ£ Predict on training and validation -----
train_meta_proba = meta_model.predict_proba(stack_train)[:, 1]
val_meta_proba = meta_model.predict_proba(stack_val)[:, 1]

# ----- 5ï¸�âƒ£ Compute ROC-AUC -----
train_fpr, train_tpr, _ = roc_curve(y_train, train_meta_proba)
train_roc_auc = auc(train_fpr, train_tpr)

val_fpr, val_tpr, _ = roc_curve(y_val, val_meta_proba)
val_roc_auc = auc(val_fpr, val_tpr)

print(f"Train ROC AUC: {train_roc_auc:.4f}")
print(f"Validation ROC AUC: {val_roc_auc:.4f}")



test_pred_proba = ensemble_model.predict_proba(test_tf)[:, 1]
test_pred_proba


id = pd.read_csv('/kaggle/input/playground-series-s4e10/test.csv')


ensemble_df = pd.DataFrame({'id': id['id'], 'loan_status':test_pred_proba })

ensemble_df.head()


ensemble_df.to_csv('submission.csv', index=False)




