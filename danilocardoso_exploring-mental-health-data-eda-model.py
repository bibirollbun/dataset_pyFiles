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


import warnings

import matplotlib.pyplot as plt

import seaborn as sns
from matplotlib import cm 

from scipy.stats import chi2_contingency, stats
#import researchpy as rp
from itertools import combinations

from sklearn.preprocessing import LabelEncoder

import statsmodels.api as sm
import statsmodels.formula.api as smf 

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, ConfusionMatrixDisplay, accuracy_score,\
 precision_score, recall_score, f1_score

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


warnings.filterwarnings('ignore')


train = pd.read_csv('/kaggle/input/playground-series-s4e11/train.csv', index_col=0)
test = pd.read_csv('/kaggle/input/playground-series-s4e11/test.csv', index_col=0)
original = pd.read_csv('/kaggle/input/depression-surveydataset-for-analysis/final_depression_dataset_1.csv')
pd.set_option('display.max_columns', None)


print(f'Number of rows: {train.shape[0]} \nNumber of cols: {train.shape[1]}')

print('\n')

negative = train['Depression'][train['Depression'] == 0].count()
positive = train['Depression'][train['Depression'] == 1].count()

print(f'Negative: {negative}\nPositve: {positive} \n%Positive: {positive/(positive+negative):.2f}')


train.head()


train.info()


train.describe().T.round(2)


train.isna().sum()


train[train.duplicated()]


print(f'Number of rows: {test.shape[0]} \nNumber of cols: {test.shape[1]}')


test.head()


test.describe().T.round(2)


print(f'Number of rows: {original.shape[0]} \nNumber of cols: {original.shape[1]}')

print('\n')

negative = original['Depression'][original['Depression'] == 'No'].count()
positive = original['Depression'][original['Depression'] == 'Yes'].count()

print(f'Negative: {negative}\nPositve: {positive} \n%Positive: {positive/(positive+negative):.2f}')


original.head()


original.info()


original.describe().T.round(2)


original.isna().sum()


original[original.duplicated()]


# check for duplicates between dep_dataset and train dataset
(original[pd.concat([original[['Name', 'Gender', 'Age', 'City']], train[['Name', 'Gender', 'Age', 'City']]])
             .reset_index()
             .duplicated()]
)


pd.set_option('display.max_rows', 10)


new_cols = (train.columns.str.lower()
              .str.replace(' ', '_')
              .str.replace('/', '_')
              .str.replace('have_you_ever_had_suicidal_thoughts_?', 'suicidal_thoughts')
        )


train.columns = new_cols
original.columns = new_cols
test.columns = new_cols[0:18]


# Check the list of values for each column in the synthetic data
pd.set_option('display.max_rows', None)
for col in train.columns:
    if col != 'name':
        print(train[col].value_counts())


# Check the list of values for each column in the original data
pd.set_option('display.max_rows', None)
for col in original.columns:
    if col != 'name':
        print(original[col].value_counts())


# Remove the wrong values
train_subset = train[train['city'].isin(original['city'])]

train_subset = train_subset[train_subset['profession'].isin(original['profession'])]

train_subset = train_subset[train_subset['sleep_duration'].isin(original['sleep_duration'])]

train_subset = train_subset[train_subset['dietary_habits'].isin(original['dietary_habits'])]

train_subset = train_subset[train_subset['degree'].isin(original['degree'])]


# % data removed
1-(len(train_subset)/len(train))


cols_corrected = ['city','profession','sleep_duration', 'dietary_habits', 'degree']

# Double check the summary
pd.set_option('display.max_rows', None)

for col in cols_corrected:
     print(train_subset[col].value_counts())


# split the data into Synthetic and Original Survey Data to check data similarities
train_subset['Source'] = 'Synthetic'
original['Source'] = 'Original'


# label depression var
depression_map = {'No':0, 'Yes':1}

original['depression'] = original['depression'].map(depression_map)


# Union both datasets
mental_health = pd.concat([train_subset, original], axis=0).reset_index(drop=True)


# create an age group
mental_health['age_group'] = ['<=30' if val <= 30 else '31 - 40'
                          if val <= 40 else '41 - 50'
                          if val <= 50 else '51 - 60'
                          if val <=60 else '> 60'
                          for val in mental_health['age']]


# Combine Working Professional and Student vars
def set_student(df):
    df['working_professional_or_student'] = df['working_professional_or_student'].map({'Working Professional':'No', 'Student':'Yes'})
    df.rename(columns={'working_professional_or_student':'student'}, inplace=True)


set_student(mental_health)

set_student(test)


# Combine Profession and student vars
def set_profession(df):
    df['profession'].loc[(df['student'] == 'Yes') & (df['profession'].isna())] = 'Student'
    df['profession'].loc[(df['student'] == 'No') & (df['profession'].isna())] = 'Unknown'


set_profession(mental_health)
set_profession(test)


# Combine academic pressure and work pressure vars
def set_work_academic_pressure(df):
    df['academic_pressure'].loc[(df['student'] == 'No') & (df['academic_pressure'].isna())] = df['work_pressure']
   # df['Academic Pressure'].loc[df['Academic Pressure'].isna()] = mental_health['Academic Pressure'].mode()
    df.drop(columns='work_pressure', inplace=True)
    df.rename(columns={'academic_pressure':'academic_work_pressure'}, inplace=True)    


set_work_academic_pressure(mental_health)
set_work_academic_pressure(test)


# Replace students withtout cgpa with the average of cgpa
def set_cgpa(df):
    df['cgpa'].loc[(df['student'] == 'Yes') & (df['cgpa'].isna())] = df['cgpa'].mean()
    #df['CGPA'].loc[df['Student'] == 'No'] = df['CGPA'].mean() 


set_cgpa(mental_health)
set_cgpa(test)


# Combine study and job satisfaction vars
def set_study_job_satisfaction(df):
    df['study_satisfaction'].loc[(df['student'] == 'No') & (df['study_satisfaction'].isna())] = df['job_satisfaction']
    df['study_satisfaction'].loc[df['study_satisfaction'].isna()] = df['study_satisfaction'].mode()[0]
    df.drop(columns='job_satisfaction', inplace=True)
    df.rename(columns={'study_satisfaction':'study_job_satisfaction'}, inplace=True)
    


set_study_job_satisfaction(mental_health)
set_study_job_satisfaction(test)


# Replace NA's in financial stress with the mode
def set_financial_stress(df):
    df['financial_stress'].fillna(df['financial_stress'].mode()[0], inplace=True)



set_financial_stress(mental_health)
set_financial_stress(test)


# Binning work/study hours
def group_study_hrs(df):
    df['study_hours_groupped'] = ['< 5' if st < 5 else '5-8' if st <8 else '>8' for st in df['work_study_hours']] 


group_study_hrs(mental_health)


# Binning cgpa 
def group_cgpa(df):
    df['cgpa_groupped'] = ['Low' if st < 6 else 'Medium' if st <8 else 'High' for st in df['cgpa']] 


group_cgpa(mental_health)


#mental_health['overall_pressure_satisfaction_ratio'] = mental_health['Academic Work Pressure']/ (1-(mental_health['Study Job Satisfaction']))


categorical_var = ['gender', 'city', 'student', 'profession', 'sleep_duration', 'dietary_habits', 'degree', 'suicidal_thoughts',\
                   'family_history_of_mental_illness', 'age_group', 'degree_category', 'cgpa_groupped', 'study_hours_groupped']

discrete_var = ['academic_work_pressure', 'study_job_satisfaction', 'work_study_hours','financial_stress', 'age']

continous_var = ['cgpa']


# Binning Degree
degree = {
          'Undergraduate': ['B.Com', 'BE', 'BA', 'BCA', 'B.Ed', 'LLB', 'B.Arch', 'BBA', 'BHM', 'B.Tech', 'B.Pharm', 'BSc'],
          'High School': ['Class 12'],
          'Postgraduate' :['MA', 'M.Com', 'MCA', 'M.Tech', 'ME', 'MBA', 'M.Pharm', 'MSc', 'MHM', 'M.Ed', 'LLM'],
          'Professional/Doctorate' :['MBBS', 'MD', 'PhD']
         }

degree_cat = []
for d in mental_health['degree']:
    for k, v in degree.items():
        if d in v:
            degree_cat.append(k)
            
mental_health['degree_category'] = degree_cat


pd.set_option('display.max_rows', 10)


# Split he data into Original and Synthetic
mental_health_original = mental_health[mental_health['Source'] == 'Original']
mental_health_synthetic = mental_health[mental_health['Source'] == 'Synthetic']


pd.set_option('display.max_rows', 5)


col_palette = ['Yellow', 'Red']


# function to plot a pie chart 
def pie_chart(col):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 15))   

    ax1.pie(mental_health_synthetic[col].value_counts(),
                   colors=col_palette,
                   autopct='%.1f%%')
    plt.legend(mental_health[col].unique())
    ax1.set_title(f'% {col} Synthetic Data')


    ax2.pie(mental_health_original[col].value_counts(),
                   colors=col_palette,
                   autopct='%.1f%%')
    plt.legend(mental_health['depression'].unique())
    ax2.set_title(f'% {col} Original Data');


# function to plot a stacked bar chart %
def stacked_chart(col):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5)) 

    df_grouped = mental_health_original.groupby(col)['depression'].value_counts(normalize=True).unstack('depression').sort_values(col, ascending=False)
    df_grouped.plot.bar(stacked=True, cmap=cm.get_cmap('viridis'), width=0.75 ,color=col_palette,ax=ax1)
    ax1.set_title(f'% {col} Original Data')
    ax1.set_ylabel('')

    df_grouped = mental_health_synthetic.groupby(col)['depression'].value_counts(normalize=True).unstack('depression').sort_values(col, ascending=False)
    df_grouped.plot.bar(stacked=True, cmap=cm.get_cmap('viridis'), width=0.75, color=col_palette, ax=ax2)
    ax2.set_title(f'% {col} Synthetic Data')
    ax2.set_ylabel('');


pie_chart('depression')


for col in categorical_var:
    stacked_chart(col)


for col in categorical_var:  
        plt.figure(figsize=(10,5))
        sns.barplot(data = mental_health[col].value_counts().reset_index().sort_values('count'),
                   x=col,
                   y='count')
        plt.show()


for num_var in discrete_var+continous_var:    
   
    fig, ax = plt.subplots(1, 2, figsize=(12, 3))   
    
    # Histograms    
    sns.histplot(data=mental_health, 
                      x=num_var,
                      bins = 30,
                      hue = 'depression',
                      palette=col_palette,
                      kde=True,
                      alpha=0.3,
                      ax=ax[0]                          
                ) 
    # Mean vertical line
   # ax[0].axvline(np.mean(mental_health_viz[num_var]), color="red")   
    
    # Boxplots
    sns.boxplot(data=mental_health, 
                     y=num_var,
                     hue = 'depression',
                     palette=col_palette,
                     showfliers=False,
                     ax=ax[1]
                   );


mental_health_subset = mental_health.copy().drop(columns=['name', 'Source']) 


sns.barplot(data=mental_health_subset[['depression', 'academic_work_pressure']].groupby('depression').mean().reset_index(),
           x='depression',
           y='academic_work_pressure');


#pd.set_option('display.max_rows', None)
sns.barplot(data=mental_health_subset[['depression', 'academic_work_pressure']].groupby('academic_work_pressure').mean().reset_index().sort_values('depression', ascending=False),
            x='academic_work_pressure',
            y='depression');


sns.barplot(data=mental_health_subset[['depression', 'study_job_satisfaction']].groupby('depression').mean().reset_index(),
           x='depression',
           y='study_job_satisfaction');


#pd.set_option('display.max_rows', None)
sns.barplot(data=mental_health_subset[['depression', 'study_job_satisfaction']].groupby('study_job_satisfaction').mean().reset_index().sort_values('depression', ascending=False),
            x='study_job_satisfaction',
            y='depression');


### Depressed people are more likely to have an negative perspective about everything and also, they are less productive due to lack of engagement and they have some cognitive losses, such as: Attention, Focus, Working memory, etc. 


sns.barplot(data=mental_health_subset[['depression', 'student']].groupby('student').mean().reset_index(),
           x='student',
           y='depression');


plt.figure(figsize=(10,5))
sns.barplot(data=mental_health_subset[['depression', 'sleep_duration']].groupby('sleep_duration').mean().reset_index(),
           x='sleep_duration',
           y='depression');


plt.figure(figsize=(10,5))
sns.barplot(data=mental_health_subset[['depression', 'dietary_habits']].groupby('dietary_habits').mean().reset_index(),
           x='dietary_habits',
           y='depression');


plt.figure(figsize=(10,5))
sns.barplot(data=mental_health_subset[['depression', 'work_study_hours']].groupby('work_study_hours').mean().reset_index(),
           x='work_study_hours',
           y='depression');


plt.figure(figsize=(10,5))
sns.barplot(data=mental_health_subset[['depression', 'financial_stress']].groupby('financial_stress').mean().reset_index().round(1),
           x='financial_stress',
           y='depression');


plt.figure(figsize=(10,5))
sns.barplot(data=mental_health_subset[['depression', 'family_history_of_mental_illness']].groupby('family_history_of_mental_illness').mean().reset_index().round(1),
           x='family_history_of_mental_illness',
           y='depression');


mental_health_original['gender'].value_counts(normalize=True)


mental_health_original[['gender', 'depression']].groupby('gender').mean().reset_index()


from scipy.stats import chi2_contingency

def cont_tb(var1, var2):
    return pd.crosstab(mental_health_original[var1], mental_health_original[var2])    


ct_gender = cont_tb('gender','depression')


def chi_tb(cont_tb):
    chi2, p, dof, expected = chi2_contingency(cont_tb)
    df= pd.DataFrame({"Metric": ["Chi-square statistic (χ²)", "p-value", "Degrees of Freedom (dof)"],
             "Value": [chi2, p, dof]})
    return df


chi_tb(ct_gender)


mental_health_original['age_group'].value_counts(normalize=True)


mental_health_original[['age_group', 'depression']].groupby('age_group').mean().reset_index()


ct_age = cont_tb('age_group','depression')


chi_tb(ct_age)


def chi_post_hoc(var1, var2):
    """
    Perform pairwise Chi-square tests and compute odds ratios for post-hoc analysis.

    Args:
    - var1: Independent categorical variable
    - var2: Dependent binary variable
    - data: DataFrame containing the variables

    Returns:
    - DataFrame with pairwise comparisons, Chi-square statistics, p-values, Bonferroni corrected p-values, significance, and odds ratios.
    """
    # Convert to DataFrame
    df = pd.DataFrame(mental_health_original)

    # Get unique categories for the independent variable
    categories = df[var1].unique()

    # Generate all pairwise combinations of categories
    category_pairs = list(combinations(categories, 2))

    # Initialize a list to store results
    post_hoc_results = []

    # Perform pairwise Chi-square tests
    for pair in category_pairs:
        # Filter data for the pair of categories
        filtered_data = df[df[var1].isin(pair)]

        # Create contingency table
        contingency_table = pd.crosstab(filtered_data[var1], filtered_data[var2])

        # Perform Chi-square test
        chi2, p, dof, expected = chi2_contingency(contingency_table)

        # Calculate odds ratio
        if contingency_table.shape == (2, 2):  # Ensure 2x2 table for odds ratio calculation
            (a, b), (c, d) = contingency_table.values
            odds_ratio = (a / b) / (c / d)
        else:
            odds_ratio = None

        # Store results
        post_hoc_results.append({
            "Comparison": f"{pair[0]} vs {pair[1]}",
            "Chi-square Statistic (χ²)": chi2,
            "p-value": p,
            "Odds Ratio": odds_ratio
        })

    # Convert results to a DataFrame
    post_hoc_df = pd.DataFrame(post_hoc_results)

    # Apply Bonferroni Correction
    alpha = 0.05
    post_hoc_df['Bonferroni Corrected p-value'] = alpha / len(category_pairs)

    # Determine significance
    post_hoc_df['Significant'] = post_hoc_df['p-value'] < post_hoc_df['Bonferroni Corrected p-value']

    # Display the results
    return post_hoc_df



pd.set_option('display.max_rows', None)
chi_post_hoc('age_group','depression')


mental_health_original['student'].value_counts(normalize=True)


mental_health_original[['student', 'depression']].groupby('student').mean().reset_index()


ct_student = cont_tb('student','depression')


def odds_ratio(var1, var2, cat1, cat2):
    # Create a contingency table for Gender vs Depression
    contingency_table = pd.crosstab(mental_health_original[var1], mental_health_original[var2])

    a = contingency_table.loc[cat1, 1]   
    b = contingency_table.loc[cat1, 0]   
    c = contingency_table.loc[cat2, 1]  
    d = contingency_table.loc[cat2, 0]  

    # Calculate the odds ratio (OR)
    odds_ratio = (a * d) / (b * c)

    # Display the odds ratio
    print(f"Odds Ratio (OR) for {contingency_table.index.name} and Depression: {odds_ratio:.4f}")



odds_ratio('student', 'depression', 'Yes', 'No')


mental_health_original['suicidal_thoughts'].value_counts(normalize=True)


mental_health_original[['suicidal_thoughts', 'depression']].groupby('suicidal_thoughts').mean().reset_index()


ct_suic_thoughts = cont_tb('suicidal_thoughts','depression')


chi_tb(ct_suic_thoughts)


odds_ratio('suicidal_thoughts', 'depression', 'Yes', 'No')


mental_health_original['dietary_habits'].value_counts(normalize=True)


mental_health_original[['dietary_habits', 'depression']].groupby('dietary_habits').mean().reset_index()


ct_diet_habits = cont_tb('dietary_habits','depression')


chi_tb(ct_diet_habits)


chi_post_hoc('dietary_habits', 'depression')


mental_health_original['family_history_of_mental_illness'].value_counts(normalize=True)


mental_health_original[['family_history_of_mental_illness', 'depression']].groupby('family_history_of_mental_illness').mean().reset_index()


ct_fam_illness = cont_tb('family_history_of_mental_illness','depression')


chi_tb(ct_fam_illness)


mental_health_original['sleep_duration'].value_counts(normalize=True)


mental_health_original[['sleep_duration', 'depression']].groupby('sleep_duration').mean().reset_index()


ct_sleep_duration = cont_tb('sleep_duration','depression')


chi_tb(ct_sleep_duration)


chi_post_hoc('sleep_duration', 'depression')


mental_health_original['degree_category'].value_counts(normalize=True)


mental_health_original[['degree_category', 'depression']].groupby('degree_category').mean().reset_index()


ct_degree_cat = cont_tb('degree_category','depression')


chi_tb(ct_degree_cat)


chi_post_hoc('degree_category', 'depression')


mental_health_original['study_hours_groupped'][mental_health_original['student'] == 'Yes'].value_counts(normalize=True)


mental_health_original[['study_hours_groupped','depression']][mental_health_original['student'] == 'Yes'].groupby('study_hours_groupped').mean().reset_index()


ct_study = cont_tb('study_hours_groupped','depression')


chi_tb(ct_study)


chi_post_hoc('study_hours_groupped', 'depression')


cat_cols = ['city', 'student','profession', 'sleep_duration', 'sleep_duration', 'dietary_habits',\
            'suicidal_thoughts', 'family_history_of_mental_illness', 'depression']

cont_cols = ['age', 'academic_work_pressure', 'study_job_satisfaction', 'cgpa', 'work_study_hours', 'financial_stress',]


mental_health_enc = mental_health_subset.copy()

enc = LabelEncoder()

for col in cat_cols:
    mental_health_enc[col] = enc.fit_transform(mental_health_enc[col])
    
for col in cat_cols:
    if col != 'depression':
        test[col] = enc.fit_transform(test[col])


pd.set_option('display.max_rows', 5)
mental_health_enc.drop(columns=['degree', 'age_group'], inplace=True)


from sklearn.model_selection import train_test_split

X = mental_health_enc.drop(columns='depression')

y = mental_health_enc['depression']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, stratify=y)


mental_health_enc.fillna(0, inplace=True)
test.fillna(0, inplace=True)


df = pd.DataFrame(mental_health_enc)

# Define independent variables (X) and dependent variable (y)
X = df[['age', 'academic_work_pressure', 'cgpa', 'study_job_satisfaction',
        'suicidal_thoughts', 'work_study_hours', 'financial_stress', 'family_history_of_mental_illness']]
y = df['depression']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the logistic regression model
logreg = LogisticRegression()

# Train the model
logreg.fit(X_train, y_train)

# Predict on the test set
y_pred = logreg.predict(X_test)
y_pred_proba = logreg.predict_proba(X_test)[:, 1]

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot();

# Evaluate the model
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nROC-AUC Score:")
print(roc_auc_score(y_test, y_pred_proba))


# Initialize XGBoost model with default parameters
xgb = XGBClassifier()

xgb.fit(X_train, y_train)

y_pred_xgb = xgb.predict(X_test)

# Evaluate the model


cm = confusion_matrix(y_test, y_pred_xgb)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot();

# Evaluate the model
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_xgb))
print("\nClassification Report:")
print(classification_report(y_test, y_pred_xgb))
#print("\nROC-AUC Score:")
#print(roc_auc_score(y_test, y_pred_proba))



importances = xgb.feature_importances_
indices = np.argsort(importances)
features = X.columns

plt.title('Feature Importances XGB')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()


rf = RandomForestClassifier(random_state=0)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)


cm = confusion_matrix(y_test, y_pred_rf)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot();

# Evaluate the model
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_rf))
print("\nClassification Report:")
print(classification_report(y_test, y_pred_rf))
#print("\nROC-AUC Score:")
#print(roc_auc_score(y_test, y_pred_proba))


importances = rf.feature_importances_
indices = np.argsort(importances)

plt.title('Feature Importances Random Forest')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()


y_pred = xgb.predict(test[X.columns])


submission_df = pd.DataFrame({'id': test.index, 
                              'Target':y_pred})


submission_df.to_csv('submission.csv', index=False)
submission_df.head()

