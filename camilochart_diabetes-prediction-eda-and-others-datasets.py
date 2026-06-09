import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')


df_train = pd.read_csv('/kaggle/input/playground-series-s5e12/train.csv', index_col='id')
df_train.head()


df_train.info()


df_train.describe()


df_train.isnull().sum()


df_train.nunique()


from matplotlib.colors import LinearSegmentedColormap

colors = ['#2ABEF4','#F4602A']
custom_cmap = LinearSegmentedColormap.from_list('CustomMap', colors)


target_column = 'diagnosed_diabetes'


bins = [0, 0.5, 1.0]
labels = ['negative diabetes','positive diabetes']


df_train['Range_'+target_column] = pd.cut(df_train[target_column],bins=bins, labels=labels, include_lowest=True)
df_train['Range_'+target_column].value_counts().sort_index()


df_train['Range_'+target_column].value_counts().sort_index().plot.bar(color=colors)


df_train['Range_'+target_column].value_counts().sort_index().plot.pie(cmap=custom_cmap,autopct='%1.1f%%',)


str_columns = list(df_train.select_dtypes(include=["object"]).columns)
str_columns


def print_stacked_columns(df,column):

    col_to_stack = 'Range_'+target_column
    df_1 = df[[column,col_to_stack]]
    df_1.groupby([column,col_to_stack]).size().unstack().plot(kind='bar', stacked=True,cmap=custom_cmap)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.title('column: ' + column)
    plt.show()


for co in str_columns :
    print_stacked_columns(df_train,co)


num_columns = list(df_train.select_dtypes(exclude=["object",'bool']).columns)


num_columns.remove(target_column)
num_columns.remove('Range_'+target_column)


num_columns


for co in num_columns :
    print_stacked_columns(df_train,co)


from scipy.stats import shapiro


val_shapiro = []
for co in num_columns :
    stat_shapiro, p_shapiro = shapiro(df_train[co])
    val_shapiro.append(stat_shapiro)    


df_shapiro = pd.DataFrame({'Column':num_columns,'Shapiro (stat)': val_shapiro})
df_shapiro.head(20).style.background_gradient(cmap='Blues',axis=0)


df_ldl = df_train[df_train['age'] <= 25]
print('age under 25 years')
print_stacked_columns(df_ldl,'age')


df_procesed = df_train.copy()
columns_to_encode = str_columns
df_procesed[columns_to_encode] = df_procesed[columns_to_encode].apply(lambda col : pd.Categorical(col).codes)


df_procesed = df_procesed.drop(['Range_'+target_column],axis=1)


df_corr = df_procesed.corr()
np.fill_diagonal(df_corr.values,0)


fig, ax = plt.subplots(figsize=(6, 6))
cax = ax.matshow(df_corr, cmap='RdBu')

plt.colorbar(cax)

# 5. Etiquetas en los ejes
ax.set_xticks(range(len(df_corr.columns)))
ax.set_yticks(range(len(df_corr.columns)))
ax.set_xticklabels(df_corr.columns, rotation='vertical', ha='center')
ax.set_yticklabels(df_corr.columns)

plt.title('Correlation Matrix', pad=20)
plt.show()


vmin_corr = df_corr.min().min()
vmax_corr = df_corr.max().max()


corr_column = 'bmi'
corr_result = df_corr.loc[(df_corr[corr_column] > 0.1) | (df_corr[corr_column] < -0.1), [corr_column]]
corr_result.style.background_gradient(cmap='RdBu',axis=0,vmin = vmin_corr,vmax = vmax_corr)


corr_column = 'cholesterol_total'
corr_result = df_corr.loc[(df_corr[corr_column] > 0.1) | (df_corr[corr_column] < -0.1), [corr_column]]
corr_result.style.background_gradient(cmap='RdBu',axis=0,vmin = vmin_corr,vmax = vmax_corr)


corr_column = 'physical_activity_minutes_per_week'
corr_result = df_corr.loc[(df_corr[corr_column] > 0.1) | (df_corr[corr_column] < -0.1), [corr_column]]
corr_result.style.background_gradient(cmap='RdBu',axis=0,vmin = vmin_corr,vmax = vmax_corr)



np.fill_diagonal(df_corr.values,0)
df_corr.drop(index = target_column, inplace=True)
df_corr[[target_column]].style.background_gradient(cmap='RdBu',axis=0)


ds_col_1 = [
    ["Preg", "Number of times pregnant."],
    ["Glucose", "Plasma glucose concentration 2 h in an oral glucose tolerance test."],
    ["BP", "Diastolic blood pressure (mm Hg)."],
    ["SkinThickness", "Triceps skinfold thickness (mm)."],
    ["Insulin", "2-hour serum insulin (ytIU/mL)."],
    ["BMI", "Body mass index (kg/m2 )."],
    ["DPF", "Diabetes pedigree function."],
    ["Age", "Age (years)."],
    ["Outcome", "Diabetes diagnose results (tested_positive: 1, tested_negative: 0)"]
]

df_ds_col_1 = pd.DataFrame(ds_col_1, columns=["Attribute", "Variable Description"])

df_ds_col_1.head(10)



ds_col_2 = [
    ["Demographics Data", "Gender of the participant"],
    ["Demographics Data", "Age in years at screening"],
    ["Demographics Data", "Race/ Hispanic origin"],
    ["Demographics Data", "Country of birth"],
    ["Demographics Data", "Length of time in US"],
    ["Demographics Data", "Education level"],
    ["Demographics Data", "Marital status"],
    ["Demographics Data", "Pregnancy status at exam"],
    ["Demographics Data", "Ratio of family income to poverty"],
    ["Dietary Data", "Energy (kcal)"],
    ["Dietary Data", "Protein (gm)"],
    ["Dietary Data", "Carbohydrate (gm)"],
    ["Dietary Data", "Total Sugars (gm)"],
    ["Dietary Data", "Dietary fiber (gm)"],
    ["Dietary Data", "Total fat (gm)"],
    ["Dietary Data", "Cholesterol (mg)"],
    ["Examination Data", "Body Mass Index"],
    ["Examination Data", "Systolic"],
    ["Examination Data", "Diastolic"],
    ["Laboratory Data (outcome)", "Fasting Glucose (mmol/L)"],
    ["Questionnaire Data", "Ever had a drink of any kind of alcohol"],
    ["Questionnaire Data", "Physical Activity"],
    ["Questionnaire Data", "Smoked at least 100 cigarettes in life"],
    ["Questionnaire Data", "Sleep hours-weekdays or workdays"],
    ["Questionnaire Data", "Mental Health-Depression Screener"]
]

df_ds_col_2 = pd.DataFrame(ds_col_2, columns=["Type", "Variable Description"])
df_ds_col_2.head(25)


