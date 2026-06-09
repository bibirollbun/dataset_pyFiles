# Supress Warnings

import warnings
warnings.filterwarnings('ignore')


!pip install Ipython


#Import the required packages

import os
import calendar
from datetime import datetime
import math
import pandas as pd
import numpy as np

from IPython.display import display_markdown
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import IncrementalPCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go


# reading the dataset
telecom = pd.read_csv("/content/train.csv")




# summary of the dataset:
telecom.info(verbose=True, show_counts=True)


# view the top 5 rows of the data

telecom.head()


#function to plot data as table
def PlotAsTable(df, figName):
    print(f"----------------------------------------------------------------------------------------------------- \n Note: If you dont see the table '{figName}' below,\n please ensure the Jupyter Notebook is marked Trusted (File --> Trusted Notebook) \n-----------------------------------------------------------------------------------------------------")
    display_markdown(f'''#### {figName} ''',  raw=True)
    display_markdown("---",  raw=True)
    display_markdown(df.to_markdown(index = False), raw=True)
    #return df.to_markdown(index = False)


# read the data dictonary
data_dict = pd.read_csv("/content/data_dictionary.csv")
PlotAsTable(data_dict, "Data Dictionary" )


# Convert followng columns to datetime
# last_date_of_month_6, last_date_of_month_7, last_date_of_month_8,
# date_of_last_rech_6, date_of_last_rech_7, date_of_last_rech_8
# date_of_last_rech_data_6, date_of_last_rech_data_7, date_of_last_rech_data_8

dateColumns = telecom.select_dtypes(include='object')

#function that converts required columns to datetime
def ConvertDateTimeColumns(df):
    for dCol in dateColumns.columns:
        df[dCol] = pd.to_datetime(df[dCol])

#update the datetime columns
ConvertDateTimeColumns(telecom)


#Rename Columns with Meaning full Names
# aug_vbc_3g jul_vbc_3g jun_vbc_3g

#function that renames required columns
def ApplyMeaningfulName(df):
    df.rename(columns={'jun_vbc_3g': 'vbc_3g_6', 'jul_vbc_3g': 'vbc_3g_7', 'aug_vbc_3g': 'vbc_3g_8'}, inplace=True)

ApplyMeaningfulName(telecom)


def ShowSummary(df):
    df.info(verbose=True, show_counts=True)
# summary of the updated dataset:
ShowSummary(telecom)


telecom.describe()


colDesrciption = []
data_dict['Acronyms'] = data_dict['Acronyms'].str.replace(" ","")
for colName in telecom.columns:
    colDescr = ""
    acronymDescription = data_dict.loc[data_dict['Acronyms'] == colName.upper()].head(1)
    if (not acronymDescription.empty):
        colDescr = f"{acronymDescription.iloc[0]['Description']}"
    else:
        for acronyms in colName.split("_"):
            acronymDescription = data_dict.loc[data_dict['Acronyms'] == acronyms.upper()].head(1)
            if (not acronymDescription.empty):
                colDescr = f" {colDescr} {acronyms}: {acronymDescription.iloc[0]['Description']},"
            elif (acronyms in ["6" , "7", "8"]):
                colDescr = f" {colDescr} {acronyms}: {calendar.month_name[int(acronyms)]},"
            elif (acronyms == "fb"):
                colDescr = f" {colDescr} fb_user: Service scheme to avail services of Facebook and similar social networking sites"
    colDesrciption.append(colDescr)

columnDescription = pd.DataFrame({'ColumnName': telecom.columns, 'Description':colDesrciption})
PlotAsTable(columnDescription, "Column Description" )


# see the top 5 rows of the data
telecom.head()


def ReturnColumnsMissingPercentage(df):
    missing = df.isnull().sum() * 100 / len(df)
    missing.name = "MissingPercentage"
    missing = missing.to_frame().reset_index()
    return missing

percent_missing = ReturnColumnsMissingPercentage(telecom)
percent_missing["Description"] = columnDescription["Description"]
PlotAsTable(percent_missing.sort_values(by=['MissingPercentage'], ascending = False), "Missing Percentage")


def GetHighMissingCols(df):
    high_missing = df.loc[(df["MissingPercentage"] > 70.0)]["index"]
    return high_missing

high_missing_cols = GetHighMissingCols(percent_missing)
telecom.loc[:,high_missing_cols].describe()


recharge_cols_missing = ['total_rech_data_6','total_rech_data_7','total_rech_data_8',
                         'av_rech_amt_data_6','av_rech_amt_data_7','av_rech_amt_data_8']
def ReplaceRechargeColsForNoRecharge(df):
    for col in recharge_cols_missing:
        df[col] = df[col].replace(np.NaN,0.0)


ReplaceRechargeColsForNoRecharge(telecom)
ShowSummary(telecom)


# Calclate new column total recharge amount for data:  av_rech_amt_data * total_rech_data

#function that create total recharge amount data columns
def CreateTotalRechAmtDataCols(df):
    df['total_rech_amt_data_6'] = df.av_rech_amt_data_6 * df.total_rech_data_6
    df['total_rech_amt_data_7'] = df.av_rech_amt_data_7 * df.total_rech_data_7
    df['total_rech_amt_data_8'] = df.av_rech_amt_data_8 * df.total_rech_data_8

CreateTotalRechAmtDataCols(telecom)
telecom.describe()



print(telecom.quantile(np.arange(0.9, 1.01, 0.01)))



#c
print(quantile_series.head())
print(quantile_series.index)
print(type(quantile_series.index))



#c
quantile_series.index = range(len(quantile_series))  # Replace with integer index



#c
quantile_series.index = quantile_series.index.astype(str)



#c
percentage_change = quantile_series.pct_change().mul(100)



print(quantile_series)
print(quantile_series.index)



#c
quantile_series = telecom.quantile(np.arange(0.9, 1.01, 0.01))
print(type(quantile_series))
print(quantile_series.index)
print(quantile_series.head())



#c

# Select only numeric columns
numeric_columns = quantile_series.select_dtypes(include=[np.number])

# Calculate percentage change for numeric columns
percentage_change = numeric_columns.pct_change().mul(100)

# Display the result
print(percentage_change)



def CalculateOutlierColumns(df, perc):
    # Calculate quantiles for numeric columns only
    numeric_quantiles = df.select_dtypes(include=[np.number]).quantile(np.arange(perc, 1.01, 0.01))

    # Calculate percentage change in quantiles
    pct_change_99_1 = numeric_quantiles.pct_change().mul(100).iloc[-1]

    # Identify columns with percentage change > 100
    outlier_condition = pct_change_99_1 > 100
    return pct_change_99_1[outlier_condition].index.values



def HandleOutliers(data, perc=0.9, quant=0.99):
    # Capping outliers to 99th percentile values
    outlier_treatment_records = []

    # Identify columns with outliers
    columns_with_outliers = CalculateOutlierColumns(data, perc)
    print('Columns with outliers:\n', columns_with_outliers)

    for col in columns_with_outliers:
        outlier_threshold = data[col].quantile(quant)  # Calculate threshold
        condition = data[col] > outlier_threshold  # Identify outliers
        num_outliers = condition.sum()  # Count outliers

        # Replace outliers with the threshold
        data.loc[condition, col] = outlier_threshold

        # Record outlier treatment details
        outlier_treatment_records.append({
            'Column': col,
            'Outlier Threshold': outlier_threshold,
            'Outliers Replaced': num_outliers
        })

    # Convert records to a DataFrame
    outlier_treatment = pd.DataFrame(outlier_treatment_records)

    # Optionally visualize or return the treatment summary
    if 'PlotAsTable' in globals():
        PlotAsTable(outlier_treatment, f"Outliers handled (above {str(quant * 100)} percentile)")

    return outlier_treatment






#check for columns with high missing values and drop them
percent_missing = ReturnColumnsMissingPercentage(telecom)
high_missing_cols = GetHighMissingCols(percent_missing)
high_missing_df = telecom.loc[:,high_missing_cols]

def DropHighMissingColumns(df):
    df.drop(high_missing_df.columns, axis=1,  inplace=True)

DropHighMissingColumns(telecom)



#Plot the missing percentage
percent_missing = ReturnColumnsMissingPercentage(telecom)
percent_missing["Description"] = columnDescription["Description"]
PlotAsTable(percent_missing, "Missing Percentage(After Imputation & dropping >70% missing columns)")


# Checking information about data.
def metadata_display(df, sort_by="Null_Percentage", sortAsc = False) :
    return pd.DataFrame({
                'Columns' : df.columns,
                'Datatype' : df.dtypes.astype(str),
                'Non_Null_Count': df.count(axis = 0).astype(int),
                'Null_Count': df.isnull().sum().astype(int),
                'Null_Percentage': round(df.isnull().sum()/len(df) * 100 , 2),
                'Unique_Values_Count': df.nunique().astype(int)
                 }).sort_values(by=sort_by, ascending=sortAsc)

PlotAsTable(metadata_display(telecom, "Unique_Values_Count", True), "Telecom Metadata (Sorted by Unique Values)")


unique_val_cols =['circle_id', 'loc_og_t2o_mou', 'std_og_t2o_mou', 'loc_ic_t2o_mou',
                          'last_date_of_month_6', 'last_date_of_month_7', 'last_date_of_month_8',
                          'std_og_t2c_mou_6', 'std_og_t2c_mou_7',
                          'std_og_t2c_mou_8',  'std_ic_t2o_mou_6',
                          'std_ic_t2o_mou_7', 'std_ic_t2o_mou_8']

def DropUniqueValColumns(df):
    df.drop(unique_val_cols, axis=1,  inplace=True)

DropUniqueValColumns(telecom)

PlotAsTable(metadata_display(telecom, "Unique_Values_Count", True), "Telecom Metadata (Dropped All 1 Unique Value columns)")


#Converting columns into appropriate data types and extracting singe value columns.
# Columns with unique values < 29 are considered as categorical variables.
# The number 30 is arrived at, by looking at the above metadata output.

def IdentifyCategoricalColumns(data):
    cat_col_df = metadata_display(data, "Unique_Values_Count", True)
    cat_col_df = cat_col_df[cat_col_df["Unique_Values_Count"] < 30]
    PlotAsTable(cat_col_df, "Categorical Columns")
    return cat_col_df

cat_df = IdentifyCategoricalColumns(telecom)
change_to_cat = cat_df["Columns"]


sachet_columns = telecom.filter(regex='.*sachet.*', axis=1).columns.values
# Converting all the above columns having <=29 unique values into categorical data type.
def ConverToCategorical(data, cat = change_to_cat):
    data[cat]= data[cat].astype('category')
    data[sachet_columns] = data[sachet_columns].astype('category')

ConverToCategorical(telecom)


HandleOutliers(telecom, 0.9, 0.99)


ShowSummary(telecom)


# create box plot for  Jun, Jul and Aug months
def plot_month_wise_box_chart(attribute, df):
    plt.clf()
    plt.figure(figsize=(20,16))
    plt.subplot(2,3,1)
    sns.boxplot(data=df, y=attribute+"_6",x="churn_probability",hue="churn_probability",
                showfliers=False,palette=("plasma"))
    plt.subplot(2,3,2)
    sns.boxplot(data=df, y=attribute+"_7",x="churn_probability",hue="churn_probability",
                showfliers=False,palette=("plasma"))
    plt.subplot(2,3,3)
    sns.boxplot(data=df, y=attribute+"_8",x="churn_probability",hue="churn_probability",
                showfliers=False,palette=("plasma"))
    plt.show()


# Plot Recharge Amount related columns
recharge_amnt_columns =  telecom.columns[telecom.columns.str.contains('rech_amt')]
recharge_amnt_columns.tolist()


# Ploting for total recharge amount:
plot_month_wise_box_chart('total_rech_amt', telecom)


# Ploting for total recharge amount for data:
plot_month_wise_box_chart('total_rech_amt_data', telecom)


# Ploting for max recharge amount for data:
plot_month_wise_box_chart('max_rech_amt', telecom)


vbc_column = telecom.columns[telecom.columns.str.contains('vbc_3g',regex=True)]
vbc_column.tolist()


ax = sns.kdeplot(telecom.vbc_3g_6[(telecom["churn_probability"] == 0)],
                color="Red", shade = True)
ax = sns.kdeplot(telecom.vbc_3g_6[(telecom["churn_probability"] == 1)],
                ax =ax, color="Blue", shade= True)
ax.legend(["Not Churn","Churn"],loc='upper right')
ax.set_ylabel('Density')
ax.set_xlabel('Volume based cost')
ax.set_title('Distribution of Volume based cost by churn June')


ax = sns.kdeplot(telecom.vbc_3g_7[(telecom["churn_probability"] == 0)],
                color="Red", shade = True)
ax = sns.kdeplot(telecom.vbc_3g_7[(telecom["churn_probability"] == 1)],
                ax =ax, color="Blue", shade= True)
ax.legend(["Not Churn","Churn"],loc='upper right')
ax.set_ylabel('Density')
ax.set_xlabel('Volume based cost')
ax.set_title('Distribution of Volume based cost by churn July')


ax = sns.kdeplot(telecom.vbc_3g_8[(telecom["churn_probability"] == 0)],
                color="Red", shade = True)
ax = sns.kdeplot(telecom.vbc_3g_8[(telecom["churn_probability"] == 1)],
                ax =ax, color="Blue", shade= True)
ax.legend(["Not Churn","Churn"],loc='upper right')
ax.set_ylabel('Density')
ax.set_xlabel('Volume based cost')
ax.set_title('Distribution of Volume based cost by churn Aug')


# Checking columns for average revenue per user
arpu_cols = telecom.columns[telecom.columns.str.contains('arpu_')]

# Plotting arpu
plot_month_wise_box_chart('arpu', telecom)


mou_cols = telecom.columns[telecom.columns.str.contains('mou')]
mou_cols



PlotAsTable(metadata_display(telecom[mou_cols], "Null_Percentage", False), "MOU Null Percentage Sorted")




# replacing null values by 0 for minutes of usage variables
def ImputeMOUCols(df):
    df.loc[:,mou_cols] = df.loc[:,mou_cols].replace(np.NaN,0)

ImputeMOUCols(telecom)


percent_missing = ReturnColumnsMissingPercentage(telecom)

PlotAsTable(percent_missing, "Missing Percentage EDA")


telecom.info()


numeric_col = ['og_others_8', 'ic_others_8', 'og_others_6','ic_others_6', 'og_others_7', 'ic_others_7']


def ImputeNumericColumns(df):
    for i in numeric_col:
        df.loc[df[i].isnull(),i]=0

ImputeNumericColumns(telecom)


# Visalizing the correlation between the numerical variables
corr = telecom.corr()



# function to correlate variables
def correlation(cor0) :
    type(cor0)
    cor0.where(np.triu(np.ones(cor0.shape),k=1).astype(bool))  # Change np.bool to bool
    cor0=cor0.unstack().reset_index()
    cor0.columns=['VAR1','VAR2','CORR']
    cor0.dropna(subset=['CORR'], inplace=True)
    cor0.CORR=round(cor0['CORR'],2)
    cor0.CORR=cor0.CORR.abs()
    cor0.sort_values(by=['CORR'],ascending=False)
    cor0=cor0[~(cor0['VAR1']==cor0['VAR2'])]

    # removing duplicate correlations
    cor0['pair'] = cor0[['VAR1', 'VAR2']].apply(lambda x: '{}-{}'.format(*sorted((x[0], x[1]))), axis=1)

    cor0 = cor0.drop_duplicates(subset=['pair'], keep='first')
    cor0 = cor0[['VAR1', 'VAR2','CORR']]
    return pd.DataFrame(cor0.sort_values(by=['CORR'],ascending=False))



cor_0 = correlation(corr)

# filtering for correlations >= 40%
condition = cor_0['CORR'] > 0.4
cor_0 = cor_0[condition]
PlotAsTable(cor_0, "Correlation Matrix (>40%) Sorted")


# Derived variables to measure change in usage between August and previous months average (June, July)

# Usage
def CalculateDeltaCols(data):
    data['delta_vol_2g'] = data['vol_2g_mb_8'] - data['vol_2g_mb_6'].add(data['vol_2g_mb_7']).div(2)
    data['delta_vol_3g'] = data['vol_3g_mb_8'] - data['vol_3g_mb_6'].add(data['vol_3g_mb_7']).div(2)
    data['delta_total_og_mou'] = data['total_og_mou_8'] - data['total_og_mou_6'].add(data['total_og_mou_7']).div(2)
    data['delta_total_ic_mou'] = data['total_ic_mou_8'] - data['total_ic_mou_6'].add(data['total_ic_mou_7']).div(2)
    data['delta_vbc_3g'] = data['vbc_3g_8'] - data['vbc_3g_6'].add(data['vbc_3g_7']).div(2)

    # Revenue
    data['delta_arpu'] = data['arpu_8'] - data['arpu_6'].add(data['arpu_7']).div(2)
    data['delta_total_rech_amt'] = data['total_rech_amt_8'] - data['total_rech_amt_6'].add(data['total_rech_amt_7']).div(2)
    # Removing variables used for derivation :
    data.drop(columns=[
     'vol_2g_mb_8', 'vol_2g_mb_6', 'vol_2g_mb_7',
      'vol_3g_mb_8'  , 'vol_3g_mb_6', 'vol_3g_mb_7' ,
        'total_og_mou_8','total_og_mou_6', 'total_og_mou_7',
        'total_ic_mou_8','total_ic_mou_6', 'total_ic_mou_7',
        'vbc_3g_8','vbc_3g_6','vbc_3g_7',
        'arpu_8','arpu_6','arpu_7',
        'total_rech_amt_8', 'total_rech_amt_6', 'total_rech_amt_7'

    ], inplace=True)

CalculateDeltaCols(telecom)


# AON convert from days to months

def ConvertAONToMonths(df):
    df["aon_months"] = (df.aon / 30).astype(int, errors='ignore')
    df.drop(columns=['aon'], inplace=True)


ConvertAONToMonths(telecom)


telecom.head()


ShowSummary(telecom)


categorical = telecom.dtypes == 'category'
categorical_vars = telecom.columns[categorical].to_list()
ind_categorical_vars = list(set(categorical_vars) - {'churn_probability'}) #independent categorical variables
ind_categorical_vars


# Finding & Grouping categories with less than 1% contribution in each column into "Others"
def GroupAsOtherInCategorical(data):
    for col in ind_categorical_vars :
        category_counts = 100*data[col
                                  ].value_counts(normalize=True)
        PlotAsTable(pd.DataFrame(category_counts),f"{col} categorical_counts")
        low_count_categories = category_counts[category_counts <= 1].index.to_list()
        print(f"Replaced {low_count_categories} in {col} with category : Others")
        data[col].replace(low_count_categories,'Others',inplace=True)


GroupAsOtherInCategorical(telecom)
ShowSummary(telecom)


def CreateDummyVars(data):
    dummy_vars = pd.get_dummies(data[ind_categorical_vars], drop_first=False, prefix=ind_categorical_vars, prefix_sep='_')
    dummy_vars.head()
    reference_cols = dummy_vars.filter(regex='.*Others$').columns.to_list() # Using category 'Others' in each column as reference.
    dummy_vars.drop(columns=reference_cols, inplace=True)
    reference_cols
    data.drop(columns=ind_categorical_vars, inplace=True) # dropping original categorical columns
    data = pd.concat([data, dummy_vars], axis=1)
    data.head()
    dummy_cols = dummy_vars.columns.to_list()
    data[dummy_cols] = data[dummy_cols].astype('category')
    return data

def DropDateCols(data):
    data.drop(data.filter(regex='date_').columns,axis=1,inplace=True)

telecom = CreateDummyVars(telecom)
DropDateCols(telecom)
ShowSummary(telecom)


telecom.shape


telecom.dropna()
y = telecom["churn_probability"].astype(int, errors='ignore')
X = telecom.drop(["churn_probability","id"],axis=1)
X.index = telecom["id"]

X_train, X_test, y_train, y_test = train_test_split(X,y, train_size=0.7, random_state=42)




#Churn Distribution
pie_chart = y.value_counts()*100.0 /len(y)
ax = pie_chart.plot.pie(autopct='%.1f%%', labels = ['No', 'Yes'],figsize =(8,6), fontsize = 14 )
ax.set_ylabel('Churn',fontsize = 12)
ax.set_title('Churn Distribution', fontsize = 12)
plt.show()


# columns with numerical data
condition1 = X.dtypes == 'int'
condition2 = X.dtypes == 'float'
numerical_vars = X.columns[condition1 | condition2].to_list()




# Standard scaling
scaler = StandardScaler()

# Fit and transform train set
X_train[numerical_vars] = scaler.fit_transform(X_train[numerical_vars])

# Transform test set
X_test[numerical_vars] = scaler.transform(X_test[numerical_vars])

X_train.describe()


baseline_model = LogisticRegression(random_state=100, class_weight='balanced') # `weight of class` balancing technique used
baseline_model = baseline_model.fit(X_train, y_train)

baseline_coefficients = baseline_model.coef_

y_train_pred = baseline_model.predict_proba(X_train)[:,1]
y_test_pred  = baseline_model.predict_proba(X_test)[:,1]


y_train_pred = pd.Series(y_train_pred,index = X_train.index, ) # converting test and train to a series to preserve index
y_test_pred = pd.Series(y_test_pred,index = X_test.index)


# Prediction at threshold of 0.5
classification_threshold = 0.5

y_train_pred_classified = y_train_pred.map(lambda x : 1 if x > classification_threshold else 0)
y_test_pred_classified = y_test_pred.map(lambda x : 1 if x > classification_threshold else 0)



train_matrix = confusion_matrix(y_train, y_train_pred_classified)
print('Confusion Matrix for train:\n', train_matrix)
test_matrix = confusion_matrix(y_test, y_test_pred_classified)
print('\nConfusion Matrix for test: \n', test_matrix)


def DisplayClassificationReport(y, y_p):
    rep = classification_report(y, y_p, labels=[0, 1])
    print(rep)
    return rep



report_train = DisplayClassificationReport(y_train, y_train_pred_classified)
report_test = DisplayClassificationReport(y_test, y_test_pred_classified)




coeffient_df = pd.DataFrame({"columns" : list(X_train.columns) , "baseline_coefficients" : list(baseline_coefficients[0]) })

PlotAsTable(coeffient_df.sort_values(by='baseline_coefficients', ascending = False, key=abs), "Baseline model Coefficients")



def model_metrics(matrix, name) :
    TN = matrix[0][0]
    TP = matrix[1][1]
    FP = matrix[0][1]
    FN = matrix[1][0]
    print(f'{name} : \n' )
    accuracy = round((TP + TN)/float(TP+TN+FP+FN),3)
    print('Accuracy :' ,accuracy )
    sensitivity = round(TP/float(FN + TP),3)
    print('Sensitivity / True Positive Rate / Recall :', sensitivity)
    specificity = round(TN/float(TN + FP),3)
    print('Specificity / True Negative Rate : ', specificity)
    precision = round(TP/float(TP + FP),3)
    print('Precision / Positive Predictive Value :', precision)
    f1_score = round(2*precision*sensitivity/(precision + sensitivity),3)
    print('F1-score :', f1_score)
    return accuracy,sensitivity,specificity,precision,f1_score


# Baseline Model Performance :
model_perf_df = pd.DataFrame(columns=['model','accuracy','sensitivity/recall', 'specificity',  'precision', 'f1-score'])

def PlotMetricsComparission(t = "Metrics by Model (sorted by accuracy)"):
    PlotAsTable(model_perf_df.sort_values(by='accuracy', ascending = False), t)

def AppendToModelPerfDF(corr_matrix, name):
    metrics = model_metrics(corr_matrix, name)
     #Combining the "name" value with the NumPy array metrics
    new_row = np.concatenate([[name], metrics])
    # Inserting the new row into the DataFrame using the loc method
    new_row_index = len(model_perf_df)
    model_perf_df.loc[new_row_index] = new_row

AppendToModelPerfDF(train_matrix,"Logistic Regr Train (0.5 Thres)")
AppendToModelPerfDF(test_matrix,"Logistic Regr Test (0.5 Thres)")


# Specificity / Sensitivity Tradeoff

# Classification at probability thresholds between 0 and 1
def thresholder(x, thresh) :
    if x > thresh :
        return 1
    else :
        return 0

def CalculateProbablityThresholds(y_pred_now, X_now):
  y_train_pred_thres = pd.DataFrame(index=X_now.index)
  thresholds = [float(x)/10 for x in range(10)]
  for i in thresholds:
      y_train_pred_thres[i]= y_pred_now.map(lambda x : thresholder(x,i))
  print("Y pred by threshold: \n")
  print(y_train_pred_thres.head())
  return (thresholds, y_train_pred_thres)




def ViewMetricsByThreshold(y_act, y_pred_now, X_now):
    metrics_df = pd.DataFrame(columns=['precision','sensitivity/recall', 'specificity', 'accuracy'])
    (thresholds, y_train_pred_thres) = CalculateProbablityThresholds(y_pred_now, X_now)

    # generating a data frame for metrics for each threshold
    for thres,column in zip(thresholds,y_train_pred_thres.columns.to_list()):
        confusion = confusion_matrix(y_act, y_train_pred_thres.loc[:,column])
        sensitivity,specificity,accuracy,precision = model_metrics_thres(confusion)

        # Create a temporary row to append to the DataFrame
        temp_df = pd.DataFrame({
            'precision': [precision],
            'sensitivity/recall': [sensitivity],
            'specificity': [specificity],
            'accuracy': [accuracy]
        }, index=[thres])

        # Concatenate the new row with the existing DataFrame
        metrics_df = pd.concat([metrics_df, temp_df])

    print("Y pred metrics by threshold: \n")
    print(metrics_df)

    # Plot the metrics for each threshold
    metrics_df.plot(kind='line', figsize=(24,8), grid=True, xticks=np.arange(0,1,0.02),
                    title='Specificity-Sensitivity TradeOff')

# Now call the function with your actual data
ViewMetricsByThreshold(y_train, y_train_pred, X_train)



# Function for calculation of metrics for each threshold
def model_metrics_thres(matrix):
    TN = matrix[0][0]
    TP = matrix[1][1]
    FP = matrix[0][1]
    FN = matrix[1][0]
    accuracy = round((TP + TN) / float(TP + TN + FP + FN), 3)
    sensitivity = round(TP / float(FN + TP), 3)
    specificity = round(TN / float(TN + FP), 3)
    precision = round(TP / float(TP + FP), 3)
    return sensitivity, specificity, accuracy, precision

def ViewMetricsByThreshold(y_act, y_pred_now, X_now):
    metrics_df = pd.DataFrame(columns=['precision', 'sensitivity/recall', 'specificity', 'accuracy'])
    (thresholds, y_train_pred_thres) = CalculateProbablityThresholds(y_pred_now, X_now)

    # generating a data frame for metrics for each threshold
    for thres, column in zip(thresholds, y_train_pred_thres.columns.to_list()):
        confusion = confusion_matrix(y_act, y_train_pred_thres.loc[:, column])
        sensitivity, specificity, accuracy, precision = model_metrics_thres(confusion)

        # Create a temporary DataFrame for the row
        temp_df = pd.DataFrame({
            'precision': [precision],
            'sensitivity/recall': [sensitivity],
            'specificity': [specificity],
            'accuracy': [accuracy]
        }, index=[thres])

        # Concatenate the new row with the existing DataFrame
        metrics_df = pd.concat([metrics_df, temp_df])

    metrics_df.index = thresholds
    print("Y pred metrics by threshold: \n")
    print(metrics_df)

    # Plot the metrics for each threshold
    metrics_df.plot(kind='line', figsize=(24, 8), grid=True, xticks=np.arange(0, 1, 0.02),
                    title='Specificity-Sensitivity TradeOff')

# Now call the function with your actual data
ViewMetricsByThreshold(y_train, y_train_pred, X_train)



optimum_cutoff = 0.53
y_train_pred_final = y_train_pred.map(lambda x : 1 if x > optimum_cutoff else 0)
y_test_pred_final = y_test_pred.map(lambda x : 1 if x > optimum_cutoff else 0)

train_matrix = confusion_matrix(y_train, y_train_pred_final)
print('Confusion Matrix for train:\n', train_matrix)
test_matrix = confusion_matrix(y_test, y_test_pred_final)
print('\nConfusion Matrix for test: \n', test_matrix)


AppendToModelPerfDF(train_matrix,f"Logistic Regr Train (Optimal:{str(optimum_cutoff)} Thres)")
AppendToModelPerfDF(test_matrix,f"Logistic Regr Test (Optimal:{str(optimum_cutoff)} Thres)")


report_train = DisplayClassificationReport(y_train, y_train_pred_classified)
report_test = DisplayClassificationReport(y_test, y_test_pred_classified)



PlotMetricsComparission()


lr = LogisticRegression(random_state=100 , class_weight='balanced')
rfe = RFE(estimator=lr, n_features_to_select=15)
results = rfe.fit(X_train,y_train)
results.support_


# DataFrame with features supported by RFE
rfe_support = pd.DataFrame({'Column' : X.columns.to_list(), 'Rank' : rfe.ranking_,
                                      'Support' :  rfe.support_}).sort_values(by=
                                       'Rank', ascending=True)
PlotAsTable(rfe_support, "RFE Feature Selection")


# RFE Selected columns
rfe_selected_columns = rfe_support.loc[rfe_support['Rank'] == 1,'Column'].to_list()
rfe_selected_columns


y_train = y_train.reset_index(drop=True)
X_train_rfe = X_train[rfe_selected_columns].reset_index(drop=True)


print(y_train.head())
print(X_train_rfe.head())


logr = sm.GLM(y_train, (sm.add_constant(X_train_rfe)), family=sm.families.Binomial())
logr_fit = logr.fit()
logr_fit.summary()


X_train_copy = X_train.copy(deep=True)


from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

   # Reset index again for the y_train, just to be certain
y_train = y_train.reset_index(drop=True)
   #Reset index for X_train_copy too
X_train_copy = X_train_copy.reset_index(drop=True)

   #RFE on X_train_copy to find the relevant features
lr = LogisticRegression(random_state=100, class_weight='balanced')
rfe = RFE(estimator=lr, n_features_to_select=15)
results = rfe.fit(X_train_copy, y_train)

   #Features selected by RFE
rfe_selected_columns = X_train_copy.columns[results.support_]


y_train = y_train.reset_index(drop=True)
X_train_rfe = X_train_rfe.reset_index(drop=True)


logr = sm.GLM(y_train, X_train_rfe, family=sm.families.Binomial())
logr_fit = logr.fit()
logr_fit.summary()



def vif(X_train_resampled, logr_fit, selected_columns) :
    vif = pd.DataFrame()
    vif['Features'] = rfe_selected_columns
    vif['VIF'] = [variance_inflation_factor(X_train_resampled[selected_columns].values, i) for i in range(X_train_resampled[selected_columns].shape[1])]
    vif['VIF'] = round(vif['VIF'], 2)
    vif = vif.set_index('Features')
    vif['P-value'] = round(logr_fit.pvalues,4)
    vif = vif.sort_values(by = ["VIF",'P-value'], ascending = [False,False])
    return vif


# vif and p-values
vif(X_train, logr_fit,  rfe_selected_columns)


selected_columns = rfe_selected_columns.tolist()


selected_columns.remove('loc_ic_mou_8')


selected_columns = rfe_selected_columns.tolist()  # Convert to list
selected_columns.remove('loc_ic_mou_8')  # Remove element
print(selected_columns)  # Print modified list


print(X_train_aligned.isnull().sum())  # Check for NaNs
print(np.isinf(X_train_aligned).sum())


X_train_aligned.replace([np.inf, -np.inf], np.nan, inplace=True)  # Replace inf with NaN
X_train_aligned.fillna(0, inplace=True)


logr2 = sm.GLM(y_train, sm.add_constant(X_train_aligned), family=sm.families.Binomial())
logr2_fit = logr2.fit()
logr2_fit.summary()


# vif and p-values
vif(X_train,logr2_fit,selected_columns)


if 'std_og_mou_8' in selected_columns:
    selected_columns.remove('std_og_mou_8')
else:
    print("std_og_mou_8 not found in selected_columns")


X_train_aligned.replace([np.inf, -np.inf], np.nan, inplace=True)  # Replace inf with NaN


X_train_aligned.fillna(0, inplace=True)  # Replace NaN with 0


logr3 = sm.GLM(y_train, sm.add_constant(X_train_aligned), family=sm.families.Binomial())
logr3_fit = logr3.fit()
logr3_fit.summary()


# vif and p-values
vif(X_train,logr3_fit,selected_columns)


y_train = y_train.reset_index(drop=True)
X_train = X_train.reset_index(drop=True)


X_train_aligned = X_train[selected_columns].reindex(y_train.index)


X_train_aligned.replace([np.inf, -np.inf], np.nan, inplace=True)


X_train_aligned.fillna(0, inplace=True)


logr5 = sm.GLM(y_train, sm.add_constant(X_train_aligned), family=sm.families.Binomial())
logr5_fit = logr5.fit()
logr5_fit.summary()


# vif and p-values
vif(X_train,logr5_fit,selected_columns)


logr5_fit.summary()



# Prediction
y_train_pred_lr = logr5_fit.predict(sm.add_constant(X_train[selected_columns]))
y_train_pred_lr.head()


print(X_test.info())


for col in X_test.select_dtypes(exclude=['number']).columns:
    X_test[col] = pd.to_numeric(X_test[col], errors='coerce')
    X_test[col] = X_test[col].fillna(0)


print(X_test.info())


# Convert boolean columns to integers
for col in X_test.select_dtypes(include=['bool']).columns:
    X_test[col] = X_test[col].astype(int)


# Convert boolean columns to integers
for col in X_test.select_dtypes(include=['bool']).columns:
    X_test[col] = X_test[col].astype(int)

# Assuming selected_columns is a list of column names
# Filter X_test to include only selected columns
X_test_selected = X_test[selected_columns]

# Prediction on test set
y_test_pred_lr = logr5_fit.predict(sm.add_constant(X_test_selected))

y_test_pred_lr.head()




# View metricsthreshold
ViewMetricsByThreshold(y_train, y_train_pred_lr, X_train)





optimum_cutoff = 0.19
y_train_pred_lr_final = y_train_pred_lr.map(lambda x : 1 if x > optimum_cutoff else 0)
y_test_pred_lr_final = y_test_pred_lr.map(lambda x : 1 if x > optimum_cutoff else 0)

train_matrix = confusion_matrix(y_train, y_train_pred_lr_final)
print('Confusion Matrix for train:\n', train_matrix)
test_matrix = confusion_matrix(y_test, y_test_pred_lr_final)
print('\nConfusion Matrix for test: \n', test_matrix)


AppendToModelPerfDF(train_matrix,f"Logistic Regr RFE Train (Optimal:{str(optimum_cutoff)} Thres)")
AppendToModelPerfDF(test_matrix,f"Logistic Regr RFE Test (Optimal:{str(optimum_cutoff)} Thres)")


PlotMetricsComparission()


type(logr5_fit.summary().tables[1])
val = logr5_fit.summary().tables[1]
valdf = pd.DataFrame(val.data)
valdf.columns = valdf.iloc[0]
valdf.rename(columns= {'' : "columns", "coef": "rfe_coef" }, inplace = True)
valdf = valdf[2:]

valdf["rfe_coef"] = pd.to_numeric(valdf["rfe_coef"])
valdf.head(len(valdf))





coeffient_df = pd.merge(coeffient_df, valdf, how="left", on=["columns"])
coeffient_df.drop(columns=coeffient_df.columns[-5:], axis=1,  inplace=True)
PlotAsTable(coeffient_df.sort_values(by='rfe_coef', ascending = False, key=abs), "Important predictors of Churn: BaseLine & RFE Coefficients (Sorted by RFE Coeff)")


pca = PCA(random_state = 42)
pca.fit(X_train)
pca.components_


pca.explained_variance_ratio_



def Plot_ScreePlot(evr):
  var_cumu = np.cumsum(evr)
  fig = plt.figure(figsize=[12,8])
  plt.vlines(x=5, ymax=1, ymin=0, colors="r", linestyles="--")
  plt.hlines(y=0.1, xmax=140, xmin=0, colors="g", linestyles="--")
  plt.vlines(x=4, ymax=1, ymin=0, colors="r", linestyles="--")
  plt.hlines(y=0.95, xmax=140, xmin=0, colors="g", linestyles="--")
  plt.plot(var_cumu)
  plt.ylabel("Cumulative variance explained")
  plt.show()


Plot_ScreePlot(pca.explained_variance_ratio_)


# Perform PCA using the first 45 components
comp_count = 31
pca_final = PCA(n_components=comp_count, random_state=42)
transformed_pca_df = pca_final.fit_transform(X_train)
print(transformed_pca_df.shape)
corrmat = np.corrcoef(transformed_pca_df.transpose())
plt.figure(figsize=[45,20])
sns.heatmap(corrmat, annot=True)


X_train_pca = pd.DataFrame(transformed_pca_df, columns=["PC_"+str(x) for x in range(1,comp_count + 1)], index = X_train.index)
telecom_train_pca = pd.concat([X_train_pca, y_train], axis=1)

telecom_train_pca.head()
## Plotting principal components
sns.pairplot(data=telecom_train_pca, x_vars=["PC_1"], y_vars=["PC_2"], hue = "churn_probability", size=8);


def ApplyPCATransformation(df):
  return pca_final.transform(df)


# Transforming test set with pca ( 45 components)
X_test_pca = ApplyPCATransformation(X_test)


# Logistic Regression
lr_pca = LogisticRegression(random_state=100, class_weight='balanced')
lr_pca.fit(X_train_pca, y_train)


# y_train predictions
y_train_pred_lr_pca = lr_pca.predict(X_train_pca)
y_train_pred_lr_pca[:5]


# Test Prediction
y_test_pred_lr_pca = lr_pca.predict(X_test_pca)
y_test_pred_lr_pca[:5]


train_matrix = confusion_matrix(y_train, y_train_pred_lr_pca)
print('Confusion Matrix for train:\n', train_matrix)
test_matrix = confusion_matrix(y_test, y_test_pred_lr_pca)
print('\nConfusion Matrix for test: \n', test_matrix)


AppendToModelPerfDF(train_matrix,f"Logistic Regr PCA Train Baseline)")
AppendToModelPerfDF(test_matrix,f"Logistic Regr PCA Test Baseline")


param_grid = {
    'C': [0,1,2,3,4,5,10,50,100],  # Regularization strength
    'solver': ['lbfgs', 'liblinear', 'sag', 'saga'],  # Solver options
    'penalty' : ['l1','l2','none']
}

# Create the GridSearchCV object
grid_search = GridSearchCV(lr_pca, param_grid, scoring='accuracy', cv=20, verbose=True, n_jobs=-1)

# Fit the GridSearchCV to your data
grid_search.fit(X_train_pca, y_train)

# Get the best hyperparameters and best accuracy
best_params = grid_search.best_params_
best_accuracy = grid_search.best_score_

print("Best Hyperparameters:", best_params)
print("Best Accuracy:", best_accuracy)





y_train_pred_lr_pca_grid = grid_search.best_estimator_.predict(X_train_pca)
y_test_pred_lr_pca_grid = grid_search.best_estimator_.predict(X_test_pca)

train_matrix = confusion_matrix(y_train, y_train_pred_lr_pca_grid)
print('Confusion Matrix for train:\n', train_matrix)
test_matrix = confusion_matrix(y_test, y_test_pred_lr_pca_grid)
print('\nConfusion Matrix for test: \n', test_matrix)


AppendToModelPerfDF(train_matrix,f"Logistic Regr PCA GRIDSEARCH Train Baseline")
AppendToModelPerfDF(test_matrix,f"Logistic Regr PCA GRIDSEARCH Test Baseline")


# Ratio of classes
class_0 = y[y == 0].count()
class_1 = y[y == 1].count()

pca_rf = RandomForestClassifier(random_state=42, class_weight= {0 : class_1/(class_0 + class_1) , 1 : class_0/(class_0 + class_1) } , oob_score=True, n_jobs=-1,verbose=1)
pca_rf


# Hyper parameter Tuning
params = {
    'n_estimators'  : [30,40,50,100],
    'max_depth' : [3,4,5,6,7],
    'min_samples_leaf' : [15,20,25,30]
}
folds = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
pca_rf_model_search = GridSearchCV(estimator=pca_rf, param_grid=params,
                                   cv=folds, scoring='roc_auc', verbose=True, n_jobs=-1 )

pca_rf_model_search.fit(X_train_pca, y_train)


# Optimum Hyperparametersprint('Best ROC-AUC score :', pca_rf_model_search.best_score_)
print('Best Parameters :', pca_rf_model_search.best_params_)


# Modelling using the best PCA-RandomForest Estimator
pca_rf_best = pca_rf_model_search.best_estimator_
pca_rf_best_fit = pca_rf_best.fit(X_train_pca, y_train)

# Prediction on Train set
y_train_pred_pca_rf_best = pca_rf_best_fit.predict(X_train_pca)


# Prediction on test set
y_test_pred_pca_rf_best = pca_rf_best_fit.predict(X_test_pca)


## PCA - RandomForest Model Performance - Hyper Parameter Tuned

train_matrix = confusion_matrix(y_train, y_train_pred_pca_rf_best)
test_matrix = confusion_matrix(y_test, y_test_pred_pca_rf_best)


AppendToModelPerfDF(train_matrix,f"Randomforest PCA GRIDSEARCH Train Baseline")
AppendToModelPerfDF(test_matrix,f"Randomforest PCA GRIDSEARCH Test Baseline")


PlotMetricsComparission()


pca_xgb = xgb.XGBClassifier(random_state=42, scale_pos_weight= class_0/class_1 ,
                                    tree_method='hist',
                                   objective='binary:logistic',


                                  )  # scale_pos_weight takes care of class imbalance
pca_xgb.fit(X_train_pca, y_train)


print('Baseline Train AUC Score')
print(roc_auc_score(y_train, pca_xgb.predict_proba(X_train_pca)[:, 1]))

print('Baseline Test AUC Score')
print(roc_auc_score(y_test, pca_xgb.predict_proba(X_test_pca)[:, 1]))



# A parameter grid for XGBoost
params = {
        'n_estimators' : [400, 500, 600], # no of trees
        'learning_rate' : [0.04, 0.05, 0.07, 0.06],  # et
        'gamma': [0.05, 0.1, 0.2],
        'max_depth': [ 11, 12, 14, 15]
        }

## Hyper parameter Tuning
parameters = {
              'learning_rate': [0.1, 0.2, 0.3],
              'gamma' : [10,20,50],
              'max_depth': [2,3,4]}
folds = 3

param_comb = 100

#pca_xgb_search = RandomizedSearchCV(pca_xgb, param_distributions=params, n_iter=param_comb, scoring='accuracy', n_jobs=-1, cv=3, verbose=3, random_state=42)
pca_xgb_search = GridSearchCV(estimator=pca_xgb , param_grid=params,scoring='roc_auc', cv=3, n_jobs=-1, verbose=1)
pca_xgb_search.fit(X_train_pca, y_train)


# Optimum Hyperparameters
print('Best ROC-AUC score :', pca_xgb_search.best_score_)
print('Best Parameters :', pca_xgb_search.best_params_)





# Modelling using the best PCA-XGBoost Estimator
pca_xgb_best_1 = pca_xgb_search.best_estimator_
pca_xgb_best_fit = pca_xgb_best_1.fit(X_train_pca, y_train)

# Prediction on Train set
y_train_pred_pca_xgb_best = pca_xgb_best_fit.predict(X_train_pca)
y_test_pred_pca_xgb_best = pca_xgb_best_fit.predict(X_test_pca)


## PCA - XGBOOST [Hyper parameter tuned] Model Performance

train_matrix = confusion_matrix(y_train, y_train_pred_pca_xgb_best)
test_matrix = confusion_matrix(y_test, y_test_pred_pca_xgb_best)


AppendToModelPerfDF(train_matrix,f"XGBoost PCA RandomSEARCH Train ")
AppendToModelPerfDF(test_matrix,f"XGBoost PCA RandomSEARCH Test ")


PlotMetricsComparission()


print(pca_xgb_best_fit.scale_pos_weight)


print(pca_xgb.scale_pos_weight)


# reading the dataset
telecom_test = pd.read_csv("/kaggle/input/telecom-churn-case-study-hackathon-cc50/test.csv")


ShowSummary(telecom_test)


ConvertDateTimeColumns(telecom_test)
ApplyMeaningfulName(telecom_test)
ShowSummary(telecom_test)

ReplaceRechargeColsForNoRecharge(telecom_test)
ShowSummary(telecom_test)

CreateTotalRechAmtDataCols(telecom_test)
telecom_test.describe()


DropHighMissingColumns(telecom_test)

DropUniqueValColumns(telecom_test)

ConverToCategorical(telecom_test,change_to_cat.drop("churn_probability"))

HandleOutliers(telecom, 0.9, 0.99)

ShowSummary(telecom_test)

ImputeMOUCols(telecom_test)

ImputeNumericColumns(telecom_test)

CalculateDeltaCols(telecom_test)
ConvertAONToMonths(telecom_test)

GroupAsOtherInCategorical(telecom_test)
ShowSummary(telecom_test)

telecom_test = CreateDummyVars(telecom_test)
DropDateCols(telecom_test)
ShowSummary(telecom_test)

telecom_test.dropna()


X_K_test = telecom_test.drop(["id", "sachet_2g_6_4"],axis=1)
X_K_test.index = telecom_test["id"]

# Transform test set
X_K_test[numerical_vars] = scaler.transform(X_K_test[numerical_vars])

# Transforming test set with pca
X_K_test_pca = ApplyPCATransformation(X_K_test)


# From train test split
print("Train Test Split: ")
ShowSummary(X_test)

print("\n \ n Kaggle dataset: ")
# Kaggle dataset
ShowSummary(X_K_test)


print("Mismatched columns if any:")
X_K_test.columns.difference(X_test.columns).tolist()


def ConvertToDFAndSave(ser, name_file):
    df = ser.to_frame().reset_index()
    ShowSummary(df)
    t = datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")
    df.to_csv(f"/kaggle/working/Godwin-Paul-Vincent_{name_file}_{t}.csv", index = False)


y_train_pred_kaggle = baseline_model.predict_proba(X_K_test)[:,1]

y_train_pred_kaggle = pd.Series(y_train_pred_kaggle,index = X_K_test.index, name='churn_probability') # converting test and train to a series to preserve index

optimum_cutoff = 0.55
y_train_pred_kaggle_final = y_train_pred_kaggle.map(lambda x : 1 if x > optimum_cutoff else 0)
y_train_pred_kaggle_final = pd.Series(y_train_pred_kaggle_final,index = X_K_test.index, name='churn_probability') # converting test and train to a series to preserve index

ConvertToDFAndSave(y_train_pred_kaggle_final, "baseline_log_regr")


y_train_pred_kaggle_final


# Prediction on RandomForest test set
y_k_test_pred_pca_rf_best = pca_rf_best_fit.predict(X_K_test_pca)
y_k_test_pred_pca_rf_best = pd.Series(y_k_test_pred_pca_rf_best,index = X_K_test.index, name='churn_probability')

ConvertToDFAndSave(y_k_test_pred_pca_rf_best, "rf_pca")


y_k_test_pred_pca_xgb_best = pca_xgb_best_fit.predict(X_K_test_pca)
y_k_test_pred_pca_xgb_best = pd.Series(y_k_test_pred_pca_xgb_best,index = X_K_test.index, name='churn_probability')
ConvertToDFAndSave(y_k_test_pred_pca_xgb_best, "xgb_pca")



class_0 = y[y == 0].count()
class_1 = y[y == 1].count()

pca_xgb_org = xgb.XGBClassifier(random_state=42, scale_pos_weight= class_0/class_1 ,
                                    tree_method='hist',
                                   objective='binary:logistic',enable_categorical = True


                                  )  # scale_pos_weight takes care of class imbalance
pca_xgb_org.fit(X_train, y_train)

print('Baseline Train AUC Score')
print(roc_auc_score(y_train, pca_xgb_org.predict_proba(X_train)[:, 1]))

print('Baseline Test AUC Score')
print(roc_auc_score(y_test, pca_xgb_org.predict_proba(X_test)[:, 1]))


# Prediction on RandomForest test set
y_k_test_pred_pca_rf_best = pca_rf_best_fit.predict(X_K_test_pca)
y_k_test_pred_xgb_org_best = pca_xgb_org.predict(X_K_test)
y_k_test_pred_xgb_org_best = pd.Series(y_k_test_pred_xgb_org_best,index = X_K_test.index, name='churn_probability')

ConvertToDFAndSave(y_k_test_pred_pca_rf_best, "xgb_org_data")


# A parameter grid for XGBoost
params = {
        'n_estimators' : [400, 500, 600], # no of trees
        'learning_rate' : [0.04, 0.05, 0.07, 0.06],  # et
        'gamma': [0.05, 0.1, 0.2],
        'max_depth': [ 11, 12, 14, 15]
        }

## Hyper parameter Tuning
parameters = {
              'learning_rate': [0.1, 0.2, 0.3],
              'gamma' : [10,20,50],
              'max_depth': [2,3,4]}
folds = 3

param_comb = 100

#pca_xgb_search = RandomizedSearchCV(pca_xgb, param_distributions=params, n_iter=param_comb, scoring='accuracy', n_jobs=-1, cv=3, verbose=3, random_state=42)
pca_xgb_search_org = GridSearchCV(estimator=pca_xgb_org , param_grid=params,scoring='roc_auc', cv=3, n_jobs=-1, verbose=1)
pca_xgb_search_org.fit(X_train, y_train)


pca_xgb_best_1 = pca_xgb_search_org.best_estimator_
pca_xgb_best_fit = pca_xgb_best_1.fit(X_train, y_train)

# Prediction on Train set
y_train_pred_xgb_org = pca_xgb_best_fit.predict(X_train)
y_test_pred_xgb_org = pca_xgb_best_fit.predict(X_test)


train_matrix = confusion_matrix(y_train, y_train_pred_xgb_org)
test_matrix = confusion_matrix(y_test, y_test_pred_xgb_org)

AppendToModelPerfDF(train_matrix,f"XGBoost ORG Grid data Train ")
AppendToModelPerfDF(test_matrix,f"XGBoost ORG Grid data Test ")


#prediction
y_k_test_pred_xgb_org_best = pca_xgb_best_fit.predict(X_K_test)
y_k_test_pred_xgb_org_best = pd.Series(y_k_test_pred_xgb_org_best,index = X_K_test.index, name='churn_probability')

ConvertToDFAndSave(y_k_test_pred_xgb_org_best, "xgb_org_data_grid_search")


testterms = ['Test', 'test']
model_performance_testdata = model_perf_df[model_perf_df['model'].str.contains('|'.join(testterms))]
model_performance_testdata.sort_values(by='accuracy', ascending = False).head(len(model_performance_testdata))


## Important predictors of Churn: Logistic Regression BaseLine & RFE Coefficients (Sorted by RFE Coeff)

coeffient_df.sort_values(by='rfe_coef', ascending = False, key=abs).head(25)




