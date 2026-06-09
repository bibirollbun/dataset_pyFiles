#loading all the necessary packages
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.feature_selection import VarianceThreshold

#reading in a dataframe (df)
#train_solutions = pd.read_csv('training.csv')
#train_categorical = pd.read_csv('training.csv')
#train_quantitative = pd.read_csv('training.csv')

#train = train.reset_index(drop=True)
#train.head(10)


%%time
def get_feats(mode='train'):
   
    feats=pd.read_excel(f"/kaggle/input/traindata/{mode}_QUANTITATIVE_METADATA.xlsx")
    
    if mode=='TRAIN':
        cate=pd.read_excel(f"/kaggle/input/traindata/{mode}_CATEGORICAL_METADATA.xlsx")
    else:
        cate=pd.read_excel(f"/kaggle/input/traindata/{mode}_CATEGORICAL.xlsx")
    feats=feats.merge(cate,on='participant_id',how='left')
    
    func=pd.read_csv(f"/kaggle/input/traindata/{mode}_FUNCTIONAL_CONNECTOME_MATRICES.csv")
    feats=feats.merge(func,on='participant_id',how='left')

    if mode=='TRAIN':
        solution=pd.read_excel("/kaggle/input/traindata/TRAINING_SOLUTIONS.xlsx")
        feats=feats.merge(solution,on='participant_id',how='left')
        
    return feats
    
train=get_feats(mode='TRAIN')
#test=get_feats(mode='TEST')
#sub = pd.read_excel('/kaggle/input/widsdatathon2025/SAMPLE_SUBMISSION.xlsx')
#y = pd.read_excel(f"/kaggle/input/widsdatathon2025/TRAIN/TRAINING_SOLUTIONS.xlsx")_QUANTITATIVE_METADATA.xlsx")
 


%%time
def get_feats(mode='test'):
   
    feats=pd.read_excel(f"/kaggle/input/testdata/{mode}_QUANTITATIVE_METADATA.xlsx")
    
    if mode=='TRAIN':
        cate=pd.read_excel(f"/kaggle/input/testdata/{mode}_CATEGORICAL_METADATA.xlsx")
    else:
        cate=pd.read_excel(f"/kaggle/input/testdata/{mode}_CATEGORICAL.xlsx")
    feats=feats.merge(cate,on='participant_id',how='left')
    
    func=pd.read_csv(f"/kaggle/input/testdata/{mode}_FUNCTIONAL_CONNECTOME_MATRICES.csv")
    feats=feats.merge(func,on='participant_id',how='left')

    if mode=='TRAIN':
        solution=pd.read_excel("/kaggle/input/testdata/TRAINING_SOLUTIONS.xlsx")
        feats=feats.merge(solution,on='participant_id',how='left')
        
    return feats
    

test=get_feats(mode='TEST')
#sub = pd.read_excel('/kaggle/input/widsdatathon2025/SAMPLE_SUBMISSION.xlsx')
#y = pd.read_excel(f"/kaggle/input/widsdatathon2025/TRAIN/TRAINING_SOLUTIONS.xlsx")_QUANTITATIVE_METADATA.xlsx")
 


train.head()


# patient at index 3
print(train.loc[3])


# columns participant enroll year and MRI tracking scan location
print(train.loc[:, ["Basic_Demos_Enroll_Year", "MRI_Track_Scan_Location"]])


# rows 5:8, columns 10:15
print(train.iloc[5:8, 10:15])


#rows 6, 1002 and columns 8 and 42
print(train.iloc[[6, 1002], [8, 42]])


train.info(verbose=False)


train.columns


#race is Black/African American
train[train["PreInt_Demos_Fam_Child_Race"] == 2] #"Black/African American"


# patient is male 
train[train["Sex_F"] == 0]


# patient's parent 1 education level is less than 7th grade and color vision score is not null (NaN)
train[(train["Barratt_Barratt_P1_Edu"] == 3) & (train['ColorVision_CV_Score'].notna())]


# sort patients youngest to oldest
train.sort_values(by='MRI_Track_Age_at_Scan')


# sort patients oldest to youngest
train.sort_values(by='MRI_Track_Age_at_Scan', ascending=False)


#notice if you just create the object, you cannot view it. You must choose a method to show the groupby function
race_gb = train.groupby('PreInt_Demos_Fam_Child_Race')['participant_id']
race_gb


type(race_gb)


# counts of patients of different races
race_gb.count()


#we can take this further to see information, like mean age by race - this is super helpful for gathering data 
#for visualizations and gleaning further understanding into a particular subset of data. This also shows one way we can 
# leverage grouping functions for numeric data

train.groupby('PreInt_Demos_Fam_Child_Race')['MRI_Track_Age_at_Scan'].mean()


train['race'] = np.where(train['PreInt_Demos_Fam_Child_Race'] == 0, 'White/Caucasin',
                np.where(train['PreInt_Demos_Fam_Child_Race'] == 1, 'Black/African American',
                np.where(train['PreInt_Demos_Fam_Child_Race'] == 2, 'Hispanic',
                np.where(train['PreInt_Demos_Fam_Child_Race'] == 3, 'Asian',
                np.where(train['PreInt_Demos_Fam_Child_Race'] == 4, 'Indian',
                np.where(train['PreInt_Demos_Fam_Child_Race'] == 5, 'Native American Indian',
                np.where(train['PreInt_Demos_Fam_Child_Race'] == 6, 'American Indian/Alaskan Native',
                np.where(train['PreInt_Demos_Fam_Child_Race'] == 7, 'Native Hawaiian/Other Pacific Islander',
                np.where(train['PreInt_Demos_Fam_Child_Race'] == 8, 'Two or more races',
                np.where(train['PreInt_Demos_Fam_Child_Race'] == 9, 'Other race',
                np.where(train['PreInt_Demos_Fam_Child_Race'] == 10, 'Unknown',
                np.where(train['PreInt_Demos_Fam_Child_Race'] == 11, 'Choose not to specify',
                'unknown'))))))))))))


train['ethnicity'] = np.where(train['PreInt_Demos_Fam_Child_Ethnicity'] == 0, 'Not Hispanic or Latino',
                np.where(train['PreInt_Demos_Fam_Child_Ethnicity'] == 1, 'Hispanic or Latino',
                np.where(train['PreInt_Demos_Fam_Child_Ethnicity'] == 2, 'Decline to specify',
                np.where(train['PreInt_Demos_Fam_Child_Ethnicity'] == 3, 'Unknown',
                'Unknown'))))


ethnicity_race_gb = train.groupby([ 'race', 'ethnicity'])['participant_id'].count()
ethnicity_race_gb


print(len(train))
train = train.drop_duplicates()
print(len(train))


print(train.isna().sum())


train['payer_type'] = train['payer_type'].fillna('UNSPECIFIED')
test['payer_type'] = test['payer_type'].fillna('UNSPECIFIED')
train['payer_type']


median_bmi = train['bmi'].median()
print(f"Median BMI is {median_bmi}")

mean_bmi = train['bmi'].mean()
print(f"Mean BMI is {mean_bmi}")

train['bmi'] = train['bmi'].fillna(median_bmi)
test['bmi'] = test['bmi'].fillna(median_bmi)



median_bmi = train['bmi'].median()
print(f"Median BMI is {median_bmi}")

mean_bmi = train['bmi'].mean()
print(f"Mean BMI is {mean_bmi}")


cols_to_drop_from = []
for col, v in train.isna().sum().items():
    #if more than 10 NaNs
    if 0 <v < 10:
        print(f"{col}, {v}")
        cols_to_drop_from.append(col)


#drop from all those columns
train =train.dropna(subset=cols_to_drop_from)

#double check no columns left with only a few NaNs
for col, v in train.isna().sum().items():
    #if more than 10 NaNs
    if 0 <v:
        print(f"{col}, {v}")


#import the commonly used visualization packages
import matplotlib.pyplot as plt
import seaborn as sns


train['scan_location'] = np.where(train['MRI_Track_Scan_Location'] == 1, 'Staten Island',
                np.where(train['MRI_Track_Scan_Location'] == 2, 'RUBIC',
                np.where(train['MRI_Track_Scan_Location'] == 3, 'CBIC',
                np.where(train['MRI_Track_Scan_Location'] == 4, 'CUNY',
                'Unknown'))))


#using seaborn

#Q: What is the geographic makeup of the patients in the dataset by region
sns.countplot(x="scan_location", data=train)
plt.show()


train.columns


train['parent2_education'] = np.where(train['Barratt_Barratt_P2_Edu'] == 3, 'Less Than 7th Grade',
                np.where(train['Barratt_Barratt_P2_Edu'] == 6, 'Junior high/middle School (9th grade)',
                np.where(train['Barratt_Barratt_P2_Edu'] == 9, 'Partial high school (10th or 11th grade)',
                np.where(train['Barratt_Barratt_P2_Edu'] == 12, 'High school graduate',
                np.where(train['Barratt_Barratt_P2_Edu'] == 15, 'Partial college (at least a year)',
                np.where(train['Barratt_Barratt_P2_Edu'] == 18, 'College education',
                np.where(train['Barratt_Barratt_P2_Edu'] == 21, 'Graduate degree',
                'Unknown')))))))


# pie chart using matplotlib
#Q: What is the percent of parents with an education higher than high school?

#1. get counts of the 0s and 1s
counts = train['parent2_education'].value_counts()

#2. Write function to format how to get counts (its okay to steal from stackoverflow -- just cite your code :))
#https://stackoverflow.com/a/71515035/2901002
def fmt(values):
        def my_format(pct):
            total = sum(values)
            val = int(round(pct*total/100.0))
            return '{:.1f}%\n({v:d})'.format(pct, v=val)
        return my_format

#3. make pie chart
plt.pie(counts, labels=counts.index, autopct=fmt(counts))
plt.show()


#scatter plot of two numeric variables 

#Q: What is the distribution of age at scan and the positive parenting score
plt.scatter(train['MRI_Track_Age_at_Scan'], train['APQ_P_APQ_P_PP'])
plt.show()


train['sex'] = np.where(train['Sex_F'] == 0, 'Male',
                np.where(train['Sex_F'] == 1, 'Female',
                'Unknown'))


train.sex.value_counts()


#Boxplots: 
#Q: Is there a difference in gender between age of participant and peer problems?

#1. make separate datasets for those under and those over 90 days
female = train[train['sex'] == 'Male']
male = train[train['sex'] == 'Female']

#2. Make grid to plot variables 
fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12, 9))

#2. plot the boxplots
#age
axs[0].boxplot([female['MRI_Track_Age_at_Scan'], male['MRI_Track_Age_at_Scan']])
axs[0].set_title("Age of Participants")
axs[0].set_xlabel('Gender')
axs[0].set_ylabel('Participant Age')

axs[1].violinplot([female['SDQ_SDQ_Peer_Problems'], male['SDQ_SDQ_Peer_Problems']])
axs[1].set_title("Peer Problems")
axs[1].set_xlabel('Gender')
axs[1].set_ylabel('Peer Problems')

plt.show()


#one hot encoding of parent 2s education data
train = pd.get_dummies(train, columns=["Barratt_Barratt_P1_Edu"], dummy_na=True)
train .head()


#dummy encoding of race data
train = pd.get_dummies(train, columns=["Barratt_Barratt_P2_Occ"], dummy_na=True, drop_first=True)
test= pd.get_dummies(test, columns=["Barratt_Barratt_P2_Occ"], dummy_na=True, drop_first=True)
train.head()


from sklearn.preprocessing import LabelEncoder

# Create a LabelEncoder object
le = LabelEncoder()

# Fit and transform the categorical data
train['breast_cancer_diagnosis_code'] = le.fit_transform(train['breast_cancer_diagnosis_code'])
test['breast_cancer_diagnosis_code'] = le.fit_transform(test['breast_cancer_diagnosis_code'])

# Fit and transform the categorical data
train['payer_type'] = le.fit_transform(train['payer_type'])
test['payer_type'] = le.fit_transform(test['payer_type'])


train.head()


from scipy import stats
from matplotlib import pylab

def plots(df, variable):
    plt.figure(figsize=(15,6))
    plt.subplot(1, 2, 1)
    df[variable].hist()
    plt.subplot(1, 2, 2)
    stats.probplot(df[variable], dist="norm", plot=pylab)
    plt.show()
plots(train, 'income_household_median')


# lets try a square root transformation
train['income_household_median_sqr'] = train.income_household_median**(1/2)
plots(train, 'income_household_median_sqr')


#logarithmic
import math
# lets try a square root transformation
train['income_household_median_log'] = [math.log(x) for x in train.income_household_median]
plots(train, 'income_household_median_log')


#identifying rows with outliers using z-score method 
#depending how many standard deviations you want, you can set that t anything! Oftentimes 3 is used. I went up to 5 because
#I only want things that are super likely to be an outlier

#see rows with outliers
train [np.abs(stats.zscore(train['bmi'])) > 5]


#filter them out of dataset
train = train [np.abs(stats.zscore(train['bmi'])) < 5]


train_y = train['DiagPeriodL90D']
train_x = train.drop( ['DiagPeriodL90D'], axis=1)

from sklearn.linear_model import LogisticRegression
x = train_x[['patient_age', 'payer_type']]
classifier = LogisticRegression(random_state=42)

classifier.fit(x, train_y)

predictions = classifier.predict(test[[ 'patient_age', 'payer_type']])


predictions_with_ids = pd.DataFrame(zip(test['patient_id'], predictions), columns=['patient_id','DiagPeriodL90D']).set_index("patient_id")
predictions_with_ids.to_csv("predictions.csv")

