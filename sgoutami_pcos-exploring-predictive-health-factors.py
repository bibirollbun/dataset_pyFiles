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


import seaborn as sns                       
import matplotlib.pyplot as plt             
%matplotlib inline
sns.set(color_codes=True)
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import altair as alt


import warnings
warnings.filterwarnings('ignore')


df_train = pd.read_csv("/kaggle/input/exploring-predictive-health-factors/train.csv")


df_train.head(5)


df_train.dtypes


print(df_train.isnull().sum())


num_entries = df_train.shape[0] 
print(f"Size of train data : {num_entries}")


for i,col in enumerate(['Age','PCOS','Hormonal_Imbalance','Hyperandrogenism','Hirsutism','Conception_Difficulty','Insulin_Resistance','Exercise_Frequency','Exercise_Type','Exercise_Duration','Sleep_Hours','Exercise_Benefit']):
    print(col, ':', df_train[col].unique(), '\n') 

 


cat_cols = ['Age','Hormonal_Imbalance','Hyperandrogenism','Hirsutism','Conception_Difficulty','Insulin_Resistance','Exercise_Frequency','Exercise_Duration','Sleep_Hours','Exercise_Benefit']


df_train[cat_cols] = df_train[cat_cols].fillna("No Data")


def clean_exercise_duration(duration):
    if duration in ['30 minutes', '30 minutes to 1 hour']:
        return '30 mins'
    elif duration in ['Less than 30 minutes', '20 minutes', 'Less than 6 hours']:
        return '< 30 mins'
    elif duration in ['45 minutes']:
        return '45 mins'
    elif duration in ['More than 30 minutes']:
        return '> 30 mins'
    elif pd.isna(duration) or duration == 'Not Applicable':  # Handle NaN and "Not Applicable"
        return 'Not Applicable'
    return duration

def clean_sleep_hours(hours):
    if hours in ['Less than 6 hours', '3-4 hours']:
        return '< 6 hours'
    elif hours == '6-8 hours':
        return '6-8 hours'
    elif hours == '9-12 hours':
        return '9-12 hours'
    elif hours == 'More than 12 hours':
        return '> 12 hours'
    elif pd.isna(hours):
        return 'No Data'
    return hours

def clean_exercise_type(exercise_type):
    if pd.isna(exercise_type):
        return "No Data"

    exercise_type = str(exercise_type).lower()

    if "cardio" in exercise_type and "strength" in exercise_type and "flexibility" in exercise_type:
        return "Cardio, Strength, and Flexibility"
    elif "cardio" in exercise_type and "strength" in exercise_type:
        return "Cardio and Strength"
    elif "cardio" in exercise_type and "flexibility" in exercise_type:
        return "Cardio and Flexibility"
    elif "strength" in exercise_type and "flexibility" in exercise_type:
        return "Strength and Flexibility"
    elif "cardio" in exercise_type:
        return "Cardio"
    elif "strength" in exercise_type:
        return "Strength"
    elif "flexibility" in exercise_type:
        return "Flexibility"
    elif "no exercise" in exercise_type or "none" in exercise_type:
        return "No Exercise"
    elif "hiit" in exercise_type:
        return "HIIT"
    elif "somewhat" in exercise_type:
        return "Somewhat"
    return "Other"

def clean_exercise_frequency(frequency):
    if frequency in ['Rarely', 'Never', 'Less than usual', 'Less than 6 hours']:
        return 'Rarely'
    elif frequency == '1-2 Times a Week':
        return '1-2 Times/Week'
    elif frequency == '3-4 Times a Week':
        return '3-4 Times/Week'
    elif frequency == '6-8 Times a Week' or frequency == '6-8 hours':
        return '6-8 Times/Week'
    elif pd.isna(frequency):
        return 'No Data'
    return frequency

def clean_age(age):
    if age == 'Less than 20' or age == 'Less than 20-25' or age == '15-20':
        return '0-20'
    elif age == '20-25' or age == '30-25':
        return '20-25'
    elif age == '25-30':
        return '25-30'
    elif age == '30-35' or age == '30-40':
        return '30-35'
    elif age == '35-44':
        return '35-44'
    elif age == '45 and above':
        return '45 <'
    elif age == 'No Data':
        return 'No Data'
    return age

def clean_category(value, category_name):
    if category_name == 'Hormonal_Imbalance':
        if value in ['No', 'No, Yes, not diagnosed by a doctor']:
            return 'No'
        elif value in ['Yes', 'Yes Significantly']:
            return 'Yes'
        elif value == 'No Data':
            return 'No Data'
    elif category_name == 'Hyperandrogenism':
        if value in ['No']:
            return 'No'
        elif value in ['Yes']:
            return 'Yes'
        elif value == 'No Data':
            return 'No Data'
    elif category_name == 'Hirsutism':
        if value in ['No', 'No, Yes, not diagnosed by a doctor']:
            return 'No'
        elif value in ['Yes']:
            return 'Yes'
        elif value == 'No Data':
            return 'No Data'
    elif category_name == 'Conception_Difficulty':
        if value in ['No', 'No, Yes, not diagnosed by a doctor']:
            return 'No'
        elif value in ['Yes', 'Yes, diagnosed by a doctor']:
            return 'Yes'
        elif value == 'No Data':
            return 'No Data'
    elif category_name == 'Insulin_Resistance':
        if value in ['No', 'No, Yes, not diagnosed by a doctor']:
            return 'No'
        elif value in ['Yes']:
            return 'Yes'
        elif value == 'No Data':
            return 'No Data'
    return value 


def create_weight_buckets(df, interval=10):

    df['Weight_kg'] = pd.to_numeric(df['Weight_kg'], errors='coerce')  # Convert to numeric

    min_weight = df['Weight_kg'].min()
    max_weight = df['Weight_kg'].max()

    if pd.isna(min_weight) or pd.isna(max_weight):
        print("Weight_kg column contains only NaN values. Cannot create weight buckets.")
        df['Weight_Bucket'] = pd.NA
        return df

    bins = list(range(int(min_weight), int(max_weight) + interval, interval))
    labels = [f'{bins[i]}-{bins[i+1]-1} kg' for i in range(len(bins) - 1)]

    df['Weight_Bucket'] = pd.cut(df['Weight_kg'], bins=bins, labels=labels, right=False, include_lowest=True)
    return df

df_train = create_weight_buckets(df_train, interval=10)  # 10 kg intervals
print(df_train['Weight_Bucket'].unique())


df_train['Exercise_Duration'] = df_train['Exercise_Duration'].apply(clean_exercise_duration)
df_train['Sleep_Hours'] = df_train['Sleep_Hours'].apply(clean_sleep_hours)
df_train['Exercise_Frequency'] = df_train['Exercise_Frequency'].apply(clean_exercise_frequency)
df_train['Exercise_Type'] = df_train['Exercise_Type'].apply(clean_exercise_type)
df_train['Age'] = df_train['Age'].apply(clean_age)


categories = ['Age', 'Hormonal_Imbalance', 'Hyperandrogenism', 'Hirsutism', 'Conception_Difficulty', 'Insulin_Resistance']

fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(20, 12)) 
axes = axes.flatten()

for i, category in enumerate(categories):
    if category != 'Age':
        df_train[category] = df_train[category].apply(lambda x: clean_category(x, category))

    grouped = df_train.groupby([category, 'PCOS']).size().unstack()

    ax = axes[i] 

    width = 0.5
    cat_groups = grouped.index
    x = np.arange(len(cat_groups))

    if grouped.columns.size == 2:
        pcos_0 = grouped.iloc[:, 0].fillna(0)
        pcos_1 = grouped.iloc[:, 1].fillna(0)
        rects1 = ax.bar(x - width/2, pcos_0, width, label=grouped.columns[0])
        rects2 = ax.bar(x + width/2, pcos_1, width, label=grouped.columns[1])

        ax.set_xticks(x)
        ax.set_xticklabels(cat_groups,  ha='right', fontsize=8)
        ax.set_ylabel('Count', fontsize=10)
        ax.set_title(f'PCOS Distribution by {category}', fontsize=12)
        ax.legend(fontsize=10)

        def autolabel(rects):
            for rect in rects:
                height = rect.get_height()
                ax.annotate('{}'.format(height),
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=8)

            autolabel(rects1)
            autolabel(rects2)

            fig.tight_layout()

    elif grouped.columns.size == 1:
        pcos_0 = grouped.iloc[:, 0].fillna(0)
        rects1 = ax.bar(x, pcos_0, width, label=grouped.columns[0])

        ax.set_xticks(x)
        ax.set_xticklabels(cat_groups, ha='right', fontsize=8)
        ax.set_ylabel('Count', fontsize=10)
        ax.set_title(f'PCOS Distribution by {category}', fontsize=12)
        ax.legend(fontsize=10)

        def autolabel(rects):
            for rect in rects:
                height = rect.get_height()
                ax.annotate('{}'.format(height),
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=8)

        autolabel(rects1)

        fig.tight_layout()
    else:
        print("error")

plt.show() 


categories = ['Age','Exercise_Duration', 'Sleep_Hours', 'Exercise_Frequency'] 
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(23, 8)) 
axes = axes.flatten()

for i, category in enumerate(categories):
    grouped = df_train.groupby([category, 'PCOS']).size().unstack()
    ax = axes[i]

    width = 0.35
    cat_groups = grouped.index
    x = np.arange(len(cat_groups))

    if grouped.columns.size == 2:
        pcos_0 = grouped.iloc[:, 0].fillna(0)
        pcos_1 = grouped.iloc[:, 1].fillna(0)
        rects1 = ax.bar(x - width / 2, pcos_0, width, label=grouped.columns[0])
        rects2 = ax.bar(x + width / 2, pcos_1, width, label=grouped.columns[1])

        ax.set_xticks(x)
        ax.set_xticklabels(cat_groups,  ha='right', fontsize=8)
        ax.set_ylabel('Count', fontsize=10)
        ax.set_title(f'PCOS Distribution by {category}', fontsize=12)
        ax.legend(fontsize=10)

        def autolabel(rects):
            for rect in rects:
                height = rect.get_height()
                ax.annotate('{}'.format(height),
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=8)

            autolabel(rects1)
            autolabel(rects2)
            fig.tight_layout()


    elif grouped.columns.size == 1:
        pcos_0 = grouped.iloc[:, 0].fillna(0)
        rects1 = ax.bar(x, pcos_0, width, label=grouped.columns[0])

        ax.set_xticks(x)
        ax.set_xticklabels(cat_groups, rotation=45, ha='right', fontsize=8)
        ax.set_ylabel('Count', fontsize=10)
        ax.set_title(f'PCOS Distribution by {category}', fontsize=12)
        ax.legend(fontsize=10)

        def autolabel(rects):
            for rect in rects:
                height = rect.get_height()
                ax.annotate('{}'.format(height),
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=8)

        autolabel(rects1)
        fig.tight_layout()

    else:
        print(f"More than 2 categories in {category} column. Please handle it accordingly.")

plt.show()


grouped = df_train.groupby(['Exercise_Type', 'PCOS']).size().unstack()

fig, ax = plt.subplots(figsize=(12, 6)) 

width = 0.35
exercise_types = grouped.index
x = np.arange(len(exercise_types))

if grouped.columns.size == 2:
    pcos_no = grouped.iloc[:, 0].fillna(0)
    pcos_yes = grouped.iloc[:, 1].fillna(0)
    rects1 = ax.bar(x - width/2, pcos_no, width, label='No PCOS')
    rects2 = ax.bar(x + width/2, pcos_yes, width, label='PCOS')

    ax.set_xticks(x)
    ax.set_xticklabels(exercise_types, rotation=45, ha='right', fontsize=10) 
    ax.set_ylabel('Count', fontsize=12) 
    ax.set_title('PCOS Distribution by Exercise Type', fontsize=14) 
    ax.legend(fontsize=12) 

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9) 

    autolabel(rects1)
    autolabel(rects2)

    fig.tight_layout()
    plt.show()
elif grouped.columns.size == 1: 
    pcos_0 = grouped.iloc[:, 0].fillna(0)
    rects1 = ax.bar(x, pcos_0, width, label=grouped.columns[0]) 

    ax.set_xticks(x)
    ax.set_xticklabels(exercise_types, rotation=45, ha='right', fontsize=10) 
    ax.set_ylabel('Count', fontsize=12)  #increased fontsize
    ax.set_title('PCOS Distribution by Exercise Type', fontsize=14) 
    ax.legend(fontsize=12) 

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3), 
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)  

    autolabel(rects1)

    fig.tight_layout()
    plt.show()
else:
    print("error")


print("\n The size of the dot is directly proportional to the number of data entries present in that category.\n ")
df_train = create_weight_buckets(df_train)

grouped = df_train.groupby(['Age', 'Weight_Bucket', 'PCOS']).size().unstack(fill_value=0)

plt.figure(figsize=(12, 8))

for age_group in grouped.index.get_level_values('Age').unique():
    for weight_bucket in grouped.index.get_level_values('Weight_Bucket').unique():
        if pd.notna(weight_bucket):
            try: 
                data_point = grouped.loc[(age_group, weight_bucket)] 

                if isinstance(data_point, pd.Series):
                    pcos_no = data_point.get('No', 0)
                    pcos_yes = data_point.get('Yes', 0)
                else:
                    if data_point.name == 'No':
                        pcos_no = data_point
                        pcos_yes = 0
                    else:
                        pcos_yes = data_point
                        pcos_no = 0

                total_count = pcos_no + pcos_yes
                if total_count > 0:
                    plt.scatter(age_group, weight_bucket, s=total_count * 50, alpha=0.7, label=f'age:{age_group}, weight:{weight_bucket}')

            except KeyError:  
                pass  
                #print(f"No data for {age_group}, {weight_bucket}")

plt.xlabel('Age Group', fontsize=12)
plt.ylabel('Weight Bucket', fontsize=12)
plt.title('PCOS Distribution by Age and Weight (Scatter Plot)', fontsize=14)
plt.xticks(rotation=90, ha='right', fontsize=10)
plt.yticks(fontsize=10)
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.tight_layout()
plt.subplots_adjust(right=0.8)
plt.show()

