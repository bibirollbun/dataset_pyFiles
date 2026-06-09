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


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


path = "/kaggle/input/phems-hackathon-early-sepsis-prediction/training_data/"

proceduresoccurrences_df = pd.read_csv(path + 'proceduresoccurrences_train.csv')
devices_df = pd.read_csv(path + "devices_train.csv")
drugsexpesure_df = pd.read_csv(path + "drugsexposure_train.csv")
measurement_lab_df = pd.read_csv(path + "measurement_lab_train.csv")
measurement_meds_df = pd.read_csv(path + "measurement_meds_train.csv")
measurement_observation_df = pd.read_csv(path + "measurement_observation_train.csv")
observation_df = pd.read_csv(path + "observation_train.csv")
person_demographics_episode_df = pd.read_csv(path + "person_demographics_episode_train.csv")
sepsis_label_df = pd.read_csv(path + "SepsisLabel_train.csv")


proceduresoccurrences_df.info()


print(proceduresoccurrences_df["procedure_datetime_hourly"].nunique())
print(proceduresoccurrences_df["procedure"].nunique())


category_num = proceduresoccurrences_df['procedure'].value_counts()

plt.bar(category_num.index, category_num.values, color='blue', edgecolor='green')
plt.title('Procedure Frequency')
plt.xlabel('Category')
plt.ylabel('Frequency')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()


devices_df.info()


print(devices_df["device_datetime_hourly"].nunique())
print(devices_df["device"].nunique())


category_num = devices_df['device'].value_counts()

plt.bar(category_num.index, category_num.values, color='blue', edgecolor='green')
plt.title('Device Frequency')
plt.xlabel('Category')
plt.ylabel('Frequency')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()


drugsexpesure_df.info()


print('drug_concept_id columns')
print(drugsexpesure_df["drug_concept_id"].nunique())

print('route_concept_id columns')
print(drugsexpesure_df["route_concept_id"].nunique())


plt.figure(figsize=(30, 8))
plt.subplot(1, 2, 1)
category_num = drugsexpesure_df['drug_concept_id'].value_counts()

plt.bar(category_num.index, category_num.values, color='blue', edgecolor='green')
plt.title('Drug Concept ID Frequency')
plt.xlabel('Category')
plt.ylabel('Frequency')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

plt.subplot(1, 2, 2)
category_num = drugsexpesure_df['route_concept_id'].value_counts()

plt.bar(category_num.index, category_num.values, color='blue', edgecolor='green')
plt.title('Route Concept ID Frequency')
plt.xlabel('Category')
plt.ylabel('Frequency')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()


person_demographics_episode_df.info()


person_demographics_episode_df['age_in_months'].describe()


plt.hist(person_demographics_episode_df['age_in_months'], color='blue', edgecolor='green')
plt.title('Histogram of Age in Months')
plt.xlabel('Age in Months')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()


category_num = person_demographics_episode_df['gender'].value_counts()

plt.pie(category_num, labels=category_num.index, autopct='%1.1f%%', startangle=90)
plt.title('Gender')
plt.axis('equal')
plt.show()


observation_df.info()


print(f"Observation Concept Id Nunique {observation_df['observation_concept_id'].nunique()}")
print(f"Observation Concept Name Nunique {observation_df['observation_concept_name'].nunique()}")
print(f"Value Filled Nunique {observation_df['valuefilled'].nunique()}")


category_num = observation_df['valuefilled'].value_counts()

plt.pie(category_num, labels=category_num.index, autopct='%1.1f%%', startangle=90)
plt.title('Value Filled')
plt.axis('equal')
plt.show()


measurement_lab_df.info()


print(measurement_lab_df.isnull().sum())



measurement_lab_df.isnull().mean().sort_values(ascending=False).round(4) * 100


import seaborn as sns


sütunlar_çıkartılacak = ['measurement_datetime', 'person_id', 'visit_occurrence_id']
correlation = measurement_lab_df.drop(columns=sütunlar_çıkartılacak).corr()

plt.figure(figsize=(30, 30))
sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt='.2f', linewidths=1, linecolor='black')
plt.title('Korelasyon Matrisini Görselleştir')
plt.show()


measurement_meds_df.info()


measurement_meds_df.head()


measurement_meds_df.isnull().sum()



plt.hist(measurement_meds_df['Body temperature'], color='blue', edgecolor='green', bins=5)
plt.title('Histogram of Body Temperature')
plt.xlabel('Body Temperatures')
plt.ylabel('Frequency')
plt.show()


df = measurement_meds_df['Body temperature'].dropna()
result = df.idxmax()
print(result)


df = df.apply(lambda x: 40.0 if x > 40 else x)
result = df.idxmax()
print(result)
print(df.max())


hist = df.hist(bins=500)

count_sravan = df[df == 40.0].count()
print("Occurrences of 'sravan':", count_sravan)


values = ['Cannulation','Non-invasive ventilation','Invasive ventilation','Exteriorization of trachea']
proceduresoccurrences_df = proceduresoccurrences_df[proceduresoccurrences_df['procedure'].isin(values)]
print(proceduresoccurrences_df.head())


category_num = proceduresoccurrences_df['procedure'].value_counts()

plt.bar(category_num.index, category_num.values, color='blue', edgecolor='green')
plt.title('Procedure Frequency')
plt.xlabel('Category')
plt.ylabel('Frequency')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()


values = ['Endotracheal tube']
devices_df = devices_df[~devices_df['device'].isin(values)]
print(devices_df.head())


category_num = devices_df['device'].value_counts()

plt.bar(category_num.index, category_num.values, color='blue', edgecolor='green')
plt.title('Device Frequency')
plt.xlabel('Category')
plt.ylabel('Frequency')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()


values = ['amoxicillin','prednisolone','methylprednisolone','ampicillin','ciprofloxacin','meropenem','dexamethasone','milrinone','epinephrine','norepinephrine','cefotaxime','piperacillin','dopamine','vancomycin','hydrocortisone','trimethoprim']
drugsexpesure_df = drugsexpesure_df[drugsexpesure_df['drug_concept_id'].isin(values)]

values = ['Intravenous', 'Oral', 'Topical']
drugsexpesure_df = drugsexpesure_df[drugsexpesure_df['route_concept_id'].isin(values)]
print(drugsexpesure_df.head())


plt.figure(figsize=(30, 8))
plt.subplot(1, 2, 1)
category_num = drugsexpesure_df['drug_concept_id'].value_counts()

plt.bar(category_num.index, category_num.values, color='blue', edgecolor='green')
plt.title('Drug Concept ID Frequency')
plt.xlabel('Category')
plt.ylabel('Frequency')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

plt.subplot(1, 2, 2)
category_num = drugsexpesure_df['route_concept_id'].value_counts()

plt.bar(category_num.index, category_num.values, color='blue', edgecolor='green')
plt.title('Route Concept ID Frequency')
plt.xlabel('Category')
plt.ylabel('Frequency')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()


valid_person_ids = pd.concat([
    proceduresoccurrences_df['person_id'],
    devices_df['person_id'],
    drugsexpesure_df['person_id']
]).unique()

observation_df = observation_df[observation_df['person_id'].isin(valid_person_ids)]
sepsis_label_df = sepsis_label_df[sepsis_label_df['person_id'].isin(valid_person_ids)]
observation_df


sepsis_label_df = sepsis_label_df[sepsis_label_df['person_id'].isin(valid_person_ids)]
sepsis_label_df


category_num = observation_df['valuefilled'].value_counts()

plt.pie(category_num, labels=category_num.index, autopct='%1.1f%%', startangle=90)
plt.title('Value Filled')
plt.axis('equal')
plt.show()


category_num = sepsis_label_df['SepsisLabel'].value_counts()

plt.pie(category_num, labels=category_num.index, autopct='%1.1f%%', startangle=90)
plt.title('Sepsis Label')
plt.axis('equal')
plt.show()


tempdf = measurement_lab_df.merge(measurement_meds_df.drop_duplicates(['person_id', 'measurement_datetime']), how="inner")
all_measurement_merge = tempdf.merge(measurement_observation_df.drop_duplicates(['person_id', 'measurement_datetime']), how="inner")


# measurement ve personu birleştirmek için visit_start_date ve measurement_datetime
person_demographics_episode_df['visit_start_date'] = pd.to_datetime(person_demographics_episode_df['visit_start_date'])
person_demographics_episode_df['visit_start_date'] = person_demographics_episode_df['visit_start_date'].dt.date
all_measurement_merge['measurement_datetime'] = pd.to_datetime(all_measurement_merge['measurement_datetime'])
all_measurement_merge['measurement_datetime'] = all_measurement_merge['measurement_datetime'].dt.date


# 2 veriyi birleştirme
all_measurement_merge = all_measurement_merge.drop_duplicates(['visit_occurrence_id'])
person_demographics_episode_df = person_demographics_episode_df.drop_duplicates(['visit_occurrence_id'])
person_demographics_episode_df = person_demographics_episode_df[person_demographics_episode_df['visit_occurrence_id'].isin(all_measurement_merge['visit_occurrence_id'])]
person_and_measurement_df = pd.merge(all_measurement_merge, person_demographics_episode_df, how='inner', left_on=['visit_occurrence_id', 'measurement_datetime'], right_on=['visit_occurrence_id', 'visit_start_date'])



person_and_measurement_df.isnull().mean().sort_values(ascending=False).round(4) * 100

