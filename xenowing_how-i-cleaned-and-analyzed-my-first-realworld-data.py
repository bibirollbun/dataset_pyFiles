import numpy as np
import pandas as pd
import seaborn as sns


training_data=pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv")
testing_data=pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv")


training_data


testing_data


training_data.isnull()


training_data


testing_data


training_data.dropna().shape,training_data.shape


numerical_cols=[col for col in training_data.columns if training_data[col].dtype!='object']
numerical_cols


categorical_cols=[col for col in training_data.columns if training_data[col].dtype=='object']
categorical_cols


# Fill NaN in numerical columns with 0
training_data[numerical_cols] = training_data[numerical_cols].fillna(0)
training_data[numerical_cols].isnull()


for col in categorical_cols:
    mode_value = training_data[col].mode()[0]
    training_data[col] = training_data[col].fillna(mode_value)
training_data[categorical_cols].isnull()


# Fill NaN in numerical columns with 0
testing_data[numerical_cols] = testing_data[numerical_cols].fillna(0)

# Fill NaN in categorical columns with mode of each column
for col in categorical_cols:
    if col !='Personality':
        mode_value = testing_data[col].mode()[0]
        testing_data[col] = testing_data[col].fillna(mode_value)


testing_data.isnull()


training_data['Time_spent_Alone'].value_counts()


import seaborn as sns
import matplotlib.pyplot as plt

time_counts=training_data['Time_spent_Alone'].value_counts().sort_index()

sns.barplot(x=time_counts.index,y=time_counts.values)
plt.xlabel("Time Spent Alone (hts)")
plt.ylabel("Number of People")
plt.title("Distribution of Time Spent Alone")
plt.show()


training_data['Social_event_attendance'].value_counts()


event_attended=training_data['Social_event_attendance'].value_counts().sort_index()

sns.barplot(x=event_attended.index,y=event_attended.values)
plt.xlabel("Social Event Attendance (Out of 10)")
plt.ylabel("Number of People")
plt.title("Distribution of Social Event Attendance")
plt.show


training_data['Going_outside'].value_counts()


go_outside=training_data['Going_outside'].value_counts().sort_index()

sns.barplot(x=go_outside.index,y=go_outside.values)
plt.xlabel("Number of Times Goes Outside ")
plt.ylabel("Number of People")
plt.title("Distribution of Going Outside")
plt.show


training_data['Friends_circle_size'].value_counts()


training_data['Friends_circle_size'].value_counts()


friends_circle=training_data['Friends_circle_size'].value_counts().sort_index()
plt.figure(figsize=(10,5))
sns.barplot(x=friends_circle.index,y=friends_circle.values)
plt.xlabel("Number of Friendds Circle size")
plt.ylabel("Number of People")
plt.title("Distribution of Friends Circle Size")
plt.show


training_data['Post_frequency'].value_counts().sort_index()


post_freq=training_data['Post_frequency'].value_counts().sort_index()
plt.figure(figsize=(10,5))
sns.barplot(x=post_freq.index,y=post_freq.values)
plt.xlabel("Number of Post Frequency")
plt.ylabel("Number of People")
plt.title("Distribution of Post Frequency")
plt.show


training_data['Stage_fear'].value_counts()


training_data['Drained_after_socializing'].value_counts()


stage_fear=training_data['Stage_fear'].value_counts().sort_index()
drain_after_social=training_data['Drained_after_socializing'].value_counts().sort_index()

plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
sns.barplot(x=stage_fear.index,y=stage_fear.values)
plt.xlabel("Has Stage Fear")
plt.ylabel("Number of People")
plt.title("Distribution of poeple who fear stage or not")

plt.subplot(1,2,2)

sns.barplot(x=drain_after_social.index,y=drain_after_social.values)
plt.xlabel("Drained After Socializing")
plt.ylabel("Number of People")
plt.title("Distribution of Drained after Socializing")

plt.tight_layout()
plt.show()


person=training_data['Personality'].value_counts().sort_index()
sns.barplot(x=person.index,y=person.values)
plt.xlabel("Personality Type")
plt.ylabel("Number of People")
plt.title("Distribution of Personality of people")
plt.show()


plt.figure(figsize=(10,6))
sns.boxplot(data=training_data,x='Personality',y='Time_spent_Alone')
plt.title("Distribution of Time Spent Type by Personality Type")
plt.xlabel("Personality")
plt.ylabel("Time spent Alone")
plt.tight_layout()
plt.show()


plt.figure(figsize=(10,6))
sns.boxplot(data=training_data,x='Personality',y='Social_event_attendance')
plt.title("Distibution of Social Event Attendance Vs Personality type")
plt.xlabel("Personality Type")
plt.ylabel("Social Event Attendance")
plt.show()


plt.figure(figsize=(10,6))
sns.boxplot(data=training_data,x='Personality',y='Going_outside')
plt.title("Distribution of Going Outside Vs Personality Type")
plt.xlabel("Personality")
plt.ylabel("Going Outside")
plt.show()


plt.figure(figsize=(10,6))
sns.boxplot(data=training_data,x='Personality',y='Friends_circle_size')
plt.title("Distribution of Friends Circle Size Vs Personality")
plt.xlabel("Personality")
plt.ylabel("Friends Circle Size")
plt.show()


plt.figure(figsize=(10,6))
sns.boxplot(data=training_data,x='Personality',y='Post_frequency')
plt.title("Distribution of Post Frequency Vs Personality Type")
plt.xlabel("Personality")
plt.ylabel("Post Frequency")
plt.show()


plt.figure(figsize=(10,6))
sns.countplot(data=training_data,x='Personality',hue='Stage_fear')
plt.title("Stage Fear By Personality Type")
plt.xlabel("Personality")
plt.ylabel("Stage Fear")
plt.legend(title='Stage Fear')
plt.tight_layout()
plt.show()


categorical_cols


plt.figure(figsize=(10,6))
sns.countplot(data=training_data,x='Personality',hue='Drained_after_socializing')
plt.title("Drained After Socializing Vs Personality Type")
plt.xlabel("Personality")
plt.ylabel("Drained after Socializing")
plt.legend(title='Drained After Socializing')
plt.tight_layout()
plt.show()



training_data.to_csv("clean_training_data.csv",index=False)
testing_data.to_csv("clean_testing_data.csv",index=False)

