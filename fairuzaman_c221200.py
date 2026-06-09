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

# Replace with your actual file path
df = pd.read_csv('/kaggle/input/machine-learning-and-data-mining-lab-exam-spring/train_dataset.csv')

# Show the first few rows
df.head(10)



# Count of missing values per column
print(df.isnull().sum())



# Fill numeric missing values
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

# Fill categorical missing values
categorical_cols = df.select_dtypes(include=['object']).columns
df[categorical_cols] = df[categorical_cols].fillna(df[categorical_cols].mode().iloc[0])



from sklearn.preprocessing import LabelEncoder

label_cols = ['Gender', 'Customer Type', 'Type of Travel', 'Class', 'satisfaction']
le = LabelEncoder()

for col in label_cols:
    df[col] = le.fit_transform(df[col])



import matplotlib.pyplot as plt
import seaborn as sns

# Plot to visually inspect
sns.boxplot(x=df['Flight Distance'])
plt.show()

# You can remove outliers manually or with IQR
Q1 = df['Flight Distance'].quantile(0.25)
Q3 = df['Flight Distance'].quantile(0.75)
IQR = Q3 - Q1
df = df[(df['Flight Distance'] >= Q1 - 1.5 * IQR) & (df['Flight Distance'] <= Q3 + 1.5 * IQR)]


print(df.isnull().sum())     # Should show 0
print(df.dtypes)             # Ensure all formats are correct
print(df.head())             # See final data



validation_report = {}

# Example columns and rules (customize as needed)
expected_values = {
    'Gender': ['Male', 'Female'],
    'Customer Type': ['Loyal Customer', 'disloyal Customer'],
    'Type of Travel': ['Business travel', 'Personal Travel'],
    'Class': ['Business', 'Eco', 'Eco Plus'],
    'satisfaction': ['satisfied', 'neutral or dissatisfied']
}

for column, valid_values in expected_values.items():
    col_data = df[column]
    total = len(col_data)
    missing = col_data.isnull().sum()
    mismatched = (~col_data.isin(valid_values)).sum()
    valid = total - missing - mismatched

    validation_report[column] = {
        'Valid': valid,
        'Mismatched': mismatched,
        'Missing': missing
    }



report_df = pd.DataFrame(validation_report).T
report_df.reset_index(inplace=True)
report_df.rename(columns={'index': 'Column'}, inplace=True)
print(report_df)


plt.figure(figsize=(12, 6))
report_df.set_index('Column')[['Valid', 'Mismatched', 'Missing']].plot(
    kind='bar', stacked=True, colormap='Set2', figsize=(12, 6)
)

plt.title('Data Validity Report by Column')
plt.xlabel('Column')
plt.ylabel('Number of Records')
plt.xticks(rotation=45)
plt.legend(title='Status')
plt.tight_layout()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()


report_df.to_csv("validation_report.csv", index=False)


gender_counts = df['Gender'].value_counts()
print(gender_counts)



plt.figure(figsize=(6, 4))
sns.barplot(x=gender_counts.index, y=gender_counts.values, palette='Set2')

plt.title('Gender Distribution')
plt.xlabel('Gender')
plt.ylabel('Number of Passengers')
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()


plt.figure(figsize=(6, 6))
plt.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%', colors=['#66b3ff', '#ff9999'], startangle=140)

plt.title('Gender Distribution')
plt.axis('equal')  # Equal aspect ratio makes the pie circular
plt.show()


customer_counts = df['Customer Type'].value_counts()
print(customer_counts)


plt.figure(figsize=(6, 4))
sns.barplot(x=customer_counts.index, y=customer_counts.values, palette='coolwarm')

plt.title('Customer Type Distribution')
plt.xlabel('Customer Type')
plt.ylabel('Number of Customers')
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()



plt.figure(figsize=(6, 6))
plt.pie(customer_counts, labels=customer_counts.index, autopct='%1.1f%%', startangle=90, colors=['#00b894', '#d63031'])
plt.title('Customer Type Percentage')
plt.axis('equal')  # Ensures the pie chart is a circle
plt.show()



print(df['Age'].describe())



age_bins = [0, 18, 30, 45, 60, 100]
age_labels = ['Teen (0–18)', 'Young Adult (19–30)', 'Adult (31–45)', 'Middle Age (46–60)', 'Senior (61+)']
df['Age Group'] = pd.cut(df['Age'], bins=age_bins, labels=age_labels)

# Count number of passengers in each age group
age_group_counts = df['Age Group'].value_counts().sort_index()
print(age_group_counts)



plt.figure(figsize=(8, 5))
sns.barplot(x=age_group_counts.index, y=age_group_counts.values, palette='viridis')

plt.title('Passenger Count by Age Group')
plt.xlabel('Age Group')
plt.ylabel('Number of Passengers')
plt.xticks(rotation=30)
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()



plt.figure(figsize=(10, 6))
sns.histplot(df['Age'], bins=30, kde=True, color='steelblue')

plt.title('Distribution of Passenger Ages')
plt.xlabel('Age')
plt.ylabel('Number of Passengers')
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()


satisfied_df = df[df['satisfaction'] == 'satisfied']



plt.figure(figsize=(10, 6))
sns.histplot(satisfied_df['Age'], bins=30, kde=True, color='green')

plt.title('Distribution of Satisfied Customers by Age')
plt.xlabel('Age')
plt.ylabel('Number of Satisfied Passengers')
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()


age_bins = [0, 18, 30, 45, 60, 100]
age_labels = ['Teen (0–18)', 'Young Adult (19–30)', 'Adult (31–45)', 'Middle Age (46–60)', 'Senior (61+)']
satisfied_df['Age Group'] = pd.cut(satisfied_df['Age'], bins=age_bins, labels=age_labels)

# Count satisfied passengers per age group
group_counts = satisfied_df['Age Group'].value_counts().sort_index()
print(group_counts)


plt.figure(figsize=(8, 5))
sns.barplot(x=group_counts.index, y=group_counts.values, palette='Greens')

plt.title('Number of Satisfied Passengers by Age Group')
plt.xlabel('Age Group')
plt.ylabel('Number of Satisfied Passengers')
plt.xticks(rotation=30)
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()


travel_type_counts = df['Type of Travel'].value_counts()
print(travel_type_counts)


plt.figure(figsize=(6, 4))
sns.barplot(x=travel_type_counts.index, y=travel_type_counts.values, palette='pastel')

plt.title('Distribution of Passengers by Type of Travel')
plt.xlabel('Type of Travel')
plt.ylabel('Number of Passengers')
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()


plt.figure(figsize=(6, 6))
plt.pie(travel_type_counts, labels=travel_type_counts.index, autopct='%1.1f%%', startangle=90, colors=['#ffcc99', '#66b3ff'])
plt.title('Passenger Distribution by Type of Travel')
plt.axis('equal')  # Circle shape
plt.show()


class_counts = df['Class'].value_counts()
print(class_counts)


plt.figure(figsize=(7, 5))
sns.barplot(x=class_counts.index, y=class_counts.values, palette='muted')

plt.title('Passenger Distribution by Flight Class')
plt.xlabel('Class')
plt.ylabel('Number of Passengers')
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()


plt.figure(figsize=(6, 6))
plt.pie(class_counts, labels=class_counts.index, autopct='%1.1f%%', startangle=90, colors=['#ff9999','#66b3ff','#99ff99'])
plt.title('Passenger Percentage by Flight Class')
plt.axis('equal')  # Equal aspect ratio ensures pie chart is circular
plt.show()


print(df['Flight Distance'].describe())



plt.figure(figsize=(10,6))
sns.histplot(df['Flight Distance'], bins=50, kde=True, color='skyblue')

plt.title('Distribution of Flight Distances')
plt.xlabel('Flight Distance')
plt.ylabel('Number of Passengers')
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()


plt.figure(figsize=(8,4))
sns.boxplot(x=df['Flight Distance'], color='lightgreen')

plt.title('Boxplot of Flight Distances')
plt.xlabel('Flight Distance')
plt.tight_layout()
plt.show()


print(df['Inflight wifi service'].value_counts().sort_index())


plt.figure(figsize=(8, 5))
sns.countplot(x='Inflight wifi service', data=df, palette='Blues')

plt.title('Inflight Wifi Service Ratings Distribution')
plt.xlabel('Wifi Service Rating (1=Poor, 5=Excellent)')
plt.ylabel('Number of Passengers')
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()


plt.figure(figsize=(6, 4))
sns.barplot(x='satisfaction', y='Inflight wifi service', data=df, palette='coolwarm')

plt.title('Average Inflight Wifi Service Rating by Satisfaction')
plt.xlabel('Satisfaction')
plt.ylabel('Average Wifi Service Rating')
plt.tight_layout()
plt.show()


print(df['Departure/Arrival time convenient'].value_counts().sort_index())


plt.figure(figsize=(8, 5))
sns.countplot(x='Departure/Arrival time convenient', data=df, palette='magma')

plt.title('Departure/Arrival Time Convenience Ratings Distribution')
plt.xlabel('Rating (1=Poor, 5=Excellent)')
plt.ylabel('Number of Passengers')
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()


plt.figure(figsize=(6, 4))
sns.barplot(x='satisfaction', y='Departure/Arrival time convenient', data=df, palette='viridis')

plt.title('Average Departure/Arrival Time Convenience Rating by Satisfaction')
plt.xlabel('Satisfaction')
plt.ylabel('Average Rating')
plt.tight_layout()
plt.show()


print(df['satisfaction'].value_counts())


plt.figure(figsize=(6, 4))
sns.countplot(x='satisfaction', data=df, palette='Set1')

plt.title('Customer Satisfaction Distribution')
plt.xlabel('Satisfaction')
plt.ylabel('Number of Customers')
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()


avg_departure_delay = df['Departure Delay in Minutes'].mean()
avg_arrival_delay = df['Arrival Delay in Minutes'].mean()

print(f"Average Departure Delay: {avg_departure_delay:.2f} minutes")
print(f"Average Arrival Delay: {avg_arrival_delay:.2f} minutes")


delays = {'Departure Delay': avg_departure_delay, 'Arrival Delay': avg_arrival_delay}

plt.figure(figsize=(6, 4))
sns.barplot(x=list(delays.keys()), y=list(delays.values()), palette='coolwarm')

plt.title('Average Flight Delays')
plt.ylabel('Minutes')
plt.ylim(0, max(delays.values()) + 10)
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()


# Group by 'satisfaction' and calculate mean delays
delay_means = df.groupby('satisfaction')[['Departure Delay in Minutes', 'Arrival Delay in Minutes']].mean().reset_index()
print(delay_means)


plt.figure(figsize=(8, 5))

# Melt the dataframe for easier plotting with seaborn
delay_melted = delay_means.melt(id_vars='satisfaction', 
                                value_vars=['Departure Delay in Minutes', 'Arrival Delay in Minutes'],
                                var_name='Delay Type',
                                value_name='Average Delay (Minutes)')

sns.barplot(data=delay_melted, x='satisfaction', y='Average Delay (Minutes)', hue='Delay Type', palette='Set2')

plt.title('Average Flight Delays by Customer Satisfaction')
plt.xlabel('Customer Satisfaction')
plt.ylabel('Average Delay (Minutes)')
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()


