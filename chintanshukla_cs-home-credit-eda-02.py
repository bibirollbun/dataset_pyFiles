import pandas as pd
import numpy as np 

import matplotlib
import matplotlib.pyplot as plt # for plotting

import seaborn as sns 
color = sns.color_palette()


import os
print(os.listdir("/kaggle/input/home-credit-default-risk"))


application_train = pd.read_csv('/kaggle/input/home-credit-default-risk/application_train.csv')
POS_CASH_balance = pd.read_csv('/kaggle/input/home-credit-default-risk/POS_CASH_balance.csv')
bureau_balance = pd.read_csv('/kaggle/input/home-credit-default-risk/bureau_balance.csv')
previous_application = pd.read_csv('/kaggle/input/home-credit-default-risk/previous_application.csv')
installments_payments = pd.read_csv('/kaggle/input/home-credit-default-risk/installments_payments.csv')
credit_card_balance = pd.read_csv('/kaggle/input/home-credit-default-risk/credit_card_balance.csv')
bureau = pd.read_csv('/kaggle/input/home-credit-default-risk/bureau.csv')
application_test = pd.read_csv('/kaggle/input/home-credit-default-risk/application_test.csv')


print('Size of application_train data', application_train.shape)
print('Size of POS_CASH_balance data', POS_CASH_balance.shape)
print('Size of bureau_balance data', bureau_balance.shape)
print('Size of previous_application data', previous_application.shape)
print('Size of installments_payments data', installments_payments.shape)
print('Size of credit_card_balance data', credit_card_balance.shape)
print('Size of bureau data', bureau.shape)


application_train.head()


application_train.columns.values


POS_CASH_balance.head()


bureau_balance.head()


previous_application.head()


previous_application.columns.values


installments_payments.head()


credit_card_balance.head()


credit_card_balance.columns.values


bureau.head()


# checking missing data
total = application_train.isnull().sum().sort_values(ascending = False)
percent = (application_train.isnull().sum()/application_train.isnull().count()*100).sort_values(ascending = False)
missing_application_train_data  = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_application_train_data.head(20)


# checking missing data
total = POS_CASH_balance.isnull().sum().sort_values(ascending = False)
percent = (POS_CASH_balance.isnull().sum()/POS_CASH_balance.isnull().count()*100).sort_values(ascending = False)
missing_POS_CASH_balance_data  = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_POS_CASH_balance_data.head(3)


# checking missing data
total = bureau_balance.isnull().sum().sort_values(ascending = False)
percent = (bureau_balance.isnull().sum()/bureau_balance.isnull().count()*100).sort_values(ascending = False)
missing_bureau_balance_data  = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_bureau_balance_data.head(3)


# checking missing data
total = previous_application.isnull().sum().sort_values(ascending = False)
percent = (previous_application.isnull().sum()/previous_application.isnull().count()*100).sort_values(ascending = False)
missing_previous_application_data  = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_previous_application_data.head(15)


# checking missing data
total = installments_payments.isnull().sum().sort_values(ascending = False)
percent = (installments_payments.isnull().sum()/installments_payments.isnull().count()*100).sort_values(ascending = False)
missing_installments_payments_data  = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_installments_payments_data.head(3)


# checking missing data
total = credit_card_balance.isnull().sum().sort_values(ascending = False)
percent = (credit_card_balance.isnull().sum()/credit_card_balance.isnull().count()*100).sort_values(ascending = False)
missing_credit_card_balance_data  = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_credit_card_balance_data.head(10)


# checking missing data
total = bureau.isnull().sum().sort_values(ascending = False)
percent = (bureau.isnull().sum()/bureau.isnull().count()*100).sort_values(ascending = False)
missing_bureau_data  = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_bureau_data.head(8)


plt.figure(figsize=(12,5))
plt.title("Distribution of AMT_CREDIT")
ax = sns.distplot(application_train["AMT_CREDIT"])


plt.figure(figsize=(12,5))
plt.title("Distribution of AMT_INCOME_TOTAL")
ax = sns.distplot(application_train["AMT_INCOME_TOTAL"].dropna())


plt.figure(figsize=(12,5))
plt.title("Distribution of AMT_GOODS_PRICE")
ax = sns.distplot(application_train["AMT_GOODS_PRICE"].dropna())


# Value counts
temp = application_train["NAME_TYPE_SUITE"].value_counts()
print("Total number of states : ",len(temp))


print(temp)
print("Sum:", temp.sum())


# Convert to percentages
percentages = (temp / temp.sum()) * 100

# Plot
plt.figure(figsize=(10, 6))
bars = plt.bar(percentages.index, percentages.values, color='skyblue')
plt.title("Who accompanied client when applying for the application (%)")
plt.xlabel("Type of Suite")
plt.ylabel("Percentage")
plt.xticks(rotation=45, ha='right')

# Add percentage labels on top of bars
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.5, f'{yval:.2f}%', ha='center', va='bottom')

plt.tight_layout()
plt.show()


# Value counts
temp = application_train["TARGET"].value_counts()

# Labels and values
labels = ['Loan Repaid (0)', 'Loan Not Repaid (1)']
values = temp.values

# Plot
plt.figure(figsize=(6, 6))
plt.pie(values, labels=labels, autopct='%1.1f%%', startangle=90, colors=['#66b3ff','#ff9999'])
plt.title('Loan Repaid or Not')
plt.axis('equal')  # Ensures pie is a circle
plt.show()


# Value counts
temp = application_train["NAME_CONTRACT_TYPE"].value_counts()

# Labels and values
labels = temp.index
values = temp.values

# Plot
fig, ax = plt.subplots(figsize=(6, 6))
wedges, texts, autotexts = ax.pie(values,
                                  labels=labels,
                                  autopct='%1.1f%%',
                                  startangle=90,
                                  pctdistance=0.85,
                                  colors=['#66b3ff', '#ff9999'])

# Draw center circle to make it a donut
centre_circle = plt.Circle((0, 0), 0.70, fc='white')
fig.gca().add_artist(centre_circle)

# Add annotation in the center
ax.text(0, 0, 'Loan Types', ha='center', va='center', fontsize=16)

# Equal aspect ratio ensures that pie is drawn as a circle
ax.axis('equal')
plt.title('Types of Loan')
plt.tight_layout()
plt.show()


# Value counts
temp1 = application_train["FLAG_OWN_CAR"].value_counts()
temp2 = application_train["FLAG_OWN_REALTY"].value_counts()

# Labels and values
labels1 = ['Does Not Own Car', 'Owns Car'] if 'N' in temp1.index else temp1.index
values1 = temp1.values

labels2 = ['Does Not Own Realty', 'Owns Realty'] if 'N' in temp2.index else temp2.index
values2 = temp2.values

# Create subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# Donut chart for Own Car
wedges1, texts1, autotexts1 = ax1.pie(values1,
                                      labels=labels1,
                                      autopct='%1.1f%%',
                                      startangle=90,
                                      pctdistance=0.85,
                                      colors=['#ff9999','#66b3ff'])
centre_circle1 = plt.Circle((0, 0), 0.60, fc='white')
ax1.add_artist(centre_circle1)
ax1.set_title('Own Car')
ax1.axis('equal')

# Donut chart for Own Realty
wedges2, texts2, autotexts2 = ax2.pie(values2,
                                      labels=labels2,
                                      autopct='%1.1f%%',
                                      startangle=90,
                                      pctdistance=0.85,
                                      colors=['#99ff99','#ffcc99'])
centre_circle2 = plt.Circle((0, 0), 0.60, fc='white')
ax2.add_artist(centre_circle2)
ax2.set_title('Own Realty')
ax2.axis('equal')

# Overall title
plt.suptitle('Ownership Status: Car vs Realty', fontsize=16)
plt.tight_layout()
plt.show()


# Value counts
temp = application_train["NAME_INCOME_TYPE"].value_counts()

# Labels and values
labels = temp.index
values = temp.values

# Plot
fig, ax = plt.subplots(figsize=(8, 6))
wedges, texts, autotexts = ax.pie(values,
                                  labels=labels,
                                  autopct='%1.1f%%',
                                  startangle=90,
                                  pctdistance=0.85,
                                  colors=plt.cm.Paired.colors)

# Draw center circle to make it a donut
centre_circle = plt.Circle((0, 0), 0.50, fc='white')
fig.gca().add_artist(centre_circle)

# Add annotation in the center
ax.text(0, 0, "Income\nSources", ha='center', va='center', fontsize=14)

# Equal aspect ratio ensures that pie is drawn as a circle
ax.axis('equal')
plt.title("Income Sources of Applicants")
plt.tight_layout()
plt.show()


# Value counts
temp = application_train["NAME_FAMILY_STATUS"].value_counts()

# Labels and values
labels = temp.index
values = temp.values

# Plot
fig, ax = plt.subplots(figsize=(8, 6))
wedges, texts, autotexts = ax.pie(values,
                                  labels=labels,
                                  autopct='%1.1f%%',
                                  startangle=90,
                                  pctdistance=0.85,
                                  colors=plt.cm.Set2.colors)

# Draw center circle to make it a donut
centre_circle = plt.Circle((0, 0), 0.50, fc='white')
fig.gca().add_artist(centre_circle)

# Add annotation in the center
ax.text(0, 0, "Family\nStatus", ha='center', va='center', fontsize=14)

# Equal aspect ratio ensures that pie is drawn as a circle
ax.axis('equal')
plt.title("Family Status of Applicants")
plt.tight_layout()
plt.show()


# Value counts
temp = application_train["OCCUPATION_TYPE"].value_counts()

# Plot
plt.figure(figsize=(10, 6))
bars = plt.bar(temp.index, temp.values, color='green')
plt.title("Occupation of Applicants Who Applied for Loan")
plt.xlabel("Occupation")
plt.ylabel("Count")
plt.xticks(rotation=45, ha='right')

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 500, f'{yval}', ha='center', va='bottom')

plt.tight_layout()
plt.show()


import matplotlib.pyplot as plt

# Value counts
temp = application_train["NAME_EDUCATION_TYPE"].value_counts()
labels = temp.index
values = temp.values

# Plot
fig, ax = plt.subplots(figsize=(8,6))
wedges, texts, autotexts = ax.pie(values,
                                  labels=None,  # Hide inline labels
                                  autopct='%1.1f%%',
                                  startangle=90,
                                  pctdistance=0.85,
                                  colors=plt.cm.Set3.colors,
                                  wedgeprops=dict(width=0.4))  # donut effect

# Add center label
ax.text(0, 0, "Education", ha='center', va='center', fontsize=14)

# Add legend instead of labels on chart
ax.legend(wedges, labels, title="Education Type", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))

# Title
plt.title("Education of Applicants", y=1.05)
plt.tight_layout()
plt.show()


# Get value counts
temp = application_train["NAME_HOUSING_TYPE"].value_counts()

# Prepare labels and values
labels = temp.index
values = temp.values

# Create pie chart
plt.figure(figsize=(8, 8))
plt.pie(values, labels=labels, autopct='%1.1f%%', startangle=140, wedgeprops={'width': 0.5})
plt.title('Type of House')
plt.axis('equal')  # Ensures pie is drawn as a circle
plt.show()


# Get value counts
temp = application_train["ORGANIZATION_TYPE"].value_counts()

# Prepare labels and values
labels = temp.index
values = temp.values

# Create bar chart
plt.figure(figsize=(12, 6))
plt.bar(labels, values, color='red')

# Add titles and labels
plt.title('Types of Organizations who applied for loan')
plt.xlabel('Organization Name')
plt.ylabel('Count')

# Rotate x-axis labels for readability
plt.xticks(rotation=90)

plt.tight_layout()
plt.show()


# Get income type counts
temp = application_train["NAME_INCOME_TYPE"].value_counts()

# Initialize lists for TARGET = 0 and 1
temp_y0 = []
temp_y1 = []

# Populate counts for each income type
for val in temp.index:
    temp_y1.append(np.sum(application_train["TARGET"][application_train["NAME_INCOME_TYPE"] == val] == 1))
    temp_y0.append(np.sum(application_train["TARGET"][application_train["NAME_INCOME_TYPE"] == val] == 0))

# Convert to percentages
total = np.array(temp_y0) + np.array(temp_y1)
percent_y0 = (np.array(temp_y0) / total) * 100
percent_y1 = (np.array(temp_y1) / total) * 100

# Set up bar positions
x = np.arange(len(temp.index))
width = 0.35

# Create the plot
plt.figure(figsize=(12, 6))
plt.bar(x - width/2, percent_y1, width, label='YES', color='tab:blue')
plt.bar(x + width/2, percent_y0, width, label='NO', color='tab:orange')

# Add labels and title
plt.xlabel('Income source')
plt.ylabel('Count in %')
plt.title("Income sources of Applicants in terms of loan repaid or not (%)")
plt.xticks(ticks=x, labels=temp.index, rotation=45, ha='right')
plt.legend(title='Loan Repaid')
plt.tight_layout()
plt.show()



# Get family status counts
temp = application_train["NAME_FAMILY_STATUS"].value_counts()

# Initialize lists for TARGET = 0 and 1
temp_y0 = []
temp_y1 = []

# Populate counts for each family status
for val in temp.index:
    temp_y1.append(np.sum(application_train["TARGET"][application_train["NAME_FAMILY_STATUS"] == val] == 1))
    temp_y0.append(np.sum(application_train["TARGET"][application_train["NAME_FAMILY_STATUS"] == val] == 0))

# Convert to percentages within each group
total = np.array(temp_y0) + np.array(temp_y1)
percent_y0 = (np.array(temp_y0) / total) * 100
percent_y1 = (np.array(temp_y1) / total) * 100

# Set up bar positions
x = np.arange(len(temp.index))
width = 0.35

# Create the plot
plt.figure(figsize=(12, 6))
plt.bar(x - width/2, percent_y1, width, label='YES', color='tab:blue')
plt.bar(x + width/2, percent_y0, width, label='NO', color='tab:orange')

# Add labels and title
plt.xlabel('Family Status')
plt.ylabel('Count in %')
plt.title("Family Status of Applicants in terms of loan repaid or not (%)")
plt.xticks(ticks=x, labels=temp.index, rotation=45, ha='right')
plt.legend(title='Loan Repaid')
plt.tight_layout()
plt.show()



# Get occupation type counts
temp = application_train["OCCUPATION_TYPE"].value_counts()

# Initialize lists for TARGET = 0 and 1
temp_y0 = []
temp_y1 = []

# Populate counts for each occupation type
for val in temp.index:
    temp_y1.append(np.sum(application_train["TARGET"][application_train["OCCUPATION_TYPE"] == val] == 1))
    temp_y0.append(np.sum(application_train["TARGET"][application_train["OCCUPATION_TYPE"] == val] == 0))

# Convert to percentages within each group
total = np.array(temp_y0) + np.array(temp_y1)
percent_y0 = (np.array(temp_y0) / total) * 100
percent_y1 = (np.array(temp_y1) / total) * 100

# Set up bar positions
x = np.arange(len(temp.index))
width = 0.35

# Create the plot
plt.figure(figsize=(12, 6))
plt.bar(x - width/2, percent_y1, width, label='YES', color='tab:blue')
plt.bar(x + width/2, percent_y0, width, label='NO', color='tab:orange')

# Add labels and title
plt.xlabel("Occupation of Applicant's")
plt.ylabel('Count in %')
plt.title("Occupation of Applicants in terms of loan repaid or not (%)")
plt.xticks(ticks=x, labels=temp.index, rotation=45, ha='right')
plt.legend(title='Loan Repaid')
plt.tight_layout()
plt.show()



# Get education type counts
temp = application_train["NAME_EDUCATION_TYPE"].value_counts()

# Initialize lists for TARGET = 0 and 1
temp_y0 = []
temp_y1 = []

# Populate counts for each education type
for val in temp.index:
    temp_y1.append(np.sum(application_train["TARGET"][application_train["NAME_EDUCATION_TYPE"] == val] == 1))
    temp_y0.append(np.sum(application_train["TARGET"][application_train["NAME_EDUCATION_TYPE"] == val] == 0))

# Convert to percentages within each group
total = np.array(temp_y0) + np.array(temp_y1)
percent_y0 = (np.array(temp_y0) / total) * 100
percent_y1 = (np.array(temp_y1) / total) * 100

# Set up bar positions
x = np.arange(len(temp.index))
width = 0.35

# Create the plot
plt.figure(figsize=(12, 6))
plt.bar(x - width/2, percent_y1, width, label='YES', color='tab:blue')
plt.bar(x + width/2, percent_y0, width, label='NO', color='tab:orange')

# Add labels and title
plt.xlabel("Education of Applicant's")
plt.ylabel('Count in %')
plt.title("Education of Applicants in terms of loan repaid or not (%)")
plt.xticks(ticks=x, labels=temp.index, rotation=45, ha='right')
plt.legend(title='Loan Repaid')
plt.tight_layout()
plt.show()



# Get housing type counts
temp = application_train["NAME_HOUSING_TYPE"].value_counts()

# Initialize lists for TARGET = 0 and 1
temp_y0 = []
temp_y1 = []

# Populate counts for each housing type
for val in temp.index:
    temp_y1.append(np.sum(application_train["TARGET"][application_train["NAME_HOUSING_TYPE"] == val] == 1))
    temp_y0.append(np.sum(application_train["TARGET"][application_train["NAME_HOUSING_TYPE"] == val] == 0))

# Convert to percentages within each group
total = np.array(temp_y0) + np.array(temp_y1)
percent_y0 = (np.array(temp_y0) / total) * 100
percent_y1 = (np.array(temp_y1) / total) * 100

# Set up bar positions
x = np.arange(len(temp.index))
width = 0.35

# Create the plot
plt.figure(figsize=(12, 6))
plt.bar(x - width/2, percent_y1, width, label='YES', color='tab:blue')
plt.bar(x + width/2, percent_y0, width, label='NO', color='tab:orange')

# Add labels and title
plt.xlabel("Types of House")
plt.ylabel("Count in %")
plt.title("For which types of house higher applicants applied for loan in terms of loan repaid or not (%)")
plt.xticks(ticks=x, labels=temp.index, rotation=45, ha='right')
plt.legend(title='Loan Repaid')
plt.tight_layout()
plt.show()



# Get organization type counts
temp = application_train["ORGANIZATION_TYPE"].value_counts()

# Initialize lists for TARGET = 0 and 1
temp_y0 = []
temp_y1 = []

# Populate counts for each organization type
for val in temp.index:
    temp_y1.append(np.sum(application_train["TARGET"][application_train["ORGANIZATION_TYPE"] == val] == 1))
    temp_y0.append(np.sum(application_train["TARGET"][application_train["ORGANIZATION_TYPE"] == val] == 0))

# Convert to percentages within each group
total = np.array(temp_y0) + np.array(temp_y1)
percent_y0 = (np.array(temp_y0) / total) * 100
percent_y1 = (np.array(temp_y1) / total) * 100

# Set up bar positions
x = np.arange(len(temp.index))
width = 0.35

# Create the plot
plt.figure(figsize=(14, 6))
plt.bar(x - width/2, percent_y1, width, label='YES', color='tab:blue')
plt.bar(x + width/2, percent_y0, width, label='NO', color='tab:orange')

# Add labels and title
plt.xlabel("Types of Organizations")
plt.ylabel("Count in %")
plt.title("Types of Organizations in terms of loan repaid or not (%)")
plt.xticks(ticks=x, labels=temp.index, rotation=90, fontsize=10)
plt.legend(title='Loan Repaid')
plt.tight_layout()
plt.show()


# Get suite type counts
temp = application_train["NAME_TYPE_SUITE"].value_counts()

# Initialize lists for TARGET = 0 and 1
temp_y0 = []
temp_y1 = []

# Populate counts for each suite type
for val in temp.index:
    temp_y1.append(np.sum(application_train["TARGET"][application_train["NAME_TYPE_SUITE"] == val] == 1))
    temp_y0.append(np.sum(application_train["TARGET"][application_train["NAME_TYPE_SUITE"] == val] == 0))

# Convert to percentages within each group
total = np.array(temp_y0) + np.array(temp_y1)
percent_y0 = (np.array(temp_y0) / total) * 100
percent_y1 = (np.array(temp_y1) / total) * 100

# Set up bar positions
x = np.arange(len(temp.index))
width = 0.35

# Create the plot
plt.figure(figsize=(12, 6))
plt.bar(x - width/2, percent_y1, width, label='YES', color='tab:blue')
plt.bar(x + width/2, percent_y0, width, label='NO', color='tab:orange')

# Add labels and title
plt.xlabel("Name of type of the Suite")
plt.ylabel("Count in %")
plt.title("Distribution of Suite Types in terms of loan repaid or not (%)")
plt.xticks(ticks=x, labels=temp.index, rotation=45, ha='right')
plt.legend(title='Loan Repaid')
plt.tight_layout()
plt.show()



# Get suite type counts
temp = application_train["NAME_TYPE_SUITE"].value_counts()

# Initialize lists for TARGET = 0 and 1
temp_y0 = []
temp_y1 = []

# Populate counts for each suite type
for val in temp.index:
    temp_y1.append(np.sum(application_train["TARGET"][application_train["NAME_TYPE_SUITE"] == val] == 1))
    temp_y0.append(np.sum(application_train["TARGET"][application_train["NAME_TYPE_SUITE"] == val] == 0))

# Convert to percentages within each group
total = np.array(temp_y0) + np.array(temp_y1)
percent_y0 = (np.array(temp_y0) / total) * 100
percent_y1 = (np.array(temp_y1) / total) * 100

# Set up bar positions
x = np.arange(len(temp.index))
width = 0.35

# Create the plot
plt.figure(figsize=(12, 6))
plt.bar(x - width/2, percent_y1, width, label='YES', color='tab:blue')
plt.bar(x + width/2, percent_y0, width, label='NO', color='tab:orange')

# Add labels and title
plt.xlabel("Name of type of the Suite")
plt.ylabel("Count in %")
plt.title("Distribution of Suite Types in terms of loan repaid or not (%)")
plt.xticks(ticks=x, labels=temp.index, rotation=45, ha='right')
plt.legend(title='Loan Repaid')
plt.tight_layout()
plt.show()



import matplotlib.pyplot as plt
import seaborn as sns

# Select only numeric columns
numeric_df = application_train.select_dtypes(include=[np.number])

# Compute correlation matrix
corr_matrix = numeric_df.corr()

# Plot the heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, cmap='viridis', annot=False, fmt=".2f", linewidths=0.5)

plt.title('Pearson Correlation of Features')
plt.tight_layout()
plt.show()

