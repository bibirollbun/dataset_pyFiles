import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns


train_df = pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv", index_col='id')
orginal_df = pd.read_csv("/kaggle/input/calories-burnt-prediction/calories.csv", index_col='User_ID').rename_axis('id')
test_df = pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv", index_col='id')


train_df.shape, orginal_df.shape, test_df.shape


train_df.shape


train_df.head()


orginal_df.head()


orginal_df = orginal_df.rename(columns={'Gender': 'Sex'})


train_df.info() 


train_df.describe().T


orginal_df.describe().T


data = pd.concat([train_df, orginal_df], axis=0)
data


numeric_data = data.select_dtypes(include='number')

sns.heatmap(numeric_data.corr(),annot=True,cmap='RdYlGn',linewidths=0.2)  
fig=plt.gcf()
fig.set_size_inches(10,8)
plt.show()


plt.figure(figsize=(12,5))
plt.title("Distribution of Calories")
ax = sns.distplot(data["Calories"], color = 'r')


data.groupby([ 'Sex'])['Calories'].agg(['mean', 'median', 'std'])


sns.boxplot(data=data, x='Sex', y='Calories', palette='Set2')
plt.title('Calories Burnt by Gender')
plt.show()


 
plt.figure(figsize=(10, 5))

sns.kdeplot(data=data, x='Calories', hue='Sex', fill=True, common_norm=False, palette='Set2', alpha=0.5, linewidth=2)

plt.title('Distribution of Calories Burnt by Gender')
plt.xlabel('Calories')
plt.ylabel('Density')
plt.show()



plt.figure(figsize=(12,5))
plt.title("Distribution of age")
ax = sns.distplot(data["Age"], color = 'g')


bins = [20, 30, 40, 50, 60, 70, 80]
labels = ['A (20-29)', 'B (30-39)', 'C (40-49)', 'D (50-59)', 'E (60-69)', 'F (70-79)']
data['Age_Group'] = pd.cut(data['Age'], bins=bins, labels=labels, right=False)
 

data.groupby(['Age_Group'])['Calories'].agg(['mean', 'median', 'std'])


plt.figure(figsize=(12,6))
sns.boxplot(data=data, x='Age_Group', y='Calories', hue='Sex', palette='Set2')
plt.title('Calories Burnt by Age Group and Gender')
plt.xlabel('Age Group')
plt.ylabel('Calories')
plt.legend(title='Sex')
plt.show()


data.groupby(['Age_Group', 'Sex'])['Calories'].agg(['mean', 'median', 'std'])


plt.figure(figsize=(12,5))
plt.title("Distribution of duration")
ax = sns.distplot(data["Duration"], color = 'g')


sns.regplot(x="Duration", y="Calories", data=data)
plt.title("Calories vs Duratio")


# how well a simple linear regression model fits the data
sns.residplot(x="Duration", y="Calories", data=data)
plt.axhline(0, color='gray', linestyle='--')
plt.title("Residual Plot: Calories vs Duration")


sns.scatterplot(x=data['Duration'], y=data['Calories'], hue=data['Sex'], palette="Set1")


 
sns.set(style="whitegrid")
g = sns.FacetGrid(data, col="Sex", height=5, aspect=1.2)
g.map_dataframe(sns.scatterplot, x="Duration", y="Calories", hue="Age_Group", palette="viridis", alpha=0.7)
 
g.set_axis_labels("Duration (min)", "Calories Burned")
g.add_legend(title="Duration")
g.fig.suptitle("Calories Burned vs Duration by Sex, colored by Age_Group", fontsize=16, y=1.05)
plt.tight_layout()
plt.show()



 
sns.set(style="whitegrid")
 
g = sns.FacetGrid(data, col="Age_Group", height=5, aspect=1.2)
g.map_dataframe(sns.scatterplot, x="Duration", y="Calories", hue="Sex", palette="viridis", alpha=0.7)
 
g.set_axis_labels("Duration (min)", "Calories Burned")
g.add_legend(title="Age")

g.fig.suptitle("Calories Burned vs Duration by Age_Group, colored by Sex", fontsize=16, y=1.05)
plt.tight_layout()
plt.show()



 
bins = [1, 5, 10, 15, 20, 25, 30]
labels = [
    'D1 (1–5)', 
    'D2 (5–10)', 
    'D3 (10–15)', 
    'D4 (15–20)', 
    'D5 (20–25)', 
    'D6 (25–30)'
]
 
data['Duration_bin'] = pd.cut(data['Duration'], bins=bins, labels=labels, right=False)
 



plt.figure(figsize=(12,5))
plt.title("Distribution of Heart_Rate")
ax = sns.distplot(data["Heart_Rate"], color = 'r')


sns.regplot(x="Heart_Rate", y="Calories", data=data)
plt.title("Calories vs Heart_Rate")



sns.residplot(x="Heart_Rate", y="Calories", data=data)
plt.axhline(0, color='gray', linestyle='--')
plt.title("Residual Plot: Calories vs Heart_Rate")


sns.scatterplot(x=data['Heart_Rate'], y=data['Calories'], hue=data['Sex'], palette="Set1")


plt.figure(figsize=(12,5))
plt.title("Distribution of Body_Temp")
ax = sns.distplot(data["Body_Temp"], color = 'b')


sns.scatterplot(x=data['Body_Temp'], y=data['Calories'], hue=data['Sex'], palette="Set1")


sns.regplot(x="Body_Temp", y="Calories", data=data)
plt.title("Calories vs Heart_Rate")


sns.residplot(x="Body_Temp", y="Calories", data=data)
plt.axhline(0, color='gray', linestyle='--')
plt.title("Residual Plot: Calories vs Body_Temp")


plt.figure(figsize=(12,5))
plt.title("Distribution of Height")
ax = sns.distplot(data["Height"], color = 'r')


sns.scatterplot(x=data['Height'], y=data['Calories'], palette="Set1")


plt.figure(figsize=(12,5))
plt.title("Distribution of Weight")
ax = sns.distplot(data["Weight"], color = 'g')


sns.scatterplot(x=data['Weight'], y=data['Calories'], palette="Set1")


 
plt.figure(figsize=(8, 6))
scatter = plt.scatter(data['Height'], data['Weight'], c=data['Calories'], cmap='viridis', alpha=0.7)
plt.xlabel('Height')
plt.ylabel('Weight')
plt.title('Calories Burned by Height and Weight')
cbar = plt.colorbar(scatter)
cbar.set_label('Calories Burned')
plt.grid(True)
plt.show()



height_bins = [126, 150, 160, 165, 170, 175, 180, 185, 223]  
weight_bins = [36, 50, 60, 70, 80, 90, 133]   
 
data['Height_bin'] = pd.cut(data['Height'], bins=height_bins, labels=['126-150', '150-160', '160-165', '165-170', '170-175', '175-180', '180-185', '185-223'], right=False)
data['Weight_bin'] = pd.cut(data['Weight'], bins=weight_bins, labels=['36-50', '50-60', '60-70', '70-80', '80-90', '90-133'], right=False)


sexes = data['Sex'].unique()
n = len(sexes)

fig, axes = plt.subplots(1, n, figsize=(6*n, 5), sharey=True)

for i, sex in enumerate(sexes):
    subset = data[data['Sex'] == sex]

    pivot = subset.pivot_table(index='Height_bin', columns='Weight_bin', values='Calories', aggfunc='mean')
    
    sns.heatmap(pivot, ax=axes[i], cmap='coolwarm', annot=True, fmt=".1f", cbar=i==n-1)
    axes[i].set_title(f"Sex: {sex}")
    axes[i].set_xlabel("Weight Bin")
    axes[i].set_ylabel("Height Bin")

plt.suptitle("Mean Calories Burned by Height and Weight Bins (Per Sex)", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

