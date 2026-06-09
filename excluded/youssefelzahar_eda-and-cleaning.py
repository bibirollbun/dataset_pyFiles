import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


data=pd.read_csv('/kaggle/input/equity-post-HCT-survival-predictions/train.csv')
data.columns


data.replace([float('inf'), -float('inf')], pd.NA, inplace=True)



number_data=data.select_dtypes(["int64","float64"])
cat_data=data.select_dtypes(["object"])


number_data=number_data.fillna(number_data.mean())
cat_data = cat_data.apply(lambda col: col.fillna(col.mode()[0]))



data=pd.concat([number_data,cat_data],axis=1)


cat_data.isnull().sum()


sns.histplot(data['donor_age'],bins=30,kde=True,color='blue')
plt.title=("Distribution of Donnr Age")
plt.xlabel("Donner Age")
plt.ylabel("frequence")
plt.show()


sns.scatterplot(data=data,x='efs_time',y='cyto_score',hue='hla_match_c_high')
#plt.title('EFS Time vs. Cytogenetic Score')
plt.show()


sns.boxplot(data=data,x='diabetes',y="efs_time")
#plt.title("efs time vs diabetes groups")
plt.show()


data["diabetes"]=data["diabetes"].replace({"Yes":1,"No":0,"Not done":2})
data["obesity"]=data["obesity"].replace({"Yes":1,"No":0,"Not done":2})
data["prior_tumor"]=data["prior_tumor"].replace({"Yes":1,"No":0,"Not done":2})



binary_vars = ['diabetes', 'obesity', 'prior_tumor']
corr = data[binary_vars].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
#plt.title('Correlation Heatmap for Binary Variables')
plt.show()



sns.violinplot(data=data,x="race_group",y="efs_time",palette='muted')
plt.show()


data.groupby('year_hct')['efs'].mean().plot(kind='line')
plt.show()


from sklearn import preprocessing 
#data["dri_score"].unique()
data["dri_score"]=data["diabetes"].replace({"Yes":1,"No":0,"Not done":2})
label=preprocessing.LabelEncoder()
data["dri_score"]=label.fit_transform(data["dri_score"])
data["dri_score"].unique()


num_vars = ['efs_time', 'donor_age', 'karnofsky_score', 'dri_score']
corr=data[num_vars].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.show()


sns.pairplot(data=data,vars=['efs_time','donor_age','dri_score'],hue='race_group')
plt.show()

