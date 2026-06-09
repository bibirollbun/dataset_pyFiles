#importing pandas for exploring data and matplotlib for visualisations
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
print("Imported libraries pandas",pd.__version__,"matplotlib ",matplotlib.__version__)


#load the training data
data_df= pd.read_csv("/kaggle/input/map-charting-student-math-misunderstandings/train.csv") #path to data as input


#print top 5 elemnents of data 
data_df.head(5)


#displaying feature columns
print(data_df.columns)
print("no of feature columns: {}".format(len(data_df.columns)))


#describing data
print("no of available data records :{}".format(len(data_df)))


print(data_df.shape)


#understand data types of data
print(data_df.dtypes)


#data columns info
print(data_df.info())


#identify count of null values
print(data_df.isnull().sum())


#identify duplicate values
print(data_df.duplicated().sum())


#unique values in each column
print(data_df.nunique())


#lets plot and visualize distribution of data in each misconception category
category_counts=data_df["Category"].value_counts()
categories=category_counts.index
values=category_counts.values
plt.figure(figsize=(10, 6))
plt.bar(categories,values,color='red')
plt.xlabel('Category')
plt.ylabel('count')
plt.tight_layout()
plt.title('Counts of Each Category')



#visualize distribution of Misconceptions
misconception_counts=data_df['Misconception'].value_counts()
misconceptions=misconception_counts.index
misconceptions_values=misconception_counts.values
plt.figure(figsize=(16,10))
plt.bar(misconceptions,misconceptions_values)
plt.xticks(rotation=90)
plt.xlabel("Misconceptions")
plt.ylabel("count")
plt.tight_layout()
plt.title("Misconceptions data distribution")



#lets seperate category to two columns one having true/false of answer and another columns having whether explaination category
data_df["Answer_Category"]=data_df["Category"].apply(lambda x: x.split("_")[0])
data_df["Explation_Category"]=data_df["Category"].apply(lambda x:x.split("_")[1])


data_df.head(4)


#let's visualize distributions of explanation category and misconceptions based on answer category
true_df=data_df[data_df['Answer_Category']=="True"]
false_df=data_df[data_df['Answer_Category']=="False"]
true_df.shape



#lets plot and visualize distribution of Explanation Category data in True answer category
category_counts=true_df["Explation_Category"].value_counts()
categories=category_counts.index
values=category_counts.values
plt.figure(figsize=(10, 6))
plt.bar(categories,values,color='red')
plt.xlabel('Explanation Category')
plt.ylabel('count')
plt.tight_layout()
plt.title('Counts of Each explanation Category When Student Answer is True')


#lets plot and visualize distribution of Explanation Category data in True answer category
category_counts=false_df["Explation_Category"].value_counts()
categories=category_counts.index
values=category_counts.values
plt.figure(figsize=(10, 6))
plt.bar(categories,values,color='green')
plt.xlabel('Explanation Category')
plt.ylabel('count')
plt.tight_layout()
plt.title('Counts of Each Category when student answer is false')

