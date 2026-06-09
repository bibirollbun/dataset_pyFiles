import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns                                                    

train_df = pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv")  
                                                                         

print(f"Dataset Shape: {train_df.shape}")                                 
                                                                          

print("\nData Info:")
train_df.info()                                                           


print("\nNumerical Features Summary:")
display(train_df.describe()) 


print("\nFirst 10 Rows of the Dataset:")
display(train_df.head())


numerical_features = ["Age","Height","Weight","Duration","Heart_Rate","Body_Temp","Calories"]
                                                        
for feature in numerical_features:
    plt.figure(figsize=(12, 5))                         

    plt.subplot(1, 2, 1)                                 
                                                        
                                                        
    
    sns.histplot(train_df[feature], kde=True, bins=30)  
                                                        
                                                        
                                                        
    plt.title(f"Histogram of {feature}")                
    plt.xlabel(feature)                                 
    plt.ylabel("Frequency")                             

    plt.subplot(1, 2, 2)                                
                                                        
    
    sns.boxplot(x=train_df[feature])                    

    plt.tight_layout()                                  
    plt.show()                                          

    print(f"\nStatistics for {feature}:")                                    
    print(f"Skewness: {train_df[feature].skew():.2f}")                       
    print(f"Number of Missing Values: {train_df[feature].isnull().sum()}")


sex_counts = train_df["Sex"].value_counts()  
plt.figure(figsize=(4, 4))       
plt.pie(sex_counts, labels=sex_counts.index, autopct='%1.4f%%', startangle=90) 



plt.title("Distribution of Sex")
plt.axis("equal")
plt.show()

print(f"Number of Unique {feature}: {train_df[feature].nunique()}")           
                            
print(f"Missing Values in {feature}: {train_df[feature].isnull().sum()}")   


import seaborn as sns
import matplotlib.pyplot as plt

colors = sns.color_palette('flare', len(numerical_features))    

rows = -(-len(numerical_features) // 4)                         
plt.figure(figsize=(20, 5 * rows))                              

for i, (col, color) in enumerate(zip(numerical_features, colors), 1):
    plt.subplot(rows, 4, i)
    sns.kdeplot(data=train_df, x=col, fill=True, color=color)   
                                                               
    
    plt.title(f'KDE Plot of {col}', fontsize=14, color=color)   
    plt.xlabel(col)
    plt.ylabel('Density')

plt.tight_layout()
plt.show()


import seaborn as sns
import matplotlib.pyplot as plt

numeric_df = train_df.select_dtypes(include='number')              

sns.pairplot(numeric_df, corner=True, plot_kws={'alpha': 0.25})    
                                 
plt.suptitle('Scatter Plots', y=1)                                 
        
plt.show()

