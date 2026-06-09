import pandas as pd 
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt


train = pd.read_csv("/kaggle/input/playground-series-s5e3/train.csv")
train 


original = train.copy()


train.drop("id",axis=1, inplace = True)


train 


for label,content in train.items():
    if pd.api.types.is_numeric_dtype(content):
        print(f'Here is the Label Name : {label}')
        print(train[label].value_counts())
        print("--"*40)


train.isnull().sum()


train.head()


train.describe().T


import warnings
warnings.filterwarnings("ignore")



sns.set_style("white")


# lets check out the skewed data 
for label, content in train.items():
    if pd.api.types.is_numeric_dtype(content):
        plt.subplots(figsize=(9,6))
        sns.distplot(train[label],bins = 10,kde= False,color = "darkblue")
        plt.ylabel("Frequency")



# Function to map wind direction degrees to cardinal directions
def wind_direction_label(degrees):
    directions = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
                  "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW", "N"]
    return directions[int(np.round(degrees / 22.5)) % 16]

# Convert wind direction degrees to labels
train["winddirection_label"] = train["winddirection"].apply(wind_direction_label)




plt.subplots(figsize=(20,10))
sns.boxplot(x=train["winddirection_label"], y=train["windspeed"], color="orange")
plt.title("DIRECTION WISE WIND SPEED", fontsize=12, fontweight='bold')
plt.show()



plt.subplots(figsize=(20,15))
sns.scatterplot(x=train["temparature"], y=train["humidity"],color="green")
plt.title("TEMPERATURE vs HUMIDITY", fontsize=12, fontweight='bold')
plt.show()



plt.subplots(figsize=(30,20))
sns.boxplot(x=train["cloud"], y=train["temparature"],color="#4169E1")
plt.title("TEMPERATURE DISTRIBUTION ACROSS CLOUD COVER", fontsize=15, fontweight='bold')
plt.xlabel('Cloud', fontsize=15)        # X-axis label size
plt.ylabel('Temperature (°C)', fontsize=15)  # Y-axis label size
plt.xticks(range(0,95,15),fontsize=15)  # X-axis tick labels size
plt.yticks(fontsize=15)  # Y-axis tick labels size
plt.show()




# Create the pairplot
g = sns.pairplot(train[["pressure", "temparature", "humidity", "windspeed"]], height=5)

# Increase tick size for all subplots
for ax in g.axes.flatten():  
    if ax is not None:
        ax.tick_params(axis='both', labelsize=10)  # Adjust label size (Increase for bigger ticks)
plt.show()






plt.figure(figsize=(25,20))
sns.heatmap(original.corr(), annot=True, cmap="coolwarm", linewidths=0.5)
plt.title("CORRELATION ANALYSIS", fontsize=12, fontweight='bold')
plt.show()



plt.subplots(figsize=(30,20))
sns.lineplot(x=train["day"], y=train["temparature"])
plt.show()





