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
import matplotlib.pyplot as plt
from io import StringIO
data="""Location,Nitrate,Phosphate,Lead
Site 1,3.5,0.5,0.02
Site 2,4.1,0.7,0.03
Site 3,2.8,0.4,0.01
Site 4,3.9,0.6,0.02
Site 5,4.5,0.8,0.4
"""
df=pd.read_csv(StringIO(data)) 
class EcoWaterAgent:
    def __init__(self,dataframe):
        self.df=dataframe
    def collect_data(self):
        print("Data loaded successfully!\n") 
    def analyze_data(self):
        print("Data Analysis:\n") 
        print(self.df.describe()) 
    def report(self):
        print("\n Visualization:") 
        self.df[["Nitrate","Phosphate","Lead"]].plot(marker="o") 
        plt.title("Water Quality Indicators") 
        plt.xlabel("Samples") 
        plt.ylabel("Level") 
        plt.grid(True) 
        plt.show() 
    def run(self):
        self.collect_data() 
        self.analyze_data() 
        self.report() 
agent=EcoWaterAgent(df) 
agent.run()


mean_values=df.mean(numeric_only=True) 
plt.figure(figsize=(8,5)) 
plt.bar(mean_values.index,mean_values.values) 
plt.xticks(rotation=45) 
plt.title("Average Water Pollutant Levels") 
plt.xlabel("Pollutant") 
plt.ylabel("Average Value") 
plt.tight_layout() 
plt.show() 


with open("submission.txt","w") as f:f.write("EcoWaterAgent submission completed successfully.") 
print("Submission file created!")


print("Zeinab test") 


import pandas as pd
import matplotlib.pyplot as plt
from io import StringIO
data =""" Location,Nitrate,Phosphate,Lead
Site 1,3.5,0.5,0.02
Site 2,4.1,0.7,0.03
Site 3,2.8,0.4,0.01
Site 4,3.9,0.6,0.02
Site 5,4.5,0.8,0.04
"""
df=pd.read_csv(StringIO(data)) 
class EcoWaterAgent:
    def __init__ (self,dataframe):
        self.df=dataframe
    def collect_data(self): 
        print("تم تحميل البيانات!\n") 
    def analyze_data(self):
        print("تحليل البيانات:\n") 
        print(self.df.describe()) 
    def report(self):
        print("\nعرض رسم بياني:") 
        self.df[["Nitrate","Phosphate","Lead"]].plot(marker="o") 
        plt.title("Water Quality Indicators") 
        plt.xlabel("Samples") 
        plt.ylabel("Level") 
        plt.grid(True) 
        plt.show() 
    def run(self):
        self.collect_data() 
        self.analyze_data() 
        self.report() 
agent=EcoWaterAgent(df) 
agent.run()


agent.report()


import matplotlib.pyplot as plt
mean_values=df.mean(numeric_only=True) 
plt.figure(figsize=(8,5)) 
plt.bar(mean_values.index,mean_values.values) 
plt.xticks(rotation=45) 
plt.title("Average Water Pollutant Levels") 
plt.xlabel("Pollutant") 
plt.ylabel("Average Value") 
plt.tight_layout() 
plt.show() 




