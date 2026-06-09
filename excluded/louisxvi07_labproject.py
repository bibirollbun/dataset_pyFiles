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


import warnings
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image as PILImage
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
warnings.filterwarnings("ignore")


img = PILImage.open("/kaggle/input/images-for-ppt/kAGGLE/import pandas.webp")
plt.imshow(img)
plt.axis('off')  
plt.show()


try:
    df = pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv")

except Exception as e:
    print("error occurred:", e)
df.head()


df.columns


df.info()


df.describe()


df["Sex"].value_counts()


df["Height"].mean()


df[df["Age"]<30].head()


df[df["Sex"]=="male"]["Height"].mean()


df.head()


d={"female":0, "male":1}
df['Sex']=df['Sex'].map(d)
df.head()


feat=[ "Sex","Age", "Height", "Weight", "Duration","Heart_Rate", "Body_Temp" ]
featt=[ "Sex","Age", "Height", "Weight", "Duration","Heart_Rate", "Body_Temp", "Calories"]


img = PILImage.open("/kaggle/input/images-for-ppt/kAGGLE/list.webp")
plt.imshow(img)
plt.axis('off')  
plt.show()


df[feat].hist(figsize=(10,9));


img = PILImage.open("/kaggle/input/images-for-ppt/kAGGLE/i_p_c_191919s_630q_90.webp")
plt.imshow(img)
plt.axis('off')  
plt.show()


cor_m=df[featt].corr()
sns.heatmap(cor_m, annot=True, annot_kws={"size": 10})


nc=df["Weight"]/((df["Height"]/100)**2)
df.insert(loc=len(df.columns)-1, column= "BMI", value=nc)


df.head()


feat.append("BMI")
featt.append("BMI")
featt[-1] = "Calories"
featt[-2] = "BMI"



cor_m=df[featt].corr()
sns.heatmap(cor_m, annot=True, annot_kws={"size": 10})


df.info()


img = PILImage.open("/kaggle/input/images-for-ppt/kAGGLE/graph.webp")
plt.imshow(img)
plt.axis('off')  # Hide axes
plt.show()


sns.set(style="whitegrid", palette="muted", font_scale=1.1)


plt.figure(figsize=(8,5))
sns.histplot(data=df, x="Age", bins=30, hue="Sex")
plt.title("Age Distribution by Sex")
plt.show()



plt.figure(figsize=(7,5))
sns.histplot(df["BMI"], bins=30, color='teal',kde=True)
plt.title("Distribution of BMI")
plt.show()



plt.figure(figsize=(8,6))
sns.scatterplot(data=df, x="BMI", y="Calories", hue="Sex", alpha=0.5)
plt.title("Calories vs BMI by Sex")
plt.show()



img = PILImage.open("/kaggle/input/images-for-ppt/kAGGLE/i am machine keaerning.webp")
plt.imshow(img)
plt.axis('off')  
plt.show()


import random


target="Calories"


for col in featt:
    if col not in df.columns:
        raise ValueError(f"Missing column: {col}")


X = df[feat].values
y = df[target].values


split_index = int(0.8 * len(X))
X_train, X_val = X[:split_index], X[split_index:]
y_train, y_val = y[:split_index], y[split_index:]
print(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")


weights = [random.uniform(-1, 1) for _ in range(len(feat))] 
bias = random.uniform(-1, 1)
print(f"Initial Weights: {weights}, Bias: {bias:.3f}")


predict = lambda X, W, b: np.dot(X, W) + b


def mean_absolute_error(y_true, y_pred):
    errors = [abs(a - b) for a, b in zip(y_true, y_pred)]
    return sum(errors) / len(errors)


best_weights = weights.copy()
best_bias = bias
best_mae = float("inf")


for epoch in range(50): 
    new_weights = [w + random.uniform(-0.05, 0.05) for w in best_weights]
    new_bias = best_bias + random.uniform(-0.05, 0.05)

    y_pred = predict(X_train, new_weights, new_bias)
    mae = mean_absolute_error(y_train, y_pred)

    if mae < best_mae:  
        best_mae = mae
        best_weights = new_weights
        best_bias = new_bias

    if epoch % 10 == 0:
        print(f"Epoch {epoch:02d} | MAE: {mae:.3f}")


print(f"Best MAE: {best_mae:.3f}")
print(f"Final Weights: {best_weights}")
print(f"Final Bias: {best_bias:.3f}")


y_val_pred = predict(X_val, best_weights, best_bias)


mean_absolute_error(y_val_pred,y_val)


plt.figure(figsize=(14, 6))  
plt.bar(feat, np.abs(best_weights), color='teal', edgecolor='black')
plt.title("Feature Importance (|Weight|)")
plt.xlabel("Features")
plt.ylabel("Absolute Weight Value")
plt.show()


img = PILImage.open("/kaggle/input/images-for-ppt/kAGGLE/stop-using-linear-regression.webp")
plt.imshow(img)
plt.axis('off')  
plt.show()


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


model = LinearRegression()
model.fit(X_train, y_train)


y_pred = model.predict(X_test)


mae=mean_absolute_error(y_pred,y_test)
mae


plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred, alpha=0.5, color='teal')
plt.xlabel("Actual Calories")
plt.ylabel("Predicted Calories")
plt.title("Predicted vs Actual Calories (Linear Regression)")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')  
plt.show()




