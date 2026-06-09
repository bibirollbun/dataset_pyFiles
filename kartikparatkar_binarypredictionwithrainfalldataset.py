import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


train = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv')


test.isnull().sum()


from sklearn.impute import SimpleImputer


imputer = SimpleImputer(strategy="mean")
test = imputer.fit_transform(test)


x_test=test


train.info()


train.head()


print(type(test))


x_train = train.drop(columns = ["rainfall"])
y_train = train["rainfall"]


from sklearn.preprocessing import StandardScaler


scaler=StandardScaler()


x_train=scaler.fit_transform(x_train)
x_test=scaler.fit_transform(x_test)


from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay


svm_model=SVC(kernel='linear',C=1.0)
svm_model.fit(x_train,y_train)


y_pred = svm_model.predict(x_test)


print(type(test))


test=pd.DataFrame(test)



result = pd.DataFrame({"id": test.index,"rainfall": y_pred})


result.to_csv("output.csv",index=False)


print("Output of the problem is saved in output.scv file")


output= pd.read_csv('/kaggle/working/output.csv')


print(output)


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the CSV file
df = pd.read_csv("output.csv")

# Scatter plot
plt.figure(figsize=(8, 5))
sns.scatterplot(x=df.index, y=df["rainfall"], alpha=0.6)
plt.xlabel("Sample Index")
plt.ylabel("Predicted Target (0 or 1)")
plt.title("Scatter Plot of ML Predictions")
plt.yticks([0, 1])  # Ensure only 0 and 1 on Y-axis
plt.show()




output.head()











