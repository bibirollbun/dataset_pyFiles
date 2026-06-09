import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns


tmp1 = pd.read_csv("/kaggle/input/rainfall-prediction-using-machine-learning/Rainfall.csv")
ori = pd.read_csv("/kaggle/input/hongkongrainfall/hongkong.csv",encoding="gbk")
tmp2 = ori.copy()


#Remove spaces from column names.
tmp1.columns = [col.replace(" ", "") for col in tmp1.columns]

# replace  values to approximate the original dataset.
tmp1.rainfall = tmp1.rainfall.replace({'yes': 1, 'no': 0}).astype(int)
tmp2.rainfall = tmp2.rainfall.apply(lambda x: 1 if str(x).replace('.', '', 1).isdigit() else x)  # 将数字字符替换为1
tmp2.rainfall = tmp2.rainfall.replace({'微量': 1, '-': 0}).astype(int)
tmp2.sunshine = tmp2.sunshine.replace('-', 0).astype(float)





print(tmp2[(tmp2.year==2016)&(tmp2.month<8)].windspeed.unique(),tmp2[(tmp2.year==2015)&(tmp2.month>=8)].windspeed.unique())

print("there is no nan value!")


for col in tmp1.columns:
    check_col = col
    print("\n",check_col,":")
    print("The number of  the original dataset from 08 to 12 different from the data from 08 to 12 2015:")
    print("is",((tmp1[check_col][213:].values)!=tmp2[(tmp2.year==2015)&(tmp2.month>=8)][check_col].values).sum())
    
    print("The number of  the original dataset from 01 to 07 different from the data from 01 to 07 2016:")
    print("is",((tmp1[check_col][:213].values)!=tmp2[(tmp2.year==2016)&(tmp2.month<8)][check_col].values).sum())



tmp3 = pd.concat([ori[(ori.year==2016)&(ori.month<8)],ori[(ori.year==2015)&(ori.month>=8)]],axis=0)
tmp3.rainfall = tmp3.rainfall.apply(lambda x:"yes" if str(x).replace('.', '', 1).isdigit() else x) 
tmp3.rainfall = tmp3.rainfall.replace({'微量': "micro", '-': "no"})


import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression

df = tmp1.copy()
df = df.fillna(0)
feature_columns = [col for col in df.columns if col != 'rainfall']
label_column = 'rainfall'


model = LogisticRegression()


model.fit(df[feature_columns], df[label_column])
y_total_proba = model.predict_proba(df[feature_columns])[:, 1]
total_auc = roc_auc_score(df[label_column], y_total_proba)
print(f'Total AUC on entire dataset: {total_auc:.4f}')

tmp3['pred_proba'] = y_total_proba
sns.kdeplot(data=tmp3, x='pred_proba', hue=label_column, fill=True, common_norm=False)
plt.title('Predicted Probability Density by Class')
plt.xlabel('Predicted Probability')
plt.ylabel('Density')
plt.savefig("1.jpg")
plt.show()


tmp3 = pd.concat([ori[(ori.year==2016)&(ori.month<8)],ori[(ori.year==2015)&(ori.month>=8)]],axis=0)
tmp3.rainfall = tmp3.rainfall.apply(lambda x:2 if str(x).replace('.', '', 1).isdigit() else x) 
tmp3.rainfall = tmp3.rainfall.replace({'微量': 0, '-': 1})
tmp3.sunshine = tmp3.sunshine.replace('-', 0).astype(float)


import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier


df = tmp3.copy()
df = df.fillna(0)
feature_columns = [col for col in tmp1.columns if col != 'rainfall']
label_column = 'rainfall'

X = df[feature_columns]
y = df[label_column]
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from lightgbm import LGBMClassifier


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

clf = LogisticRegression(max_iter=1000, random_state=42, multi_class='ovr')

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
class_names = ['micro', 'no', 'yes']
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()




