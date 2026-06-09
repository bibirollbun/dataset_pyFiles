import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score , roc_curve 
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')


responders = pd.read_csv('../input/jane-street-real-time-market-data-forecasting/responders.csv')


responders.head()


responders.info()


responders.describe()


responders_new = responders.drop(columns=['responder'])


responders_new.head()


sns.heatmap(responders_new.corr(), annot=True,fmt='0.1f')


for column in responders_new.columns:
    sns.countplot(x=column, data=responders_new)
    plt.show()


x = responders_new.drop(columns=['tag_0'])
y = responders_new['tag_0']


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)


model = LinearRegression()


model.fit(x_train, y_train)  
pre = model.predict(x_test)


pre=model.predict(x_test)


roc_score=  roc_auc_score(y_test,pre)
r_c=  roc_curve(y_test,pre)
print(model)
print('roc_ is :',roc_score)
df = pd.DataFrame(r_c)
sns.heatmap(df)
plt.show()

