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


train_df = pd.read_csv("/kaggle/input/playground-series-s5e1/train.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e1/test.csv")

#datetime.strptime(datetime_str, '%y/%m/%d')
train_df['date']= pd.to_datetime(train_df['date'])
test_df['date']= pd.to_datetime(test_df['date'])
train_df.tail(10)


test_df.head(10)


train_df.describe


test_df.describe


train_df.groupby(["country"]).count()


train_df[train_df["country"]=="Canada"].groupby(["store"]).count()


train_df[train_df["country"]=="Finland"].groupby(["store","product"]).count()


train_df[train_df["country"]=="Canada"].groupby(["store","product"]).count()


test_df[test_df["country"]=="Canada"].groupby(["store","product"]).count()


train_df[(train_df["country"]=="Canada") & (train_df["store"]=="Discount Stickers") & (train_df["product"]=="Kaggle")]["date"]


# 2012, 2016 are leap years          Plot Canada, Discount Stickers, Kaggle
import matplotlib.pyplot as plt
import datetime

x = np.array(train_df[(train_df["country"]=="Canada") & (train_df["store"]=="Discount Stickers") & (train_df["product"]=="Kaggle")]["date"])
y = np.array(train_df[(train_df["country"]=="Canada") & (train_df["store"]=="Discount Stickers") & (train_df["product"]=="Kaggle")]["num_sold"])

#plt.plot(x,y)

fig, axs = plt.subplots(4, 2, figsize=(12, 4.7), layout='constrained')
axs[0][0].plot(x,y)
xx = np.array(train_df[(train_df["country"]=="Canada") & (train_df["store"]=="Discount Stickers") & (train_df["product"]=="Kaggle") & 
              (train_df["date"]<datetime.datetime(2011, 1, 1))]["date"])
yy = np.array(train_df[(train_df["country"]=="Canada") & (train_df["store"]=="Discount Stickers") & (train_df["product"]=="Kaggle") & 
              (train_df["date"]<datetime.datetime(2011, 1, 1))]["num_sold"])
axs[0][1].plot(xx,yy)
x = np.array(train_df[(train_df["country"]=="Canada") & (train_df["store"]=="Discount Stickers") & (train_df["product"]=="Kaggle") & 
              (train_df["date"]>datetime.datetime(2010, 12, 31)) & (train_df["date"]<datetime.datetime(2012, 1, 1))]["date"])
y = np.array(train_df[(train_df["country"]=="Canada") & (train_df["store"]=="Discount Stickers") & (train_df["product"]=="Kaggle") & 
              (train_df["date"]>datetime.datetime(2010, 12, 31)) & (train_df["date"]<datetime.datetime(2012, 1, 1))]["num_sold"])
axs[1][0].plot(x,y)
xx = np.array(train_df[(train_df["country"]=="Canada") & (train_df["store"]=="Discount Stickers") & (train_df["product"]=="Kaggle") & 
              (train_df["date"]>datetime.datetime(2011, 12, 31)) & (train_df["date"]<datetime.datetime(2013, 1, 1))]["date"])
yy = np.array(train_df[(train_df["country"]=="Canada") & (train_df["store"]=="Discount Stickers") & (train_df["product"]=="Kaggle") & 
              (train_df["date"]>datetime.datetime(2011, 12, 31)) & (train_df["date"]<datetime.datetime(2013, 1, 1))]["num_sold"])
axs[1][1].plot(xx,yy)
x = np.array(train_df[(train_df["country"]=="Canada") & (train_df["store"]=="Discount Stickers") & (train_df["product"]=="Kaggle") & 
              (train_df["date"]>datetime.datetime(2010, 12, 31)) & (train_df["date"]<datetime.datetime(2011, 2, 1))]["date"])
y = np.array(train_df[(train_df["country"]=="Canada") & (train_df["store"]=="Discount Stickers") & (train_df["product"]=="Kaggle") & 
              (train_df["date"]>datetime.datetime(2010, 12, 31)) & (train_df["date"]<datetime.datetime(2011, 2, 1))]["num_sold"])
axs[2][0].plot(x,y,marker='x')
xx = np.array(train_df[(train_df["country"]=="Canada") & (train_df["store"]=="Discount Stickers") & (train_df["product"]=="Kaggle") & 
              (train_df["date"]>datetime.datetime(2011, 12, 31)) & (train_df["date"]<datetime.datetime(2012, 2, 1))]["date"])
yy = np.array(train_df[(train_df["country"]=="Canada") & (train_df["store"]=="Discount Stickers") & (train_df["product"]=="Kaggle") & 
              (train_df["date"]>datetime.datetime(2011, 12, 31)) & (train_df["date"]<datetime.datetime(2012, 2, 1))]["num_sold"])
axs[2][1].plot(xx,yy,marker='x')
x = np.array(train_df[(train_df["country"]=="Canada") & (train_df["store"]=="Discount Stickers") & (train_df["product"]=="Kaggle") & 
              (train_df["date"]>datetime.datetime(2011, 1, 31)) & (train_df["date"]<datetime.datetime(2011, 3, 1))]["date"])
y = np.array(train_df[(train_df["country"]=="Canada") & (train_df["store"]=="Discount Stickers") & (train_df["product"]=="Kaggle") & 
              (train_df["date"]>datetime.datetime(2011, 1, 31)) & (train_df["date"]<datetime.datetime(2011, 3, 1))]["num_sold"])
axs[3][0].plot(x,y,marker='x')
xx = np.array(train_df[(train_df["country"]=="Canada") & (train_df["store"]=="Discount Stickers") & (train_df["product"]=="Kaggle") & 
              (train_df["date"]>datetime.datetime(2012, 1, 31)) & (train_df["date"]<datetime.datetime(2012, 3, 1))]["date"])
yy = np.array(train_df[(train_df["country"]=="Canada") & (train_df["store"]=="Discount Stickers") & (train_df["product"]=="Kaggle") & 
              (train_df["date"]>datetime.datetime(2012, 1, 31)) & (train_df["date"]<datetime.datetime(2012, 3, 1))]["num_sold"])
axs[3][1].plot(xx,yy,marker='x')
plt.show()


# 2012, 2016 are leap years            Plot Canada, Premium Sticker Mart, Holographic Goose
import matplotlib.pyplot as plt
import datetime

c="Canada"
s="Premium Sticker Mart"
p="Holographic Goose"

x = np.array(train_df[(train_df["country"]==c) & (train_df["store"]==s) & (train_df["product"]==p)]["date"])
y = np.array(train_df[(train_df["country"]==c) & (train_df["store"]==s) & (train_df["product"]==p)]["num_sold"])

#plt.plot(x,y)

fig, axs = plt.subplots(4, 2, figsize=(12, 4.7), layout='constrained')
axs[0][0].plot(x,y)
xx = np.array(train_df[(train_df["country"]==c) & (train_df["store"]==s) & (train_df["product"]==p) & 
              (train_df["date"]<datetime.datetime(2011, 1, 1))]["date"])
yy = np.array(train_df[(train_df["country"]==c) & (train_df["store"]==s) & (train_df["product"]==p) & 
              (train_df["date"]<datetime.datetime(2011, 1, 1))]["num_sold"])
axs[0][1].plot(xx,yy)
x = np.array(train_df[(train_df["country"]==c) & (train_df["store"]==s) & (train_df["product"]==p) & 
              (train_df["date"]>datetime.datetime(2010, 12, 31)) & (train_df["date"]<datetime.datetime(2012, 1, 1))]["date"])
y = np.array(train_df[(train_df["country"]==c) & (train_df["store"]==s) & (train_df["product"]==p) & 
              (train_df["date"]>datetime.datetime(2010, 12, 31)) & (train_df["date"]<datetime.datetime(2012, 1, 1))]["num_sold"])
axs[1][0].plot(x,y)
xx = np.array(train_df[(train_df["country"]==c) & (train_df["store"]==s) & (train_df["product"]==p) & 
              (train_df["date"]>datetime.datetime(2011, 12, 31)) & (train_df["date"]<datetime.datetime(2013, 1, 1))]["date"])
yy = np.array(train_df[(train_df["country"]==c) & (train_df["store"]==s) & (train_df["product"]==p) & 
              (train_df["date"]>datetime.datetime(2011, 12, 31)) & (train_df["date"]<datetime.datetime(2013, 1, 1))]["num_sold"])
axs[1][1].plot(xx,yy)
x = np.array(train_df[(train_df["country"]==c) & (train_df["store"]==s) & (train_df["product"]==p) & 
              (train_df["date"]>datetime.datetime(2010, 12, 31)) & (train_df["date"]<datetime.datetime(2011, 2, 1))]["date"])
y = np.array(train_df[(train_df["country"]==c) & (train_df["store"]==s) & (train_df["product"]==p) & 
              (train_df["date"]>datetime.datetime(2010, 12, 31)) & (train_df["date"]<datetime.datetime(2011, 2, 1))]["num_sold"])
axs[2][0].plot(x,y,marker='x')
xx = np.array(train_df[(train_df["country"]==c) & (train_df["store"]==s) & (train_df["product"]==p) & 
              (train_df["date"]>datetime.datetime(2011, 12, 31)) & (train_df["date"]<datetime.datetime(2012, 2, 1))]["date"])
yy = np.array(train_df[(train_df["country"]==c) & (train_df["store"]==s) & (train_df["product"]==p) & 
              (train_df["date"]>datetime.datetime(2011, 12, 31)) & (train_df["date"]<datetime.datetime(2012, 2, 1))]["num_sold"])
axs[2][1].plot(xx,yy,marker='x')
x = np.array(train_df[(train_df["country"]==c) & (train_df["store"]==s) & (train_df["product"]==p) & 
              (train_df["date"]>datetime.datetime(2011, 1, 31)) & (train_df["date"]<datetime.datetime(2011, 3, 1))]["date"])
y = np.array(train_df[(train_df["country"]==c) & (train_df["store"]==s) & (train_df["product"]==p) & 
              (train_df["date"]>datetime.datetime(2011, 1, 31)) & (train_df["date"]<datetime.datetime(2011, 3, 1))]["num_sold"])
axs[3][0].plot(x,y,marker='x')
xx = np.array(train_df[(train_df["country"]==c) & (train_df["store"]==s) & (train_df["product"]==p) & 
              (train_df["date"]>datetime.datetime(2012, 1, 31)) & (train_df["date"]<datetime.datetime(2012, 3, 1))]["date"])
yy = np.array(train_df[(train_df["country"]==c) & (train_df["store"]==s) & (train_df["product"]==p) & 
              (train_df["date"]>datetime.datetime(2012, 1, 31)) & (train_df["date"]<datetime.datetime(2012, 3, 1))]["num_sold"])
axs[3][1].plot(xx,yy,marker='x')
plt.show()


# Look at num_sold = nan
c="Canada"
s="Premium Sticker Mart"
p="Holographic Goose"

x = np.array(train_df[(train_df["country"]==c) & (train_df["store"]==s) & (train_df["product"]==p) &
             (train_df["date"]>datetime.datetime(2011, 5, 31)) & (train_df["date"]<datetime.datetime(2011, 7, 1))]["date"])
y = np.array(train_df[(train_df["country"]==c) & (train_df["store"]==s) & (train_df["product"]==p) &
             (train_df["date"]>datetime.datetime(2011, 5, 31)) & (train_df["date"]<datetime.datetime(2011, 7, 1))]["num_sold"])
for i in range(len(x)):
    print(x[i], y[i])


# count num_sold = nan
cty = train_df["country"].unique()
sto = train_df["store"].unique()
pro = train_df["product"].unique()
for c in cty:
    n = 0
    for s in sto:
        for p in pro:
            l = len(train_df[(train_df["country"]==c) & (train_df["store"]==s) & (train_df["product"]==p)])
            ic = 0
            df = train_df[(train_df["country"]==c) & (train_df["store"]==s) & (train_df["product"]==p)]
            for i in df.index:
                if str(df["num_sold"].loc[i]) == "nan":
                    ic = ic + 1
            print(c,s,p,l,ic)
            n = n + ic
    print(c,"nan = ",n)


# all num_sold = nan
# for, Canada, Discount Stickers, Holographic Goose
# for, Kenya, Discount Stickers, Holographic Goose
skip = []
skip.append({"cty" : "Canada", "sto" : "Discount Stickers", "pro": "Holographic Goose"})
skip.append({"cty" : "Kenya", "sto" : "Discount Stickers", "pro": "Holographic Goose"})


# 2012, 2016 are leap years            Plot Canada, Stickers for Less, Holographic Goose
import matplotlib.pyplot as plt
import datetime

c="Canada"
s="Stickers for Less"
p="Holographic Goose"

x = np.array(train_df[(train_df["country"]==c) & (train_df["store"]==s) & (train_df["product"]==p)]["date"])
y = np.array(train_df[(train_df["country"]==c) & (train_df["store"]==s) & (train_df["product"]==p)]["num_sold"])

#plt.plot(x,y)

fig, axs = plt.subplots(4, 2, figsize=(12, 4.7), layout='constrained')
axs[0][0].plot(x,y)
xx = np.array(train_df[(train_df["country"]==c) & (train_df["store"]==s) & (train_df["product"]==p) & 
              (train_df["date"]<datetime.datetime(2011, 1, 1))]["date"])
yy = np.array(train_df[(train_df["country"]==c) & (train_df["store"]==s) & (train_df["product"]==p) & 
              (train_df["date"]<datetime.datetime(2011, 1, 1))]["num_sold"])
axs[0][1].plot(xx,yy)
x = np.array(train_df[(train_df["country"]==c) & (train_df["store"]==s) & (train_df["product"]==p) & 
              (train_df["date"]>datetime.datetime(2010, 12, 31)) & (train_df["date"]<datetime.datetime(2012, 1, 1))]["date"])
y = np.array(train_df[(train_df["country"]==c) & (train_df["store"]==s) & (train_df["product"]==p) & 
              (train_df["date"]>datetime.datetime(2010, 12, 31)) & (train_df["date"]<datetime.datetime(2012, 1, 1))]["num_sold"])
axs[1][0].plot(x,y)
xx = np.array(train_df[(train_df["country"]==c) & (train_df["store"]==s) & (train_df["product"]==p) & 
              (train_df["date"]>datetime.datetime(2011, 12, 31)) & (train_df["date"]<datetime.datetime(2013, 1, 1))]["date"])
yy = np.array(train_df[(train_df["country"]==c) & (train_df["store"]==s) & (train_df["product"]==p) & 
              (train_df["date"]>datetime.datetime(2011, 12, 31)) & (train_df["date"]<datetime.datetime(2013, 1, 1))]["num_sold"])
axs[1][1].plot(xx,yy)
x = np.array(train_df[(train_df["country"]==c) & (train_df["store"]==s) & (train_df["product"]==p) & 
              (train_df["date"]>datetime.datetime(2010, 12, 31)) & (train_df["date"]<datetime.datetime(2011, 2, 1))]["date"])
y = np.array(train_df[(train_df["country"]==c) & (train_df["store"]==s) & (train_df["product"]==p) & 
              (train_df["date"]>datetime.datetime(2010, 12, 31)) & (train_df["date"]<datetime.datetime(2011, 2, 1))]["num_sold"])
axs[2][0].plot(x,y,marker='x')
xx = np.array(train_df[(train_df["country"]==c) & (train_df["store"]==s) & (train_df["product"]==p) & 
              (train_df["date"]>datetime.datetime(2011, 12, 31)) & (train_df["date"]<datetime.datetime(2012, 2, 1))]["date"])
yy = np.array(train_df[(train_df["country"]==c) & (train_df["store"]==s) & (train_df["product"]==p) & 
              (train_df["date"]>datetime.datetime(2011, 12, 31)) & (train_df["date"]<datetime.datetime(2012, 2, 1))]["num_sold"])
axs[2][1].plot(xx,yy,marker='x')
x = np.array(train_df[(train_df["country"]==c) & (train_df["store"]==s) & (train_df["product"]==p) & 
              (train_df["date"]>datetime.datetime(2011, 1, 31)) & (train_df["date"]<datetime.datetime(2011, 3, 1))]["date"])
y = np.array(train_df[(train_df["country"]==c) & (train_df["store"]==s) & (train_df["product"]==p) & 
              (train_df["date"]>datetime.datetime(2011, 1, 31)) & (train_df["date"]<datetime.datetime(2011, 3, 1))]["num_sold"])
axs[3][0].plot(x,y,marker='x')
xx = np.array(train_df[(train_df["country"]==c) & (train_df["store"]==s) & (train_df["product"]==p) & 
              (train_df["date"]>datetime.datetime(2012, 1, 31)) & (train_df["date"]<datetime.datetime(2012, 3, 1))]["date"])
yy = np.array(train_df[(train_df["country"]==c) & (train_df["store"]==s) & (train_df["product"]==p) & 
              (train_df["date"]>datetime.datetime(2012, 1, 31)) & (train_df["date"]<datetime.datetime(2012, 3, 1))]["num_sold"])
axs[3][1].plot(xx,yy,marker='x')
plt.show()


# Replace NaN values with zeros
train_df["num_sold"] = train_df["num_sold"].fillna(0)
c="Canada"
s="Premium Sticker Mart"
p="Holographic Goose"

x = np.array(train_df[(train_df["country"]==c) & (train_df["store"]==s) & (train_df["product"]==p) &
             (train_df["date"]>datetime.datetime(2011, 5, 31)) & (train_df["date"]<datetime.datetime(2011, 7, 1))]["date"])
y = np.array(train_df[(train_df["country"]==c) & (train_df["store"]==s) & (train_df["product"]==p) &
             (train_df["date"]>datetime.datetime(2011, 5, 31)) & (train_df["date"]<datetime.datetime(2011, 7, 1))]["num_sold"])
for i in range(len(x)):
    print(x[i], y[i])


# https://facebook.github.io/prophet/docs/quick_start.html
!pip install prophet


from prophet import Prophet

c="Canada"
s="Premium Sticker Mart"
p="Holographic Goose"

x = np.array(train_df[(train_df["country"]==c) & (train_df["store"]==s) & (train_df["product"]==p)]["date"])
y = np.array(train_df[(train_df["country"]==c) & (train_df["store"]==s) & (train_df["product"]==p)]["num_sold"])
data = {'ds': x,'y': y}
df = pd.DataFrame(data)

m = Prophet()
m.fit(df)
future = m.make_future_dataframe(periods=3*365)
forecast = m.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].head()


train_df["num_sold"].max()


# do forecast using last three years of training data and growth='logistic', floor=0, cap=max*1.2
from prophet import Prophet
import datetime

four_dict = {}
test_df["num_sold"] = 0.0
cty = train_df["country"].unique()
sto = train_df["store"].unique()
pro = train_df["product"].unique()
for c in cty:
    print("county = ",c)
    for s in sto:
        for p in pro:
            sk = {"cty" : c, "sto" : s, "pro": p}
            skp = False
            data = []
            for i in range(len(skip)):
                if sk["cty"] == skip[i]["cty"] and sk["sto"] == skip[i]["sto"] and sk["pro"] == skip[i]["pro"]:
                    skp = True
            if not skp:
                x = np.array(train_df[(train_df["country"]==c) & (train_df["store"]==s) & (train_df["product"]==p) &
                             (train_df["date"]>datetime.datetime(2013, 12, 31))]["date"])
                y = np.array(train_df[(train_df["country"]==c) & (train_df["store"]==s) & (train_df["product"]==p) &
                             (train_df["date"]>datetime.datetime(2013, 12, 31))]["num_sold"])
                d = {'ds': x,'y': y}
                df = pd.DataFrame(d)
                df['floor'] = 0.0
                df['cap'] = train_df[(train_df["country"]==c) & (train_df["store"]==s) & (train_df["product"]==p) &
                             (train_df["date"]>datetime.datetime(2013, 12, 31))]["num_sold"].max()*1.2
                m = Prophet(growth='logistic')
                m.fit(df)
                future = m.make_future_dataframe(periods=3*365)
                future['floor'] = 0.0
                future['cap'] = train_df[(train_df["country"]==c) & (train_df["store"]==s) & (train_df["product"]==p) &
                             (train_df["date"]>datetime.datetime(2013, 12, 31))]["num_sold"].max()*1.2
                forecast = m.predict(future)
                forecast.drop(index=forecast[forecast["ds"]<test_df["date"].iloc[0]].index, axis=0, inplace=True)
                for i in forecast.index:
                    four_dict[str(forecast["ds"].loc[i])+","+c+","+s+","+p] = forecast["yhat"].loc[i]


for i in test_df.index:
    key = str(test_df["date"].loc[i])+","+test_df["country"].loc[i]+","+test_df["store"].loc[i]+\
        ","+test_df["product"].loc[i]
    if key in four_dict.keys():
        test_df["num_sold"].at[i] = four_dict[key]
test_df.head()


test_df[["id","num_sold"]].to_csv("submission.csv", index=False)

