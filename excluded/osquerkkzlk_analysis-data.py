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


import numpy as np
import pandas as pd
import os
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import random
%matplotlib inline
warnings.filterwarnings("ignore")


# import file

dir=r"/kaggle/input/playground-series-s5e11"
dir2 =r"/kaggle/input/loan-prediction-dataset-2025"
train=pd.read_csv(os.path.join(dir,"train.csv"))
test=pd.read_csv(os.path.join(dir,"test.csv"))
sub_=pd.read_csv(os.path.join(dir,"sample_submission.csv"))
orig=pd.read_csv(os.path.join(dir2,"loan_dataset_20000.csv"))


display(train.head())
display(test.head())
display(orig.head())


# display shape
display(f"shape of train<{train.shape}>")
display(f"shape of test<{test.shape}>")
display(f"shape of orig<{orig.shape}>")


# revise orig data
orig["id"]=test.id.max()+1
test["loan_paid_back"]=0

display(f"shape of train<{train.shape}>")
display(f"shape of test<{test.shape}>")
display(f"shape of orig<{orig.shape}>")

data=pd.concat([train,test],axis=0)
display(f"shape of merged data <{data.shape}>")


# display data

CAT,NUM,TARGET=[],[],"loan_paid_back"
for col in data.columns.drop(["id","loan_paid_back"]):
    temp="NUM"
    if data[col].dtype=="object":
        CAT.append(col)
        temp="CAT"
    else:
        NUM.append(col)
    print(f"{col:20}[{temp}] {data[col].nunique():10} catgory{data[col].isna().sum():10} NAN")
Features=train.columns.drop(["id","loan_paid_back"])

len(CAT),len(NUM),len(Features)


train["education_level"].value_counts().index.tolist()


# display  the distributions of train\test\orig
# < 1 > analysis of object features
for cat_features in CAT:
    plt.figure(figsize=(12,6))

    plt.subplot(131)
    sns.countplot(x=cat_features,order=train[cat_features].value_counts().index.tolist(),\
                  label="trian",data=train,color="g",alpha=0.5,width=0.5)
    sns.countplot(x=cat_features,order=test[cat_features].value_counts().index.tolist(),\
                  label="test",data=test,color="b",alpha=0.5,width=0.5)
    plt.title(cat_features)
    plt.legend()
    plt.xlabel("Count")

    plt.subplot(132)
    counts=train[train["loan_paid_back"]==1][cat_features].value_counts(normalize=True)
    plt.pie(counts,labels=counts.index,autopct="%.2f%%")
    plt.title("pay back")

    plt.subplot(133)
    counts=train[train["loan_paid_back"]==0][cat_features].value_counts(normalize=True)
    plt.pie(counts,labels=counts.index,autopct="%.2f%%")
    plt.title("Not pay back")
    
    plt.tight_layout()
    plt.show()


# display relations of between object features and target

for col in CAT:
    prop=train.groupby(col)[TARGET].value_counts(normalize=True).unstack(fill_value=0)
    prop.plot(kind="bar",stacked=True,figsize=(8,4),colormap="Set2")
    plt.title(f"{col} VS {TARGET[0]}")
    plt.ylabel("Proportion")
    plt.xlabel(col)
    plt.legend(title=TARGET)
    plt.show()


len(CAT)


temp=train.copy()
_,axes=plt.subplots(nrows=5,ncols=5,figsize=(20,20))

for i in range(len(CAT)):
    for j in range(len(CAT)):
        if i>=j: continue
        name=CAT[i]+"_"+CAT[j]
        temp[name]=temp[CAT[i]]+"_"+temp[CAT[j]]

        prop=temp.groupby(name)[TARGET].value_counts(normalize=True).unstack(fill_value=0)
        prop.plot(kind="bar",stacked=True,ax=axes[i,j-1],colormap="Set3")
        axes[i,j-1].set_title(name)
        axes[i,j-1].axis("off")
plt.legend(title=True)
plt.tight_layout()
plt.show()
            
            


sns.histplot(x="loan_paid_back",kde=True,data=train)
plt.title("The distributions of Target")
plt.show()


for col in NUM:
    sns.kdeplot(x=col,data=train,fill=False,label="train")
    sns.kdeplot(x=col,data=test,fill=False,label="test")
    sns.kdeplot(x=col,data=orig,fill=False,label="orig")
    plt.legend()
    plt.title(col)
    plt.show()


# < 2 > analysis of numeric features
from tqdm.notebook import tqdm
# import matplotlib
# matplotlib.use("Agg")
pbar=tqdm(desc="bar",total=len(NUM)**2)

_,axes=plt.subplots(nrows=len(NUM),ncols=len(NUM),figsize=(15,15))

for i in range(len(NUM)):
    for j in range(len(NUM)):
        pbar.update(1)
        ax_=axes[i,j]
        if i==j:
            sns.kdeplot(x=NUM[i],hue="loan_paid_back",data=train.sample(20_0000),ax=ax_,fill=False)
            corr=train[NUM[i]].corr(train["loan_paid_back"])
            ax_.set_title(f"Distribution of {NUM[i]} \ncorr is {corr:.4f}")
        else:
            sns.scatterplot(x=NUM[i],y=NUM[j],hue="loan_paid_back",data=train.sample(20_0000),ax=ax_,alpha=0.3)
            ax_.set_title(f"{NUM[i]} VS {NUM[j]}")
print("Rendering the interface (which takes a relatively long time)")
plt.tight_layout()
plt.show()
pbar.close()


metrix=train[NUM].corr(method="spearman")
sns.heatmap(metrix,annot=True,cmap="coolwarm")


display(orig.head())
display(f"the shape of original data {orig.shape}")


# display data

CAT_O,NUM_O,TARGET_O=[],[],["loan_paid_back"]
for col in orig.columns.drop(["id","loan_paid_back"]):
    temp="NUM"
    if orig[col].dtype=="object":
        CAT_O.append(col)
        temp="CAT"
    else:
        NUM_O.append(col)
    print(f"{col:20}[{temp}] {orig[col].nunique():10} catgory{orig[col].isna().sum():10} NAN")

len(CAT_O),len(NUM_O),len(TARGET_O)


train.columns.drop(["id","loan_paid_back"])


metrix=orig[NUM_O].corr(method="spearman")
plt.figure(figsize=(12,12))
sns.heatmap(metrix,annot=True,cmap="coolwarm",fmt=".3f")
plt.xticks(rotation=45)
plt.title("original data")
plt.show()




