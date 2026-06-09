
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns


from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.svm import SVC


import os
np.set_printoptions(suppress=True)
pd.options.display.float_format = '{:.4f}'.format
pd.set_option('display.max_colwidth', None)


BASE_DIR = "/kaggle/input/playground-series-s4e7"


TRAIN_DIR = os.path.join(
    BASE_DIR , "train.csv"
)

COL_DESCRIPTION = {
    "id": "Unique identifier for each customer",
    "Gender": "Customer gender",
    "Age": "Age of the customer in years",
    "Driving_License": "Indicates whether the customer has a driving license (1 = Yes, 0 = No)",
    "Region_Code": "Code representing the customer's region",
    "Previously_Insured": "Indicates whether the customer already has vehicle insurance (1 = Yes, 0 = No)",
    "Vehicle_Age": "Age of the customer's vehicle",
    "Vehicle_Damage": "Indicates whether the vehicle was damaged in the past",
    "Annual_Premium": "Yearly insurance premium amount",
    "Policy_Sales_Channel": "Channel used to sell the insurance policy",
    "Vintage": "Number of days the customer has been associated with the company",
    "Response": "Target variable indicating customer interest in vehicle insurance (1 = Interested, 0 = Not Interested)",
}




train_df = pd.read_csv(TRAIN_DIR).drop(
    columns = ["id"]
)

pos_class = train_df[train_df["Response"] == 1].reset_index(drop = True)
neg_class_sampled = train_df[train_df["Response"]==0].sample(frac=0.2101).reset_index(drop = True)
train_df = pd.concat([pos_class , neg_class_sampled]).sample(frac=1).reset_index(drop = True)

train_df



print(
    f"Duplicate Rows Count : {train_df.duplicated().sum()} \n\n\n"\
    f"Missing Values Count : \n\n {train_df.isnull().sum()}"
)



cat_columns = [
    'Gender', 'Vehicle_Age', 'Vehicle_Damage' , "Response" , "Previously_Insured" , "Region_Code" , "Policy_Sales_Channel" , "Driving_License"
]

num_columns = [
    "Age" , "Annual_Premium" , "Vintage"
]


print("###########"*2 + "  Categorical Columns  " + "###########"*2)
print( "###########"*6 , end = "\n")



for column in cat_columns:
    col_data = train_df[column].copy()
    print(
        f"{column} --> {col_data.unique() if len(col_data.unique()) < 5 else len(col_data.unique())} --> "
        f"{COL_DESCRIPTION.get(column, None)}"
    )
print( "###########"*6 , end = "\n")


description = train_df[num_columns].describe().T
description["description"] = [
    COL_DESCRIPTION.get(column, None) for column in description.index.tolist()
]
description.head()


fig, ax = plt.subplots(1, 3, figsize=(12, 8))

for i, col in enumerate(num_columns):
    sns.boxplot(data=train_df, y=col, x="Response", ax=ax[i])
    ax[i].set_title(f"{col} vs Response", fontsize=12)
    ax[i].set_xlabel("Response")
    ax[i].set_ylabel(col)

fig.suptitle("Distribution of Numerical Features by Response", fontsize=16)
plt.tight_layout()


train_df.hist(
        column = num_columns,
        figsize = (18, 4),
        layout=(1 , 3),
        grid = False,
    )
plt.suptitle("Distribution of Numerical Features", fontsize=16)
plt.show()



corr_df = train_df[num_columns + ["Response"]].corr()
fig = plt.figure(figsize = (12 , 8))
sns.heatmap(corr_df , annot=True)
plt.show()




cols_to_display = ['Gender',
                 'Vehicle_Age',
                 'Vehicle_Damage',
                 'Previously_Insured',
                 'Driving_License'
                ]



def make_autopct(total_count):
    def my_autopct(pct):
        count = int(round(pct * total_count / 100.0))
        return f"{count} ({pct:.1f}%)"
    return my_autopct

fig, axes = plt.subplots(len(cols_to_display), 2, figsize=(14, 18))




for row, col in enumerate(cols_to_display): # 0 , Gender
    for col_idx, resp in enumerate([0, 1]): # [0 , 1] Response 
        data = (
            train_df[train_df['Response'] == resp][col]
            .value_counts()
        )
        
        total_count = sum(data)

        axes[row, col_idx].pie(
            data.values,
            labels=data.index,
            autopct=make_autopct(total_count),
            startangle=90
        )

        axes[row, col_idx].set_title(f"{col} | Response = {resp}")




fig.suptitle("Categorical Feature Distribution by Response", fontsize=18)

plt.tight_layout()
plt.show()










data = train_df["Response"].value_counts()
plt.pie(
        data.values,
        labels=data.index,
        autopct=make_autopct(total_count),
        startangle=90
        )

plt.title("Target Value counts (Response)")


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
def show_confusion_and_report(y_true, y_pred, labels=None, normalize=None, figsize=(6, 5)):

    cm = confusion_matrix(y_true, y_pred, labels=labels, normalize=normalize)
    disp_labels = labels if labels is not None else np.unique(y_true)

    fig, ax = plt.subplots(figsize=figsize)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=disp_labels)
    disp.plot(ax=ax, cmap=plt.cm.Blues)
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    plt.show()

    print("Classification Report\n")
    print(classification_report(y_true, y_pred, labels=labels, digits=4))


import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
from sklearn.metrics import f1_score , roc_auc_score
from sklearn.preprocessing import StandardScaler

def cross_val_predict(
    estimator,
    X,
    y,
    n_splits=5,
    random_state=42,
    shuffle=True,
    verbose=0,
):

    X = np.asarray(X)
    y = np.asarray(y)



    n_samples = X.shape[0]

    oof_probs = np.zeros((n_samples, 2), dtype=float)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y) , start = 1):

        print(f"[fold {fold}/{n_splits}] train={len(train_idx)} val={len(val_idx)}")

        X_train = X[train_idx]
        X_val = X[val_idx]

        
        y_train = y[train_idx]
        y_val = y[val_idx]



        

        clf = (estimator)

        clf.fit(
            X_train,
            y_train, 
        )
        probs = clf.predict_proba(X_val)

        oof_probs[val_idx] = probs


    return oof_probs



from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder , LabelEncoder


print("Before encoding:", train_df['Gender'].unique())
le = LabelEncoder()
train_df['Gender'] = le.fit_transform(train_df['Gender'])  # Male = 1, Female = 0
print("After encoding:", train_df['Gender'].unique())



print("Before encoding:", train_df['Vehicle_Age'].unique())
ord_enc = OrdinalEncoder(categories=[['< 1 Year', '1-2 Year', '> 2 Years']])
train_df['Vehicle_Age'] = ord_enc.fit_transform(train_df[['Vehicle_Age']])
print("After encoding:", train_df['Vehicle_Age'].unique())


print("Before encoding:", train_df['Vehicle_Damage'].unique())
train_df['Vehicle_Damage'] = le.fit_transform(train_df['Vehicle_Damage'])  
print("After encoding:", train_df['Vehicle_Damage'].unique())




X = train_df.drop("Response" , axis = 1).copy()
y = train_df["Response"].copy()





model = XGBClassifier(
    n_estimators=300,          
    device = "cuda",
    random_state = 42
)


oof_probs_xgb = cross_val_predict(
    model,
    X,
    y,
    n_splits=5,
    random_state=42,
    shuffle=True,
    verbose=1,
)

classes = [0 , 1]
oof_preds_xgb = np.argmax(oof_probs_xgb, axis=1)

show_confusion_and_report(y, oof_preds_xgb, labels=classes, normalize=None, figsize=(6, 5))


model = LGBMClassifier(
    learning_rate=0.05,
    # max_depth=6,
    num_leaves=200,
    n_estimators=150,
    random_state=42,
    verbose = -1,
)


oof_probs_lgbm = cross_val_predict(
    model,
    X,
    y,
    n_splits=5,
    random_state=42,
    shuffle=True,
    verbose=1,
)

classes = [0 , 1]
oof_preds_lgbm = np.argmax(oof_probs_lgbm, axis=1)

show_confusion_and_report(y, oof_preds_lgbm, labels=classes, normalize=None, figsize=(6, 5))



from sklearn.ensemble import RandomForestClassifier

model_clf = RandomForestClassifier(
    n_estimators = 200 , random_state = 42
)

oof_probs_clf = cross_val_predict(
    model_clf,
    X,
    y,
    n_splits=5,
    random_state=42,
    shuffle=True,
    verbose=1
)

classes = [0 , 1]
oof_preds_clf = np.argmax(oof_probs_clf, axis=1)

show_confusion_and_report(y, oof_preds_clf, labels=classes, normalize=None, figsize=(6,5))


ws = {'w1': 0.1, 'w2': 0.51, 'w3': 0.89}


w1 , w2 , w3 = ws["w1"], ws["w2"] , ws["w3"]
sum_ = w1+w2+w3
meta_probs = (w1*oof_probs_clf + w2*oof_probs_lgbm + w3*oof_probs_xgb)/sum_


meta_preds = np.argmax(meta_probs , axis = 1)



show_confusion_and_report(y, meta_preds, labels=classes, normalize=None, figsize=(6,5))



