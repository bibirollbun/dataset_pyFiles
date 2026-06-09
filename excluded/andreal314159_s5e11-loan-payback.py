# imports 

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestRegressor

import warnings
warnings.simplefilter(action="ignore", category = FutureWarning)


loans = pd.read_csv("/kaggle/input/playground-series-s5e11/train.csv", index_col = 0)
test = pd.read_csv("/kaggle/input/playground-series-s5e11/test.csv", index_col=0)
loans.head()


# First glance at the data
display(loans.info())
display(loans.describe(include=np.number))
display(loans.describe(include=["object"]))


loans[["gender", "marital_status", "employment_status", "loan_purpose", "grade_subgrade"]] = loans[["gender", "marital_status", "employment_status", "loan_purpose", "grade_subgrade"]].astype("category")

loans.education_level.unique()
loans["education_level"] = pd.Categorical(loans.education_level, ordered = True, categories = ["Other", "High School", "Bachelor's", "Master's", "PhD"])
loans.head()


loans.info()


fig, ax = plt.subplots(figsize = (4,6))
plt.pie(loans.loan_paid_back.value_counts().sort_index(ascending=False), labels = ["paid", "unpaid"], autopct="%.2f%%", startangle=90)
ax.set_title("Distribution of the target variable - paid vs unpaid loans")
plt.show()


%%time
if False: # It takes roughly 1 minute to generate the pairplot so I disable it
    g = sns.pairplot(loans.sample(100000, random_state =0).select_dtypes(include=[int, float]), 
                    hue= "loan_paid_back", 
                    plot_kws={"s":8, 
                              "alpha": 0.1},
                    corner = True
                    )
    g.fig.suptitle("Numeric features by loan payback status")
    plt.show()


for col in loans.select_dtypes("category").columns:
    fig, ax = plt.subplots(1,2, figsize=(10,6))  
    fig.suptitle(f"Loan repayment by {col}", fontsize=16)
    
    # Counts
    g = sns.countplot(data=loans, y=col, hue="loan_paid_back", ax = ax[0], palette = "Blues")
    #ax[0].set_title(f"Loan repayment status by {col}")
    ax[0].set_ylabel(f"{col}")
    ax[0].set_xlabel("Count")
    ax[0].legend(title="Loan paid back", labels=["Unpaid (0)", "Paid (1)"])

    # Ratios
    h = sns.barplot(data= loans, y = col, x = "loan_paid_back", ax = ax[1], palette = "Purples")
    ax[1].set_xlabel(f"Ratio of payback")
    
    sns.despine()
    plt.show()
    
    for container in ax[1].containers:
        ax[1].bar_label(container)
    
    plt.show()





loans["grade"] = loans.grade_subgrade.str[0]

test["grade"] = test.grade_subgrade.str[0]
loans.head()


from sklearn.ensemble import ExtraTreesRegressor


%%time

X = loans.drop(columns = ["loan_paid_back"]).select_dtypes(float, int)
y = loans.loan_paid_back

model = LinearRegression()
model = RandomForestRegressor(random_state=0) # very time consuming
model = ExtraTreesRegressor(random_state = 0)

def get_score(model, X, y):
    """
    Calculate the cross validated roc auc score of a model.
    """
    scores = cross_val_score(model, X, y, scoring = "roc_auc", cv = 5)
    consolidated_score = np.mean(scores) - np.std(scores)
    print(f"{model=}: {consolidated_score} based on scores {scores}")
    return consolidated_score

get_score(model, X, y)


model.fit(X, y)
predictions = model.predict(test[X.columns])


submission = pd.DataFrame({"id": test.index, "loan_paid_back": predictions})
submission.to_csv("submission.csv", index=False)




