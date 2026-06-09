import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import CategoricalNB
from xgboost import XGBClassifier


df_train = pd.read_csv("/kaggle/input/playground-series-s5e11/train.csv")
df_test = pd.read_csv("/kaggle/input/playground-series-s5e11/test.csv")
df_sample = pd.read_csv("/kaggle/input/playground-series-s5e11/sample_submission.csv")


df_train.head() # head(, tail() ,sample()


df_train.info()


df_train.drop(columns = "id", inplace = True)


df_train.head()


df_train.isnull().sum()


print(df_train["gender"].value_counts())

label_encoder = LabelEncoder()
df_train["gender"] = label_encoder.fit_transform(df_train["gender"])


print(df_train["marital_status"].value_counts())

label_encoder = LabelEncoder()
df_train["marital_status"] = label_encoder.fit_transform(df_train["marital_status"])


print(df_train["education_level"].value_counts())

label_encoder = LabelEncoder()
df_train["education_level"] = label_encoder.fit_transform(df_train["education_level"])


print(df_train["employment_status"].value_counts())

label_encoder = LabelEncoder()
df_train["employment_status"] = label_encoder.fit_transform(df_train["employment_status"])



print(df_train["loan_purpose"].value_counts())

label_encoder = LabelEncoder()
df_train["loan_purpose"] = label_encoder.fit_transform(df_train["loan_purpose"])


print(df_train["grade_subgrade"].value_counts())

ordinal_encoder = OrdinalEncoder(categories = [["A1","A2","A3", "A4","A5","B1","B2","B3", "B4","B5","C1","C2","C3","C4","C5","D1","D2","D3","D4","D5","E1","E2","E3","E4","E5","F1","F2","F3","F4","F5"]])
df_train["grade_subgrade"] = ordinal_encoder.fit_transform(df_train[["grade_subgrade"]])


df_train.head()


print(df_train["annual_income"].value_counts())
print(df_train["annual_income"].nunique())


print(df_train["debt_to_income_ratio"].value_counts())
print(df_train["debt_to_income_ratio"].nunique())


print(df_train["credit_score"].value_counts())
print(df_train["credit_score"].nunique())


print(df_train["loan_amount"].value_counts())
print(df_train["loan_amount"].nunique())


print(df_train["interest_rate"].value_counts())
print(df_train["interest_rate"].nunique())


continuous_cols = ["annual_income","debt_to_income_ratio","credit_score","loan_amount","interest_rate"]

for col in continuous_cols:
    plt.figure(figsize = (25,6), dpi = 100 , facecolor = "white", edgecolor = "black")
    plt.hist(
        df_train[col],
        bins = int(np.sqrt(df_train[col].nunique()) + 10),
        color = "grey"
    )

    plt.title(f"Hist Plot of {col} with skewness {df_train[col].skew()}", fontsize = 20, fontweight = "bold", color = "black", loc = "center")
    plt.xlabel(f"{col}", fontsize = 16, fontweight = "bold", color = "black")
    plt.ylabel(f"Frequency", fontsize = 16, fontweight = "bold", color = "black")
    plt.grid()
    plt.tight_layout()
    plt.show()


categorical_cols = ["gender", "marital_status", "education_level", "employment_status","loan_purpose", "grade_subgrade","loan_paid_back"]

for col in categorical_cols:
    plt.figure(figsize = (10,6), dpi = 100, facecolor = "white", edgecolor = "black")
    plt.bar(
        df_train[col].value_counts().index,
        df_train[col].value_counts().values,
        color = "grey",
        linewidth = 0.8,
        width = 0.8
        
    )
    plt.title(f"Bar Plot of {col}", fontsize = 20, color = "black", fontweight = "bold", loc = "center")
    plt.xlabel(f"{col}",fontsize = 16, color = "black", fontweight = "bold" )
    plt.ylabel("Count",fontsize = 16, color = "black", fontweight = "bold")
    plt.tight_layout()
    plt.show()



print(df_train.groupby("gender")["annual_income"].mean().reset_index(name = "Average_income"))
print("\n---------------------------------------------******----------------------------------------\n")

print(df_train.groupby("gender")["debt_to_income_ratio"].mean().reset_index(name = "Average_ratio"))
print("\n---------------------------------------------******----------------------------------------\n")

print(df_train.groupby("gender")["credit_score"].mean().reset_index(name = "Average_credit_score"))
print("\n---------------------------------------------******----------------------------------------\n")

print(df_train.groupby("gender")["loan_amount"].mean().reset_index(name = "Average_loan_amount"))
print("\n---------------------------------------------******----------------------------------------\n")

print(df_train.groupby("gender")["interest_rate"].mean().reset_index(name = "Average_interest_rate"))



print(df_train.groupby("education_level")["annual_income"].mean().reset_index(name = "Average_income"))
print("\n---------------------------------------------******----------------------------------------\n")

print(df_train.groupby("education_level")["debt_to_income_ratio"].mean().reset_index(name = "Average_ratio"))
print("\n---------------------------------------------******----------------------------------------\n")

print(df_train.groupby("education_level")["credit_score"].mean().reset_index(name = "Average_credit_score"))
print("\n---------------------------------------------******----------------------------------------\n")

print(df_train.groupby("education_level")["loan_amount"].mean().reset_index(name = "Average_loan_amount"))
print("\n---------------------------------------------******----------------------------------------\n")

print(df_train.groupby("education_level")["interest_rate"].mean().reset_index(name = "Average_interest_rate"))


print(df_train.groupby("employment_status")["annual_income"].mean().reset_index(name = "Average_income"))
print("\n---------------------------------------------******----------------------------------------\n")

print(df_train.groupby("employment_status")["debt_to_income_ratio"].mean().reset_index(name = "Average_ratio"))
print("\n---------------------------------------------******----------------------------------------\n")

print(df_train.groupby("employment_status")["credit_score"].mean().reset_index(name = "Average_credit_score"))
print("\n---------------------------------------------******----------------------------------------\n")

print(df_train.groupby("employment_status")["loan_amount"].mean().reset_index(name = "Average_loan_amount"))
print("\n---------------------------------------------******----------------------------------------\n")

print(df_train.groupby("employment_status")["interest_rate"].mean().reset_index(name = "Average_interest_rate"))


print(df_train.groupby("loan_paid_back")["annual_income"].mean().reset_index(name = "Average_income"))
print("\n---------------------------------------------******----------------------------------------\n")

print(df_train.groupby("loan_paid_back")["debt_to_income_ratio"].mean().reset_index(name = "Average_ratio"))
print("\n---------------------------------------------******----------------------------------------\n")

print(df_train.groupby("loan_paid_back")["credit_score"].mean().reset_index(name = "Average_credit_score"))
print("\n---------------------------------------------******----------------------------------------\n")

print(df_train.groupby("loan_paid_back")["loan_amount"].mean().reset_index(name = "Average_loan_amount"))
print("\n---------------------------------------------******----------------------------------------\n")

print(df_train.groupby("loan_paid_back")["interest_rate"].mean().reset_index(name = "Average_interest_rate"))


sns.heatmap(df_train.corr(), cmap = "rocket")


plt.figure(figsize = (25,6), dpi = 100, facecolor = "white", edgecolor = "black")
plt.boxplot(
    df_train["annual_income"],
    vert = False
)
plt.title("Box Plot of annual income (Before Outlier Handling)", fontsize = 20, fontweight = "bold", color = "black", loc = "center")
plt.xlabel("Count", fontsize = 16, fontweight = "bold", color = "black")
plt.tight_layout()
plt.show()

#==============================================================================

Q1 = np.percentile(df_train["annual_income"], 25)
Q3 = np.percentile(df_train["annual_income"], 75)
IQR = Q3- Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

df_train.loc[(df_train["annual_income"] < lower_bound) | (df_train["annual_income"] > upper_bound ), "annual_income"] = df_train["annual_income"].median()

plt.figure(figsize = (25,6), dpi = 100, facecolor = "white", edgecolor = "black")
plt.boxplot(
    df_train["annual_income"],
    vert = False
)
plt.title("Box Plot of annual income (After Outlier Handling 1)", fontsize = 20, fontweight = "bold", color = "black", loc = "center")
plt.xlabel("Count", fontsize = 16, fontweight = "bold", color = "black")
plt.tight_layout()
plt.show()

#==============================================================================

Q1 = np.percentile(df_train["annual_income"], 25)
Q3 = np.percentile(df_train["annual_income"], 75)
IQR = Q3- Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

df_train.loc[(df_train["annual_income"] < lower_bound) | (df_train["annual_income"] > upper_bound ), "annual_income"] = df_train["annual_income"].median()

plt.figure(figsize = (25,6), dpi = 100, facecolor = "white", edgecolor = "black")
plt.boxplot(
    df_train["annual_income"],
    vert = False
)
plt.title("Box Plot of annual income (After Outlier Handling 2)", fontsize = 20, fontweight = "bold", color = "black", loc = "center")
plt.xlabel("Count", fontsize = 16, fontweight = "bold", color = "black")
plt.tight_layout()
plt.show()

#==============================================================================

Q1 = np.percentile(df_train["annual_income"], 25)
Q3 = np.percentile(df_train["annual_income"], 75)
IQR = Q3- Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

df_train.loc[(df_train["annual_income"] < lower_bound) | (df_train["annual_income"] > upper_bound ), "annual_income"] = df_train["annual_income"].median()

plt.figure(figsize = (25,6), dpi = 100, facecolor = "white", edgecolor = "black")
plt.boxplot(
    df_train["annual_income"],
    vert = False
)
plt.title("Box Plot of annual income (After Outlier Handling 3)", fontsize = 20, fontweight = "bold", color = "black", loc = "center")
plt.xlabel("Count", fontsize = 16, fontweight = "bold", color = "black")
plt.tight_layout()
plt.show()


#==============================================================================


Q1 = np.percentile(df_train["annual_income"], 25)
Q3 = np.percentile(df_train["annual_income"], 75)
IQR = Q3- Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

df_train.loc[(df_train["annual_income"] < lower_bound) | (df_train["annual_income"] > upper_bound ), "annual_income"] = df_train["annual_income"].median()

plt.figure(figsize = (25,6), dpi = 100, facecolor = "white", edgecolor = "black")
plt.boxplot(
    df_train["annual_income"],
    vert = False
)
plt.title("Box Plot of annual income (After Outlier Handling 4)", fontsize = 20, fontweight = "bold", color = "black", loc = "center")
plt.xlabel("Count", fontsize = 16, fontweight = "bold", color = "black")
plt.tight_layout()
plt.show()


#==============================================================================


Q1 = np.percentile(df_train["annual_income"], 25)
Q3 = np.percentile(df_train["annual_income"], 75)
IQR = Q3- Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

df_train.loc[(df_train["annual_income"] < lower_bound) | (df_train["annual_income"] > upper_bound ), "annual_income"] = df_train["annual_income"].median()

plt.figure(figsize = (25,6), dpi = 100, facecolor = "white", edgecolor = "black")
plt.boxplot(
    df_train["annual_income"],
    vert = False
)
plt.title("Box Plot of annual income (After Outlier Handling 5)", fontsize = 20, fontweight = "bold", color = "black", loc = "center")
plt.xlabel("Count", fontsize = 16, fontweight = "bold", color = "black")
plt.tight_layout()
plt.show()


plt.figure(figsize = (25,6), dpi = 100, facecolor = "white", edgecolor = "black")
plt.boxplot(
    df_train["debt_to_income_ratio"],
    vert = False
)
plt.title("Box Plot of annual income (Before Outlier Handling)", fontsize = 20, fontweight = "bold", color = "black", loc = "center")
plt.xlabel("Count", fontsize = 16, fontweight = "bold", color = "black")
plt.tight_layout()
plt.show()

# ===========================================================================

Q1 = np.percentile(df_train["debt_to_income_ratio"], 25)
Q3 = np.percentile(df_train["debt_to_income_ratio"], 75)
IQR = Q3- Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

df_train.loc[(df_train["debt_to_income_ratio"] < lower_bound) | (df_train["debt_to_income_ratio"] > upper_bound ), "debt_to_income_ratio"] = df_train["debt_to_income_ratio"].median()


plt.figure(figsize = (25,6), dpi = 100, facecolor = "white", edgecolor = "black")
plt.boxplot(
    df_train["debt_to_income_ratio"],
    vert = False
)
plt.title("Box Plot of debt_to_income_ratio (After Outlier Handling 1)", fontsize = 20, fontweight = "bold", color = "black", loc = "center")
plt.xlabel("Count", fontsize = 16, fontweight = "bold", color = "black")
plt.tight_layout()
plt.show()

#==============================================================================

Q1 = np.percentile(df_train["debt_to_income_ratio"], 25)
Q3 = np.percentile(df_train["debt_to_income_ratio"], 75)
IQR = Q3- Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

df_train.loc[(df_train["debt_to_income_ratio"] < lower_bound) | (df_train["debt_to_income_ratio"] > upper_bound ), "debt_to_income_ratio"] = df_train["debt_to_income_ratio"].median()

plt.figure(figsize = (25,6), dpi = 100, facecolor = "white", edgecolor = "black")
plt.boxplot(
    df_train["debt_to_income_ratio"],
    vert = False
)
plt.title("Box Plot of debt_to_income_ratio (After Outlier Handling 2)", fontsize = 20, fontweight = "bold", color = "black", loc = "center")
plt.xlabel("Count", fontsize = 16, fontweight = "bold", color = "black")
plt.tight_layout()
plt.show()


#==============================================================================

Q1 = np.percentile(df_train["debt_to_income_ratio"], 25)
Q3 = np.percentile(df_train["debt_to_income_ratio"], 75)
IQR = Q3- Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

df_train.loc[(df_train["debt_to_income_ratio"] < lower_bound) | (df_train["debt_to_income_ratio"] > upper_bound ), "debt_to_income_ratio"] = df_train["debt_to_income_ratio"].median()


plt.figure(figsize = (25,6), dpi = 100, facecolor = "white", edgecolor = "black")
plt.boxplot(
    df_train["debt_to_income_ratio"],
    vert = False
)
plt.title("Box Plot of debt_to_income_ratio (After Outlier Handling 3)", fontsize = 20, fontweight = "bold", color = "black", loc = "center")
plt.xlabel("Count", fontsize = 16, fontweight = "bold", color = "black")
plt.tight_layout()
plt.show()

#==============================================================================


Q1 = np.percentile(df_train["debt_to_income_ratio"], 25)
Q3 = np.percentile(df_train["debt_to_income_ratio"], 75)
IQR = Q3- Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

df_train.loc[(df_train["debt_to_income_ratio"] < lower_bound) | (df_train["debt_to_income_ratio"] > upper_bound ), "debt_to_income_ratio"] = df_train["debt_to_income_ratio"].median()


plt.figure(figsize = (25,6), dpi = 100, facecolor = "white", edgecolor = "black")
plt.boxplot(
    df_train["debt_to_income_ratio"],
    vert = False
)
plt.title("Box Plot of debt_to_income_ratio (After Outlier Handling 4)", fontsize = 20, fontweight = "bold", color = "black", loc = "center")
plt.xlabel("Count", fontsize = 16, fontweight = "bold", color = "black")
plt.tight_layout()
plt.show()





plt.figure(figsize = (25,6), dpi = 100, facecolor = "white", edgecolor = "black")
plt.boxplot(
    df_train["credit_score"],
    vert = False
)
plt.title("Box Plot of credit_score (Before Outlier Handling)", fontsize = 20, fontweight = "bold", color = "black", loc = "center")
plt.xlabel("Count", fontsize = 16, fontweight = "bold", color = "black")
plt.tight_layout()
plt.show()


mean = df_train["credit_score"].mean()
std = df_train["credit_score"].std()
z_score = (df_train["credit_score"] - mean) / std

threshold = 2

outliers = np.abs(z_score) > threshold

df_train.loc[outliers,"credit_score"] = df_train["credit_score"].median() 


plt.figure(figsize = (25,6), dpi = 100, facecolor = "white", edgecolor = "black")
plt.boxplot(
    df_train["credit_score"],
    vert = False
)
plt.title("Box Plot of credit_score (After Outlier Handling )", fontsize = 20, fontweight = "bold", color = "black", loc = "center")
plt.xlabel("Count", fontsize = 16, fontweight = "bold", color = "black")
plt.tight_layout()
plt.show()




plt.figure(figsize = (25,6), dpi = 100, facecolor = "white", edgecolor = "black")
plt.boxplot(
    df_train["loan_amount"],
    vert = False
)
plt.title("Box Plot of loan_amount (Before Outlier Handling)", fontsize = 20, fontweight = "bold", color = "black", loc = "center")
plt.xlabel("Count", fontsize = 16, fontweight = "bold", color = "black")
plt.tight_layout()
plt.show()


mean = df_train["loan_amount"].mean()
std = df_train["loan_amount"].std()
z_score = (df_train["loan_amount"] - mean) / std

threshold = 2

outliers = np.abs(z_score) > threshold

df_train.loc[outliers,"loan_amount"] = df_train["loan_amount"].median() 

plt.figure(figsize = (25,6), dpi = 100, facecolor = "white", edgecolor = "black")
plt.boxplot(
    df_train["loan_amount"],
    vert = False
)
plt.title("Box Plot of loan_amount (After Outlier Handling )", fontsize = 20, fontweight = "bold", color = "black", loc = "center")
plt.xlabel("Count", fontsize = 16, fontweight = "bold", color = "black")
plt.tight_layout()
plt.show()



plt.figure(figsize = (25,6), dpi = 100, facecolor = "white", edgecolor = "black")
plt.boxplot(
    df_train["interest_rate"],
    vert = False
)
plt.title("Box Plot of interest_rate (Before Outlier Handling)", fontsize = 20, fontweight = "bold", color = "black", loc = "center")
plt.xlabel("Count", fontsize = 16, fontweight = "bold", color = "black")
plt.tight_layout()
plt.show()


mean = df_train["interest_rate"].mean()
std = df_train["interest_rate"].std()
z_score = (df_train["interest_rate"] - mean) / std

threshold = 2

outliers = np.abs(z_score) > threshold

df_train.loc[outliers,"interest_rate"] = df_train["interest_rate"].median() 

plt.figure(figsize = (25,6), dpi = 100, facecolor = "white", edgecolor = "black")
plt.boxplot(
    df_train["interest_rate"],
    vert = False
)
plt.title("Box Plot of interest_rate (After Outlier Handling )", fontsize = 20, fontweight = "bold", color = "black", loc = "center")
plt.xlabel("Count", fontsize = 16, fontweight = "bold", color = "black")
plt.tight_layout()
plt.show()



df_train


df_train.describe()


sns.heatmap(df_train.corr(), cmap = "rocket")


X = df_train.drop(columns = "loan_paid_back")
Y = df_train["loan_paid_back"]


X_train, X_test, Y_train, Y_test = train_test_split(X, Y ,test_size = 0.2,stratify = Y ,random_state = 42)


model = LogisticRegression()

model.fit(X_train, Y_train)

Y_pred = model.predict(X_test)

accuracy_score(Y_test, Y_pred)


model = XGBClassifier(
    n_estimators=1000, 
    learning_rate=0.01, 
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss'
)

model.fit(X_train, Y_train)

Y_pred = model.predict(X_test)

accuracy_score(Y_test, Y_pred)


df_test


df_sample


print(df_test["gender"].value_counts())

label_encoder = LabelEncoder()
df_test["gender"] = label_encoder.fit_transform(df_test["gender"])


print(df_test["marital_status"].value_counts())

label_encoder = LabelEncoder()
df_test["marital_status"] = label_encoder.fit_transform(df_test["marital_status"])


print(df_test["education_level"].value_counts())

label_encoder = LabelEncoder()
df_test["education_level"] = label_encoder.fit_transform(df_test["education_level"])


print(df_test["employment_status"].value_counts())

label_encoder = LabelEncoder()
df_test["employment_status"] = label_encoder.fit_transform(df_test["employment_status"])



print(df_test["loan_purpose"].value_counts())

label_encoder = LabelEncoder()
df_test["loan_purpose"] = label_encoder.fit_transform(df_test["loan_purpose"])


print(df_test["grade_subgrade"].value_counts())

ordinal_encoder = OrdinalEncoder(categories = [["A1","A2","A3", "A4","A5","B1","B2","B3", "B4","B5","C1","C2","C3","C4","C5","D1","D2","D3","D4","D5","E1","E2","E3","E4","E5","F1","F2","F3","F4","F5"]])
df_test["grade_subgrade"] = ordinal_encoder.fit_transform(df_test[["grade_subgrade"]])


df_test


df_test["loan_paid_back"] = model.predict(df_test.drop(columns = "id"))


submission = pd.DataFrame({
    "id" : df_test["id"],
    "loan_paid_back" : df_test["loan_paid_back"]
})


submission.to_csv("submission.csv", index = False)


submission.to_csv("/kaggle/working/submission.csv", index = False)

