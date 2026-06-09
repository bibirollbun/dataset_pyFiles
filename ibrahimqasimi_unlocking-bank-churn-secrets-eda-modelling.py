import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from catboost import CatBoostClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, auc
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")


#Loading the data
train= pd.read_csv('/kaggle/input/playground-series-s4e1/train.csv')
test= pd.read_csv('/kaggle/input/playground-series-s4e1/test.csv')
sample_submission= pd.read_csv('/kaggle/input/playground-series-s4e1/sample_submission.csv')



#Viewing the first 5 rows
train.head()


#Viewing the last 5 rows
train.tail()


#Viewing the shape
train.shape


#Viewing the summary
train.describe().round(2).style.format(precision=2).background_gradient(cmap="Reds")


#Viewing the information
train.info()


#missing values
train.isnull().sum()


#Viewing the target variable
plt.pie(
    train["Exited"].value_counts(),
    labels=train["Exited"].value_counts().index,
    textprops={"fontsize": 14},
    colors=["#ff9999", "#495680", "#de6264"],
    autopct="%.0f%%",
    explode=[0.05, 0.05],
)
plt.title("Target Variable Distribution", fontsize=14)
plt.show()


plt.figure(figsize=(6, 6))
labels =["Churn: Yes","Churn:No"]
values = [1869,5163]
labels_gender = ["F","M","F","M"]
sizes_gender = [939,930 , 2544,2619]
colors = ['#ff6666', '#66b3ff']
colors_gender = ['#c2c2f0','#ffb3e6', '#c2c2f0','#ffb3e6']
explode = (0.3,0.3)
explode_gender = (0.1,0.1,0.1,0.1)
textprops = {"fontsize":15}
#Plot
plt.pie(values, labels=labels,autopct='%1.1f%%',pctdistance=1.08, labeldistance=0.8,colors=colors, startangle=90,frame=True, explode=explode,radius=10, textprops =textprops, counterclock = True, )
plt.pie(sizes_gender,labels=labels_gender,colors=colors_gender,startangle=90, explode=explode_gender,radius=7, textprops =textprops, counterclock = True, )
#Draw circle
centre_circle = plt.Circle((0,0),5,color='black', fc='white',linewidth=0)
fig = plt.gcf()
fig.gca().add_artist(centre_circle)

plt.title('Churn Distribution w.r.t Gender: Male(M), Female(F)', fontsize=15, y=1.1)

# show plot

plt.axis('equal')
plt.tight_layout()
plt.show()


#Viewing the attributes
train.attrs["name"] = "train"
test.attrs["name"] = "test"
#DATA PREPARATION

def prep(df):
    df.columns = df.columns.str.lower()
    df["zero_bal_flag"] = np.where(df["balance"] == 0, "z", "nz")
    df["zero_bal_flag"] = df["zero_bal_flag"].astype("category")
    df["gender_by_geo"] = df["gender"] + "_" + df["geography"]
    df["gender_by_geo"] = df["gender_by_geo"].astype("category")
    df["is_senior"] = np.where(df["age"] > 63, 1, 0)
    df["is_senior"] = df["is_senior"].astype("int32").astype("category")
    df["isactive_by_creditcard"] = np.multiply(df["hascrcard"], df["isactivemember"]).astype("int32").astype("category")
    df["hascrcard"] = df["hascrcard"].astype("int32").astype("category")
    df["products_by_tenure"] = np.divide(df["tenure"], df["numofproducts"])
    df["balance_and_salary_ratio"] =  np.divide(df["balance"], df["estimatedsalary"])
    df["age_cat"] = np.round(np.divide(df["age"], 20)).astype("int").astype("category")
    df["credit_score_cat"] = pd.cut(
        df["creditscore"],
        bins=[0, 450, 650, 750, 850],
        labels=["very_low", "low", "medium", "high"],
    )
    df["credit_score_cat"] = df["credit_score_cat"].astype("category")
    df["isactivemember"] = df["isactivemember"].astype("int32").astype("category")
    df["customer_relationships"] = np.where(df["tenure"] > 1, "new_customer", "long-lasting")
    df["customer_relationships"] = df["customer_relationships"].astype("category")
    df = df.drop("id", axis=1)
    categorical_vars = df.loc[:, df.nunique() <= 4].columns[0:5]
    numerical_vars = df[
        ["customerid", "creditscore", "age", "tenure", "balance", "estimatedsalary"]
    ].columns
    df[categorical_vars] = df[categorical_vars].astype("category")

    return df, categorical_vars, numerical_vars


#DATA PREPARATION
train, categorical_vars, numerical_vars = prep(train)
test = prep(test)
test = test[0]


#making  zeero balance flag w.r.t. gender, geography, credit score and products by tenure and active member.
for i in categorical_vars:
    g = sns.FacetGrid(data=train, col=i, height=2.5, aspect=1.5, margin_titles=True)
    g.map_dataframe(
        sns.countplot,
        x="zero_bal_flag",
        hue="gender_by_geo",
        width=0.4,
        palette="Set2",
    )
    g.set_titles(
        col_template="\n---------------------\n{col_var} = {col_name}\n---------------------\n",
        size=8,
    )
    g.add_legend(fontsize=8)
    g.tick_params(labelsize=8)
    g.set_axis_labels(x_var="zero balance Flag", y_var="Counts", fontsize=10)

plt.show()



# distribution of exited w.r.t. customer_relationships.
for i in categorical_vars:
    g = sns.FacetGrid(data=train, col=i, height=2.5, aspect=1.5, margin_titles=True)
    g.map_dataframe(
        sns.countplot,
        x="exited",
        hue="customer_relationships",
        width=0.4,
        palette="Set2",
    )
    g.set_titles(
        col_template="\n---------------------\n{col_var} = {col_name}\n---------------------\n",
        size=8,
    )
    g.add_legend(fontsize=8)
    g.tick_params(labelsize=8)
    g.set_axis_labels(x_var="Exited", y_var="Counts", fontsize=10)

plt.show()


plt.style.use("default")
corr_mat = train[numerical_vars].corr()
mask = np.triu(np.ones_like(corr_mat, dtype=bool))
cmap = sns.diverging_palette(230, 30, as_cmap=True)
f, ax = plt.subplots(figsize=(9, 7))
sns.heatmap(
    corr_mat,
    mask=mask,
    cbar=False,
    cmap=cmap,
    fmt="0.2f",
    center=0,
    square=False,
    annot=True,
    linewidths=0.5,
    cbar_kws={"shrink": 0.8},
)
plt.title("Correlation Matrix (Train Datase)\n")
plt.xticks(fontsize=9)
plt.yticks(fontsize=9)
plt.show()


#Viewing the first 5 rows
test.head()


# tail
test.tail()


#Viewing the shape
test.shape


# summary
test.describe().round(2).style.format(precision=2).background_gradient(cmap="Reds")


#Viewing the information
test.info()


plt.style.use("default")
corr_mat = test[numerical_vars].corr()
mask = np.triu(np.ones_like(corr_mat, dtype=bool))
cmap = sns.diverging_palette(200, 60, as_cmap=True)
f, ax = plt.subplots(figsize=(9, 7))
sns.heatmap(
    corr_mat,
    mask=mask,
    cbar=False,
    cmap=cmap,
    fmt="0.2f",
    center=0,
    square=False,
    annot=True,
    linewidths=0.5,
    cbar_kws={"shrink": 0.8},
)
plt.title("Correlation Matrix (Test Dataset)\n")
plt.xticks(fontsize=9)
plt.yticks(fontsize=9)
plt.show()


train.info()


train.head().T


categorical_features = train.select_dtypes(
    include=["object", "category"]
).columns.to_list()
train[categorical_features].info()


rand_state = 1
# making X and Y
test_df = test.drop(columns=["customerid"], axis=1)
X = train.drop(columns=["customerid", "exited"])
y = train["exited"]
# splitting
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.25, 
                                                    random_state=rand_state, 
                                                    stratify=y)


# define model
catboost_model = CatBoostClassifier(
    random_seed=rand_state, cat_features=categorical_features, verbose=100, n_estimators=2000
)


# fit the model
catboost_model.fit(X_train, y_train)
y_pred_prob = catboost_model.predict_proba(X_test)[:, 1]


# roc curve and auc
roc_auc = roc_auc_score(y_test, y_pred_prob)
print(f"roc auc score: {roc_auc:.6f}")


# feature importance of the model
catboost_model.get_feature_importance(prettified=True).round(2).style.format(
    precision=2
).background_gradient(cmap="Reds")


feature_importance = catboost_model.get_feature_importance(prettified=True)

plt.figure(figsize=(10, 8))
sns.barplot(x="Importances", y="Feature Id", data=feature_importance, palette="Dark2")
plt.title("Feature Importance")
plt.show()


plt.style.use("default")
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)


plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkgreen', label='ROC')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend()
plt.show()


y_pred_test = catboost_model.predict_proba(test_df)[:, 1]
y_pred_test


sample_submission["Exited"] = y_pred_test
sample_submission.to_csv("submission(cat).csv", index=False)
sample_submission.head()

