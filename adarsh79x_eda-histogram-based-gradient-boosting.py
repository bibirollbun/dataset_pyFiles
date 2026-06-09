!pip install shap


pip install --upgrade scikit-learn



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()

import shap
#import ipynbname


from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.metrics import root_mean_squared_error

import warnings
warnings.filterwarnings('ignore')


!pip list| grep scikit-learn


ROOT_PATH = "/kaggle/input/playground-series-s5e4/"


podcast_df = pd.read_csv(ROOT_PATH + "train.csv")
podcast_df.set_index("id", inplace=True)


podcast_df.head()


def summarize_df(df):

    print("DataFrame Info:")
    print("-----------------------") 
    display(df.info()) 
    print("\n")

    print("DataFrame Description:")
    print("----------------------") 
    display(df.describe(percentiles=[.25, .50, .75, .99]).T)
    print("\n")

    print("Number of Null Values:")
    print("----------------------") 
    display(df.isnull().sum()) 
    print("\n")

    print("Number of Duplicated Rows:")
    print("---------------------------")
    display(int(df.duplicated().sum())) 
    print("\n")

    print("Number of Unique Values:")
    print("-------------------------")
    display(df.nunique()) 
    print("\n")

    print("DataFrame Shape:")
    print("----------------")
    print(f"No. of Rows:    {df.shape[0]}\nNo. of Columns: {df.shape[1]}")


summarize_df(podcast_df)


podcast_df["Genre"].unique()


NUM_FEATURES = podcast_df.select_dtypes(include="number").columns
TEXT_FEATURES = podcast_df.select_dtypes(include="object").columns

NON_CAT_FEATURES = ["Podcast_Name", "Episode_Title"]
CAT_FEATURES = [feature for feature in TEXT_FEATURES if feature not in NON_CAT_FEATURES]

#converting certain features into category dtype
for feature in CAT_FEATURES:
    podcast_df[feature] = podcast_df[feature].astype("category")     


# removing all rows where 'Episode_Length_minutes' feature has null entries
podcast_df = podcast_df[~podcast_df["Episode_Length_minutes"].isnull()]


# plt.figure(figsize=(10,12))
# n_features = len(NUM_FEATURES)
# for i ,col in zip(range(1,n_features*2 - 1, 2), NUM_FEATURES):
#     plt.subplot(n_features,2,i)
#     sns.boxplot(podcast_df[col])
    
#     plt.subplot(n_features,2, i+1)
#     sns.kdeplot(podcast_df[col])
# plt.tight_layout()
# plt.show()


# plt.figure(figsize=(10,6))
# for i, feature in enumerate(CAT_FEATURES,1):
#     order = podcast_df[feature].value_counts().index # sort by count descending
#     plt.subplot(2,2,i)
#     sns.countplot(podcast_df[feature], order=order)
# plt.tight_layout() #is used to ensure that the subplots fit well in the figure and do not overlap.
# plt.show()


sns.pairplot(podcast_df)
plt.show()


plt.figure(figsize=(5, 3))
sns.heatmap(podcast_df.select_dtypes(include="number").corr(), annot=True, cmap="coolwarm", center=0)
plt.show()


# To split podcast df into feature matrix and targets
def split_features_target(dataset:pd.DataFrame,subs=False): #subs -> is it final submission file or not
    if not subs:
        X = dataset.drop(columns=["Listening_Time_minutes",'Podcast_Name', 'Episode_Title'])
        y = dataset["Listening_Time_minutes"]
        return X, y
        
    X = dataset.drop(columns=['Podcast_Name', 'Episode_Title'])
    return X


X, y = split_features_target(podcast_df)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)


X_train.head(1)


hgb_reg = make_pipeline(
    make_column_transformer((OrdinalEncoder(), CAT_FEATURES), remainder="passthrough"),
    
    HistGradientBoostingRegressor(
        max_iter=500,
        #categorical_features=(X_train.dtypes == "category").values,
        validation_fraction=0.15,
        early_stopping=True,      #not necessary, if dataset > 10000 rows : automatically sets to True
        n_iter_no_change=15,
        tol=1e-5,
        random_state=42
            )
)

hgb_reg.fit(X_train, y_train)


points = hgb_reg.named_steps['histgradientboostingregressor'].train_score_.size
points


x = np.linspace(1, points+1, points, dtype="int")
plt.style.use("default")
plt.plot(x, np.sqrt(-hgb_reg.named_steps['histgradientboostingregressor'].train_score_),'g-', lw=3,label="train score") #clm color|line|marker
plt.plot(x, np.sqrt(-hgb_reg.named_steps['histgradientboostingregressor'].validation_score_), 'r--', lw=2, label="validation score") #negative of validation score is used as  
plt.xlim([0,points+20])                                                                 #validation scores are utility functions
plt.xlabel("no of trees")
plt.ylabel("loss (root sqaured error)")
plt.legend(loc="upper right")
plt.grid()
plt.show()


test_error = root_mean_squared_error(hgb_reg.predict(X_train), y_train)
f"Test error: {test_error:.3f}"


ans = np.c_[hgb_reg.predict(X_train), y_train]


ans[:25]


test_set = pd.read_csv(ROOT_PATH + "test.csv")
test_set.set_index("id",inplace=True)
Xs = split_features_target(test_set, subs=True)


final_preds = hgb_reg.predict(Xs)


final_subs = pd.DataFrame(np.c_[test_set.index, final_preds]).rename(columns={0: "id", 1: "Listening_Time_minutes"})
final_subs.to_csv("subsmission.csv", index=False)


# model = hgb_reg.named_steps['histgradientboostingregressor']
# X_transformed = hgb_reg.named_steps['columntransformer'].transform(X_train)

# explainer = shap.TreeExplainer(model) 
# shap_values = explainer.shap_values(X_transformed)

# feature_names = hgb_reg.named_steps['columntransformer'].get_feature_names_out()

# shap.summary_plot(shap_values, X_transformed, feature_names=feature_names)

