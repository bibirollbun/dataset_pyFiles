import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
import warnings 
warnings.filterwarnings("ignore")


train_path="/kaggle/input/playground-series-s5e7/train.csv"
test_path="/kaggle/input/playground-series-s5e7/test.csv"
sub_path="/kaggle/input/playground-series-s5e7/sample_submission.csv"


train_df=pd.read_csv(train_path)
test_df=pd.read_csv(test_path)


train_df.info()


train_df.isnull().sum()


train_df.isnull().any(axis=1).sum()


test_df.isnull().sum()


train_df.drop("id",axis=1,inplace=True)



numerical_columns=train_df.select_dtypes(include=["number"]).columns.to_list()
categorical_columns=train_df.select_dtypes(exclude=["number"]).columns.to_list()
target_column="Personality"
print("numerical cols:",numerical_columns)
print("categorical_cols:",categorical_columns)


for col in categorical_columns:
    print(f"unique values in column: {col} are : {train_df.dropna()[col].unique()}")


categorical_columns.remove(target_column)


# for numerical columns 
train_df[numerical_columns]=train_df[numerical_columns].fillna(train_df[numerical_columns].mean())

#for categorical
for col in categorical_columns:
    train_df[col].fillna(train_df[col].mode()[0], inplace=True)



# for numerical columns 
test_df[numerical_columns]=test_df[numerical_columns].fillna(train_df[numerical_columns].mean())

#for categorical
for col in categorical_columns:
    test_df[col].fillna(test_df[col].mode()[0], inplace=True)


fig,axs=plt.subplots(1,3,figsize=(12,8))
axs[0].pie(
    train_df[target_column].value_counts(),
    labels=train_df[target_column].value_counts().index,
    autopct='%1.1f%%',
    startangle=90,
    colors=sns.color_palette('Set2'),
    wedgeprops={'edgecolor': 'black'},
)
axs[0].set_title("Perosnality Distribution")
for i,col in enumerate(categorical_columns):
    axs[i+1].pie(
    train_df[col].value_counts(),
    labels=train_df[col].value_counts().index,
    autopct='%1.1f%%',
    startangle=90,
    colors=sns.color_palette('Set2'),
    wedgeprops={'edgecolor': 'black'},
)
    axs[i+1].set_title(f"{col} Distribution")

plt.tight_layout()
plt.subplots_adjust(top=0.85)
plt.show()


fig,axs=plt.subplots(1,2,figsize=(8,4))
sns.histplot(data=train_df,x="Stage_fear",hue="Personality",ax=axs[0],stat="percent", multiple='dodge')
sns.histplot(data=train_df,x="Drained_after_socializing",hue="Personality",ax=axs[1],stat="percent",multiple='dodge')
plt.tight_layout()
plt.subplots_adjust(top=0.85)
plt.show()


fig, axs = plt.subplots(1, len(numerical_columns), figsize=(5 * len(numerical_columns), 6))

if len(numerical_columns) == 1:
    axs = [axs]

palette = sns.color_palette("Set2")  # Soft pastel colors

for i, col in enumerate(numerical_columns):
    sns.boxplot(
        data=train_df,
        y=col,
        x="Personality",
        ax=axs[i],
        palette=palette,
        showfliers=True,  # Show outliers
        linewidth=1.5     # Thicker box borders
    )
    axs[i].set_title(f'Distribution of {col}', fontsize=14, fontweight='bold')
    axs[i].set_xlabel('')
    axs[i].set_ylabel(col, fontsize=12)
    axs[i].tick_params(axis='x', rotation=30)  # Rotate x labels if needed

plt.tight_layout()
plt.subplots_adjust(top=0.85, wspace=0.35)

fig.suptitle('Boxplots of Numerical Features by Personality', fontsize=18, fontweight='bold')
plt.show()


sns.set_theme(style="ticks", palette="pastel")

# Assuming you have a DataFrame `df` and a target column 'target' to hue by
# Replace 'target' with your categorical column or remove `hue` if not needed

pairplot = sns.pairplot(
    train_df,
    hue='Personality',          # categorical variable for color coding
    diag_kind='kde',       # KDE on the diagonal instead of histogram
    kind='scatter',        # scatter plots in off-diagonal
    plot_kws={'alpha':0.7, 's':40, 'edgecolor':'k'},  # transparency, point size, edge color
    diag_kws={'shade':True}  # smooth KDE with shading
)

# Improve legend and plot aesthetics
pairplot.fig.suptitle("Pairplot", y=1.02, fontsize=16, fontweight='bold')

plt.show()


from sklearn.preprocessing import OneHotEncoder,LabelEncoder,StandardScaler
from sklearn.model_selection import StratifiedKFold



features=numerical_columns+categorical_columns
X=train_df[features]
y=train_df[[target_column]]


for col in categorical_columns:
    X[col]=X[col].map({"Yes":1,"No":1})
y[target_column]=y[target_column].map({"Extrovert":0,"Introvert":1})


scaler=StandardScaler()
X[numerical_columns]=scaler.fit_transform(X[numerical_columns])


from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import time
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score,precision_score,recall_score,f1_score,ConfusionMatrixDisplay


def train_model(model,X,y,return_model=False):
    skf=StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
    models=[]
    for i, (train_index, test_index) in enumerate(skf.split(X, y)):
        print(f"Training the Fold {i}:\n-----------------------------------------------------")
        start=time.time()
        X_train,X_val=X[train_index],X[test_index]
        y_train,y_val=y[train_index],y[test_index]
        curr_model=model
        curr_model.fit(X_train,y_train)
        y_pred=curr_model.predict(X_val)
        end=time.time()
        print(f"time taken to train the model: {end-start}")
        print(f"Metrics on the {i}th Fold is \n {classification_report(y_val,y_pred)}")
        cm = confusion_matrix(y_val,y_pred, labels=curr_model.classes_)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=curr_model.classes_)
        disp.plot()
        plt.show()
        models.append(curr_model)
    if return_model:
        return models
        


log_model=LogisticRegression(penalty='l2', C=0.01,  random_state=None, solver='lbfgs', max_iter=1000 )


train_model(log_model,X.values,y.values)





final_model=log_model.fit(X.values,y.values)


## converting testing dataframe
def convert_test(df):
    for col in categorical_columns:
        df[col]=df[col].map({"Yes":1,"No":1})
    df[numerical_columns]=scaler.transform(df[numerical_columns])
    return df.values

X_test=test_df[features]
X_test=convert_test(X_test)


y_test=final_model.predict(X_test)


sub_df=pd.read_csv(sub_path)
sub_df["Personality"]=y_test
sub_df["Personality"]=sub_df["Personality"].map({0:"Extrovert",1:"Introvert"})
sub_df.to_csv("/kaggle/working/submission.csv",index=False)




