#Load data set and import libaries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
import warnings
from pandas.errors import PerformanceWarning
warnings.simplefilter(action="ignore", category=PerformanceWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


train_df = pd.read_csv('/kaggle/input/playground-series-s5e8/train.csv')
sample = pd.read_csv("/kaggle/input/playground-series-s5e8/sample_submission.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e8/test.csv")
original = pd.read_csv("/kaggle/input/bank-marketing-dataset-full/bank-full.csv", sep=';')
train_df=train_df.drop("id",axis=1)
test_df=test_df.drop("id",axis=1)
original['y'] = original['y'].map({'no': 0, 'yes': 1})


train_df.describe()


categorical_col=["job","marital","education","default","housing","loan","contact","month","poutcome"]
print(categorical_col)
numerical_col=["age","balance","day","duration","campaign","pdays","previous"]
print(numerical_col)


print(train_df.isna().sum())
print(test_df.isna().sum())


#Count plot for categorical columns
for col in categorical_col:
    plt.figure(figsize=(8,6))
    sns.countplot(x=col, data=train_df)
    plt.xticks(rotation=45)
    plt.show()

 


%matplotlib inline

for col in numerical_col:
    plt.figure(figsize=(12,10))
    sns.histplot(train_df[col],kde=True,bins=30)
    plt.show()


#Relation between continous feature and target feature
plt.figure(figsize=(16,10))
plt.subplot(1,3,1)
sns.boxplot(data=train_df,x="y",y="age",palette='Set2')
plt.subplot(1,3,2)
sns.boxplot(data=train_df,x="y",y="duration",palette='Set2')
plt.subplot(1,3,3)
sns.boxplot(data=train_df,x="y",y="balance",palette='Set2')
plt.show()


#relation between import feature and target feature
plt.figure(figsize=(10,8))
sns.countplot(data=train_df,x="job",hue="y",palette="Set2")
plt.xticks(rotation=45)
plt.show()
sns.countplot(data=train_df,x="marital",hue="y",palette="Set2")
plt.show()



COLS = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing',
       'loan', 'contact', 'day', 'month', 'duration', 'campaign', 'pdays',
       'previous', 'poutcome',]
def new_fe(df):
    df['balance_log'] = np.log1p(df['balance'].clip(lower=0))
    df["contact_before"]=(df["pdays"]!=-1).astype(int)
    df["duration_sin"]=np.sin(2*np.pi*df["duration"]/800)
    df["duration_cos"]=np.cos(2*np.pi*df["duration"]/800)
    return df
train_df=new_fe(train_df)
test_df=new_fe(test_df)



contact_before_count=train_df.groupby("contact_before")["y"].value_counts(normalize=True).unstack().fillna(0)
contact_before_count.plot(kind="bar",stacked=False,figsize=(10,8))
plt.legend(title="contact before and after accept",labels=["no","yes"])
plt.show()



print(train_df.dtypes)


cat_cols = ['job','marital', "education", 'contact', 'poutcome','month','default','housing','loan','job_edu']

mean = train_df['y'].mean() 

for c in COLS:
    new_col = f"{c}_mean_target_orig"
    train_df[new_col] = train_df[c].map(original.groupby(c)['y'].mean())
    train_df[new_col] = train_df[new_col].fillna(mean)
    test_df[new_col] = test_df[c].map(original.groupby(c)['y'].mean())
    test_df[new_col] = test_df[new_col].fillna(mean)


 # Label encode categorical columns safe

for col in categorical_col:
    le=LabelEncoder()
    train_df[col+"_enc"]=le.fit_transform(train_df[col].astype(str))
    test_df[col+"_enc"]=le.fit_transform(test_df[col].astype(str))
#drop all caegorical columns
train_df = train_df.drop(categorical_col, axis=1, errors='ignore')
test_df  = test_df.drop(categorical_col, axis=1, errors='ignore')



print(train_df.dtypes)


train_df.head()


def train_lightgbm(train_df, test_df, target_col="y"):
    X = train_df.drop(columns=[target_col])
    y = train_df[target_col]
    X_test = test_df.copy()
    
    n_splits = 5
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    y_probs = np.zeros(len(X_test))
    models = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
        
        model = lgb.LGBMClassifier(
            n_estimators=30000,
            class_weight='balanced',
            learning_rate=0.06,
            num_leaves=100,
            max_depth=10,
            min_child_samples=7,
            subsample=0.8,
            colsample_bytree=0.5,
            reg_alpha=0.8,
            reg_lambda=0.3,
            max_bin=4859,
            random_state=2003,
            verbosity=-1,
            boosting_type='gbdt',
            eval_metric='auc'
        )
        
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            eval_names=["valid"],
            callbacks=[
                lgb.early_stopping(300),
                lgb.log_evaluation(500)
            ]
        )
        
        models.append(model)
        y_probs += model.predict_proba(X_test)[:, 1] / n_splits
    
    return y_probs, models


# à¦¬à§�à¦¯à¦¬à¦¹à¦¾à¦°
y_probs, models = train_lightgbm(train_df, test_df, target_col="y")


testify = pd.read_csv("/kaggle/input/playground-series-s5e8/test.csv")

# y_probs is already the averaged predictions from all folds
submission = pd.DataFrame({
    'id': testify['id'],
    'target': y_probs  # from CV loop
})

# Save to CSV
submission.to_csv('submission.csv', index=False)

print("âœ… Submission file saved as 'submission.csv'")


