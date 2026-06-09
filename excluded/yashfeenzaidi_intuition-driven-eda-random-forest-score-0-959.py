import pandas as pd
train_df = pd.read_csv("/kaggle/input/playground-series-s5e8/train.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e8/test.csv")



train_df.head(40)


##I always try to start by observing my dataset usually trying to see how many of my columns are categorical, how many are numerical, as they both are dealt in separate ways
train_df.info()


#looking for null values
train_df.isnull().sum()


test_df.isnull().sum()


a=train_df.select_dtypes(include=["object"]).columns.tolist()
## I am trying to do two things here:
## 1) see unnique values in each categorical column in order to understand each categorical column best
## 2) Filling null values in each categorical column
for i in a:
  print(i)
  train_df.fillna({i:train_df[i].mode().iloc[0]},inplace=True)
  print(train_df[i].value_counts())
  print("\n")


##Then, i just filled the null values for each column ( mainly my target was numerical columns since i have not filled values there)
a=train_df.columns
for i in a:
  train_df.fillna({i:train_df[i].mode().iloc[0]},inplace=True)



train_df.isna().sum()


#Trying to understand my data basically my numerical columns so that i can see if i need to apply scaling/ detect outliers.
train_df.describe()
#Well from the description below i do have an idea now that i need scaling but which type of scaling would be suitable to it depends upon the existence of outliers so i need to check them in next cell too.



##Let me visualize each column for detecting outliers ofc they are not going to give me outliers but they will tell me which method to use in order to detect outliers.
# Usually if your data is not skewed it is preferred to use Z score for detecting outliers and when data is skewed you should be using IQR metthod.
import matplotlib.pyplot as plt
import seaborn as sns
a=train_df.select_dtypes(include=["int64","float64"])
a=a.drop(columns=['y', 'id'], errors='ignore')
for i in a:
  print(i)
  plt.hist(train_df[i])
  plt.show()
  # As you can see almost all of my numeric columns are skewed, so i will be using IQR method for detecting the outliers.



#Using IQR method to detect outliers for each column
for i in a:
  if i!="y":
    q1=train_df[i].quantile(0.25)
    q3=train_df[i].quantile(0.75)
    iqr=q3-q1
    upper_bound=q3+1.5*iqr
    lower_bound=q1-1.5*iqr
    outliers=train_df[(train_df[i]>upper_bound)|(train_df[i]<lower_bound)]
    print(i)
    print(f"{i}: {len(outliers)} outliers, {len(outliers)/len(train_df[i]):.2%} of data")
    print("\n")


#So, i have seen that my data indeed has outliers and so i need to deal with this. As you can see some of my columns have only positive values and positive skew, i am apply log transformation on them and on the ones remaining i am applying yeo-johnson.
import numpy as np
from sklearn.preprocessing import PowerTransformer
pt=PowerTransformer(method='yeo-johnson')
skew_threshold=0.5
for col in a:
    data=train_df[col]
    skewness=data.skew()
    print(f"{col} skewness: {skewness:.3f}")
    if abs(skewness)<skew_threshold:
        print(f"Skewness is low. No transformation applied to '{col}'.\n")
        continue
    if skewness>0 and (data>=0).all():
        print(f"Right-skewed and non-negative. Applying log1p to '{col}'.")
        train_df[col]=np.log1p(data)
        test_df[col]=np.log1p(test_df[col])
    else:
        print(f"Applying Yeo-Johnson transform to '{col}'.")
        pt.fit(data.values.reshape(-1,1))
        train_df[col]=pt.transform(data.values.reshape(-1,1)).flatten()
        test_df[col]=pt.transform(test_df[col].values.reshape(-1,1)).flatten()
    print()



train_df.describe()
#Trying to see if I still need scaling


#Now that i have handled outliers, i still need to check for skewness in order to determine which scaler to use as discussed earlier.
for col in a:
    skewness = train_df[col].skew()
    print(f"{col} skewness after transform: {skewness:.3f}")




#Since i still have skewness in my data, i will be using the robust scaler.
from sklearn.preprocessing import RobustScaler
r=RobustScaler()
for i in a:
  r.fit(train_df[i].values.reshape(-1,1))
  train_df[i]=r.transform(train_df[i].values.reshape(-1,1)).flatten()
  test_df[i]=r.transform(test_df[i].values.reshape(-1,1)).flatten()


train_df.describe()
#Now, i can see my features are on more consistent scale.


b=train_df.select_dtypes(include=["int64","float64"])
a=b.corr()["y"].sort_values()
print(a)


import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(10,8))
sns.heatmap(b.corr(), annot=True, cmap='coolwarm')
plt.show()


test_df.head()


# Since You can not calculate the correlation of categorical columns with the target column, it is advised to look for association which basically kind of tells you how many "yes" (in case of a binary classification) are there for each category in a column.
b=train_df.select_dtypes(include=['object'])
for col in b:
    print(f"\nColumn: {col}")
    print(train_df.groupby(col)['y'].mean().sort_values(ascending=False))
# As you can see in column==job, students have a good association number, this shows that students are more likely to subscribe to a bank term deposit.
# I also calculated these scores, so i can think of a way to create new columns from these existing columns( this is a useful technique in feature engineering) or drop some columns if i think they are not really helpful at all.


#lets try to make some new columns
# merging 'housing' and 'loan' into one flag 'has_loan'
# reason: both indicate financial obligation, and in both cases having a loan lowered target rate
# checked target means before: housing_yes=0.075, loan_yes=0.056 -> both much lower than 'no'
# made has_loan=1 if either housing or loan is yes, else 0
# after merge: has_loan=0 -> 0.1895, has_loan=1 -> 0.0747, bigger gap than before
# stronger signal + fewer columns, so we keep 'has_loan' and drop original two
train_df['has_loan'] = ((train_df['housing'] == 'yes') | (train_df['loan'] == 'yes')).astype(int)
train_df.groupby('has_loan')['y'].mean()
test_df['has_loan'] = ((test_df['housing'] == 'yes') | (test_df['loan'] == 'yes')).astype(int)
test_df.drop(columns=['housing','loan'],axis=1,inplace=True)
train_df.drop(columns=['housing','loan'],axis=1,inplace=True)




# months with similar target rates grouped into high/mid/low to cut noise & keep main patterns
high_months=['mar','sep','dec','oct']
mid_months=['apr','feb']
low_months=['jan','aug','nov','jun','jul','may']
def group_month(m):
    if m in high_months: return 'high'
    elif m in mid_months: return 'mid'
    else: return 'low'

train_df['month_group']=train_df['month'].apply(group_month)
test_df['month_group']=test_df['month'].apply(group_month)
train_df.drop('month',axis=1,inplace=True)
test_df.drop('month',axis=1,inplace=True)
train_df.groupby('month_group')['y'].mean()
#The reason we are checking mean of the newly formed columns is to make sure the association we saw in the original column is preserved.
#Like for the column marital i could not think of a way to group them and keep the association intact as well, so i didnt do anything with that and same goes for job and education.







from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import numpy as np

X = train_df.drop(columns=['id', 'y'])
y = train_df['y']
categorical_cols = X.select_dtypes(include=['object', 'category']).columns
numeric_cols = X.select_dtypes(include=['number']).columns
preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
    ('num', 'passthrough', numeric_cols)
])
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
auc_scores = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    # RandomForest pipeline
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('rf', RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        ))
    ])

    model.fit(X_train, y_train)
    y_proba = model.predict_proba(X_val)[:, 1]

    auc = roc_auc_score(y_val, y_proba)
    auc_scores.append(auc)

    print(f"Fold {fold} AUC: {auc:.4f}")

print(f"\nMean AUC: {np.mean(auc_scores):.4f} ± {np.std(auc_scores):.4f}")



test_proba = model.predict_proba(test_df.drop(columns=['id']))[:, 1]
output = test_df[['id']].copy()
output['y'] = test_proba

output.to_csv('submission.csv', index=False)
print("Submission file saved as submission.csv")


