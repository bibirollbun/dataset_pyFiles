import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


train=pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv')
test=pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv')
org=pd.read_csv('/kaggle/input/fertilizer-prediction/Fertilizer Prediction.csv')


train.shape


test.info


test.shape


org.shape


org.head(2)


org.describe().T


test_new=test


test_new


train=train.drop('id',axis=1)
test=test.drop('id',axis=1)
train_new=pd.concat([train,org],ignore_index=True)
#test_new=pd.concat([test,org],ignore_index=True)


train_new.info()


#Checking for null values
train_new.isna().sum()


#Checking for null values
test.isna().sum()


#Counting the total no of Fertilizer
Fertilizer_count=train['Fertilizer Name'].value_counts().reset_index()
Fertilizer_count.columns = ['Fertilizer Name', 'Count']
Fertilizer_count


Soil_count=train['Soil Type'].value_counts().reset_index()
Soil_count.columns = ['Soil Type', 'Count']
Soil_count


plt.figure(figsize=(10, 5))
sns.countplot(data=train,x="Fertilizer Name" ,
              order=train["Fertilizer Name"].value_counts().index,palette="Set1")
plt.title("Fertilizer Count")
plt.xlabel("Count")
plt.ylabel("Fertilizer Name")
plt.show()


plt.figure(figsize=(12, 7))
sns.countplot(data=train, x='Soil Type', hue='Fertilizer Name', palette='tab10')
plt.title('Fertilizer Name Distribution by Soil Type')
plt.xticks(rotation=45)
plt.legend(title='Fertilizer Name', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()


#Categorical Column Distribution
cat_cols=['Soil Type','Crop Type']
for i in cat_cols:
    plt.figure(figsize=(10,5))
    sns.countplot(data=train,x=i,order=train[i].value_counts().index)
    plt.title(f"{i} distribution")  
    plt.show()



#Numeric Column Distribution
num_cols=train.select_dtypes(include="number").columns
ncols = 3
nrows = int(np.ceil(len(num_cols) / ncols))
fig, axes = plt.subplots(nrows, ncols, figsize=(16, nrows*3))
axes = axes.flatten()

for ax, col in zip(axes, num_cols):
    sns.histplot(train[col], kde=True, ax=ax)
    ax.set_title(col)

plt.tight_layout()
plt.show()



#Feature Vs Target
vc = train['Fertilizer Name'].value_counts().sort_values(ascending=False)
for i in num_cols:
    plt.figure(figsize=(10,5))
    sns.boxplot(data=train,x="Fertilizer Name",y=i,
               order=vc.index)
    plt.xticks(rotation=45)
    plt.show()


#Spearman Correlation
corr=train[num_cols].corr(method="spearman")
sns.heatmap(corr,cmap="coolwarm",annot=True,square=True)
plt.title("Spearman Correlation")
plt.show


#Pair wise Relationship
sns.pairplot(
    train[num_cols.union(['Fertilizer Name'])],
     hue="Fertilizer Name", corner=True, diag_kind="kde",
    height=1.5, plot_kws=dict(alpha=.3, linewidth=0)
)
plt.show()


x=train_new.drop(['Fertilizer Name'],axis=1)
y=train_new['Fertilizer Name']


from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


model=LabelEncoder()
y_enc=model.fit_transform(y)


cat_cols=['Soil Type','Crop Type']
x=pd.get_dummies(x,columns=cat_cols)
test=pd.get_dummies(test,columns=cat_cols)


X_train,X_val,y_train,y_val=train_test_split(x,y_enc,test_size=0.2,random_state=42)


from lightgbm import LGBMClassifier

model_lgbm = LGBMClassifier(objective='multiclass',
                            n_estimators=500,
                            learning_rate=0.05,
                            num_iterations=500,
                            min_data_in_leaf = 5000,
                            lambda_l2 = 100,
                            verbose=0,
                            random_state=0)


model_lgbm.fit(X_train,y_train).score(X_val,y_val)


from catboost import CatBoostClassifier

model_cat = CatBoostClassifier(learning_rate=0.05,
                               boosting_type='Plain',
                               grow_policy = "Depthwise",
                               min_data_in_leaf=5000,
                               verbose=50)

model_cat.fit(X_train,y_train).score(X_val,y_val)


y_probs=model_cat.predict_proba(X_val)


test_probs=model_cat.predict_proba(test)


def get_top_k_predictions(probs, k):
    return np.argsort(probs, axis=1)[:, -k:][:, ::-1]


#inversing the labelencoder values
top3_preds = get_top_k_predictions(test_probs, k=3)
top3_labels = model.inverse_transform(top3_preds.ravel()).reshape(top3_preds.shape)
top3_labels


submission = pd.DataFrame({
    'id': test_new['id'],
    'Fertilizer Name':[' '.join(preds) for preds in top3_labels]})


submission.to_csv('/kaggle/working/submission.csv',index=False)


submission

