import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


train=pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv')
test=pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv')


train.head(3)


test.head(3)


print("Size of the Training Dataset :",train.shape)
print("Size of the Testing Dataset :",test.shape)


train.info() #info of the training dataset


test.info()#info of the testing dataset


train.columns #printing the columns


test.columns #printing the columns


#Checking for null values on training dataset
train.isna().sum()


#Checking for null values on testing dataset
test.isna().sum()


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
num_cols=train.select_dtypes(include="number").columns.drop("id")
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


from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score 
from sklearn.model_selection import train_test_split


x=train.drop(['Fertilizer Name'],axis=1)
y=train['Fertilizer Name']


model=LabelEncoder()
y_enc=model.fit_transform(y)


le=LabelEncoder()
cat_cols=['Soil Type','Crop Type']
for i in cat_cols:
    x[i]=le.fit_transform(x[i])
    test[i]=le.transform(test[i])


x.dtypes


test.dtypes


X_train,X_val,y_train,y_val=train_test_split(x,y_enc,test_size=0.3,random_state=42)


from xgboost import XGBClassifier


xgb_best_params = {
   'n_estimators': 3500,
    'max_depth':12,
    'subsample': 0.9,
    'colsample_bytree':0.5,
    'learning_rate':0.03,
    'gamma':0.5,
    'max_delta_step': 5,
    'early_stopping_rounds':50,
    # 'objective':'multi:softprob',
    # 'objective':'rank:map',
    'objective': 'multi:softmax',
    'enable_categorical':True,
    'tree_method':'hist',
    'device':'cuda',
    'reg_alpha':2.7,
    'reg_lambda':1.4,
    'num_parallel_tree': 5,
    # 'disable_default_eval_metric': True,    
    # 'eval_metrics': 'accuracy',
    # 'verbose': 100
}
xgb=XGBClassifier(**xgb_best_params)


xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=50)


y_probs=xgb.predict_proba(X_val)


def get_top_k_predictions(probs, k):
    return np.argsort(probs, axis=1)[:, -k:][:, ::-1]


# Single-label MAP@K
def mapk_single_label(y_true, y_pred, k=3):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)[:, :k]
    matches = (y_true.reshape(-1, 1) == y_pred)
    ranks = np.where(matches.any(axis=1), matches.argmax(axis=1) + 1, np.inf)
    return np.mean(ranks ** -1)


# Multi-label MAP@K (each instance has one label in a list)
def apk(actual, predicted, k=10):
    if not actual:
        return 0.0
    predicted = predicted[:k]
    score = 0.0
    num_hits = 0
    seen = set()
    actual_set = set(actual)
    for i, p in enumerate(predicted):
        if p in actual_set and p not in seen:
            num_hits += 1
            score += num_hits / (i + 1)
            seen.add(p)
    return score / min(len(actual), k)
def mapk(actual, predicted, k=10):
    return np.mean([apk([a], p, k) for a, p in zip(actual, predicted)])


# Evaluate MAP@K for k = 1 to k_max
k_max = 8
k_values = range(1, k_max)
mapk_single_scores = []
mapk_multi_scores = []

for k in k_values:
    top_k_preds = get_top_k_predictions(y_probs, k)
    mapk_single_scores.append(mapk_single_label(y_val, top_k_preds, k))
    mapk_multi_scores.append(mapk(y_val, top_k_preds, k))


test_proba = xgb.predict_proba(test)


test_proba.shape


preds = np.argsort(test_proba, axis=1)[:, ::-1]
preds


test_top_3 = np.argsort(test_proba, axis=1)[:, -3:][:, ::-1]
test_top_3


test_top_3_names = model.inverse_transform(test_top_3.ravel())
test_3_picks = test_top_3_names.reshape(test_top_3.shape)

test_3_picks


preds_df = pd.DataFrame({
    'id': test['id'],
    'Fertilizer Name': [' '.join(preds) for preds in test_3_picks]
})

preds_df.head(4)


preds_df.to_csv('/kaggle/working/submission.csv', index=False)




