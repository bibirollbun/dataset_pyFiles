import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
from sklearn.model_selection import KFold, StratifiedKFold
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
from sklearn.preprocessing import LabelEncoder
from matplotlib import pyplot as plt


train = pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv')

use_original = True
if use_original:
    train_original = pd.read_csv('/kaggle/input/fertilizer-prediction/Fertilizer Prediction.csv').reset_index()
    train_original.columns = train.columns
    train = pd.concat([train, train_original])


print(f"This dataset has {train.shape[0]} rows and {train.shape[1]} columns.")
print(f"There are {train.isna().sum().sum()} NA's in the dataset.")
print(f"There are {str(train.duplicated().sum())} duplicates in the dataset.")

# quick look at the data
train.head(3)

# We don't have any NA's in this dataset.


unique_id = 'id'
target = 'Fertilizer Name'
categorical_columns = ['Soil Type', 'Crop Type']
numerical_columns = ['Temparature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous']


print('The unique identifier is unique.') if train[unique_id].nunique() == train.shape[0] else print('The unique identifier is not unique.')


skewness_threshold = .5 # can tune / experiment with this value
skewed_cols = [col for col in numerical_columns if train[col].skew() > skewness_threshold]

print(f'There are {len(skewed_cols)} skewed columns: {str(skewed_cols)}')

# According to our threshold of .5, none of the numerical features are skewed.


imbalance_threshold = 3.33
imbalanced_cols = [col for col in categorical_columns if (train[col].count() / train[col].value_counts().values.min()) > imbalance_threshold]

print(f'There are {len(imbalanced_cols)} imbalanced columns: {str(imbalanced_cols)}')

# According to our imbalance_threshold of 3.33, both categorical features are imbalanced.


high_cardinality_threshold = 8
high_cardinality_columns = [x for x in categorical_columns if train[x].nunique() > high_cardinality_threshold]

print(f'There are {len(high_cardinality_columns)} high cardinality columns: {str(high_cardinality_columns)}')

# According to our high cardinality threshold, only 'Crop Type' has a high cardinality.


train_unique_cols = [x for x in train.drop([target],axis=1).columns if x not in test.columns]
print('All columns in train exist in test.') if not train_unique_cols else print(f'The following train columns are not in test: {train_unique_cols}')

test_unique_cols = [x for x in test.columns if x not in train.columns]
print('All columns in test exist in train.') if not test_unique_cols else print(f'The following test columns are not in train: {test_unique_cols}')

# Perfect, we can use all train columns in our modelling efforts.


for col in numerical_columns:
    _, axes = plt.subplots(1,2,figsize=(10,5),sharex=False,sharey=False)
   
    sns.histplot(data=train, x=col,bins=20,ax=axes[0],log_scale=False, color = 'lightblue')
    axes[0].set_title(f'Distribution of {col}')

    sns.boxplot(data=train,x=col,ax=axes[1], showfliers=False, color = 'honeydew')
    axes[1].set_title(f'{col}')

    plt.tight_layout()
    plt.show()


for col in categorical_columns:
    if col not in high_cardinality_columns:
        
        _, ax = plt.subplots(1,1,figsize=(6,3))
    
        sns.countplot(data=train, y = col, color = 'lightblue')
        ax.set_title(f'{col}')

        plt.tight_layout()
        plt.show()


print(f"This dataset has {test.shape[0]} rows and {test.shape[1]} columns.")
print(f"There are {test.isna().sum().sum()} NA's in the dataset.")
print(f"There are {str(test.duplicated().sum())} duplicates in the dataset.")
# quick look at the data
test.head(3)

# Same as train.csv, no NA's in test.csv


# Count plot of the target

_, ax = plt.subplots(1,1,figsize=(8,4))

sns.countplot(data=train, y = target)
ax.set_title(f'{target}')

plt.show()
plt.tight_layout()

# We can see the target has two groupings, 'Urea' & 'DAP' in one group and '20-20', '28-28', '17-17-17', '10-26-26', & '14-35-14' in the other.


def pre_process(df,categorical_columns, numerical_columns, unique_id, target):

    begin_col_cnt = len(df.columns)

    for col in categorical_columns:
        df[col] = df[col].fillna('NA')
        df[col] = df[col].astype('category').cat.codes

    for col in numerical_columns:
        mean_value = df[col].mean()
        df[col] = df[col].fillna(mean_value)

    df['Temparature_bin6'] = pd.qcut(df['Temparature'], q=6, labels=[1, 2, 3, 4,5,6]).astype(int)
    df['Potassium_bin6'] = pd.qcut(df['Potassium'], q=6, labels=[1, 2, 3, 4,5,6]).astype(int)
    df['Potassium_bin6_Potassium_ratio'] = pd.qcut(df['Potassium'], q=6, labels=[1, 2, 3, 4,5,6]).astype(int) / df['Potassium']
    df['Potassium_Phosphorous_bin6_sum'] = df['Potassium'] + pd.qcut(df['Phosphorous'], q=6, labels=[1, 2, 3, 4,5,6]).astype(int)

    df['Phosphorous_Phosphorous_bin6_ratio'] = df['Phosphorous'] / pd.qcut(df['Phosphorous'], q=6, labels=[1, 2, 3, 4,5,6]).astype(int)
    df['Phosphorous_Phosphorous_bin6_diff'] = df['Phosphorous'] - pd.qcut(df['Phosphorous'], q=6, labels=[1, 2, 3, 4,5,6]).astype(int)
    df['Humidity_log1p_Humidity_bin6_ratio'] = np.log1p(df['Humidity'].astype(float)) / pd.qcut(df['Humidity'], q=6, labels=[1, 2, 3, 4,5,6]).astype(int)

    end_col_cnt = len(df.columns)
    print(f'Created {end_col_cnt-begin_col_cnt} columns.')
    
    return df

train_orig = pre_process(train.copy(),categorical_columns, numerical_columns, unique_id, target)
test_orig = pre_process(test,categorical_columns, numerical_columns, unique_id, target)



# single_apk function taken from the following notebook -- please upvote that one as well!
# https://www.kaggle.com/code/wordcards/pgs5-6-lightgbm-baseline
from sklearn.metrics import accuracy_score
def single_apk(y, oof):
    sorted_oof = np.argsort(oof, axis=1)[:, ::-1][:, :3]

    score = 0
    for i in range(3):
        score += accuracy_score(y, sorted_oof[:, i]) / (i+1)

    return score


X = train_orig.drop([target, unique_id],axis=1)
y = train_orig[target].astype('category')

test_ids = test[unique_id]
X_test = test.drop(unique_id,axis=1).copy()

# transforming y to work w/ probability prediction
le = LabelEncoder()
y = pd.DataFrame(le.fit_transform(y))


X = train_orig.drop([target, unique_id],axis=1)
y = train_orig[target].astype('category')

#test_ids = test[unique_id]
X_test = test_orig.drop(unique_id,axis=1).copy()

# transforming y to work w/ probability prediction
le = LabelEncoder()
y = pd.DataFrame(le.fit_transform(y))

scores = []
oof = np.zeros((len(y), 7))
test_preds = np.zeros((X_test.shape[0], 7))

NFOLDS = 5
cv_method = KFold(n_splits=NFOLDS, shuffle=True, random_state=1)
params = {'lambda': 0.04181394082422529, 'alpha': 1.322577130703578e-07, 'max_depth': 13, 'min_child_weight': 74, 'n_estimators': 2085, 'subsample': 0.6903543685502419, 'colsample_bytree': 0.2640850346741723, 'eta': 0.029124005831763014}
for fold, (idx_tr, idx_va) in enumerate(cv_method.split(X, y), start=1):
    X_tr = X.iloc[idx_tr]
    X_va = X.iloc[idx_va]
    y_tr = y.iloc[idx_tr]
    y_va = y.iloc[idx_va]

    model = xgb.XGBClassifier(**params, n_jobs=-1,objective='multi:softprob',booster='gbtree', device='gpu', enable_categorical=True, random_state=1,missing=np.inf)
    model.fit(X_tr, y_tr,verbose=0)
    y_pred = model.predict_proba(X_va)

    oof[idx_va] = y_pred
    test_preds += model.predict_proba(X_test) / NFOLDS
    score = single_apk(y_va, oof[idx_va])
    print(f"# Fold {fold}: {score=:.5f}")
    scores.append(score)

score = np.mean(scores)
print(f"#XGBoost_gbtree Overall score: {score}: +/- {np.std(scores)}")


# Top 3 logic from the following notebook -- please upvote that notebook!
# https://www.kaggle.com/code/aryamanvaishya/fork-of-predicting-optimal-fertilizers-eda-xgb/notebook

top_3 = np.argsort(test_preds, axis=1)[:, -3:][:, ::-1]  # top 3 in descending order

# Convert back to fertilizer names
preds = [' '.join(le.inverse_transform(row)) for row in top_3]


sample_submission = pd.read_csv('/kaggle/input/playground-series-s5e6/sample_submission.csv')
sample_submission.head(3)


# Create submission file
test_ids = test[unique_id]
submission = pd.DataFrame({'id': test_ids, 'Fertilizer Name': preds})
submission.to_csv('submission.csv', index=False)
submission.head(3)

