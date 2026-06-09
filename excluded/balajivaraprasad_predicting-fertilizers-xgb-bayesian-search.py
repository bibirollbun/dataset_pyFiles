import numpy as np 
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
from plotly.subplots import make_subplots
import plotly.graph_objects as go

from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder

from scipy.stats import chi2_contingency

import warnings
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
pd.set_option('display.max_columns', None)
warnings.filterwarnings("ignore")


train = pd.read_csv("/kaggle/input/playground-series-s5e6/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e6/test.csv")
sam_sub = pd.read_csv("/kaggle/input/playground-series-s5e6/sample_submission.csv")
original = pd.read_csv("/kaggle/input/fertilizer-prediction/Fertilizer Prediction.csv")


train.shape, test.shape, sam_sub.shape, original.shape


train.head()


original['Soil Type'].unique()


original['Crop Type'].unique()


original['Soil Type'].unique()


original.info()


num_cols = [i for i in train.columns if train[i].dtype == np.int64]
cat_cols = ['Soil Type', 'Crop Type']
num_cols.remove('id')


# Are there any outliers in numerical columns?


j = iter([[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2]])
colors = iter(["#AEC6CF", "#FFB347", "#77DD77", "#FF6961", "#CBAACB", "#FDFD96"])

fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(10, 7));
fig.suptitle('NPK proportions based on soil type', fontsize=12)

for i in num_cols:
    index = next(j)
    sns.boxplot(y = i, data = train, ax = axes[*index], color = next(colors));
    axes[*index].set_title(i)
    axes[*index].set_xlabel(None)
    
plt.subplots_adjust(top=0.9)
plt.show();


# Is there difference in NPK based on Soil Type?

fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10, 15));
fig.suptitle('NPK proportions based on soil type', fontsize=16)
j = iter([0, 1, 2])
for i in ['Nitrogen', 'Potassium', 'Phosphorous']:
    index = next(j)
    sns.boxplot(x = 'Soil Type', y = i, data = train, ax = axes[index], palette = sns.color_palette("pastel"))
    axes[index].set_title(i)
    axes[index].set_xlabel(None)
plt.subplots_adjust(top=0.93)
plt.show();


# Is there any significant impact of existing NPK levels in the soil on Fertiliers

fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10, 15));
fig.suptitle('NPK proportions based on Fertilzer Used for the Crop', fontsize=16)
j = iter([0, 1, 2])
for i in ['Nitrogen', 'Potassium', 'Phosphorous']:
    index = next(j)
    sns.boxplot(x = 'Fertilizer Name', y = i, data = train, ax = axes[index], palette = sns.color_palette("pastel"))
    axes[index].set_title(i)
    axes[index].set_xlabel(None)
plt.subplots_adjust(top=0.93)
plt.show();


contingency = pd.crosstab(train['Soil Type'], train['Fertilizer Name'])
contingency


chi2, p, dof, expected = chi2_contingency(contingency)

print(f"Chi2 statistic: {chi2}")
print(f"p-value: {np.round(p, 4)}")


def cramers_v(confusion_matrix):
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    return np.sqrt(phi2 / min(k-1, r-1))

v = cramers_v(contingency)
print(f"CramÃ©râ€™s V: {v}")



for col in cat_cols:
    combined_cats = pd.concat([train[col], test[col], original[col]]).unique()
    le = LabelEncoder().fit(combined_cats)
    train[col] = le.transform(train[col])
    test[col] = le.transform(test[col])
    original[col] = le.transform(original[col])\

target_encoder = LabelEncoder()
train["Fertilizer Name"] = target_encoder.fit_transform(train["Fertilizer Name"])
original["Fertilizer Name"] = target_encoder.fit_transform(original["Fertilizer Name"])


train['n_h_ratio'] = train['Nitrogen']**2/np.log(train['Humidity'])
train['p_h_ratio'] = train['Potassium']**2/np.log(train['Humidity'])
train['k_h_ratio'] = train['Phosphorous']**2/np.log(train['Humidity'])
train['n_h_ratio'] = train['Nitrogen']**2/np.log(train['Moisture'])
train['n_h_ratio'] = train['Potassium']**2/np.log(train['Moisture'])
train['n_h_ratio'] = train['Phosphorous']**2/np.log(train['Moisture'])

test['n_h_ratio'] = test['Nitrogen']**2/np.log(test['Humidity'])
test['p_h_ratio'] = test['Potassium']**2/np.log(test['Humidity'])
test['k_h_ratio'] = test['Phosphorous']**2/np.log(test['Humidity'])
test['n_h_ratio'] = test['Nitrogen']**2/np.log(test['Moisture'])
test['n_h_ratio'] = test['Potassium']**2/np.log(test['Moisture'])
test['n_h_ratio'] = test['Phosphorous']**2/np.log(test['Moisture'])

original['n_h_ratio'] = original['Nitrogen']**2/np.log(original['Humidity'])
original['p_h_ratio'] = original['Potassium']**2/np.log(original['Humidity'])
original['k_h_ratio'] = original['Phosphorous']**2/np.log(original['Humidity'])
original['n_h_ratio'] = original['Nitrogen']**2/np.log(original['Moisture'])
original['p_h_ratio'] = original['Potassium']**2/np.log(original['Moisture'])
original['k_h_ratio'] = original['Phosphorous']**2/np.log(original['Moisture'])


for i in cat_cols:
    train[i] = train[i].astype('category')
for i in cat_cols:
    test[i] = test[i].astype('category')


train.info()


N_FOLDS = 10
ES_ROUNDS = 75


X = train.drop(columns=["id", "Fertilizer Name"])
y = pd.Series(train["Fertilizer Name"])
X_test = test.drop(columns=["id"])


params = {
    'objective': 'multi:softprob', 
    'num_class': y.nunique(), 
    'max_depth': 7,
    'learning_rate': 0.062, 
    'subsample': 0.8, 
    'max_bin': 128, 
    'colsample_bytree': 0.3,
    'tree_method': 'gpu_hist', 
    'eval_metric': 'aucpr',
    'device': "cuda", 
    'enable_categorical': True, 
    'n_estimators': 1000,
    'early_stopping_rounds': ES_ROUNDS,
}


def mapk(actual, predicted, k=3):
    def apk(a, p, k):
        p = p[:k]
        score = 0.0
        hits = 0
        seen = set()
        for i, pred in enumerate(p):
            if pred in a and pred not in seen:
                hits += 1
                score += hits / (i + 1.0)
                seen.add(pred)
        return score / min(len(a), k)
    return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])


skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
test_preds = np.zeros(shape=(len(test), y.nunique()))
map3_scores = []

for i, (train_idx, valid_idx) in enumerate(skf.split(X, y)):
    print(  " âˆ’ " * 15, "\n", f"FOLD {i+1}".center(45),"\n",  " âˆ’ " * 15)
    x_train = pd.DataFrame(X.iloc[train_idx])
    y_train = pd.Series(y.iloc[train_idx])
    for i in range(5):
        x_train = pd.concat([x_train, original.copy().drop('Fertilizer Name', axis = 1)], axis = 0)
        y_train = pd.concat([y_train, original['Fertilizer Name'].copy()], axis = 0)
    x_valid = pd.DataFrame(X.iloc[valid_idx])
    y_valid = pd.Series(y.iloc[valid_idx])
    model = XGBClassifier(**params)
    model.fit(
        x_train, y_train,
        eval_set=[(x_valid, y_valid)],
        verbose = 500
    )
    preds = model.predict_proba(X_test) 
    test_preds += preds / N_FOLDS
    top_3_preds_indices = np.argsort(preds, axis=1)[:, ::-1][:, :3]
    actual = [[label] for label in y_valid]
    MAP3 = mapk(actual, top_3_preds_indices)
    map3_scores.append(MAP3)
    # top_3_labels = target_encoder.inverse_transform(top_3_preds_indices.ravel()).reshape(top_3_preds_indices.shape)
    # MAP3 = mean_average_precision_at_k(np.array(y_valid).reshape(-1, 1), top_3_labels, 3)
    print(f"MAP@3 value is {MAP3}")


top_3_preds_indices = np.argsort(test_preds, axis=1)[:, ::-1][:, :3]
top_3_labels = target_encoder.inverse_transform(top_3_preds_indices.ravel()).reshape(top_3_preds_indices.shape)

submission = pd.DataFrame({
    'id': sam_sub['id'],
    'Fertilizer Name': [' '.join(row) for row in top_3_labels]
})

submission.to_csv('submission_XGBoost.csv', index=False)

