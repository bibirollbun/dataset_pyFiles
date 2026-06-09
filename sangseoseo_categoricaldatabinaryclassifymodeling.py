import pandas as pd

data_path = '/kaggle/input/categorical-feature-encoding-challenge-binary-clas/'

train = pd.read_csv(data_path + 'train.csv', index_col='id')
test =  pd.read_csv(data_path + 'test.csv', index_col='id')
submission = pd.read_csv(data_path + 'sample_submission.csv', index_col='id')


train.head()


all_data = pd.concat([train, test])
all_data = all_data.drop('target', axis = 1)


def resumetable(df):
  """
  feature summary table
  """
  print(f"Dataset shape : {df.shape}")
  summary = pd.DataFrame(df.dtypes, columns = ['Data types'])
  summary = summary.reset_index()
  summary = summary.rename(columns = {'index': 'Feature'})
  summary['Missing values'] = df.isnull().sum().values
  summary['Unique values'] = df.nunique().values
  summary['1st value'] = df.loc[0].values
  summary['2nd value'] = df.loc[1].values
  summary['3rd value'] = df.loc[2].values

  return summary



summary = resumetable(all_data)
summary


all_data['bin_3'] = all_data['bin_3'].map({'F': 0, 'T': 1})
all_data['bin_4'] = all_data['bin_4'].map({'N': 0, 'Y': 1})


# ord_1, ord_2
ord1dict = {'Novice':0, 'Contributor': 1, 'Expert':2, 'Master':3, 'Grandmaster': 4}
ord2dict = {
    'Feezing': 0,
    'Cold':1,
    'Warm':2,
    'Hot':3,
    'Boiling Hot': 4,
    'Lava Hot': 5
}

all_data['ord_1'] = all_data['ord_1'].map(ord1dict)
all_data['ord_2'] = all_data['ord_2'].map(ord2dict)

# Fill potential NaN values resulting from mapping with -1
#It appears that there are missing values (NaNs) in your data,
#which the LogisticRegression model cannot handle.
#This likely happened during the manual encoding of ord_1 and ord_2 and the scaling of the ordinal features.
# all_data['ord_1'] = all_data['ord_1'].fillna(-1)
# all_data['ord_2'] = all_data['ord_2'].fillna(-1)


#ord_3, ord_4, ord_5
from sklearn.preprocessing import OrdinalEncoder
ord345 = ['ord_3','ord_4', 'ord_5']
order_encoder = OrdinalEncoder()
all_data[ord345] = order_encoder.fit_transform(all_data[ord345])

#Output feature specific encoder order
for feature, categories in zip(ord345, order_encoder.categories_):
  print(feature)
  print(categories)


from sklearn.preprocessing import OneHotEncoder

nom_features = ['nom_' + str(i) for i in range(10)]
onehot_encoder = OneHotEncoder()

# ValueError: Columns must be same length as key
# all_data[nom_features] = onehot_encoder.fit_transform(all_data[nom_features])
encoded_nom_matrix  = onehot_encoder.fit_transform(all_data[nom_features])


encoded_nom_matrix   # <Compressed Sparse Row sparse matrix of dtype 'float64'	with 5000000 stored elements and shape (500000, 16276)>


all_data = all_data.drop(nom_features, axis = 1)


date_features = ['day', 'month']

encoded_date_matrix = onehot_encoder.fit_transform(all_data[date_features])
all_data = all_data.drop(date_features, axis = 1)

encoded_date_matrix


from sklearn.preprocessing import MinMaxScaler

ord_features = ['ord_' + str(i) for i in range(6)]

# Fill any remaining NaN values in ordinal features with -1 before scaling
all_data[ord_features] = all_data[ord_features].fillna(-1)
all_data[ord_features] = MinMaxScaler().fit_transform(all_data[ord_features])


all_data[ord_features]


from scipy import sparse

#hstack() - Combine the matrices horizontally
all_data_sprs = sparse.hstack([sparse.csr_matrix(all_data), encoded_date_matrix, encoded_nom_matrix], format='csr')
all_data_sprs


num_train = len(train)

X_train = all_data_sprs[:num_train]
X_test  = all_data_sprs[num_train:]
y       = train['target']

# Separating training data and validation data
from sklearn.model_selection import train_test_split
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y, test_size = 0.1, random_state=10,  stratify = y)


%%time

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression

logistic_model = LogisticRegression()
lr_params = {'C': [0.1, 0.125, 0.2], 'max_iter': [800, 900, 1000], 'solver': ['liblinear'], 'random_state': [42]}

gridsearch_logistic_model = GridSearchCV(estimator=logistic_model, param_grid=lr_params, cv = 5, scoring='roc_auc')
gridsearch_logistic_model.fit(X_train, y_train)
print(f"Optimal Parameter : {gridsearch_logistic_model.best_params_}")



from sklearn.metrics import roc_auc_score
y_valid_preds = gridsearch_logistic_model.predict_proba(X_valid)[:, 1]

roc_auc = roc_auc_score(y_valid, y_valid_preds)
print(f"Evaluation Metrics ROC AUC : {roc_auc:.4f}")


y_preds = gridsearch_logistic_model.best_estimator_.predict_proba(X_test)[:, 1]
submission['target'] = y_preds
submission.to_csv('submission.csv')

