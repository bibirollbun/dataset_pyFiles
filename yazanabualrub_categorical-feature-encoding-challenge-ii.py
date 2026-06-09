# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


train = pd.read_csv("/kaggle/input/cat-in-the-dat-ii/train.csv")
test = pd.read_csv("/kaggle/input/cat-in-the-dat-ii/test.csv")
sample = pd.read_csv("/kaggle/input/cat-in-the-dat-ii/sample_submission.csv")

print(train.shape, test.shape)


# To hide (disable) warnings
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)



train.head()


train.info()


train.describe()


train.columns


counts = train['target'].value_counts()

plt.pie(counts, labels=counts.index,
    autopct='%1.1f%%', 
    startangle=90,
    colors=['lightgreen', 'lightblue'])

plt.title('Target Distribution')
plt.axis('equal')  # make it a circle
plt.show()

counts
# unbalanced data


# number of unique values for every Feature
for col in train.columns:
    unique_values = train[col].nunique()
    print(f"{col}: {unique_values}")



#unique values for every Feature
for col in train.columns:
    print(f"{col} ==> {train[col].unique()}")



#Distribute values for each Feature
for col in train.columns:
    print(f"\n{col} value counts:")
    print(train[col].value_counts().head(10)) 



train.duplicated().sum()


#counts of Missing values
summary = pd.DataFrame({
    'Missing Count': train.isna().sum(),
    'Unique Values': train.nunique(dropna=True)
})

print(summary)



#Make Heatmap for Missing values

plt.figure(figsize=(20,10))
sns.heatmap(train.isnull(), cbar=False, cmap='gray')
plt.title('Heatmap of Missing Data')
plt.show()



# change values of binary object columns to 0,1 
train['bin_3'] = train['bin_3'].map({'F': 0, 'T': 1})
train['bin_4'] = train['bin_4'].map({'N': 0, 'Y': 1})
test['bin_3'] = train['bin_3'].map({'F': 0, 'T': 1})
test['bin_4'] = train['bin_4'].map({'N': 0, 'Y': 1})


train[['bin_3','bin_4']].info()


# Categorical Columns 
cat_vars = [var for var in train.columns if train[var].dtype == 'O']
print(len(cat_vars), cat_vars)


# Numerical Columns 
target = 'target'
num_vars = [var for var in train.columns if var not in cat_vars + [target] + ['id']]
print(len(num_vars), num_vars)




corr_matrix = train[num_vars + ['target']].corr()
plt.figure(figsize=(10,8))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
plt.title("Correlation Heatmap")
plt.show()


id_test = test['id']
train = train.drop(columns=['id'])
test = test.drop(columns=['id'])


from sklearn.model_selection import train_test_split

X = train.drop(columns=['target'])
y = train['target']

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# cyclical encoding For Day and Month (its like Normalization)
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class CyclicDateEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, day_col='day', month_col='month'):
        self.day_col = day_col
        self.month_col = month_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=[self.day_col, self.month_col] + [f'col{i}' for i in range(X.shape[1]-2)])
        else:
            X = X.copy()

       
        if self.day_col in X.columns:
            X['day_sin'] = np.sin(2 * np.pi * X[self.day_col] / 7)
            X['day_cos'] = np.cos(2 * np.pi * X[self.day_col] / 7)
            X.drop(self.day_col, axis=1, inplace=True)

        if self.month_col in X.columns:
            X['month_sin'] = np.sin(2 * np.pi * X[self.month_col] / 12)
            X['month_cos'] = np.cos(2 * np.pi * X[self.month_col] / 12)
            X.drop(self.month_col, axis=1, inplace=True)

        return X.values  


from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer                              # to fill missing values
import category_encoders as ce                                        # for target Encoding


numeric_features = ['bin_0','bin_1','bin_2','bin_3','bin_4','ord_0','day','month']    # fill missing with mode
onehot_features = ['nom_0','nom_1','nom_2','nom_3','nom_4','ord_3','ord_4']           # fill missing with unknown + One-Hot encoding
ordinal_encoded_features = ['ord_1','ord_2']                                          # fill missing with mode + Ordinal Encoding
target_encoded_features = ['nom_5','nom_6','nom_7','nom_8','nom_9','ord_5']          # fill missing with 'unknown' + Target Encoding

ordinal_order = [
    ['Novice', 'Contributor', 'Expert', 'Master', 'Grandmaster'], 
    ['Freezing', 'Cold', 'Warm', 'Hot', 'Boiling Hot', 'Lava Hot']                    #to detect how the model understands the order
]

for col in target_encoded_features:
    for df in [X_train, X_val, test]:
        if col in df.columns:
            df[col] = df[col].fillna('Unknown')

target_encoder = ce.TargetEncoder(cols=target_encoded_features)
X_train[target_encoded_features] = target_encoder.fit_transform(X_train[target_encoded_features], y_train)
X_val[target_encoded_features] = target_encoder.transform(X_val[target_encoded_features])
test[target_encoded_features] = target_encoder.transform(test[target_encoded_features])

numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),        # 'most_frequent' its like 'mode
    ('cyclic',CyclicDateEncoder(day_col='day', month_col='month'))    # cyclical encoding For Day and Month (its like Normalization)
                                             
])

onehot_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))   # (handle_unknown='ignore') is to ignore any new data in test data
])

ordinal_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('ordinal', OrdinalEncoder(categories=ordinal_order, handle_unknown='use_encoded_value', unknown_value=-1))
])

preprocessor = ColumnTransformer([
    ('num', numeric_transformer, numeric_features),
    ('onehot', onehot_transformer, onehot_features),
    ('ordinal', ordinal_transformer, ordinal_encoded_features)
], remainder='passthrough') 




from sklearn.linear_model import LogisticRegression

from xgboost import XGBClassifier
ratio = (y_train == 0).sum() / (y_train == 1).sum()

model = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier',
     
XGBClassifier(
    n_estimators=1500,  #Number of Decision Tree          
    learning_rate=0.01,  # best one is 0,01  'alpha'       
    max_depth=8,                   
    min_child_weight=4,           
    gamma=0.3,                     
    subsample=0.11,                 
    colsample_bytree=0.8,          
    tree_method='gpu_hist',  # use GPU
    predictor='gpu_predictor',  # use GPU in predict
    grow_policy='lossguide',       
    scale_pos_weight = ratio, #for Imbalanced Data            
    eval_metric='auc',              
    use_label_encoder=False,
    n_jobs=-1,
    random_state=42
)
     
    )]
                )



model.fit(X_train, y_train)


from sklearn.metrics import roc_curve, roc_auc_score

y_val_prob = model.predict_proba(X_val)[:, 1]  # Probability of the positive class
roc_auc = roc_auc_score(y_val, y_val_prob)
print(f'ROC-AUC: {roc_auc:.4f}')


fpr, tpr, thresholds = roc_curve(y_val, y_val_prob)
plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.4f})')
plt.plot([0,1], [0,1], 'k--')  
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.grid()
plt.show()


'''
from sklearn.model_selection import RandomizedSearchCV

param_grid = {
    'classifier__n_estimators': [1000, 1500, 2000],
    'classifier__learning_rate': [0.01, 0.02, 0.05],
    'classifier__max_depth': [7, 9, 11],
    'classifier__min_child_weight': [1, 3, 5],
    'classifier__subsample': [0.7, 0.8, 0.9],
    'classifier__colsample_bytree': [0.7, 0.8, 0.9],
    'classifier__gamma': [0, 0.1, 0.2, 0.3]
}

search = RandomizedSearchCV(model, param_grid, n_iter=20, scoring='roc_auc', cv=, verbose=2)
search.fit(X_train, y_train)
print(search.best_params_)
'''




test_pred = model.predict(test)


submission = pd.DataFrame({
    'id': id_test,
    'target': test_pred
})

submission.to_csv('test_predictions_Final.csv', index=False)


# to see how the  looks like after processing
'''
X_train_transformed = model.named_steps['preprocessor'].transform(X_train)

X_train_transformed_df = pd.DataFrame(
    X_train_transformed.toarray(),  
    columns=model.named_steps['preprocessor'].get_feature_names_out()
)

'''



#X_train_transformed_df.columns

