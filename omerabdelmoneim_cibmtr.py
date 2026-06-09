import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


input_dir =  '/kaggle/input/equity-post-HCT-survival-predictions/'
train = pd.read_csv(input_dir + 'train.csv', index_col='ID')
test = pd.read_csv(input_dir + 'test.csv', index_col='ID')


train.head()


null_count = train.isna().sum()
null_count = null_count[null_count != 0] 

col_width = max(len(col) for col in null_count.index) + 2  # Find max column name length
num_width = max(len(str(count)) for count in null_count.values) + 2  # Find max count length

for i in range(0, len(null_count), 3):
    line = ""
    for j in range(3):
        if i + j < len(null_count):  # Ensure we don't go out of bounds
            col = null_count.index[i + j]
            count = null_count[i + j]
            line += f"{col.ljust(col_width)}: {str(count).ljust(num_width)}                  "
    print(line)



train_transformed = train.copy()
train.select_dtypes(object).columns



def get_from_dic(x,dic,default = None):
    if x in dic:
        return dic[x]
    elif default == None:
        return x
    else:
        return default
def apply_dic(X,dict,default = None):
    return X.map(lambda x: get_from_dic(x,dict,default))


valid_values = {"yes", "no", "not done"}

binary_columns = [
    col for col in train.columns
    if set(train[col].dropna().astype(str).str.lower().unique()).issubset(valid_values)]
train[binary_columns].dtypes


def to_number(x : str) -> int:
    x = str(x).lower()
    if x == 'yes':
        return 1
    if x == 'no':
        return 0
    return None
def transform_binary_columns(new_df,original_df,binary_column_names):
    for column in binary_column_names:
        new_df[column] = original_df[column].map(to_number)
transform_binary_columns(train_transformed,train,binary_columns)
train_transformed.head(5)


train_transformed.select_dtypes(object).columns


def transform_sex_match(new_df,original_df):
    new_df['donor_sex'] = original_df['sex_match'].map(lambda x:str(x)[0]).map(lambda x:get_from_dic(x,{'M':1,'F':0,"n":None}))
    new_df['recepient_sex'] = original_df['sex_match'].map(lambda x:str(x)[-1]).map(lambda x:get_from_dic(x,{'M':1,'F':0,"n":None}))
transform_sex_match(train_transformed,train)
print('male donor:', train_transformed['donor_sex'].mean() * 100,'%')
print('male recepient:', train_transformed['recepient_sex'].mean() * 100,'%')


from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
encoder.fit(train[['race_group']]) 
def encode_race(new_df,original_df):
    encoded_race = encoder.transform(original_df[['race_group']])
    race_columns = encoder.get_feature_names_out(['race_group'])
    race_df = pd.DataFrame(encoded_race, columns=race_columns, index=original_df.index)
    return pd.concat([new_df, race_df], axis=1)
train_transformed = encode_race(train_transformed,train)
train_transformed.head()



def add_is_hispanic(new_df,original_df):
    new_df['is_hispanic'] = original_df['ethnicity'] == 'Hispanic or Latino'
add_is_hispanic(train_transformed,train)
train_transformed['is_hispanic'].mean() * 100


def transform_graft_type(new_df,original_df):
    new_df['graft_type'] =  apply_dic(original_df['graft_type'],{'Peripheral blood':1,'Bone marrow':0})
transform_graft_type(train_transformed, train)
train_transformed['graft_type'].mean()


def transform_prod_type(new_df,original_df):
    new_df['prod_type'] =  apply_dic(original_df['prod_type'],{'PB':1,'BM':0})
transform_prod_type(train_transformed, train)
train_transformed['prod_type'].mean()


for column in train_transformed.select_dtypes(object).columns:
    print(column,': ')
    print(train[column].value_counts())
    print()


from sklearn.model_selection import cross_val_score
def eval_model(model,x,y):
    scores = cross_val_score(model,x,y,cv=4, scoring='neg_mean_squared_error')
    print("mean: ",scores.mean())
    print("median: ",np.median(scores))
    print("std: ", np.std(scores))
    print("scores: ",scores)


from sklearn.impute import SimpleImputer

num_cols_mean = train.select_dtypes(include=['number', 'bool']).columns.difference(['efs', 'efs_time'])
num_cols_median = train_transformed.select_dtypes(include=['number', 'bool']).columns.difference(['efs', 'efs_time'])
num_cols_median = num_cols_median.difference(num_cols_mean)

num_imputer_mean = SimpleImputer(strategy='mean')
num_imputer_median = SimpleImputer(strategy='median')


from sklearn.base import BaseEstimator, TransformerMixin
class TargetEncoder(BaseEstimator, TransformerMixin):
    feature_names_in_ = None
    columns = []
    # smoothness blend between the general mean and the mean of a category  based on its frequency.
    # 0 smoothness will use the category’s  mean without blending, increasing the smoothness will increase the blend effect.
    def __init__(self,smoothness = 0): # no *args or **kargs
         self.smoothness = smoothness
         return
    def fit(self, X, y):
        self.feature_names_in_ = X.columns
        self.mean = y.mean()
        dict = {}
        # calculating each category’s mean.
        for col in X.columns:
            counts = X[col].value_counts()
            d = {}
            for cat in counts.index:
                t = counts[cat] / (counts[cat] + self.smoothness)
                cat_mean = y[X[col] == cat].mean()
                d[cat] = t * cat_mean + (1-t) * self.mean
            dict[col] = d
        self.dict = dict
        self.columns = X.columns
        return self
    def transform(self, X):
        X = X.copy()  
        for col in self.columns:
            X[col] = X[col].map(self.dict[col]).fillna(self.mean)  # Handle unknown categories
        return X
    def get_feature_names_out(self,feature_names_out):
        return self.columns
target_encoding_columns = train_transformed.select_dtypes(object).columns
target_encoding_columns



from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
preprocessor = ColumnTransformer(
    transformers=[
        ('num_impute_mean', num_imputer_mean, num_cols_mean), 
        ('num_impute_median', num_imputer_median, num_cols_median),
        ('target_encoder',TargetEncoder(2),target_encoding_columns),
    ],
    remainder='passthrough'  
)
def get_pipline(model):
    return Pipeline([
    ('preprocessor', preprocessor), 
    ('scl',StandardScaler()),
    ('model',model) 
])



selected_columns = train_transformed.select_dtypes(include=['number','bool']).columns.union(target_encoding_columns)
cleaned = train_transformed[selected_columns]
x = cleaned.drop(columns=['efs','efs_time'])
y = cleaned['efs']
selected_columns


from sklearn.base import BaseEstimator, RegressorMixin

class Smoother(BaseEstimator, RegressorMixin):

    def __init__(self, model, power=3.0, epsilon=1e-6):  
        self.model = model
        self.power = power
        self.epsilon = epsilon 
        
    def fit(self, X, y=None):
        self.model.fit(X,y)
        return self
    
    def predict(self, X):
        raw_preds = self.model.predict(X)
        transformed_preds = np.clip(raw_preds, self.epsilon, 1 - self.epsilon)
        transformed_preds = (transformed_preds**self.power) / (transformed_preds**self.power + (1 - transformed_preds)**self.power)
        return transformed_preds

    def score(self, X, y):
        return self.model.score(X, y) 


from sklearn.ensemble import VotingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

base_models = [
    ('linear', LinearRegression(n_jobs=-1)),
    ('forest', RandomForestRegressor(n_jobs=-1, random_state=0,max_depth=10)),
]

ensembled_model = get_pipline(Smoother(VotingRegressor(estimators=base_models, weights=[0.5,0.5]),1.15))

eval_model(ensembled_model, x, y)


ensembled_model.fit(x,y)


z = test.copy()
transform_binary_columns(z,test,binary_columns)
transform_sex_match(z,test)
z = encode_race(z,test)
add_is_hispanic(z,test)
transform_graft_type(z,test)
transform_prod_type(z,test)
z = z[x.columns]
z


result = ensembled_model.predict(z)
submission = pd.DataFrame({'ID': z.index, 'prediction': ensembled_model.predict(z)})
submission.to_csv("submission.csv", index=False)
submission

