import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split


train = pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv')
test= pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv')
submission = pd.read_csv('/kaggle/input/playground-series-s5e2/sample_submission.csv')


train.head()


sns.scatterplot(data = train, x = 'id' , y = 'Price')


#CREATING BINS
bin_edge = np.arange(0, 310000, 10000)
train['id_bin'] = pd.cut(train['id'], bins = bin_edge, include_lowest=True, right=False)



idbin = train.groupby('id_bin')


bin_avg = idbin['Price'].mean().reset_index()


plt.figure(figsize=(12,6))
sns.barplot(data=bin_avg, x='id_bin', y='Price')
plt.xticks(rotation=90)
plt.title('Average Price per ID Bin (10K Intervals)')
plt.show()


train['price_per_compt'] = train['Price'] / train['Compartments'] +1


train['avg_price_of_sizes'] = train.groupby('Size')['Price'].transform('mean')
test['avg_price_of_sizes']  = train['avg_price_of_sizes']


train.groupby('Brand')['Price'].mean().reset_index()


train['capc_to_compt_ratio'] = train['Weight Capacity (kg)'] / train['Compartments'] + 1
test['capc_to_compt_ratio'] = test['Weight Capacity (kg)'] / test['Compartments'] + 1


train['Brand_Material'] = train['Brand']+ '_' + train['Material']
test['Brand_Material'] = test['Brand']+ '_' + test['Material']


plt.figure(figsize = (10,4))
sns.barplot(x = train['Brand_Material'].dropna(), y = train['Price'])
plt.xticks(rotation = 70)
plt.show()


train['Style'].value_counts()


train['style_waterproof'] = train['Style']+'_'+train['Waterproof']
test['style_waterproof'] = test['Style']+'_'+test['Waterproof']


train['Style_count'] = train.groupby('Style')['Style'].transform('count')
test['Style_count'] = test.groupby('Style')['Style'].transform('count')


#color popularity (pink is popular).
train.groupby('Color')['Color'].count()


num_cols = train[['Price' ,'price_per_compt'	,'avg_price_of_sizes',	'capc_to_compt_ratio', 'Style_count']]


sns.heatmap(num_cols.corr(), annot = True, cmap = 'viridis')


train['median_material_price'] = train.groupby('Material')['Price'].transform('median')
test['median_material_price'] = train['median_material_price']


train['avg_price_per_style'] = train.groupby('Style')['Price'].transform('mean')
test['avg_price_per_style'] = train['avg_price_per_style']


train.head()


train[train.duplicated()] # NO Duplicate


train.drop(columns=['price_per_compt', 'id', 'id_bin'], inplace = True)
test.drop(columns=['id'], inplace = True)


train.head()


X = train.drop(columns= 'Price')
y = train['Price']


xtrain, xtest, ytrain, ytest = train_test_split(X,y , test_size=0.2, random_state = 42)


xtrain.info()


xtrain.isnull().sum()


impute = ColumnTransformer([
    ('Simp1', SimpleImputer(strategy='most_frequent'), [0,1,2]),
    ('p1', 'passthrough', [3]),
    ('Simp2', SimpleImputer(strategy='most_frequent'), [4,5,6,7]),
    ('kimp1', KNNImputer(), [8,9,10]),
    ('Simp3', SimpleImputer(strategy='most_frequent'), [11,12]),
    ('kimp2', KNNImputer(), [13,14,15])

], remainder = 'passthrough')


encode = ColumnTransformer([
    ('enc1', OneHotEncoder(sparse_output=False, handle_unknown='ignore', drop = 'first'), [0,1,2]),
    ('p1', 'passthrough', [3]),
    ('enc2', OneHotEncoder(sparse_output=False, handle_unknown='ignore', drop = 'first'), [4,5,6,7]),
    ('p2', 'passthrough', [8,9,10]),
    ('enc3', OneHotEncoder(sparse_output=False, handle_unknown='ignore', drop = 'first'), [11,12])
    ], remainder='passthrough')


xtrain.info()


scale = ColumnTransformer([
    ('scale', StandardScaler(),[3,8,9,10,13,14,15])
], remainder = 'passthrough')


pipe = Pipeline([
    ('impute', impute),
    ('encode', encode),
    ('scale', scale),
    ('model', GradientBoostingRegressor(n_estimators=50))
])


from sklearn import set_config
set_config(display= 'diagram')


pipe.fit(xtrain, ytrain)


ypred = pipe.predict(xtest)


from sklearn.metrics import r2_score, mean_squared_error, make_scorer
print(f'rmse : {np.sqrt(mean_squared_error(ytest, ypred))}')


yp = pipe.predict(test)


submission['Price'] = yp


submission.to_csv('submission.csv', index=False)
print('Success!')

