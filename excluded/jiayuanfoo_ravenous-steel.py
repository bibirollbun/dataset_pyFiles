import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt 
import seaborn as sns




from sklearn.multiclass import OneVsRestClassifier




sample_submission_df = pd.read_csv('/kaggle/input/playground-series-s4e3/sample_submission.csv')
train_df = pd.read_csv('/kaggle/input/playground-series-s4e3/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s4e3/test.csv')

# print(f"""
# train_df.columns : {train_df.columns}
# test_df.columns : {test_df.columns}
# sample_submission_df.columns : {sample_submission_df.columns}
# """)

RANDOM_STATE = 42


train_df.info()


train_df.describe()



# targets = ['Pastry', 'Z_Scratch', 'K_Scatch', 'Stains', 'Dirtiness', 'Bumps', 'Other_Faults']

X_columns = test_df
Y_columns = [item for item in train_df.columns if item not in X_columns]
X_columns,Y_columns
X_columns = test_df.columns[1:]


# train_df_input = train_df.drop(columns=targets)
# train_df_target = train_df[targets]
train_df_X = train_df[X_columns]
train_df_Y = train_df[Y_columns]


for i in X_columns:

    print(i,train_df_X[i].nunique())

    print('---')



cat_col=[] # categorical columns

num_col=[] # numerical columns

for i in X_columns:

    if train_df[i].nunique()<=4:

        cat_col.append(i)

    else:

        num_col.append(i)

cat_col,num_col


train_df[num_col].hist(bins=20,figsize=(12,10))

plt.tight_layout()

plt.show()



cmap = sns.diverging_palette(230, 20, n=256, as_cmap=True)
train_df_corr = train_df.corr()

sns.heatmap(
    train_df_corr
)


#Outlier Equation

def outlier_threshhold(
    dataframe
    ,column
    ,q1=0.25
    ,q3=0.75):

    Q1=dataframe[column].quantile(q1)
    Q3=dataframe[column].quantile(q3)

    iqr=Q3-Q1
    up_limit=Q3+1.5*iqr
    low_limit=Q1-1.5*iqr

    return low_limit,up_limit



#Outlier Count    

def outlier_percentage(dataframe, column):

    low_limit, up_limit = outlier_threshhold(dataframe, column)
    outliers = [x for x in dataframe[column] if (x > up_limit) or (x < low_limit)]  # Changed | to or
    
    print(f"{column} Outliers percentage: {len(outliers) / dataframe[column].shape[0] * 100}%")
    print("-------------------------")



#Checking Outliers 

def check_outliers(dataframe,column):

    low_limit,up_limit=outlier_threshhold(dataframe,column)
    outliers=(dataframe[column]>up_limit) | (dataframe[column]<low_limit)

    if outliers.any():
        return True
    else:
        return False

#Replace with IQR     

def replace_with_threshholds(dataframe,dataframe2,column):

    low_limit,up_limit=outlier_threshhold(dataframe,column)

    dataframe.loc[(dataframe[column]<low_limit),column]=low_limit

    dataframe.loc[(dataframe[column]>up_limit),column]=up_limit

    dataframe2.loc[(dataframe2[column]<low_limit),column]=low_limit

    dataframe2.loc[(dataframe2[column]>up_limit),column]=up_limit  





for col in (num_col):

    outlier_percentage(train_df,col)



corr = train_df[X_columns].corr(numeric_only=True)

mask = np.triu(corr) # Upper right triangle of array, everything else zeroed
mask=mask

plt.figure(figsize=(17, 8))

sns.heatmap(corr, annot=True,mask=mask, cmap='mako', fmt='.2f')

plt.show()



def detect_skewness(dataframe, threshold=0.5):

    """

    Detects left or right skewed columns in a pandas DataFrame.


s  
    Parameters:

    dataframe (pandas DataFrame): The DataFrame to analyze.

    threshold (float): The threshold for considering a column as skewed.

                      Default is 0.5.



    Returns:

    skewed_columns (list): A list of column names that are skewed.

    """

    skewed_columns = []



    for column in dataframe.columns:

        skewness = dataframe[column].skew()

        if abs(skewness) > threshold:

            #skewed.append((column, skewness))

            skewed_columns.append(column)

            

    

    return skewed_columns 

skewed_cols = detect_skewness(train_df[num_col])

print("Skewed columns:", skewed_cols)



from sklearn.model_selection import cross_val_score,train_test_split
from sklearn.metrics import accuracy_score,classification_report,f1_score,mean_squared_error,roc_auc_score,precision_score,recall_score,roc_curve,ConfusionMatrixDisplay,confusion_matrix,auc
from sklearn.pipeline import make_pipeline,Pipeline
from sklearn.preprocessing import StandardScaler,LabelEncoder,OneHotEncoder,OrdinalEncoder,RobustScaler,MinMaxScaler
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import ColumnTransformer

from sklearn.base import BaseEstimator,TransformerMixin





# Threshold Moderating downwards
for column in num_col:

    replace_with_threshholds(train_df,test_df,column)

    print(column,check_outliers(train_df,column))



# class OrdinalEncodeColumns(BaseEstimator, TransformerMixin):
#     """
#     Transformer class to perform ordinal encoding on specified columns of a Pandas DataFrame.

#     Parameters
#     ----------
#     columns : list of str
#         The names of the ordinal columns to encode.

#     Returns
#     -------
#     pandas.DataFrame
#         A new DataFrame with the ordinal columns encoded.
#     """
#     def __init__(self, columns):
#         self.columns = columns
#         self.encoder = None

#     def fit(self, X, y=None):
#         ordinal_data = X[self.columns].values
#         self.encoder = OrdinalEncoder()
#         self.encoder.fit(ordinal_data)
#         return self

#     def transform(self, X):
#         X_new = X.copy()
#         ordinal_data = X_new[self.columns].values
#         encoded_data = self.encoder.transform(ordinal_data)
#         X_new[self.columns] = encoded_data
#         return X_new

#     def fit_transform(self, X, y=None):
#         self.fit(X)
#         return self.transform(X)

# class LogTransform(BaseEstimator, TransformerMixin):

#     """
#     A transformer class to apply a log transform to a specified column in a Pandas DataFrame.
    
#     Parameters
#     ----------
#     columns : str
#         The name of the column to apply the log transform to.
#     domain_shift : float
#         The value to be added to the column before applying the log transform.

#     return
#     ------
#         transformed feature
#     """

#     def __init__(self, columns, domain_shift=1):
#         self.columns = columns
#         self.domain_shift = domain_shift

#     def fit(self, X, y=None):
#         return self

#     def transform(self, X):
#         X[self.columns] = np.log(X[self.columns] + self.domain_shift)
#         return X

#     def fit_transform(self, X, y=None):
#         return self.transform(X)


# class StandardScaleTransform(BaseEstimator, TransformerMixin):
#     """
#     A transformer class to apply standard scaling to specified columns in a Pandas DataFrame.

#     Parameters
#     ----------
#     cols : list of str
#         The names of the columns to apply standard scaling to.
#     """

#     def __init__(self, cols):
#         self.cols = cols
#         self.scaler_ = None

#     def fit(self, X, y=None):
#         self.scaler_ = StandardScaler().fit(X.loc[:, self.cols])
#         return self

#     def transform(self, X):
#         X_copy = X.copy()
#         X_copy.loc[:, self.cols] = self.scaler_.transform(X_copy.loc[:, self.cols])
#         return X_copy

#     def fit_transform(self, X, y=None):
#         self.scaler_ = StandardScaler().fit(X.loc[:, self.cols])
#         return self.transform(X)




from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC



train_df['Outside_Global_Index']


log_cols=['Y_Minimum','Y_Maximum','Pixels_Areas','X_Perimeter','Y_Perimeter','Sum_of_Luminosity','Length_of_Conveyer','Steel_Plate_Thickness','Edges_Index','Outside_X_Index','Edges_Y_Index','LogOfAreas','Log_X_Index']
numerical_cols=['X_Minimum', 'X_Maximum', 'Y_Minimum', 'Y_Maximum', 'Pixels_Areas','X_Perimeter', 'Y_Perimeter', 'Sum_of_Luminosity','Minimum_of_Luminosity', 'Maximum_of_Luminosity','Length_of_Conveyer', 'Steel_Plate_Thickness', 'Edges_Index','Empty_Index', 'Square_Index', 'Outside_X_Index', 'Edges_X_Index','Edges_Y_Index', 'LogOfAreas', 'Log_X_Index', 'Log_Y_Index','Orientation_Index', 'Luminosity_Index', 'SigmoidOfAreas']
# categorical_cols=['TypeOfSteel_A300', 'TypeOfSteel_A400', 'Outside_Global_Index'] # Not in use as only Type Of steel needs to be one-hot encoded


log_transform = lambda x: np.log(x+1)
log_transformer = FunctionTransformer(log_transform)

preprocessor = ColumnTransformer(
    [
        ('LogTransform', log_transformer, log_cols),
        ('StandardScale-0',StandardScaler(), numerical_cols),
        ('OneHot', OneHotEncoder(handle_unknown='ignore'), 
         ['TypeOfSteel_A300', 'TypeOfSteel_A400'])
    ]
)
# pipe = Pipeline([
#     ('preprocessor', preprocessor),
#     (('StandardScale-1'), StandardScaler()),
#     ('model', OneVsRestClassifier(
#         SVC())
#     )
# ])




from sklearn.metrics import roc_auc_score

from sklearn.model_selection import train_test_split,cross_val_score






X_train, X_test, y_train, y_test = train_test_split(
    train_df_X, train_df_Y, test_size=0.3, random_state=RANDOM_STATE)


from sklearn.linear_model import LogisticRegression,SGDRegressor, ridge_regression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor,ExtraTreesRegressor,AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVC
from xgboost import XGBClassifier, XGBRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.multioutput import MultiOutputClassifier,MultiOutputRegressor
from sklearn.neighbors import KNeighborsRegressor

from lightgbm import LGBMClassifier,LGBMRegressor



models = {
    'DecisionTree': DecisionTreeRegressor(random_state=RANDOM_STATE),
    'RandomForest': RandomForestRegressor(random_state=RANDOM_STATE),
    'GradientBoost': GradientBoostingRegressor(random_state=RANDOM_STATE),
    # 'GradientBoost': MultiOutputRegressor(GradientBoostingRegressor(random_state=RANDOM_STATE)),
    'ExtraTrees': ExtraTreesRegressor(random_state=RANDOM_STATE),
    'AdaBoost' : AdaBoostRegressor(random_state=RANDOM_STATE),
    
    'XGBoost': XGBRegressor(random_state=RANDOM_STATE),
    'LogisticRegression': LogisticRegression(random_state=RANDOM_STATE),
    # 'LogisticRegression': MultiOutputRegressor(LogisticRegression(random_state=RANDOM_STATE)),
    # 'SVC': OneVsRestClassifier(SVC(probability=True)),
    'KNN': KNeighborsRegressor(),
    'CatBoostClass': CatBoostClassifier(random_state=RANDOM_STATE,verbose=False),
    'CatBoostReg': CatBoostRegressor(random_state=RANDOM_STATE,verbose=False),
    'LGBMClass': LGBMClassifier(random_state = RANDOM_STATE),
    'LGBMReg': LGBMRegressor(random_state = RANDOM_STATE),
}



results = []

for name, model in models.items():

    try:
    
        print(f"Training with Base {name}")
        pipe = Pipeline([
            ('preprocessor', preprocessor),
            (('StandardScale-1'), StandardScaler()),
            (name, model)
        ])
        print(f"Fitting {name}")
        pipe.fit(X_train,y_train)
    
    except:
        print(f"Training with MultiOutput Wrapper {name}")
        pipe = Pipeline([
            ('preprocessor', preprocessor),
            (('StandardScale-1'), StandardScaler()),
            (name, MultiOutputRegressor(model))
        ])

        print(f"Fitting {name}")
        pipe.fit(X_train,y_train)
    
    print(f"Predicting with {name}")
    y_pred = pipe.predict(X_test)
    
    score = roc_auc_score(y_test,y_pred)
    print(f"{name} scores {score}")

    results.append((name,score))
    
results


final_model = 'LGBMReg'

pipe = Pipeline([
    ('preprocessor', preprocessor),
    (('StandardScale'), StandardScaler()),
    (name, MultiOutputRegressor(model))
])

pipe.fit(X_train,y_train)





y_test


train_df


test_df


test_df_idx,test_df_X = test_df['id'],test_df[train_df_X.columns]


test_df_idx


test_df_X


submission_pred = pipe.predict(test_df[train_df_X.columns])
submission_pred


target_new = {}
for i in range(len(y_test.columns)):
    target_new[i] = y_test.columns[i]

test_pred_df = pd.DataFrame(submission_pred)
test_pred_df = test_pred_df.rename(columns = target_new)

submission_df = pd.concat(
    (test_df['id'],
    test_pred_df
    ),
    axis = 1
) 
submission_df = submission_df.set_index(
    "id"
)



submission_df


save_file = submission_df.to_csv(
    'submission.csv'
)


