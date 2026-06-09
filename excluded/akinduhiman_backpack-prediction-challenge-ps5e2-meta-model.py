import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import gc


file_path = "/kaggle/input/playground-series-s5e2/train.csv" 
train = pd.read_csv(file_path)


file_path = "/kaggle/input/playground-series-s5e2/training_extra.csv" 
train_extra = pd.read_csv(file_path)


file_path = "/kaggle/input/student-bag-price-prediction-dataset/Noisy_Student_Bag_Price_Prediction_Dataset.csv" 
original = pd.read_csv(file_path)


file_path = "/kaggle/input/playground-series-s5e2/test.csv" 
test = pd.read_csv(file_path)


ids = test['id']
train = train.drop(columns=['id'], axis=1)
train_extra = train_extra.drop(columns=['id'], axis=1)
test = test.drop(columns=['id'], axis=1)


train


train_extra


test


train = train.drop_duplicates()
train_extra = train_extra.drop_duplicates()


def customDescription(df: pd.DataFrame, numeric_only: bool = False):
    if numeric_only:
        df = df.select_dtypes(include=np.number)
    
    desc = pd.DataFrame(index=df.columns.to_list())
    desc['type'] = df.dtypes
    desc['count'] = df.count()
    desc['nunique'] = df.nunique()
    desc['null'] = df.isnull().sum()
    
    # Calculate mode and handle multiple modes
    modes = df.mode()
    desc['mode'] = np.nan  # Default to NaN
    for col in df.columns:
        if len(modes[col].dropna()) == 1:  # Single mode exists
            desc.loc[col, 'mode'] = modes[col].iloc[0]
        else:  # Multiple modes
            desc.loc[col, 'mode'] = np.nan

    # Calculate least frequent value
    desc['least_frequent'] = np.nan  # Default to NaN
    for col in df.columns:
        value_counts = df[col].value_counts(dropna=False)
        if not value_counts.empty:
            least_freq_count = value_counts.min()  # Find the minimum frequency
            least_freq_values = value_counts[value_counts == least_freq_count].index
            
            if len(least_freq_values) == 1:  # If exactly one least frequent value exists
                desc.loc[col, 'least_frequent'] = least_freq_values[0]
            else:  # Multiple least frequent values
                desc.loc[col, 'least_frequent'] = np.nan
    
    # Handle numeric columns
    numeric_cols = df.select_dtypes(include=np.number)
    if not numeric_cols.empty:
        numeric_desc = numeric_cols.describe().T.drop(columns=['count', 'std', '25%', '50%', '75%'], axis=1)
        for col in numeric_cols.columns:
            desc.loc[col, 'mean'] = numeric_desc.loc[col, 'mean']
            desc.loc[col, 'min'] = numeric_desc.loc[col, 'min']
            desc.loc[col, 'max'] = numeric_desc.loc[col, 'max']
    
    # Handle datetime columns
    datetime_cols = df.select_dtypes(include=['datetime64[ns]', 'datetime64[ns, UTC]'])
    for col in datetime_cols.columns:
        desc.loc[col, 'min'] = df[col].min()
        desc.loc[col, 'max'] = df[col].max()
    
    return desc


customDescription(train)


customDescription(train_extra)


# original.dropna(inplace=True)


# customDescription(original)


train_nw = pd.concat([train, train_extra], axis =0)
# train_nw = train


# customDescription(test)


train_nw.columns = train_nw.columns.str.lower()
test.columns = test.columns.str.lower()


train_nw.rename(columns={"weight capacity (kg)" : "weight_capacity_kg"}, inplace= True)
test.rename(columns={"weight capacity (kg)" : "weight_capacity_kg"}, inplace= True)

train_nw.rename(columns={"laptop compartment" : "laptop_compartment"}, inplace= True)
test.rename(columns={"laptop compartment" : "laptop_compartment"}, inplace= True)


train_nw.info(memory_usage='deep')


train_nw['compartments'] = train_nw['compartments'].astype('Int8')
train_nw['weight_capacity_kg'] = train_nw['weight_capacity_kg'].astype('float32')
train_nw['price'] = train_nw['price'].astype('float16')

test['compartments'] = test['compartments'].astype('Int8')
test['weight_capacity_kg'] = test['weight_capacity_kg'].astype('float32')


train_nw


test


test['set'] = 'test'
train_nw['set'] = 'train'

new_df = pd.concat([train_nw, test], axis =0)
df = new_df.copy()
df.fillna(-1, inplace=True)


import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

# Separate features and target
X = df.drop(columns=['price'])
# y = new_df['price']

# Define categorical and numerical features
categorical_features = ['brand', 'material', 'size', 'laptop_compartment', 'waterproof', 'style', 'color']
numerical_features = ['compartments', 'weight_capacity_kg']

# Convert all categorical columns to string type to ensure uniformity
X[categorical_features] = X[categorical_features].astype(str)

# Create a preprocessor with the updated OneHotEncoder using 'sparse_output'
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
    ]
)

# Choose the number of principal components
n_components = 3

# Create a pipeline that preprocesses the data and applies PCA
pca_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('pca', PCA(n_components=n_components))
])

# Fit the PCA pipeline on your features and transform
X_pca = pca_pipeline.fit_transform(X)

# Convert the resulting principal components to a DataFrame and name them
pca_columns = [f'PC{i+1}' for i in range(n_components)]
X_pca_df = pd.DataFrame(X_pca, columns=pca_columns)

# Optionally, you can concatenate these principal components back with the original data
X_new = pd.concat([X.reset_index(drop=True), X_pca_df], axis=1)

print("New features from PCA:")
print(X_new.head())


X_new


new_df['Feature_1'] = X_new['PC1']
new_df['Feature_2'] = X_new['PC2']
new_df['Feature_3'] = X_new['PC3']
new_df


# new_df['Brand_Material'] = new_df['brand'] + '_' + new_df['material']
# new_df['Brand_Size'] = new_df['brand'] + '_' + new_df['size']
# new_df['Compartments_Category'] = pd.cut(new_df['compartments'], bins=[0, 2, 5, 10, np.inf], labels=['Few', 'Moderate', 'Many', 'Very Many'])
new_df['Weight_Capacity_Ratio'] = new_df['weight_capacity_kg'] / new_df['weight_capacity_kg'].max()
new_df['Weight_to_Compartments'] = new_df['weight_capacity_kg'] / (new_df['compartments'] + 1)  
# new_df['Style_Size'] = new_df['style'] + '_' + new_df['size']


new_df


# test['set'] = 'test'
# train_nw['set'] = 'train'

# new_df = pd.concat([train_nw, test], axis =0)

new_df['size'] = pd.Categorical(new_df['size'], categories=['Small', 'Medium', 'Large'], ordered=True)
new_df['size'] = new_df['size'].cat.codes
new_df['size'] = new_df['size'].astype('Int8')


from sklearn.preprocessing import OneHotEncoder

def one_hot_encode_and_add(df, column):
    one_hot_encoder = OneHotEncoder(sparse_output=False)
    one_hot_encoded = one_hot_encoder.fit_transform(df[[column]])
    encoded_columns = pd.DataFrame(one_hot_encoded, columns=one_hot_encoder.get_feature_names_out([column]))
    encoded_columns.index = df.index
    df = pd.concat([df, encoded_columns], axis=1)
    df = df.drop(columns=[column])
    return df

columns_to_encode = ['brand', 'material', 'laptop_compartment', 'waterproof', 'style', 'color']

for col in columns_to_encode:
    new_df = one_hot_encode_and_add(new_df, col)


new_df


new_df.fillna(-1, inplace=True)


train = new_df[new_df['set'] == 'train']
test = new_df[new_df['set'] == 'test']
test = test.drop(columns=['price','set'])
train = train.drop(columns=['set'])


train


train = train.drop_duplicates()
train


label = 'price' 
X = train.drop(columns=[label], axis=1)  
y = train[label]

X = X.reset_index(drop=True)
y = y.reset_index(drop=True)


train.to_csv('train.csv', index=False)
test.to_csv('test.csv', index=False)
X.to_csv('X.csv', index=False)
y.to_csv('y.csv', index=False)


from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor

# Generalized function for obtaining model predictions using cross-validation.
def get_model_predictions(X, y, df_test, model_func):
    test_preds = np.zeros(len(df_test))
    val_preds = np.zeros(len(X))
    cv = KFold(n_splits=10, shuffle=True, random_state=9)

    for fold, (train_ind, valid_ind) in enumerate(cv.split(X, y)):
        X_train, y_train = X.iloc[train_ind], y.iloc[train_ind]
        X_val, y_val = X.iloc[valid_ind], y.iloc[valid_ind]

        model = model_func()

        # Depending on which model we are using, call fit with appropriate parameters.
        if model_func == lgb_model:
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(100)]
            )
        elif model_func == xgb_model:
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False
            )
        elif model_func == catboost_model:
            # For CatBoost, the early stopping and verbosity have been preset.
            model.fit(
                X_train, y_train,
                eval_set=(X_val, y_val)
            )
        else:
            # Fallback: use early stopping if available.
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                early_stopping_rounds=50,
                verbose=False
            )

        gc.collect()

        # Get predictions on validation set
        y_pred_val = model.predict(X_val)

        # Compute RMSE on validation set
        val_rmse = np.sqrt(mean_squared_error(y_val, y_pred_val))

        print("-" * 50)
        print(model_func.__name__, "Fold:", fold, " Val RMSE:", np.round(val_rmse, 5))
        print("-" * 50)

        val_preds[valid_ind] = y_pred_val
        test_preds += model.predict(df_test) / cv.n_splits
        gc.collect()

    return val_preds, test_preds


# def lgb_model():
#     # Return an LGBMRegressor with the best found parameters.
#     return lgb.LGBMRegressor(
#         n_estimators=350,
#         max_depth=7,
#         min_child_weight=7,
#         learning_rate=0.05536296475337918,
#         subsample=0.9450072936265559,
#         colsample_bytree=0.2844458803202738,
#         reg_alpha=2.957874183151822,
#         reg_lambda=9.128884058603495,
#         verbosity=0,
#         num_leaves=8
#     )

# def xgb_model():
#     # Return an XGBRegressor with the best found parameters.
#     return xgb.XGBRegressor(
#         n_estimators=650,
#         max_depth=3,
#         min_child_weight=3,
#         learning_rate=0.020660818326575233,
#         subsample=0.8994245030863979,
#         colsample_bytree=0.2837367232342688,
#         reg_alpha=9.517052679872931,
#         reg_lambda=9.834846014759364,
#         gamma=0.07545972777518654,
#         verbosity=0, 
#         objective = 'reg:squarederror'
#     )

# def catboost_model():
#     # Return a CatBoostRegressor with the best found parameters.
#     return CatBoostRegressor(
#         iterations=997,
#         learning_rate=0.024408249615595675,
#         depth=3,
#         l2_leaf_reg=6.041125744041331,
#         early_stopping_rounds = 94,
#         verbose=0
#     )


def lgb_model():
    return lgb.LGBMRegressor(n_estimators=1000, learning_rate=0.01, device='gpu')

def xgb_model():
    return xgb.XGBRegressor(n_estimators=1000, learning_rate=0.01, tree_method='gpu_hist',predictor='gpu_predictor')

def catboost_model():
    return CatBoostRegressor(
        iterations=1000,
        learning_rate=0.01,
        task_type='GPU'
    )


print("1. XGBRegressor")
xgb_val_preds, xgb_test_preds = get_model_predictions(X, y, test, xgb_model)


print("2. LGBMRegressor")
lgb_val_preds, lgb_test_preds = get_model_predictions(X, y, test, lgb_model)


print("3. CatBoostRegressor")
cat_val_preds, cat_test_preds = get_model_predictions(X, y, test, catboost_model)


val_preds_df = pd.DataFrame({
    'lgb': lgb_val_preds,
    'xgb': xgb_val_preds,
    'catb': cat_val_preds
})

test_preds_df = pd.DataFrame({
    'lgb': lgb_test_preds,
    'xgb': xgb_test_preds,
    'catb': cat_test_preds
})


from sklearn.linear_model import LinearRegression

# Stage 1: Base meta-model
base_meta_model = LinearRegression()
base_meta_model.fit(val_preds_df, y)

meta_train_preds = base_meta_model.predict(val_preds_df)
meta_test_preds = base_meta_model.predict(test_preds_df)

residuals = y - meta_train_preds

train_rmse = np.sqrt(mean_squared_error(y, meta_train_preds))
print("Stage 1 (base meta-model) RMSE:", np.round(train_rmse, 5))

# # Stage 2: Error correction model
# error_meta_model = LinearRegression()
# error_meta_model.fit(val_preds_df, residuals)

# predicted_error_train = error_meta_model.predict(val_preds_df)
# meta_train_preds = meta_train_preds + predicted_error_train
# predicted_error_test = error_meta_model.predict(test_preds_df)
# meta_test_preds = meta_test_preds + predicted_error_test

# train_rmse = np.sqrt(mean_squared_error(y, meta_train_preds))
# print("Stage 1 (base meta-model) RMSE:", np.round(train_rmse, 5))

#39.01228


file_path = "/kaggle/input/playground-series-s5e2/sample_submission.csv" 
submission = pd.read_csv(file_path)


submission['Price'] = meta_test_preds
submission.to_csv('submission.csv', index=False)


submission['Price'] = lgb_test_preds
submission.to_csv('lgb_submission.csv', index=False)


submission['Price'] = xgb_test_preds
submission.to_csv('xgb_submission.csv', index=False)


submission['Price'] = cat_test_preds
submission.to_csv('cat_submission.csv', index=False)

