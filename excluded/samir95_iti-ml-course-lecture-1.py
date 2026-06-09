import os

IS_KAGGLE = False
DATA_DIR = ""
if os.getcwd().startswith("/kaggle"):
    IS_KAGGLE = True
    DATA_DIR = "/kaggle/working/"

print("Notebook running on Kaggle!")


if IS_KAGGLE:
    print("Extracting competition data...")
    !unzip -o /kaggle/input/bluebook-for-bulldozers/Train.zip

    TRAIN_PATH = os.path.join(DATA_DIR, "Train.csv")


import warnings
warnings.filterwarnings('ignore')


# pd.read_csv??


import pandas as pd

df_raw = pd.read_csv(TRAIN_PATH, low_memory=False, parse_dates=["saledate"])


df_raw.saledate


def display_all(df):
    with pd.option_context("display.max_rows", 1000, "display.max_columns", 1000): 
        display(df)


display_all(df_raw.tail().T)


display_all(df_raw.describe(include='all').T)


import numpy as np

df_raw.SalePrice = np.log(df_raw.SalePrice)


from sklearn.ensemble import RandomForestRegressor

m = RandomForestRegressor(n_jobs=-1)
# The following code is supposed to fail due to string values in the input data
m.fit(df_raw.drop('SalePrice', axis=1), df_raw.SalePrice)


import re
import numpy as np

def extract_date_features(df, date_column, prefix=None, drop=True):
    # Ensure the date column is in datetime format
    df[date_column] = pd.to_datetime(df[date_column], errors='coerce')

    prefix = prefix if prefix is not None else ""
    field = df[date_column]

    attrs = ["year", "quarter", "month", "week", "day", "day_of_week", "day_of_year", "is_month_start", "is_month_end",
            "is_quarter_start", "is_quarter_end", "is_year_start", "is_year_end"]

    # Handle week attribute compatibility
    week = field.dt.isocalendar().week.astype(field.dt.day.dtype) if hasattr(field.dt, 'isocalendar') else field.dt.week
    
    # Extract features
    for attr in attrs:
        df[prefix + attr] = getattr(field.dt, attr.lower()) if attr != 'week' else week

    df[prefix + 'elapsed'] = field.astype(np.int64) // 10 ** 9

    if drop:
        df = df.drop(date_column, axis=1)
    
    return df


df_raw = extract_date_features(df_raw, "saledate", "sale_", drop=True)
df_raw.sale_year.head()


!mkdir tmp


%time df_raw.to_parquet("tmp/df_raw.parquet", index=False)


from sklearn.preprocessing import OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer 
from sklearn import set_config
set_config(transform_output = "pandas")

def create_encoding_pipeline(df, ordinal_features=None):
    if ordinal_features:
        ordinal_cols = list(ordinal_features.keys())
        encoder = OrdinalEncoder(categories=[ordinal_features[col] for col in ordinal_cols])
    else:
        ordinal_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        encoder = OrdinalEncoder()

    column_transformer = ColumnTransformer(
        transformers=[('ordinal', encoder, ordinal_cols)],
        remainder='passthrough'
    )

    # Create a pipeline
    pipeline = Pipeline(steps=[
        ('encode', column_transformer)
    ])

    return pipeline

def transform_dataframe(df, pipeline, fit=False):
    if fit:
        pipeline.fit(df)
    out = pipeline.transform(df)
    if isinstance(out, pd.DataFrame):
        out.columns = [col.split("__")[-1] for col in out.columns]
    return out

def show_column_mapping(pipeline, column_name):
    # Access the encoder from the pipeline
    encoder = pipeline.named_steps['encode'].transformers_[0][1]
    
    # Find the index of the column
    column_index = list(encoder.feature_names_in_).index(column_name)
    
    # Get the categories for the column
    categories = encoder.categories_[column_index]
    
    # Display the mapping
    print(f"Mapping for column '{column_name}':")
    for i, category in enumerate(categories):
        print(f"  {category}: {i}")


encoding_pipeline = create_encoding_pipeline(df_raw)
df_encoded = transform_dataframe(df_raw, encoding_pipeline, fit=True)


show_column_mapping(encoding_pipeline, "UsageBand")


display_all((df_encoded.isnull().sum().sort_index()/len(df_encoded)))


import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder, LabelEncoder


class OrdinalWithMissingIndicator(BaseEstimator, TransformerMixin):
    def __init__(self):
        # Impute missing values using the mode (most_frequent)
        self.first_pass_imputer = SimpleImputer(strategy='most_frequent')  # TODO: add an option to fill this with -999
        self.ordinal = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan)
        self.second_pass_imputer = SimpleImputer(strategy='most_frequent')

    def fit(self, X, y=None):
        # Fit the imputer and ordinal encoder.
        # X should be a 2D array (n_samples, n_features).
        self.first_pass_imputer.fit(X)
        X_imputed = self.first_pass_imputer.transform(X)
        self.ordinal.fit(X_imputed)
        X_ordinal = self.ordinal.transform(X_imputed)
        self.second_pass_imputer.fit(X_ordinal)
        return self

    def transform(self, X):

        # Compute missing indicator BEFORE imputation.
        if isinstance(X, pd.DataFrame):
            missing = X.isna().astype(int).values
        else:
            missing = pd.isna(X).astype(int)
            if missing.ndim == 1:
                missing = missing.reshape(-1, 1)

        # print(X.shape)

        # Impute
        X_imputed = self.first_pass_imputer.transform(X)

        # Apply ordinal encoding
        X_ordinal_encoded = self.ordinal.transform(X_imputed)

        # Compute missing indicator After ordinal.
        if isinstance(X_ordinal_encoded, pd.DataFrame):
            missing += X_ordinal_encoded.isna().astype(int).values
        else:
            missing_tmp = pd.isna(X_ordinal_encoded).astype(int)
            if missing_tmp.ndim == 1:
                missing += missing_tmp.reshape(-1, 1)
            else:
                missing += missing_tmp
                
        # Impute again
        X_imputed = self.second_pass_imputer.transform(X_ordinal_encoded)

        X_imputed = np.where(missing == 1, -1, X_imputed)

        # print(missing.shape)
        # print(np.hstack([X_imputed, missing]).shape)
        
        # Concatenate ordinal encoded values with missing indicator columns.
        # return np.hstack([X_imputed, missing])
        return X_imputed

    def get_feature_names_out(self, input_features=None):
        # If input_features is not provided, use default names.
        if input_features is None:
            input_features = [f"col{i}" for i in range(self.ordinal.n_features_in_)]
        # For each input column, we now produce two output columns:
        # one for the ordinal encoding and one for the missing indicator.
        out = []
        for name in input_features:
            out.append(name)
        # for name in input_features:
        #     out.append(f"{name}_na")
        return np.array(out)



def prepare_dataframe_for_modeling(
    dataframe,
    target_column=None,
    columns_to_skip=None,
    columns_to_ignore=None,
    scale_features=False,
    sample_size=None,
    pipeline=None,
    copy_data=True,
    max_n_cat=None
):
    """
    Processes a dataframe for modeling. In addition to basic imputation, scaling,
    and one-hot encoding, this function:
      - Extracts any ignored columns immediately.
      - For each categorical column:
           * If max_n_cat is provided and the number of unique values is > max_n_cat,
             applies ordinal encoding (using mode imputation) with a missing indicator.
           * Otherwise, uses one-hot encoding with a dummy for missing values.
      - For numeric columns, missing values are imputed using the median (with a missing indicator).
      
    Parameters:
      dataframe (pd.DataFrame): Input DataFrame.
      target_column (str): Name of the target column.
      columns_to_skip (list): Columns to drop from transformation.
      columns_to_ignore (list): Columns to extract and ignore (will be reattached later).
      scale_features (bool): If True, numeric features are scaled.
      sample_size (int): If provided, sample this many rows.
      pipeline (dict): If provided, uses the stored pipeline (transform mode).
      copy_data (bool): If True, work on a (shallow) copy; if False, modifies in place.
      max_n_cat (int): If provided, any categorical column with more than max_n_cat unique 
                       values is ordinal-encoded (with a missing indicator) rather than one-hot encoded.
                       
    Returns:
      Tuple (X_processed, y, pipeline) where:
        - X_processed (pd.DataFrame): The transformed features with ignored columns reattached.
        - y (np.array or pd.Series): Processed target variable.
        - pipeline (dict): Dictionary storing the fitted preprocessor and additional info.
    """
    # Optionally work on a copy.
    df = dataframe.copy(deep=False) if copy_data else dataframe

    # --- Optional Sampling ---
    if sample_size is not None:
        idxs = np.random.choice(df.index, size=sample_size, replace=False)
        df = df.loc[idxs]

    # --- Extract Ignored Columns Immediately ---
    if columns_to_ignore:
        ignored_df = df[columns_to_ignore]
        df.drop(columns=columns_to_ignore, inplace=True, errors='ignore')
    else:
        ignored_df = None

    # --- Process the Target Column ---
    target_encoder = None
    y = None
    if target_column is not None and target_column in df.columns:
        y = df[target_column]
        if not is_numeric_dtype(y):
            # For training mode, fit a LabelEncoder.
            if pipeline is None:
                target_encoder = LabelEncoder().fit(y)
            else:
                target_encoder = pipeline.get("target_encoder", None)
            y = target_encoder.transform(y)
        else:
            y = y.values
        # Ensure the target column is not used in features.
        if columns_to_skip is None:
            columns_to_skip = [target_column]
        elif target_column not in columns_to_skip:
            columns_to_skip.append(target_column)

    # --- Drop Columns to Skip (in place) ---
    if columns_to_skip:
        df.drop(columns=columns_to_skip, inplace=True, errors='ignore')

    # --- Identify Numeric and Categorical Features ---
    numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = df.select_dtypes(exclude=[np.number]).columns.tolist()

    # --- For categorical columns, split into low- and high-cardinality groups ---
    cat_low = []
    cat_high = []
    if max_n_cat is not None:
        for col in categorical_features:
            nunique = df[col].nunique(dropna=True)
            if nunique <= max_n_cat:
                cat_low.append(col)
            else:
                cat_high.append(col)
    else:
        cat_high = categorical_features  # all categorical get ordinal encoding

    # --- Build or Use the Pipeline ---
    if pipeline is None:
        # Numeric pipeline: impute (with missing indicator) and (optionally) scale.
        num_pipeline_steps = [
            ('imputer', SimpleImputer(strategy='median', add_indicator=True))
        ]
        if scale_features:
            num_pipeline_steps.append(('scaler', StandardScaler()))
        numeric_pipeline = Pipeline(steps=num_pipeline_steps)

        # Categorical pipeline for low-cardinality features: impute and one-hot encode.
        if cat_low:
            cat_low_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),  # TODO: change this to fill with negative values (e.g. -1 or -999)
                ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ])
        else:
            cat_low_pipeline = None

        # Categorical pipeline for high-cardinality features: impute with mode then ordinal encode
        # with an extra missing indicator.
        if cat_high:
            cat_high_pipeline = Pipeline(steps=[
                # ('ord_encoder', OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan)),
                # ('imputer', SimpleImputer(strategy="constant", fill_value="-1"))
                ('ord_encode', OrdinalWithMissingIndicator())
            ])
        else:
            cat_high_pipeline = None

        # Build the column transformer.
        transformers = []
        if numeric_features:
            transformers.append(('num', numeric_pipeline, numeric_features))
        if cat_low_pipeline is not None:
            transformers.append(('cat_low', cat_low_pipeline, cat_low))
        if cat_high_pipeline is not None:
            transformers.append(('cat_high', cat_high_pipeline, cat_high))
        
        preprocessor = ColumnTransformer(transformers=transformers, remainder='drop')

        # Fit and transform the dataframe in place.
        X_transformed = preprocessor.fit_transform(df)

        # Attempt to get feature names (scikit-learn ≥1.0).
        try:
            feature_names = preprocessor.get_feature_names_out()
        except Exception:
            feature_names = None

        # Convert transformed output into a DataFrame.
        if feature_names is not None:
            X_processed = pd.DataFrame(X_transformed, index=df.index, columns=feature_names)
        else:
            X_processed = pd.DataFrame(X_transformed, index=df.index)

        # Save the pipeline details.
        pipeline = {
            'preprocessor': preprocessor,
            'numeric_features': numeric_features,
            'categorical_features': categorical_features,
            'cat_low': cat_low,
            'cat_high': cat_high,
            'ignored_columns': columns_to_ignore,
            'columns_to_skip': columns_to_skip,
            'target_encoder': target_encoder if (target_column is not None and 
                                                  not is_numeric_dtype(dataframe[target_column]))
                                                  else None,
            'feature_names': feature_names,
            'max_n_cat': max_n_cat
        }
    else:
        # --- Transform mode: use the existing pipeline ---
        preprocessor = pipeline['preprocessor']
        # (Ignored columns have already been dropped above.)
        df.drop(columns=pipeline.get("columns_to_skip", []), inplace=True, errors='ignore')
        X_transformed = preprocessor.transform(df)
        feature_names = pipeline.get("feature_names", None)
        if feature_names is not None:
            X_processed = pd.DataFrame(X_transformed, index=df.index, columns=feature_names)
        else:
            X_processed = pd.DataFrame(X_transformed, index=df.index)

    # --- Reattach the Ignored Columns ---
    if ignored_df is not None:
        X_processed = pd.concat([ignored_df, X_processed], axis=1)

    return X_processed, y, pipeline


df, y, proc_pipeline = prepare_dataframe_for_modeling(df_raw, 'SalePrice')


df.head()


m = RandomForestRegressor(n_estimators=10, n_jobs=-1)
%time m.fit(df, y)
m.score(df,y)


df_raw = df_raw.sort_values("sale_elapsed")


def split_vals(a,n): return a[:n].copy(), a[n:].copy()

n_valid = 12000  # same as Kaggle's test set size
n_trn = len(df)-n_valid
raw_train, raw_valid = split_vals(df_raw, n_trn)

X_train, y_train, proc_pipeline = prepare_dataframe_for_modeling(raw_train, 'SalePrice')
X_valid, y_valid, _ = prepare_dataframe_for_modeling(raw_valid, 'SalePrice', pipeline=proc_pipeline)

X_train.shape, y_train.shape, X_valid.shape


import math

def rmse(x,y): return math.sqrt(((x-y)**2).mean())

def print_score(m):
    res = [rmse(m.predict(X_train), y_train), rmse(m.predict(X_valid), y_valid),
                m.score(X_train, y_train), m.score(X_valid, y_valid)]
    if hasattr(m, 'oob_score_'): res.append(m.oob_score_)
    print(res)


_, X_train = split_vals(X_train, X_train.shape[0]-20000)
_, y_train = split_vals(y_train, y_train.shape[0]-20000)


m = RandomForestRegressor(n_jobs=-1)
%time m.fit(X_train, y_train)
print_score(m)


m = RandomForestRegressor(n_estimators=1, max_depth=3, bootstrap=False, n_jobs=-1)
m.fit(X_train, y_train)
print_score(m)


import IPython
import graphviz
from sklearn.tree import export_graphviz

def draw_tree(t, df, size=10, ratio=0.6, precision=0):
    """Draws a representation of a random forest in IPython."""
    feature_names = [f.split("__")[-1] for f in df.columns]
    s=export_graphviz(t, out_file=None, feature_names=feature_names, filled=True,
                      special_characters=True, rotate=True, precision=precision)
    IPython.display.display(graphviz.Source(re.sub('Tree {',
       f'Tree {{ size={size}; ratio={ratio}', s)))


draw_tree(m.estimators_[0], X_train, precision=3)


m = RandomForestRegressor(n_estimators=1, bootstrap=False, n_jobs=-1)
m.fit(X_train, y_train)
print_score(m)


m = RandomForestRegressor(n_jobs=-1)
m.fit(X_train, y_train)
print_score(m)


preds = np.stack([t.predict(X_valid) for t in m.estimators_])
preds[:,0], np.mean(preds[:,0]), y_valid[0]


preds.shape


import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

plt.plot([r2_score(y_valid, np.mean(preds[:i+1], axis=0)) for i in range(10)]);


m = RandomForestRegressor(n_estimators=20, n_jobs=-1)
m.fit(X_train, y_train)
print_score(m)


m = RandomForestRegressor(n_estimators=40, n_jobs=-1)
m.fit(X_train, y_train)
print_score(m)


m = RandomForestRegressor(n_estimators=80, n_jobs=-1)
m.fit(X_train, y_train)
print_score(m)


m = RandomForestRegressor(n_estimators=40, n_jobs=-1, oob_score=True)
m.fit(X_train, y_train)
print_score(m)


X_train, y_train, proc_pipeline = prepare_dataframe_for_modeling(raw_train, 'SalePrice')
X_valid, y_valid, _ = prepare_dataframe_for_modeling(raw_valid, 'SalePrice', pipeline=proc_pipeline)

X_train.shape, y_train.shape, X_valid.shape


m = RandomForestRegressor(n_estimators=10, max_samples=20000, n_jobs=-1, oob_score=True)
%time m.fit(X_train, y_train)
print_score(m)


m = RandomForestRegressor(n_estimators=40, max_samples=20000, n_jobs=-1, oob_score=True)
m.fit(X_train, y_train)
print_score(m)


m = RandomForestRegressor(n_estimators=40, min_samples_leaf=3, max_samples=20000, n_jobs=-1, oob_score=True)
m.fit(X_train, y_train)
print_score(m)


m = RandomForestRegressor(n_estimators=40, min_samples_leaf=3, max_features=0.5, max_samples=20000, n_jobs=-1, oob_score=True)
m.fit(X_train, y_train)
print_score(m)




