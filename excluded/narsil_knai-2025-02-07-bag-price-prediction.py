import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


def split_numerical_categorical(df):
    """
    Splits the columns of a DataFrame into numerical and categorical features.

    Parameters:
    df (pandas.DataFrame): The DataFrame to split.

    Returns:
    tuple: A tuple containing two lists - numerical columns and categorical columns.
    """
    numerical_cols = df.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=['number']).columns.tolist()
    return numerical_cols, categorical_cols


def nums_combinations(df_, numerical_cols):
    for col1 in numerical_cols:
        for col2 in numerical_cols:
            if col1 != col2:
                df_[f'{col1}__{col2}__dzielenie'] = df_[col1]/df_[col2]
                df_[f'{col1}__{col2}__mnozenie'] = df_[col1]*df_[col2]
                df_[f'{col1}__{col2}__dodawanie'] = df_[col1]+df_[col2]
                df_[f'{col1}__{col2}__odejmowanie'] = df_[col1]-df_[col2]
    return df_


path = '/kaggle/input/playground-series-s5e2/'
df_train = pd.read_csv(f'{path}train.csv')
df_test = pd.read_csv(f'{path}test.csv')


df_train.shape


df_train_extra = pd.read_csv(f'{path}training_extra.csv')


df_train_extra.shape


df_train = pd.concat([df_train, df_train_extra], axis = 0, ignore_index = True)


# df_train.drop("Weight Capacity (kg)", axis = 1, inplace = True)
# df_test.drop("Weight Capacity (kg)", axis = 1, inplace = True)


df_train


df_train.drop('id', axis=1, inplace=True)
df_test.drop('id', axis=1, inplace=True)


# df_train_orig = pd.read_csv("/kaggle/input/student-bag-price-prediction-dataset/Noisy_Student_Bag_Price_Prediction_Dataset.csv")
# df_train = pd.concat([df_train, df_train_orig], axis = 0, ignore_index = True)


df_train.shape


numerical_cols, categorical_cols = split_numerical_categorical(df_train)


target_column = 'Price'
numerical_cols.remove(target_column)


def transform_categoricals(df_, categorical_cols):
    return pd.get_dummies(df_, columns=categorical_cols) 


for c in categorical_cols:
    print(c, "n unique:",df_train[c].nunique())


df_train['n_null'] = df_train.isnull().sum(axis=1)


df_test['n_null'] = df_test.isnull().sum(axis=1)


import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

class CategoricalTargetEncoder:
    def __init__(self, categorical_features, target, n_splits=5, random_state=42):
        """
        Parameters:
            categorical_features (list): List of names of categorical features to encode.
            target (str): Name of the target column.
            n_splits (int): Number of folds to use for out-of-fold encoding.
            random_state (int): Random state for reproducibility.
        """
        self.categorical_features = categorical_features
        self.target = target
        self.n_splits = n_splits
        self.random_state = random_state
        self.global_mean = None
        self.mapping_ = {}  # to store full train mapping for each feature

    def fit_transform(self, train_df):
        """
        Fit the encoder on the training DataFrame and perform out-of-fold transformation.

        Parameters:
            train_df (pd.DataFrame): The training data.

        Returns:
            pd.DataFrame: A copy of train_df with new columns for each encoded feature.
        """
        # Work on a copy to avoid modifying the original DataFrame
        train_df = train_df.copy()
        self.global_mean = train_df[self.target].mean()

        # Process each categorical feature
        for cat in self.categorical_features:
            encoded_col = cat + "_encoded"
            # Initialize the new column with NaNs
            train_df[encoded_col] = np.nan

            # Set up K-Fold cross-validation
            kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)

            # For each fold, compute the mean target per category using the training part
            for train_idx, val_idx in kf.split(train_df):
                # Compute means on the training fold
                means = train_df.iloc[train_idx].groupby(cat)[self.target].mean()
                # Map these means to the validation fold
                train_df.loc[train_df.index[val_idx], encoded_col] = train_df.loc[train_df.index[val_idx], cat].map(means)

            # Fill in any missing values (e.g. if a category never appears in the training fold)
            train_df[encoded_col].fillna(self.global_mean, inplace=True)

            # Also, compute the full mapping on the entire train data for later use on test data.
            self.mapping_[cat] = train_df.groupby(cat)[self.target].mean()

        return train_df

    def transform(self, df):
        """
        Transform a new DataFrame (e.g., test data) using the encoding learned from the training data.

        Parameters:
            df (pd.DataFrame): The DataFrame to transform.

        Returns:
            pd.DataFrame: A copy of df with new encoded feature columns.
        """
        # Work on a copy to avoid modifying the original DataFrame
        df = df.copy()

        for cat in self.categorical_features:
            encoded_col = cat + "_encoded"
            # Map the stored means to the new data
            df[encoded_col] = df[cat].map(self.mapping_[cat])
            # For any unseen categories, fill with the global mean
            df[encoded_col].fillna(self.global_mean, inplace=True)

        return df



df_train['Weight Capacity (kg)_category'] = df_train['Weight Capacity (kg)'].astype(str)
df_test['Weight Capacity (kg)_category'] = df_test['Weight Capacity (kg)'].astype(str)


categorical_cols += ['Weight Capacity (kg)_category']


categorical_cols


encoder = CategoricalTargetEncoder(categorical_features=categorical_cols, target=target_column, n_splits=5, random_state=42)
df_train_encoded = encoder.fit_transform(df_train)
df_test_encoded = encoder.transform(df_test)


df_train_encoded['n_null'] = df_train['n_null']


df_test_encoded['n_null'] = df_test['n_null']


cols_to_take = numerical_cols + [c for c in df_train_encoded.columns if "_encoded" in c]


cols_to_take.append('n_null')


cols_to_take


X_train_encoded = df_train_encoded[cols_to_take]


X_test_encoded = df_test_encoded[cols_to_take]


y_train = df_train[target_column]
df_train = df_train.drop([target_column], axis=1)


df_all = pd.concat([df_train, df_test], axis=0)
df_all = transform_categoricals(df_all, categorical_cols)
X_train = df_all[:df_train.shape[0]]
X_test = df_all[df_train.shape[0]:]


X_train


from sklearn.metrics import mean_squared_error


def val_loss_function(y_val, val_preds):
    return mean_squared_error(y_val, val_preds) ** 0.5


from sklearn.model_selection import KFold
import numpy as np
from lightgbm import LGBMClassifier, LGBMRegressor

# Assuming df_all, y_train, and df_train are already defined


# Prepare arrays to store out-of-fold predictions and test set predictions
oof_preds = np.zeros(X_train_encoded.shape[0])
test_preds = np.zeros(X_test_encoded.shape[0])

# Initialize 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

val_score = 0

# Loop over each fold
for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_encoded)):
    print(f"Fold {fold + 1}")
    
    # Split data into train and validation sets
    X_tr, X_val = X_train_encoded.iloc[train_idx], X_train_encoded.iloc[val_idx]
    y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
    
    # Initialize and train the model
    model = LGBMRegressor()
    model.fit(X_tr, y_tr)
    
    # Predict on validation set and test set
    val_preds = model.predict(X_val)
    oof_preds[val_idx] = val_preds
    cur_val_score = val_loss_function(y_val, val_preds)
    print('current validation score: ', cur_val_score)
    val_score += cur_val_score / kf.n_splits
    test_preds += model.predict(X_test_encoded) / kf.n_splits

# Final averaged predictions for the test set
y_pred_lgbm = test_preds


print('average validation score', val_score)
#all features = 38.88557081798126
#all features with categorical encoding 38.91
#all features with categorical encoding of weight kg 38.71
#without weight capacity = 38.914244690934424


X_train['Weight Capacity (kg)_category_encoded'] = X_train_encoded['Weight Capacity (kg)_category_encoded']
X_test['Weight Capacity (kg)_category_encoded'] = X_test_encoded['Weight Capacity (kg)_category_encoded']


X_train['n_null'] = X_train_encoded['n_null']
X_test['n_null'] = X_test_encoded['n_null']


from xgboost import XGBRegressor
oof_preds = np.zeros(X_train.shape[0])
test_preds = np.zeros(X_test.shape[0])
for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
    print(f"Fold {fold + 1}")
    
    # Split data into train and validation sets
    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
    
    # Initialize and train the model
    model = XGBRegressor(
        n_estimators=500,
        learning_rate=0.1,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='reg:squarederror',
        random_state=42,
        n_jobs=-1
    )
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=100,
        verbose=100
    )
    
    # Predict on validation set and test set
    oof_preds[val_idx] = model.predict(X_val)
    test_preds += model.predict(X_test) / kf.n_splits

# Final averaged predictions for the test set
y_pred_XGB = test_preds


#y_pred = y_pred_lgbm
y_pred = (y_pred_lgbm+y_pred_XGB)/2


y_pred_lgbm


#y_pred_XGB


ssub = pd.read_csv(f"{path}sample_submission.csv")


ssub[target_column] = y_pred
ssub.to_csv('submission.csv', index = False)







