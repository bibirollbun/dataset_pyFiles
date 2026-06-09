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


from xgboost import XGBRegressor, XGBClassifier
import xgboost as xgb
from lightgbm import LGBMRegressor,LGBMClassifier
import lightgbm as lgb
from catboost import CatBoostRegressor, CatBoostClassifier
import catboost as cb
import seaborn as sns
from sklearn.model_selection import TimeSeriesSplit, KFold
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
from pathlib import Path
from sklearn.linear_model import LinearRegression


class settings:

    train_link = "/kaggle/input/playground-series-s5e1/train.csv"
    test_link =  "/kaggle/input/playground-series-s5e1/test.csv"
    sub_link = "/kaggle/input/playground-series-s5e1/sample_submission.csv"
    
    seed = 42
    target = 'num_sold'
    n_splits = 5
    col_ignore = ['num_sold','date']


def mape(y_true,y_pred):
    """
    This function calculates MAPE

    Parameters:
    - y_true: list of real numbers, true values
    - y_pred: list of real numbers, predicted values

    Returns:
    - MAPE
    """
    return mean_absolute_percentage_error(y_true, y_pred)


class TimeSeriesModel:

    def __init__(self, model, n_splits = settings.n_splits, random_state = settings.seed):
        """
        Initialise the TimeSeriesModel training class

        Parameters:
        - model: model instance
        - n_splits: number of splits in cv
        - random_state: for reproducability
        """
        self.model = model
        self.n_splits = n_splits
        self.random_state = random_state
        self.oof = None
        self.pred = None
        self.scores = []
        self.cats = []

    def preprocessing(self, train, test):
        """
        Proprocessing the test and train data

        Parameters
        - train: training data
        - test: testing data
        """
        features = [col for col in train.columns if not col in settings.col_ignore]
        
        # Combined test and train to label encode
        combined = pd.concat([train,test], axis = 0, ignore_index = True)

        # Datetime prepreocessing
        combined["date"] = pd.to_datetime(combined["date"])
        combined["year"] = combined["date"].dt.year.astype("float32")
        combined["month"] = combined["date"].dt.month.astype("float32")
        combined["day"] = combined["date"].dt.day.astype("float32")
        combined["dow"] = combined["date"].dt.dayofweek.astype("float32")

        for col in features:
            if combined[col].dtype == "object":
                self.cats.append(col)
                combined[col] = combined[col].fillna("NAN")
                combined[col],_ = combined[col].factorize()
                combined[col] -= combined[col].min()
                combined[col] = combined[col].astype("int32")
                combined[col] = combined[col].astype("category")

            else:
                if combined[col].dtype=="float64":
                    combined[col] = combined[col].astype("float32")
                if combined[col].dtype=="int64":
                    combined[col] = combined[col].astype("int32")
                    
        print("#"*25,"Categorical Columns", "#"*25)
        print(self.cats)
        combined[settings.target] = np.log1p(combined[settings.target])
        combined[settings.target] = combined[settings.target].fillna(-1)
        train = combined.iloc[:len(train)].copy()
        test = combined.iloc[len(train):].reset_index(drop=True).copy().drop([settings.target], axis = 1)
        return train, test

    def fit_predict(self,train, test, catboost = False):
        """
        Fit the model using timeseriessplit and collect oof predictions.

        Parameters:
        - train: the training data
        - test: the test data from kaggle

        Returns:
        - oof predictions
        - Predictions for the test data
        """

        # Collect features

        features = [col for col in train.columns if not col in settings.col_ignore]
        print("#"*25,"Features","#"*25)
        print(f"There are {len(features)} features.")
        print(features)

        # Initialise k-fold cross validation
        print("#"*25,"Initialise Cross-Validation","#"*25)

        kf = KFold(n_splits = self.n_splits, shuffle=True, random_state=self.random_state)

        # Predictions for OOF and test
        self.oof = np.zeros(len(train))
        self.preds = np.zeros(len(test))

        # Cross-validation loop

        for fold, (train_idx, test_idx) in enumerate(kf.split(train)):

            print("#"*25, f"Fold {fold+1}", "#"*25)

            x_train, x_val = train.loc[train_idx, features].copy(), train.loc[test_idx, features].copy()
            y_train, y_val = train.loc[train_idx, settings.target].copy(), train.loc[test_idx, settings.target].copy()

            x_test = test[features]

            # Train
            if catboost:
                self.model.fit(x_train, y_train, self.cats, verbose = 500)
            else:
                self.model.fit(x_train, y_train)
                

            # prediction on val set
            val_preds = self.model.predict(x_val)
            self.oof[test_idx] = val_preds
        
            # Calculate score
            m = mape(np.expm1(y_val), np.expm1(val_preds))
            self.scores.append(m)
            print(f"Fold {fold+1}: Mean Absolute Percentage Error: {m}")

            self.preds += self.model.predict(x_test)/self.n_splits
                
        print("#"*25,"Cross-Validation Completed","#"*25)
        print(f"Average Mean Absolute Percentage Error: {np.mean(self.scores)}")
        print(f"Overall Mean Absolute Percentage Error: {mape(np.expm1(train[settings.target]), np.expm1(self.oof))}")
        return self.oof, self.preds
        
    def save_predictions(self, filename,train, test):
        """
        Save predictions in a csv.

        Parameters:
        - filename: model_name
        - train: train data for ID
        - test: test data for ID
        """
        directory_train = Path("/kaggle/working/train")
        directory_test = Path("/kaggle/working/test")

        directory_train.mkdir(exist_ok = True)
        print(f"Directory '{directory_train}' created successfully")
        directory_test.mkdir(exist_ok = True)
        print(f"Directory '{directory_test}' created successfully")
            
        
        # oof train preds
        train_csv = train.copy()
        train_csv["prediction"] = self.oof

        train_csv.to_csv("train/"+filename+"_train.csv", index = False)

        # test
        test_csv = test.copy()
        test_csv["prediction"] = self.preds

        test_csv.to_csv("test/"+filename+"_test.csv", index = False)


class Ensembler:
    def __init__(self, train_folder, test_folder):
        """
        Initialse class with the train and test folders.

        Parameters:
        - train_folder: Path to the train folder containing OOF predictions
        - test_folder: Path to the predictions
        """

        self.train_folder = Path(train_folder)
        self.test_folder = Path(test_folder)
        self.scores = []

    def load_files(self, folder_path):
        """
        Load the train and test files.

        Returns:
        - train: consolidated predictions for each dataset
        - test: consolidated predictions for each dataset
        """
        predictions = {}

        # loop through the csv files
        counter = 0
        for idx, file in enumerate(sorted(folder_path.glob("*.csv"))):
            try:
                df = pd.read_csv(file)
                if "prediction" in df.columns:
                    counter +=1
                    predictions[f"prediction_{counter}"] = df["prediction"].reset_index(drop=True)
                    print(f"Loaded 'prediction' column from {file.name}")
                else:
                    print(f"'prediction' column not found in {file.name}")
            except Exception as e:
                print(f"Error loading {file.name}: {e}")
        if predictions:
            predictions = pd.DataFrame(predictions)
            return predictions
        else:
            print("No predictions found.")
            return pd.DataFrame()

    def concatenate_train_test(self, train):
        """
        Concatenate train and test folder data into another train and test set.

        Parameters:
        - train: we need this to get the target variable in the training set

        Returns
        - train: Another training set based on the oof predictions.
        - test: Another test set based on preds.
        """
        print(f"Loading predictions from the train folder {self.train_folder}")
        train_predictions = self.load_files(self.train_folder)
        train_predictions[settings.target] = train[settings.target].reset_index(drop=True)

        print(f"Loading predictions from the test folder {self.test_folder}")
        test_predictions = self.load_files(self.test_folder)

        return train_predictions, test_predictions

    def stack(self, train, test):

        model = LinearRegression()
        # Collect features
        
        features = [col for col in train.columns if not col in settings.col_ignore]
        print("#"*25,"Features","#"*25)
        print(f"There are {len(features)} features.")
        print(features)

        # Initialise k-fold cross validation
        print("#"*25,"Initialise Cross-Validation","#"*25)

        kf = KFold(n_splits = settings.n_splits, shuffle=True, random_state=settings.seed)

        # Cross-validation loop
        for fold, (train_idx, test_idx) in enumerate(kf.split(train)):
            
            print("#"*25,f"Fold {fold+1}","#"*25)
            # Split data
            x_train, x_val = train.loc[train_idx, features].copy(), train.loc[test_idx, features].copy()
            y_train, y_val = train.loc[train_idx, settings.target].copy(), train.loc[test_idx, settings.target].copy()
            x_test = test[features]
             
            # Predict on validation set

            model.fit(x_train, y_train)

            val_preds = model.predict(x_val)
            
            # Calculate score
            m = mape(np.expm1(y_val), np.expm1(val_preds))
            self.scores.append(m)
            print(f"Fold {fold+1}: Mean Absolute Percentage Error: {m}")

        model.fit(train[features], train[settings.target])
        prediction = model.predict(test[features])
            
        print("#"*25,"Cross-Validation Completed","#"*25)
        print(f"Mean Absolute Percentage Error: {np.mean(self.scores)}")
        #print(f"Overall Mean Absolute Percentage Error: {mape(np.expm1(train[settings.target]), np.expm1(overall_pred))}")
        
        return prediction


train = pd.read_csv(settings.train_link, index_col = ['id'])
test = pd.read_csv(settings.test_link, index_col = ['id'])


train.shape


train = train.dropna()


train.shape


model = XGBRegressor(
        device="cuda",
        enable_categorical=True
)

pipeline = TimeSeriesModel(model)
df_train, df_test = pipeline.preprocessing(train, test)
oof_xgb, preds_xgb = pipeline.fit_predict(df_train,df_test)
pipeline.save_predictions("XGBoost_", df_train, df_test)



model = LGBMRegressor(
    device = "gpu",
    objective ="regression",
    verbose = -1
)

pipeline = TimeSeriesModel(model)
df_train, df_test = pipeline.preprocessing(train, test)
oof_lgb, preds_lgb = pipeline.fit_predict(df_train,df_test)
pipeline.save_predictions("LGBM_", df_train, df_test)


model = CatBoostRegressor(task_type = "GPU", grow_policy = 'Lossguide',random_state = settings.seed)

pipeline = TimeSeriesModel(model)
df_train, df_test = pipeline.preprocessing(train, test)
oof_cat, preds_cat = pipeline.fit_predict(df_train,df_test, catboost = True)
pipeline.save_predictions("CatBoost_", df_train, df_test)


esm = Ensembler("/kaggle/working/train", "/kaggle/working/test")
final_train, final_test = esm.concatenate_train_test(df_train)
prediction = esm.stack(final_train,final_test)


submission = pd.read_csv(settings.sub_link)
submission['num_sold'] = np.expm1(prediction)
submission.to_csv("submission.csv", index = False)
submission

