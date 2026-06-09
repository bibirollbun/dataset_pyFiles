!pip install scikit-optimize
!pip install --upgrade scikit-learn
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, OneHotEncoder, LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb
from lightgbm import LGBMClassifier
from skopt import BayesSearchCV
from skopt.space import Real, Integer
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')


train=pd.read_csv('/kaggle/input/playground-series-s5e8/train.csv')
test=pd.read_csv('/kaggle/input/playground-series-s5e8/test.csv')


train.info()


train.describe()


test.describe()


class visual:
    def __init__(self):
        self.numerical_col =[]
        self.categorical_cols = []
    
    def countplot(self,df):
        self.categorical_cols=df.select_dtypes(include=['object', 'category']).columns.tolist()
        for col in self.categorical_cols:
            plt.figure(figsize=(10, 5))
            sns.countplot(x=col, data=df)
            plt.title(f'Distribution of {col}')
            plt.xticks(rotation=45)
            plt.show()
    def grouped_bar_plot(self, df, cat_col, y_col='y', use_proportions=False, figsize=(10, 6)):
        
        # Extract column name if passed as a single-element list
        cat_col = cat_col[0] if isinstance(cat_col, list) else cat_col

        # Validate columns
        if cat_col not in df.columns:
            raise ValueError(f"Column '{cat_col}' not found in DataFrame")
        if y_col not in df.columns:
            raise ValueError(f"Column '{y_col}' not found in DataFrame")

        # Set plot style
        sns.set_style("whitegrid")

        # Create figure
        plt.figure(figsize=figsize)

        # Compute counts or proportions
        if use_proportions:
            plot_data = df.groupby([cat_col, y_col]).size().unstack(fill_value=0)
            plot_data = plot_data.div(plot_data.sum(axis=1), axis=0)
            plot_data = plot_data.stack().reset_index(name='proportion')
            sns.barplot(data=plot_data, x=cat_col, y='proportion', hue=y_col, palette='Set2')
            plt.ylabel('Proportion')
            plt.title(f'Proportion of {y_col} by {cat_col}')
        else:
            sns.countplot(data=df, x=cat_col, hue=y_col, palette='Set2')
            plt.ylabel('Count')
            plt.title(f'Count of {y_col} by {cat_col}')

        # Rotate x-axis labels for readability
        plt.xticks(rotation=45, ha='right')

        # Adjust layout
        plt.tight_layout()
        plt.show()

    def all_categorical_bar_plots(self, df, y_col='y', use_proportions=False, figsize=(10, 6)):
        
        # Validate y_col
        if y_col not in df.columns:
            raise ValueError(f"Column '{y_col}' not found in DataFrame")

        # Identify categorical columns
        cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        if not cat_cols:
            raise ValueError("No categorical columns found in DataFrame")

        print(f"Generating plots for {len(cat_cols)} categorical columns: {cat_cols}")

        # Generate a plot for each categorical column
        for col in cat_cols:
            self.grouped_bar_plot(df, col, y_col=y_col, use_proportions=use_proportions, figsize=figsize)

        
    def histplot(self,df,y_col='y'):
        self.numerical_col=df.select_dtypes(include=[np.number]).columns.tolist()
        for i in self.numerical_col:
            plt.figure(figsize=(8, 4))
            # Pass the DataFrame to the 'data' parameter and the column names
            # to 'x' and 'hue' parameters.
            sns.histplot(data=df, x=i, kde=True, hue=y_col, element='step', stat='density', common_norm=False)
            plt.title(f'Distribution of {i}')
            plt.xticks(rotation=45)
            plt.show() # Add this to display each plot
    def boxplot(self,df):
        self.numerical_col=df.select_dtypes(include=[np.number]).columns.tolist()
        for col in self.numerical_col:
            plt.figure(figsize=(8, 4))
            sns.boxplot(x='y', y=col, data=df)
            plt.title(f'{col} by Subscription Status')
            plt.show()
    def correlation(self,df):
        self.numerical_col=df.select_dtypes(include=[np.number]).columns.tolist()
        corr = df[self.numerical_col + ['y']].corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr, annot=True, cmap='coolwarm', center=0)
        plt.title('Correlation Matrix')
        plt.show()
    def pairplot(self,df,hue_col='y'):
        sns.pairplot(df, hue=hue_col, diag_kind='kde', markers=["o", "s"], height=2.5)
        plt.show()
    def jointplot(self, df, column1, column2, column3):
        # Extract column names if passed as single-element lists
        col1 = column1[0] if isinstance(column1, list) else column1
        col2 = column2[0] if isinstance(column2, list) else column2
        col3 = column3[0] if isinstance(column3, list) else column3

        # Validate columns
        for col in [col1, col2, col3]:
            if col not in df.columns:
                raise ValueError(f"Column '{col}' not found in DataFrame")

        # Create jointplot
        sns.jointplot(x=df[col1], y=df[col2], hue=df[col3], height=6, marginal_ticks=True)
        plt.show()
    def all_visual(self,df):
        # Correctly calling instance methods with `self` and passing the DataFrame `df`
        self.countplot(df)
        self.all_categorical_bar_plots(df, y_col='y', use_proportions=False)
        self.histplot(df)
        self.boxplot(df)
        self.correlation(df)
        #self.pairplot(df)


V=visual()


V.jointplot(train,['duration'],['balance'],['housing'])


df_train=train.copy()


df_train


df_train= df_train.drop(['id'],axis=1)


V.all_visual(df_train)



class preprocessing:
    def __init__(self, df):
        self.df = df.copy()

    def duratio_fetur(self):
        self.df['duration_per_campaign'] = self.df['duration'] / (self.df['campaign'] + 1)
        self.df['duration_squared'] = self.df['duration'] ** 2
        self.df['duration_log'] = np.log1p(self.df['duration'])
        self.df['duration_sqrt'] = np.sqrt(self.df['duration'])
        return self.df

    def contact_category(self):
    
        # Create a column for previous contact category
        self.df['contact_status'] = '0'  # default

        self.df.loc[(self.df['pdays'] != -1) & (self.df['pdays'] <= 30), 'contact_status'] = '1'

        # Old contact
        self.df.loc[(self.df['pdays'] > 30), 'contact_status'] = '2'

        return self.df

    def mont_encoding(self):
        month_map = {
            "jan": 1, "feb": 2, "mar": 3, "apr": 4,
            "may": 5, "jun": 6, "jul": 7, "aug": 8,
            "sep": 9, "oct": 10, "nov": 11, "dec": 12
        }
        self.df["month_num"] = self.df["month"].map(month_map)
        self.df["month_sin"] = np.sin(2 * np.pi * self.df["month_num"] / 12)
        self.df["month_cos"] = np.cos(2 * np.pi * self.df["month_num"] / 12)
        self.df = self.df.drop(columns=['month', 'month_num'])
        return self.df

    def date_encoding(self):
        self.df["day_sin"] = np.sin(2 * np.pi * self.df["day"] / 31)  # Corrected to 31 for days in a month
        self.df["day_cos"] = np.cos(2 * np.pi * self.df["day"] / 31)
        self.df = self.df.drop(columns=['day'])
        return self.df

    def enoding_continus(self):
        self.df['age'] = pd.cut(self.df['age'], bins=[17,37,56,75,95], labels=[0, 1, 2,3]).astype(int)
        self.df['balance'] = pd.cut(self.df['balance'], bins=[-8020,334,887,1600,99718], labels=[0, 1, 2,3]).astype(int)
        self.df['duration'] = pd.cut(self.df['duration'], bins=[-1,102,180,319,645,4918], labels=[0, 1, 2,3,4]).astype(int)
        self.df['campaign'] = pd.cut(self.df['campaign'], bins=[0,3,23,46,63], labels=[0, 1, 2,3]).astype(int)
        self.df['pdays'] = pd.cut(self.df['pdays'], bins=[-2,0,150,380,872], labels=[0, 1, 2,3], right=True, include_lowest=True).astype(int)
        self.df['previous'] = pd.cut(self.df['previous'], bins=[-1,13,26,58,200], labels=[0, 1, 2,3], right=True, include_lowest=True).astype(int)
        return self.df

    def Roscaling(self):
        # initialize scaler
        scaler = RobustScaler()
        cont_columns = ['pdays','previous','campaign','duration','balance','age','duration_per_campaign','duration_log','duration_squared','duration_sqrt']  # Fixed 'duration_sqr' to 'duration_sqrt'
        for i in cont_columns:
            self.df[i] = scaler.fit_transform(self.df[i].values.reshape(-1,1))
        return self.df

    def encoding_chtagrical(self):
        nominal = ['job','marital','contact','poutcome']
        encoder = OneHotEncoder(sparse_output=False)

        # Fit and transform
        encoded = encoder.fit_transform(self.df[nominal])

        # Convert to DataFrame with correct column names
        encoded_df = pd.DataFrame(encoded,
                                  columns=encoder.get_feature_names_out(nominal),
                                  index=self.df.index)

        # Concatenate with original dataframe (and drop old categorical columns)
        self.df = pd.concat([self.df.drop(columns=nominal), encoded_df], axis=1)
        for col in self.df.select_dtypes(include='object').columns:
            le = LabelEncoder()
            self.df[col] = le.fit_transform(self.df[col])
        return self.df

    def extract_features(self):

        # Balance: Store balance value where 0 < balance < 10
        if 'balance' in self.df.columns:
            self.df['balance_favorable'] = self.df['balance'].where((self.df['balance'] > 0) & (self.df['balance'] < 10), np.nan)

        # Duration: Store duration value where duration > 0.5
        if 'duration' in self.df.columns:
            self.df['duration_favorable'] = self.df['duration'].where(self.df['duration'] > 0.5, np.nan)

        # Pdays: Store pdays value where 50 < pdays < 200
        if 'pdays' in self.df.columns:
            self.df['pdays_favorable'] = self.df['pdays'].where((self.df['pdays'] > 50) & (self.df['pdays'] < 200), np.nan)

        # 2. Cyclical Features
        # Month: Store month_sin or month_cos value where month_sin <-0.75 or >0.75, or -0.75 < month_cos < 0.75
        if all(col in self.df.columns for col in ['month_sin', 'month_cos']):
            # Store month_sin if its condition is True
            month_sin_condition = (self.df['month_sin'] < -0.75) | (self.df['month_sin'] > 0.75)
            self.df['month_sin_favorable'] = self.df['month_sin'].where(month_sin_condition, np.nan)

            # Store month_cos if its condition is True
            month_cos_condition = (self.df['month_cos'] > -0.75) & (self.df['month_cos'] < 0.75)
            self.df['month_cos_favorable'] = self.df['month_cos'].where(month_cos_condition, np.nan)

        # Day: Store day_sin or day_cos value where day_sin > 0.25 or day_cos < -0.75 or day_cos > 0.25
        if all(col in self.df.columns for col in ['day_sin', 'day_cos']):
            # Store day_sin if its condition is True
            day_sin_condition = (self.df['day_sin'] > 0.25)
            self.df['day_sin_favorable'] = self.df['day_sin'].where(day_sin_condition, np.nan)

            # Store day_cos if its condition is True
            day_cos_condition = (self.df['day_cos'] < -0.75) | (self.df['day_cos'] > 0.25)
            self.df['day_cos_favorable'] = self.df['day_cos'].where(day_cos_condition, np.nan)

        # Contact: Store contact_cellular value where contact_cellular > 0.9 or contact_unknown < 0.1
        if all(col in self.df.columns for col in ['contact_cellular', 'contact_unknown']):
            contact_condition = (self.df['contact_cellular'] > 0.9) | (self.df['contact_unknown'] < 0.1)
            # Store contact_cellular value if either condition is True
            self.df['contact_favorable'] = self.df['contact_cellular'].where(contact_condition, np.nan)
        # 5. Feature Interactions
        # Store duration value where duration > 0.5 or (contact_cellular > 0.9 or contact_unknown < 0.1)
        if all(col in self.df.columns for col in ['duration', 'contact_cellular', 'contact_unknown']):
            interaction_condition = (self.df['duration'] > 0.5) | (self.df['contact_cellular'] > 0.9) | \
                                   (self.df['contact_unknown'] < 0.1)
            self.df['duration_contact_interaction'] = self.df['duration'].where(interaction_condition, np.nan)
        self.df.fillna(0, inplace=True)

        return self.df

    def combination(self):
        combinations = [
            ('poutcome_success', 'duration_favorable', 'pdays_favorable'),
        ]

        # Loop through and create new columns
        for combo in combinations:
            col_name = "_".join(combo)  # name like 'duration_pdays_poutcome_success'
            self.df[col_name] = self.df[list(combo)].max(axis=1)

        # Now new_df contains only the combined columns
        print(self.df.head())
        return self.df

    def all_prprocesing(self):
        self.duratio_fetur()
        self.contact_category()
        self.mont_encoding()
        self.date_encoding()
        self.enoding_continus()
        self.Roscaling()
        self.encoding_chtagrical()
        self.extract_features()
        self.combination()
        return self.df


Tr=preprocessing(df_train)
process_df_train=Tr.all_prprocesing()


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression

class ImbalanceModelTrainer:
    def __init__(self, X, y, test_size=0.2, random_state=42):
        self.X = X
        self.y = y
        self.test_size = test_size
        self.random_state = random_state
        self.models = {}
        self.results = {}
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

    def train_models(self):
        # Logistic Regression with class_weight
        self.models["LogisticRegression"] = LogisticRegression(
            class_weight="balanced", solver="liblinear", random_state=self.random_state
        )

        # Random Forest with class_weight
        self.models["RandomForest"] = RandomForestClassifier(
            n_estimators=200, class_weight="balanced",
            random_state=self.random_state, n_jobs=-1
        )

        # LightGBM with class_weight
        self.models["LightGBM"] = LGBMClassifier(
            n_estimators=300, class_weight="balanced",
            random_state=self.random_state, n_jobs=-1
        )

        # XGBoost with GPU support + scale_pos_weight
        scale_pos_weight = np.sum(self.y_train == 0) / np.sum(self.y_train == 1)
        self.models["XGBoost"] = XGBClassifier(
            n_estimators=300,
            objective="binary:logistic",
            tree_method="gpu_hist",  # GPU acceleration
            predictor="gpu_predictor",
            scale_pos_weight=scale_pos_weight,
            random_state=self.random_state,
            use_label_encoder=False,
            eval_metric="logloss"
        )

        # Train all models
        for name, model in self.models.items():
            model.fit(self.X_train, self.y_train)
            preds = model.predict_proba(self.X_test)[:, 1]
            roc_score = roc_auc_score(self.y_test, preds)
            self.results[name] = roc_score
            y_pred = model.predict(self.X_test)
            score = f1_score(self.y_test, y_pred, average="binary")
            print(f"✅ {name} F1-score: {score:.4f}")
            print(classification_report(self.y_test, y_pred))

    def get_best_model(self):
        best_model = max(self.results, key=self.results.get)
        return best_model, self.results[best_model]

    def summary(self):
        return pd.DataFrame.from_dict(self.results, orient="index", columns=["ROC_AUC"]).sort_values(by="ROC_AUC", ascending=False)




X=process_df_train.drop(['y'],axis=1)
y=process_df_train['y']


trainer = ImbalanceModelTrainer(X, y)
trainer.train_models()
print(trainer.summary())
print("Best Model:", trainer.get_best_model())



import optuna
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
import time

def run_xgboost_optuna(X,y, time_limit=3200):
    """
    Runs XGBoost with Optuna hyperparameter tuning.
    Stops automatically after `time_limit` seconds.
    Returns best model and score.
    """
    
    X_train,  X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "gamma": trial.suggest_float("gamma", 0, 5),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 10.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 10.0),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, 10.0),
            "random_state": 42,
            "tree_method": "gpu_hist",  # GPU acceleration
            "use_label_encoder": False,
            "eval_metric": "logloss"
        }
        model = XGBClassifier(**params)
        model.fit(X_train, y_train)
        preds = model.predict_proba(X_valid)[:, 1]
        score = roc_auc_score(y_valid, preds)
        return score

    # Define study with time limit
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, timeout=time_limit)

    print("✅ Best ROC AUC:", study.best_value)
    print("✅ Best Params:", study.best_params)

    # Train final model on full data with best params
    best_model = XGBClassifier(**study.best_params,
                               random_state=42,
                               tree_method="gpu_hist",
                               use_label_encoder=False,
                               eval_metric="logloss")
    best_model.fit(X_train, y_train)

    return best_model, study.best_value, study.best_params



model, best_score, best_params = run_xgboost_optuna(
    X,y
)


test_ids = test['id'].copy()  # Save IDs as a Series
test_process = test.drop(columns=['id'])  # DataFrame without 'ID' for processing

# Step 2: Instantiate the preprocessing class and apply all_prprocesing
Ts = preprocessing(test_process)
transformed_test = Ts.all_prprocesing()

# Step 3: Reattach 'ID' to the transformed DataFrame (insert at the beginning for clarity)
#transformed_test.insert(0, 'id', test_ids)




import optuna
import numpy as np
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from lightgbm import LGBMClassifier

def run_lightgbm_optuna(X, y, time_limit=3200):
    """
    Runs LightGBM with Optuna hyperparameter tuning using Stratified K-Fold CV.
    Optimizes ROC AUC score.
    """
    
    n_splits = 5
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 500, 30000),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 31, 300),
            "max_depth": trial.suggest_int("max_depth", -1, 15),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 1.0),
            "max_bin": trial.suggest_int("max_bin",255,4851),
            "boosting_type": trial.suggest_categorical("boosting_type", ["gbdt", "dart", "goss"]),
            "objective": "binary",
            "metric": "auc",     # <--- LightGBM uses AUC
            "random_state": 42,
            "n_jobs": -1,
            "device": "cpu"
        }

        cv_scores = []
        for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
            print(f"\n<== Training fold {fold + 1}/{n_splits} ==>")
        
            X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
            X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
        
            model = LGBMClassifier(**params)
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                eval_metric='auc',
                callbacks=[
                    lgb.early_stopping(100),
                    lgb.log_evaluation(50)
                ]
            )
            preds = model.predict_proba(X_val)[:, 1]
            cv_scores.append(roc_auc_score(y_val, preds))

        return np.mean(cv_scores)   # Optuna optimizes mean ROC AUC

    # Run Optuna
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, timeout=time_limit)

    print("✅ Best ROC AUC:", study.best_value)
    print("✅ Best Params:", study.best_params)

    # Train final model on full dataset
    best_model = LGBMClassifier(**study.best_params, n_jobs=-1)
    best_model.fit(X, y)

    return best_model, study.best_value, study.best_params




Tr=preprocessing(df_train)
process_tain=Tr.all_prprocesing()


X=process_tain.drop(['y'],axis=1)
y=process_tain['y']


best_params


best_model, best_value, best_params=run_lightgbm_optuna(X,y)


best_params


best_value


best_estimator,best_params=bayesian_opt_lgbm(X, y)


import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
import numpy as np

def train_lightgbm(train, test, target):
    X = train
    y = target
    
    X_test = test.copy()
    
    n_splits = 5
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    y_probs = np.zeros(len(X_test))
    models = []
    
    # Corrected params: 'class_weight' (not 'weights'), remove invalid 'eval_metric'/'metric'
    best_parms = {'n_estimators': 26261,
                    'learning_rate': 0.012814954358255548,
                    'num_leaves': 288,
                    'max_depth': 5,
                    'subsample': 0.6883646083074866,
                    'min_child_samples': 56,
                    'reg_alpha': 0.4043915820797841,
                    'reg_lambda': 0.7192030876825268,
                    'max_bin': 3624,
                    'boosting_type': 'gbdt',
                    'random_state': 42, }
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        print(f"\n<== Training fold {fold + 1}/{n_splits} ==>")
        
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
        
        model = lgb.LGBMClassifier(
            **best_parms,
            
            #random_state=42,
            objective= 'binary',
            metric= 'auc',
            
            class_weight= 'balanced',   # handles imbalance
            
            n_jobs=-1,
            verbosity=0  # Optional: Reduces logging verbosity (set to -1 for even quieter)
        )
        
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric='auc',  # Moved here for proper AUC evaluation
            callbacks=[
                lgb.early_stopping(200),
                lgb.log_evaluation(100)
            ]
        )
        
        models.append(model)
        y_probs += model.predict_proba(X_test)[:, 1] / n_splits
    
    print("\nLightGBM model training complete.")
    return y_probs, models


y_probs, models = train_lightgbm(X, transformed_test, y)


joblib.dump(model, 'lightgbm_model.pkl')


test_df = pd.read_csv("/kaggle/input/playground-series-s5e8/test.csv")
submission = pd.DataFrame({
    'id': test_ids,
    'target': y_probs 
})

submission.to_csv('submission.csv', index=False)
print("Submission file saved as 'submission.csv'")

