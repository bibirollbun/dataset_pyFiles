import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from colorama import Fore
import seaborn as sns

import missingno as msno
from IPython.display import display, Markdown

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split, KFold

import xgboost as xgb

from tqdm import tqdm
import os

import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class config:
    data_dir = '/kaggle/input/playground-series-s5e10'
    sub_path = '/kaggle/input/playground-series-s5e10/sample_submission.csv'
    seed = 42
    test_size = 0.2
    target = 'accident_risk'
    V = 1

cfg = config()


class UnivariateAnalysis:
    def __init__(self, df, target):
        self.data = df.copy()
        self.data = self.data[:1000]  # sample for speed
        self.target = target
        self.num_cols = [col for col in self.data.columns if self.data[col].dtype in ['int64', 'float64'] and col != target]
        self.cat_cols = [col for col in self.data.columns if self.data[col].dtype in ['O','bool'] and col != target]
        display(Markdown(f"### ğŸ”¢ There are **{len(self.num_cols)} numerical** columns and **{len(self.cat_cols)} categorical** columns"))

    # --- Plots for classification ---
    def barplots(self, target):
        plt.figure(figsize=(18, 10))
        for i, col in enumerate(self.num_cols, 1):
            plt.subplot(2, 4, i)
            sns.histplot(data=self.data, x=col, hue=target, kde=True, bins=30, common_norm=False, element='step', stat='density')
            plt.title(f"Distribution of {col}", pad=10, weight='bold')
        plt.tight_layout()
        plt.show()

    def boxplots(self, target):
        plt.figure(figsize=(18, 10))
        for i, col in enumerate(self.num_cols, 1):
            plt.subplot(2, 4, i)
            sns.boxplot(data=self.data, y=col, x=target)
            plt.title(f"{col} by Target", pad=10, weight='bold')
        plt.tight_layout()
        plt.show()

    def countplots(self, target):
        plt.figure(figsize=(18, 10))
        for i, col in enumerate(self.cat_cols, 1):
            plt.subplot(2, 5, i)
            sns.countplot(data=self.data, x=col, hue=target)
            plt.title(f"Frequency of {col}")
            plt.xticks(rotation=90)
        plt.tight_layout()
        plt.show()

    def targetDistribution(self, target):
        plt.figure(figsize=(18, 10))

        plt.subplot(1, 2, 1)
        ax = sns.countplot(x=target, data=self.data)
        plt.title("Target Distribution", pad=15)
        for p in ax.patches:
            ax.annotate(f'{int(p.get_height())}',  
                        (p.get_x() + p.get_width() / 2., p.get_height()), 
                        ha='center', va='center', xytext=(0, 10), textcoords='offset points')

        plt.subplot(1, 2, 2)
        self.data[target].value_counts().plot(kind='pie', autopct='%1.1f%%', explode=[0.05, 0], startangle=90)
        plt.title("Target Proportion", pad=15)

        plt.tight_layout()
        plt.show()

    # --- Plots for regression ---
    def targetDistributionRegression(self, target):
        plt.figure(figsize=(10, 5))
        plt.subplot(1,2,1)
        sns.histplot(data=self.data, x=target, kde=True, bins=30)
        plt.title("Target Distribution", pad=15)
        # plt.show()

        # plt.figure(figsize=(6, 5))
        plt.subplot(1,2,2)
        sns.boxplot(y=self.data[target])
        plt.title("Target Spread (Boxplot)", pad=15)

        plt.tight_layout()
        plt.show()

    def scatterplots(self, target):
        """Scatterplots of numerical features vs target"""
        plt.figure(figsize=(18, 10))
        for i, col in enumerate(self.num_cols, 1):
            plt.subplot(2, 5, i)
            sns.scatterplot(data=self.data, x=col, y=target, alpha=0.5)
            sns.regplot(data=self.data, x=col, y=target, scatter=False, color='red')  # regression line
            plt.title(f"{col} vs {target}", pad=10, weight='bold')
        plt.tight_layout()
        plt.show()

    def NumericalDistributionRegression(self,target):
        """Distribution of Numerical Columns"""
        plt.figure(figsize=(18,10))
        for i,col in enumerate(self.num_cols,1):
            plt.subplot(2,5,i)
            sns.histplot(data = self.data,x=col,kde=True,bins=30)
            plt.title(f"{col} Distribution",pad = 10, weight='bold')
        plt.tight_layout()
        plt.show()

    def boxplots_regression(self, target):
        """Boxplots of categorical features vs target"""
        plt.figure(figsize=(18, 10))
        for i, col in enumerate(self.cat_cols, 1):
            plt.subplot(2, 5, i)
            sns.boxplot(data=self.data, x=col, y=target)
            plt.title(f"{target} by {col}", pad=10, weight='bold')
            plt.xticks(rotation=90)
        plt.tight_layout()
        plt.show()

    # --- Automation methods ---
    def UAautomation(self, target):
        """For classification targets"""
        display(Markdown("## ğŸ“ˆ Analysis of Numerical Columns"))
        self.barplots(target=target)
        self.boxplots(target=target)

        display(Markdown("## ğŸ�·ï¸� Analysis of Categorical Columns"))
        self.countplots(target)

        display(Markdown("## ğŸ�¯ Analysis of Target"))
        self.targetDistribution(target=target)

    def UAautomation_regression(self, target):
        """For regression targets"""
        display(Markdown("## ğŸ�¯ Target Distribution"))
        self.targetDistributionRegression(target=target)

        display(Markdown("## ğŸ“ˆ Numerical Columns vs Target"))
        self.scatterplots(target=target)
        self.NumericalDistributionRegression(target = target)

        display(Markdown("## ğŸ�·ï¸� Categorical Columns vs Target"))
        self.boxplots_regression(target=target)


class BivariateAnalysis:
    def __init__(self, df, target, test_size, seed):
        self.test_size = test_size
        self.seed = seed
        self.target = target
        self.data = df.copy()
        self.num_cols = [col for col in self.data.columns if self.data[col].dtype in ['int64', 'float64'] and col != target]
        self.cat_cols = [col for col in self.data.columns if self.data[col].dtype == 'O' and col != target]

    def correlationPlot(self):
        display(Markdown("## ğŸ”— Feature Correlation Matrix"))

        sample_data = self.data[:1000]
        for col in self.cat_cols:
            sample_data[col], _ = pd.factorize(sample_data[col])

        corr_mat = sample_data.corr()
        mask = np.triu(np.ones_like(corr_mat, dtype=bool))

        plt.figure(figsize=(18, 10))
        sns.heatmap(corr_mat, mask=mask, fmt='.2f', cmap='winter', annot=True)
        plt.title("Feature Correlation Matrix", pad=10)
        plt.xticks(rotation=90, ha='right')
        plt.yticks(rotation=0)
        plt.show()

    def featureInteraction(self):
        display(Markdown("## ğŸ”„ Feature Interactions"))

        sample_df = self.data[:1000]
        plt.figure(figsize=(18, 10))

        plt.subplot(2, 2, 1)
        sns.violinplot(x='education', y='age', data=sample_df, inner='quartile')
        plt.title("Age Distribution by Education", pad=15)

        plt.subplot(2, 2, 2)
        sns.scatterplot(data=sample_df, x='job', y='balance', hue=self.target, alpha=0.7)
        plt.title("Distribution of Balance by Jobs", pad=10)
        plt.xticks(rotation=90)

        plt.subplot(2, 2, 3)
        sns.boxplot(data=sample_df, x='marital', y='duration', hue=self.target)
        plt.title("Distribution of Duration by Marital Status")

        plt.subplot(2, 2, 4)
        sns.kdeplot(data=sample_df, x='day', hue=self.target, fill=True, common_norm=False)
        plt.title("Day Distribution", pad=10)

        plt.tight_layout()
        plt.show()

    def feature_importance(self):
        display(Markdown("## ğŸŒŸ Feature Importance (Random Forest)"))

        sample_data = self.data[:1000]
        for col in self.cat_cols:
            sample_data[col], _ = pd.factorize(sample_data[col])

        X = sample_data.copy()
        y = X.pop(self.target)

        X_train, X_valid, Y_train, Y_valid = train_test_split(
            X, y, test_size=self.test_size, random_state=self.seed
        )

        rf = RandomForestClassifier(n_estimators=100, random_state=self.seed)
        rf.fit(X_train, Y_train)

        feature_imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values()

        plt.figure(figsize=(10, 6))
        sns.barplot(x=feature_imp, y=feature_imp.index)
        plt.title("Feature Importance")
        plt.xlabel("Relative Importance")
        plt.show()

    def feature_importance_Regression(self):
        display(Markdown("## ğŸŒŸ Feature Importance (Random Forest)"))

        sample_data = self.data[:1000]
        for col in self.cat_cols:
            sample_data[col], _ = pd.factorize(sample_data[col])

        X = sample_data.copy()
        y = X.pop(self.target)

        X_train, X_valid, Y_train, Y_valid = train_test_split(
            X, y, test_size=self.test_size, random_state=self.seed
        )

        rf = RandomForestRegressor(n_estimators=100, random_state=self.seed)
        rf.fit(X_train, Y_train)

        feature_imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values()

        plt.figure(figsize=(10, 6))
        sns.barplot(x=feature_imp, y=feature_imp.index)
        plt.title("Feature Importance")
        plt.xlabel("Relative Importance")
        plt.show()

    def BAautomation(self):
        self.correlationPlot()
        # self.featureInteraction()  # uncomment if you want interactions
        self.feature_importance_Regression()

class DataIngestion(UnivariateAnalysis,BivariateAnalysis):
    def __init__(self, data_dir,target,seed,test_size,sample):
        self.target = target
        self.train = pd.read_csv(data_dir + '/train.csv', index_col='id')
        self.test = pd.read_csv(data_dir + '/test.csv', index_col='id')
        UnivariateAnalysis.__init__(self,self.train.dropna().head(sample),target)
        BivariateAnalysis.__init__(self,self.train.dropna().head(sample),target,test_size,seed)

    def summarize(self, df, include='all'):
        if include == 'numerical':
            summarize_df = df.describe(include=['number']).T
        elif include == 'categorical':
            summarize_df = df.describe(include=['object', 'category']).T
        else:
            summarize_df = df.describe(include='all').T

        summarize_df['dtype'] = df.dtypes
        summarize_df['missing'] = df.isnull().sum()
        summarize_df['unique'] = df.nunique()
        summarize_df['duplicates'] = df.duplicated().sum()
        summarize_df['most_frequent'] = df.select_dtypes(include=['object', 'category']).apply(
            lambda col: col.value_counts().idxmax() if col.nunique() > 0 else None
        )

        def highlight(val):
            if isinstance(val, (int, float)):
                if val > 0.7*len(df):
                    return 'background-color: red'
                elif val > 0.5*len(df):
                    return 'background-color: orange'
                elif val > 0.3*len(df):
                    return 'background-color: blue'
                elif val < 1000:
                    return 'background-color: green'
            return ''

        summarize_df.drop(columns=['25%', '50%', '75%', 'count', 'most_frequent'], inplace=True)
        styled_df = summarize_df.style.applymap(highlight, subset=['missing', 'unique'])
        return styled_df

    def fetchData(self):
        display(Markdown(f"### âœ… Shape of Train: `{self.train.shape}`"))
        display(self.train.head())

        display(Markdown(f"### âœ… Shape of Test: `{self.test.shape}`"))
        display(self.test.head())

        display(Markdown("## ğŸ“Š Missing Values â€” Train"))
        plt.figure()
        msno.matrix(self.train)
        plt.show()

        display(Markdown("## ğŸ“Š Missing Values â€” Test"))
        plt.figure()
        msno.matrix(self.test)
        plt.show()
        train_summ_df = self.summarize(df = self.train)
        display(Markdown("## ğŸ‘‰ Summary â€” Train"))
        display(train_summ_df)
        test_summ_df = self.summarize(df = self.test)
        display(Markdown("## ğŸ‘‰ Summary â€” Test"))
        display(test_summ_df)
        UnivariateAnalysis.UAautomation_regression(self,self.target)
        BivariateAnalysis.BAautomation(self)


        return self.train, self.test


class AutoEDA(DataIngestion):
    def __init__(self, data_dir, target, seed, test_size,sample=10000):
        super().__init__(data_dir,target,seed,test_size,sample)
        self.data_dir = data_dir
        self.target = target
        self.seed = seed
        self.test_size = test_size

    def AutomationPipeline(self):
        train,test = super().fetchData()
        # ingester = DataIngestion(self.data_dir)
        # train, test = ingester.fetchData()

        # UA = UnivariateAnalysis(train, self.target)
        # UA.UAautomation(self.target)

        # BA = BivariateAnalysis(train,self.target,self.test_size,self.seed)
        # BA.automation()

        return train, test


auto = AutoEDA(cfg.data_dir,cfg.target,cfg.seed,cfg.test_size)
train, test = auto.AutomationPipeline()


class Transformation:
    def __init__(self,train,test):
        self.train = train.copy()
        self.test = test.copy()
        self.cat_cols = [col for col in test.columns if test[col].dtype in ['O','bool']] + ['num_lanes','speed_limit','num_reported_accidents']
        self.bool_cols = [col for col in test.columns if test[col].dtype in ['bool']]
    def encodeCategory(self):
        for col in self.cat_cols:
            self.train[col] = self.train[col].astype('category')
            self.test[col] = self.test[col].astype('category')
    def automation(self):
        self.encodeCategory()
        return self.train,self.test


trf = Transformation(train,test)
train, test = trf.automation()


FEATURES = test.columns.to_list()
print(Fore.GREEN + f'We have {len(FEATURES)} to train !!!!')

oof_preds = np.zeros(len(train))
test_preds = np.zeros(len(test))


FOLDS = 5
SEED = 42

params = {
    "objective": "reg:squarederror",   
    "eval_metric": "rmse",             
    "learning_rate": 0.002,
    "max_depth": 4,                    
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "seed": SEED,
    "device": "cuda",
}


kf = KFold(n_splits = FOLDS, random_state = SEED, shuffle = True)
X = train.copy()
y = X.pop(cfg.target)

for fold,(train_indx,valid_indx) in enumerate(kf.split(X,y)):
    
    print("#"*25)
    print(f"### Fold {fold+1} ###")
    print("#"*25)
    
    X_train, y_train = X.iloc[train_indx][FEATURES].copy(), y.iloc[train_indx]
    X_valid, y_valid = X.iloc[valid_indx][FEATURES].copy(), y.iloc[valid_indx]
    X_test = test[FEATURES].copy()
    
    dtrain = xgb.DMatrix(X_train, label=y_train, enable_categorical=True)
    dval   = xgb.DMatrix(X_valid, label=y_valid, enable_categorical=True)
    dtest  = xgb.DMatrix(X_test, enable_categorical=True)
    
    model = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=4_000,
        evals=[(dtrain, "train"), (dval, "valid")],
        early_stopping_rounds=1000,
        verbose_eval=200,
    )
    
    oof_preds[valid_indx] = model.predict(dval, iteration_range=(0, model.best_iteration + 1))
    test_preds += model.predict(dtest, iteration_range=(0, model.best_iteration + 1)) / FOLDS


m = np.sqrt( np.mean( (oof_preds - train[cfg.target].values)**2. ) )
print(f" Overall CV RMSE = {m}")


fig, ax = plt.subplots(figsize=(5, 10))
xgb.plot_importance(model, max_num_features=100, importance_type='gain',ax=ax)
plt.title("Top 100 Feature Importances (XGBoost)")
plt.show()


sub = pd.read_csv(cfg.sub_path)
sub[cfg.target] = test_preds
sub.to_csv(f"PS5E10_{cfg.V}.csv",index=False)
sub.head()

