%pip install phik missingno optuna


# Simple import

from sklearn.datasets import make_classification # Make a pseudorandom datasets
import pandas as pd # You know what is it
import numpy as np
import seaborn as sns # Make a beautiful graphs 
import matplotlib.pyplot as plt # Make a graphs
import plotly.express as px # Make a complex graphs
from statsmodels.stats.outliers_influence import variance_inflation_factor # See https://en.wikipedia.org/wiki/Variance_inflation_factor
import phik # library for PhiK correlation
import missingno # library for displaying gaps in data
import gc # garbage collector

from sklearn.preprocessing import (
    OneHotEncoder,
    OrdinalEncoder
)

from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    accuracy_score,
)
from sklearn.model_selection import StratifiedKFold
from imblearn.under_sampling import RandomUnderSampler
from catboost import *
import optuna


import warnings
warnings.filterwarnings("ignore")


class CFG:
    TARGET = 'rainfall'
    N_FOLDS = 5
    RANDOM_STATE = 52

    TRAIN_PATH = '/kaggle/input/playground-series-s5e3/train.csv'
    TEST_PATH = '/kaggle/input/playground-series-s5e3/test.csv'
    SUBMIT_PATH = '/kaggle/input/playground-series-s5e3/sample_submission.csv'
    ORIGINAL_PATH = '/kaggle/input/rainfall/Rainfall.csv' # Change for your path


class DataAnalysis:
    
    @staticmethod
    def info_df(df: pd.DataFrame) -> None:
        print('------------------------------')
        print('| Dataset information |')
        print('------------------------------')
        df.info()
        print('-----------------------------------------')
        print('| First 5 rows |')
        print('-----------------------------------------')
        display(df.head())
        print('--------------------')
        print('| Sum of duplicates |')
        print('--------------------')
        print(df.duplicated().sum())


    @staticmethod
    def view_distribution(data: pd.DataFrame, object_col = False, numeric_col = False) -> None:
        numeric_cols = data.select_dtypes(exclude=['object', 'datetime']).columns.to_list()
        object_cols = data.select_dtypes(include=['object']).columns.to_list()
        
        if numeric_col:
            _, axes = plt.subplots(nrows=len(data[numeric_cols].columns), ncols=2, figsize=(len(numeric_cols)+15,len(numeric_cols)+7))
            j = 0
            for i in data[numeric_cols].columns:
                sns.histplot(data[numeric_cols][i], ax=axes[j, 0], kde=True, bins=40, edgecolor='black')
                axes[j, 0].set_title(i, fontsize=14)
                axes[j, 0].set_xlabel('')

                sns.boxplot(x=data[numeric_cols][i], ax=axes[j, 1], orient='h', palette='pink')
                axes[j, 1].set_title(i, fontsize=14)
                axes[j, 1].set_xlabel('')
                j += 1
            plt.suptitle(f'Num features\n\n', ha='center', fontweight='bold', fontsize=20);
            plt.tight_layout();
            plt.show();

        if object_col:
            _,ax = plt.subplots(len(object_cols),1, figsize=(len(object_cols)+7,len(object_cols)+20));
            ax =ax.flatten();
            g = 0
            for k in data[object_cols].columns:
                sns.countplot(data=data, x=k,ax=ax[g]);
                ax[g].set_xticklabels(labels=ax[g].get_xticklabels());
                ax[g].set_title(k);
                ax[g].set_xlabel('');
                g += 1
            plt.suptitle(f'Categorical\n\n', ha='center', fontweight='bold', fontsize=20);
            plt.show();


    @staticmethod
    def bloating_of_variance(data: pd.DataFrame) -> None:
        num = data.select_dtypes(exclude=['object', 'datetime']).columns.to_list()
        vif_data = pd.DataFrame()
        vif_data['feature'] = data.select_dtypes(exclude=['object', 'datetime']).columns.to_list()

        vif_data['VIF'] = [variance_inflation_factor(data[num].values, i) \
                                for i in range(len(data[num].columns))]
        print(vif_data)
    
    @staticmethod
    def balance_of_target(data: pd.DataFrame, target: str) -> None:
        sns.countplot(y=target, data=data, color='green', width=0.6);

    @staticmethod
    def plot_count(df: pd.core.frame.DataFrame, col: str, title_name: str='Train') -> None:
        # Set background color
        f, ax = plt.subplots(1, 2, figsize=(16, 7))
        plt.subplots_adjust(wspace=0.2)

        s1 = df[col].value_counts()
        N = len(s1)

        outer_sizes = s1
        inner_sizes = s1/N

        colors = sns.color_palette("mako")
        
        outer_colors = ['#2e1e3b', '#413d7b', '#37659e', '#348fa7', '#40b7ad', '#8bdab2']
        inner_colors = ['#2e1e3b', '#413d7b', '#37659e', '#348fa7', '#40b7ad', '#8bdab2']

        ax[0].pie(
            outer_sizes,colors=outer_colors, 
            labels=s1.index.tolist(), 
            startangle=90, frame=True, radius=1.3, 
            explode=([0.05]*(N-1) + [.3]),
            wedgeprops={'linewidth' : 1, 'edgecolor' : 'black'}, 
            textprops={'fontsize': 12, 'weight': 'bold', 'color': 'white'}
        )

        textprops = {
            'size': 13, 
            'weight': 'bold', 
            'color': 'white'
        }

        ax[0].pie(
            inner_sizes, colors=inner_colors,
            radius=1, startangle=90,
            autopct='%1.f%%', explode=([.1]*(N-1) + [.3]),
            pctdistance=0.8, textprops=textprops
        )

        center_circle = plt.Circle((0,0), .68, color='black', fc='#243139', linewidth=0)
        ax[0].add_artist(center_circle)

        x = s1
        y = s1.index.tolist()
        sns.barplot(
            x=x, y=y, ax=ax[1],
            palette=colors, orient='horizontal'
        )

        ax[1].spines['top'].set_visible(False)
        ax[1].spines['right'].set_visible(False)
        ax[1].tick_params(
            axis='x',         
            which='both',      
            bottom=False,       
            labelbottom=False
        )

        for i, v in enumerate(s1):
            ax[1].text(v, i+0.1, str(v), color='white', fontweight='bold', fontsize=12)

        plt.setp(ax[1].get_yticklabels(), fontweight="bold")
        plt.setp(ax[1].get_xticklabels(), fontweight="bold")
        ax[1].set_xlabel(col, fontweight="bold", color='white')
        ax[1].set_ylabel('count', fontweight="bold", color='white')

        f.suptitle(f'{title_name}', fontsize=14, fontweight='bold', color='white')
        plt.tight_layout() 
        plt.show()
    
    @staticmethod
    def summary(data: pd.DataFrame) -> None:
        data = data.select_dtypes(exclude=['object', 'datetime'])
        sum = pd.DataFrame(data.dtypes, columns=['dtypes'])
        sum['missing#'] = data.isna().sum()
        sum['missing%'] = (data.isna().sum())/len(data)
        sum['uniques'] = data.nunique().values
        sum['count'] = data.count().values
        sum['skew'] = data.skew().values
        return sum
    
    @staticmethod
    def correlations(data: pd.DataFrame) -> None:
        data = data.drop(columns=CFG.TARGET)
        plt.figure(figsize=(15, 13));
        # Generate a mask for the upper triangle
        mask_pir = np.triu(np.ones_like(data.corr(method='pearson'), dtype=bool));
        mask_spi = np.triu(np.ones_like(data.corr(method='spearman'), dtype=bool));
       
        # Set up the matplotlib figure
        f, ax = plt.subplots(figsize=(11, 9));

        # Generate a custom diverging colormap
        cmap = sns.diverging_palette(230, 20, as_cmap=True);
        plt.title('PIRSON')
        sns.heatmap(data.corr(method='pearson'), annot=True, mask=mask_pir, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5}, robust=True);
        plt.show();
        
        plt.figure(figsize=(15, 13));
        f, ax = plt.subplots(figsize=(11, 9));
        plt.title('SPEARMAN')
        sns.heatmap(data.corr(method='spearman'), annot=True, mask=mask_spi, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5}, robust=True)
        plt.show();

        plt.figure(figsize=(15, 13));
        f, ax = plt.subplots(figsize=(11, 9));
        
        interval_cols = data.select_dtypes(exclude='object').columns.to_list()
        phik_overview = data.phik_matrix(interval_cols=interval_cols)
        plt.title(r'$\phi_K$')
        corr = phik_overview.round(2)
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, cmap='pink_r', vmax=.3, center=0,
                annot=True, fmt='.2f', square=True, linewidths=.5, cbar_kws={"shrink": .5})

        significance_overview  = data.significance_matrix(interval_cols=interval_cols)

        plt.figure(figsize=(15, 13));
        plt.title('Statistical significance')
        corr = significance_overview.round(2)
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, cmap='pink_r', vmax=5, vmin=-5, center=0,
                annot=True, fmt='.2f', square=True, linewidths=.5, cbar_kws={"shrink": .5})

        plt.show()
    
    @staticmethod
    def blinks(data: pd.DataFrame) -> None:
        print('Data gaps')
        missingno.matrix(data)


class DataLoader:
    def __init__(self, train: pd.DataFrame, original: pd.DataFrame, test: pd.DataFrame):
        self.train = train
        self.original= original
        self.test = test

    @staticmethod
    def ohe_encode_categorical_features(features: pd.DataFrame) -> pd.DataFrame:  
        print('--- Encoding categorical features')
        cat_features = features.select_dtypes(include=['object','category']).columns.to_list()
        
        encoder_ohe = OneHotEncoder(drop='first', handle_unknown='ignore', sparse=False)

        encoder_ohe.fit(features[cat_features])

        features[
            encoder_ohe.get_feature_names_out()
        ] = encoder_ohe.transform(features[cat_features])
        
        return features.drop(cat_features, axis=1)
    
    
    @staticmethod
    def encode_categorical_features(dataframe):  # We are known that original dataframe have a another interpretation  of target feature (yes/no), change it
        print('--- Encoding categorical features')
        
        target = {'yes': 1, 'no': 0}
        
        dataframe[CFG.TARGET] = dataframe[CFG.TARGET].map(target)
        
        return dataframe

    @staticmethod
    def reduce_mem_usage(dataframe):
        
        print('--- Reducing memory usage')
        initial_mem_usage = dataframe.memory_usage().sum() / 1024**2
        
        for col in dataframe.columns:
            col_type = dataframe[col].dtype

            if col_type.name in ['category', 'object']:
                raise ValueError(f"Column '{col}' is of type '{col_type.name}'")

            c_min = dataframe[col].min()
            c_max = dataframe[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    dataframe[col] = dataframe[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    dataframe[col] = dataframe[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    dataframe[col] = dataframe[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    dataframe[col] = dataframe[col].astype(np.int64)

        # NOT WORKING WITH NEW VERSION OF PANDAS (NotImplementedError: float16 indexes are not supported)
            # else:
            #     if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
            #         dataframe[col] = dataframe[col].astype(np.float16)
            #     elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
            #         dataframe[col] = dataframe[col].astype(np.float32)
            #     else:
            #         dataframe[col] = dataframe[col].astype(np.float64)

        final_mem_usage = dataframe.memory_usage().sum() / 1024**2
        print('------ Memory usage before: {:.2f} MB'.format(initial_mem_usage))
        print('------ Memory usage after: {:.2f} MB'.format(final_mem_usage))
        print('------ Decreased memory usage by {:.1f}%'.format(100 * (initial_mem_usage - final_mem_usage) / initial_mem_usage))

        return dataframe

    def load(self):
        print(f'Loading data')
        
        train = self.train
        origianl = self.original
        test = self.test
        
        origianl = self.encode_categorical_features(origianl)
        
        origianl = origianl.rename(columns={'pressure ': 'pressure', 
                                 'humidity ': 'humidity', 
                                 'cloud ': 'cloud'}) # Additionally some features names have a gaps
        
        train = pd.concat([train, origianl]).reset_index(drop=True)

        train['is_train'] = 1
        test['is_train'] = 0
        dataframe = pd.concat([train, test])
        del train, test
        gc.collect()
        dataframe = self.ohe_encode_categorical_features(dataframe)
        dataframe = self.reduce_mem_usage(dataframe)
        
        train = dataframe[dataframe['is_train'] == 1].drop(columns=['is_train'])
        test = dataframe[dataframe['is_train'] == 0].drop(columns=['is_train', CFG.TARGET])
        
        del dataframe
        gc.collect()
        
        train[CFG.TARGET] = train[CFG.TARGET].astype(np.int8)
        
        return train, test


train, test = DataLoader(pd.read_csv(CFG.TRAIN_PATH, index_col=['id']),
                         pd.read_csv(CFG.ORIGINAL_PATH, skipinitialspace=True),
                         pd.read_csv(CFG.TEST_PATH, index_col=['id'])).load()

spl_sub = pd.read_csv(CFG.SUBMIT_PATH)


for data in [train]: # You should add dataframes
    DataAnalysis.info_df(data)
    DataAnalysis.blinks(data)
    DataAnalysis.view_distribution(data, numeric_col=True)
    DataAnalysis.correlations(data)


for i in train.columns:
    print(i, train[i].unique())


train = train.dropna().reset_index(drop=True)


DataAnalysis.bloating_of_variance(train)


DataAnalysis.plot_count(train, col='rainfall', title_name='Rainfall Distribution of Train Data')


X = train.drop(columns=CFG.TARGET)
y = train[CFG.TARGET]
cat_features = X.select_dtypes(include=['category','object']).columns.to_list()


def build_catboost(trial):
    params = {
        'iterations': 100,
        'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-1, log=True),
        'depth': trial.suggest_int('depth', 1, 12),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', .1, 1., log=True),
        'random_strength': trial.suggest_float('random_strength', .1, 1., log=True),
        'bagging_temperature': trial.suggest_float('bagging_temperature', .1, 1., log=True),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 1, 100),
        'bootstrap_type':'Bayesian',
        'loss_function': 'Logloss',
        'eval_metric': 'AUC',
        'task_type': 'CPU', # On your own PC do in GPU
    }


    model = CatBoostClassifier(**params, silent=True, random_state=CFG.RANDOM_STATE)
    cv_data = cv(
        Pool(X, y, cat_features=cat_features),
        model.get_params(),
        verbose=False
    )
    return np.mean(cv_data['test-AUC-mean'])


study = optuna.create_study(direction="maximize")
study.optimize(build_catboost, n_trials=CFG.N_FOLDS)


print('Best hyperparameters:', study.best_params)
print('Best AUC-ROC:', study.best_value)


aucs = []
preds = []

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=CFG.RANDOM_STATE)

for CFG.N_FOLDS, (train_idx, valid_idx) in enumerate(skf.split(X, y)):
    print(f'### Fold {CFG.N_FOLDS+1} Training ###')
    X_train = X.iloc[train_idx]
    y_train = y.iloc[train_idx]
    X_valid = X.iloc[valid_idx]
    y_valid = y.iloc[valid_idx]
    X_test = test[X.columns]

    X_train_pool = Pool(X_train, y_train, cat_features=cat_features)
    X_valid_pool = Pool(X_valid, y_valid, cat_features=cat_features)
    X_test_pool = Pool(X_test, cat_features=cat_features)
    
    model = CatBoostClassifier(
        loss_function='Logloss',
        eval_metric='AUC',
        learning_rate=0.018574256376068948,
        iterations=10000,
        depth=8,
        l2_leaf_reg=0.3648329375168125,
        random_strength=0.34210492920316143,
        bagging_temperature=0.19998783773475814,
        min_data_in_leaf=97,
        bootstrap_type='Bayesian',
        task_type='CPU',
        random_seed=CFG.RANDOM_STATE,
        verbose=False
    )

    model.fit(X=X_train_pool, eval_set=X_valid_pool, verbose=500, early_stopping_rounds=500)

    pred_valid = model.predict_proba(X_valid_pool)[:, 1]
    preds.append(model.predict_proba(X_test_pool)[:, 1])

    auc = roc_auc_score(y_valid, pred_valid)
    aucs.append(auc)

    print(f'Fold {CFG.N_FOLDS+1} AUC: {auc:.5f}\n')

print(f'\nOverall AUC: {np.mean(aucs):.5f} +/- {np.std(aucs):.5f}')


pd.DataFrame(np.array(preds).T).hist(bins=100, figsize=(10,10));


submission = spl_sub[['id']]
submission[CFG.TARGET] = np.mean(preds, axis=0)

submission.to_csv('submission.csv', index=False)
submission

