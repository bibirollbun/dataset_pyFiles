import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import missingno as msno
import optuna

from datetime import datetime
from lightgbm import LGBMRegressor
from sklearn.model_selection import KFold, GroupKFold
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.preprocessing import FunctionTransformer
from numpy.polynomial import Polynomial
from scipy.optimize import minimize
import warnings


warnings.filterwarnings("ignore")


!pip install wbgapi

import wbgapi as wb


df_train = pd.read_csv('/kaggle/input/playground-series-s5e1/train.csv')

df_test = pd.read_csv('/kaggle/input/playground-series-s5e1/test.csv')


df_train.head()


# Percentages of missing rows per column
df_train.isna().mean().mul(100)


df_train[df_train['num_sold'].isna()].head()


#Checking what combinations are related to missing num_sold
df_miss = df_train.drop(['id', 'date'], axis = 1)

unique_col_values = df_miss.loc[df_miss.iloc[:, -1].isna(), df_miss.columns[:-1]].drop_duplicates()

print(unique_col_values)


gp = df_miss[df_miss.iloc[:, -1].isna()].groupby(['country', 'store', 'product'])
gp.count().rsub(gp.size(), axis=0)


for i in range(3):
    print(df_train.iloc[:, i+2].unique())


def date_extractor(df):
    df['date'] = pd.to_datetime(df['date'])
    df['Year'] = df['date'].dt.year
    df['Day'] = df['date'].dt.day
    df['Month'] = df['date'].dt.month
    df['Weekday'] = df['date'].dt.weekday+1
    df['Day Of Year'] = df['date'].dt.dayofyear
    df['Group'] = (df['Year'] - 2010) * 48 + df['Month'] * 4 + df['Day'] // 7
    
    return df


df_train = date_extractor(df_train)


def grouping_plotter(df, group_list, groupby_time, groupdistinct, makegrid=None, scale=False, title = None):
    
    df = df.dropna(axis = 0)
    
    # Filtering chosen variables
    for i in range(3):
        df = df[df.iloc[:, i].isin(group_list[i])]
    
    # Changing character variables into categorical
    df.iloc[:, :3] = df.iloc[:, :3].astype('category')


    # Grouping variables and calculating mean of num_sold    
    df['num_sold_mean'] = df.groupby([groupby_time] + groupdistinct)['num_sold'].transform('mean')
    
    # Scaling num_sold_mean based on first element of groupdistinct by its max value (0-1 scale)
    if(scale == True):
        df['num_sold_mean'] = df.groupby(groupdistinct[0])['num_sold_mean'].transform(lambda x: x / x.max())
    

    
    sns.set(font_scale=2.5)  
    sns.set_theme(style="whitegrid")
    if makegrid is None or not makegrid:
        # Creating a joint grouping variable
        df['grouping'] = df[groupdistinct].astype(str).agg('_'.join, axis=1)
        df = df.sort_values('grouping').drop_duplicates(subset='num_sold_mean')
        

        plt.figure(figsize=(14, 8))
        sns.lineplot(
            data=df, 
            x=groupby_time, 
            y='num_sold_mean', 
            hue='grouping',
            linewidth=1.75
        )
        plt.grid(color='gray', alpha=0.75)
        plt.xticks(fontsize=12, rotation=45)
        plt.yticks(fontsize=12)
        plt.xlabel(groupby_time.replace('_', ' ').title(), fontsize=14)
        plt.ylabel('Mean Num Sold', fontsize=14)
        plt.legend(
            title='Grouping', 
            fontsize=12, 
            title_fontsize=14, 
            loc='upper right',  
            bbox_to_anchor=(1.20, 1)  
        )
        if title:
              plt.title(title, fontsize=16, y=1.05)
        plt.show()
    
    elif makegrid:
        # Removing unnecessary data
        df = df[[groupby_time] + groupdistinct + ['num_sold_mean']].drop_duplicates()
        
        
        g = sns.FacetGrid(
            df, 
            col=groupdistinct[1], 
            row=groupdistinct[0], 
            margin_titles=True,
            hue = groupdistinct[1],
            height=4.5, 
            aspect=1.5
        )
        g.map_dataframe(
            sns.lineplot, 
            x=groupby_time, 
            y='num_sold_mean', 
            linewidth=3.4
        )
        
        g.set_axis_labels(groupby_time.replace('_', ' ').title(), 'Mean Num Sold', fontsize = 14)
        g.set_titles(col_template='{col_name}', row_template='{row_name}')
        g.tick_params(axis='x', which='both', rotation=45)
        g.add_legend(title=groupdistinct[1], fontsize=16, title_fontsize=18)
        
        if title:
            g.fig.suptitle(title, fontsize=22, y = 1.025)
        
        plt.show()


warnings.filterwarnings("ignore", "is_categorical_dtype")
warnings.filterwarnings("ignore", "use_inf_as_na")

grouping_plotter(df_train.iloc[:, 2:11],
                group_list = [['Canada', 'Finland', 'Italy', 'Kenya', 'Norway', 'Singapore'],
                             ['Discount Stickers', 'Stickers for Less', 'Premium Sticker Mart'],
                             ['Holographic Goose', 'Kaggle', 'Kaggle Tiers', 'Kerneler', 'Kerneler Dark Mode']],
                groupby_time = 'Month',
                groupdistinct = ['country', 'product'],
                makegrid = True,
                scale = True,
                title = "Monthly mean of num_sold, grouped and scaled by country")


grouping_plotter(df_train.iloc[:, 2:11],
                group_list = [['Canada', 'Finland', 'Italy', 'Kenya', 'Norway', 'Singapore'],
                             ['Discount Stickers', 'Stickers for Less', 'Premium Sticker Mart'],
                             ['Holographic Goose', 'Kaggle', 'Kaggle Tiers', 'Kerneler', 'Kerneler Dark Mode']],
                groupby_time = 'Day',
                groupdistinct = ['product','country'],
                makegrid = True,
                scale = False,
                title = "Daily mean of num_sold, grouped by product")


grouping_plotter(df_train.iloc[:, 2:11],
                group_list = [['Kenya'],
                             ['Discount Stickers', 'Stickers for Less', 'Premium Sticker Mart'],
                             ['Holographic Goose', 'Kaggle', 'Kaggle Tiers', 'Kerneler', 'Kerneler Dark Mode']],
                groupby_time = 'Month',
                groupdistinct = ['product'],
                makegrid = False,
                title = 'Monthly mean of num_sold, grouped by product in Kenya')


grouping_plotter(df_train.iloc[:, 2:11],
                group_list = [['Finland', 'Canada'],
                             ['Discount Stickers', 'Stickers for Less', 'Premium Sticker Mart'],
                             ['Holographic Goose']],
                groupby_time = 'Weekday',
                groupdistinct = ['country'],
                makegrid = False,
                title = 'Mean of num_sold throghout the week, grouped by country')


grouping_plotter(df_train.iloc[:, 2:11],
                group_list = [['Finland', 'Italy', 'Norway', 'Singapore'],
                             ['Discount Stickers', 'Stickers for Less', 'Premium Sticker Mart'],
                             ['Holographic Goose']],
                groupby_time = 'Month',
                groupdistinct = ['country', 'store'],
                makegrid = True,
                scale = True,
                title = 'Monthly mean of num_sold of Holographic Goose in stores, grouped by country')


grouping_plotter(df_train.iloc[:, 2:11],
                group_list = [['Finland', 'Italy', 'Norway', 'Singapore'],
                             ['Discount Stickers', 'Stickers for Less', 'Premium Sticker Mart'],
                             ['Holographic Goose']],
                groupby_time = 'Month',
                groupdistinct = ['country'],
                makegrid = None,
                scale = False,
                title = 'Monthly mean of num_sold of Holographic Goose, grouped by country')


grouping_plotter(df_train.iloc[:, 2:11],
                group_list = [['Kenya'],
                             ['Discount Stickers', 'Stickers for Less', 'Premium Sticker Mart'],
                             ['Holographic Goose', 'Kaggle', 'Kaggle Tiers', 'Kerneler', 'Kerneler Dark Mode']],
                groupby_time = 'Day Of Year',
                groupdistinct = ['product'],
                makegrid = False,
                title = 'Daily mean of num_sold, grouped by product in Kenya')


df_gdp = wb.data.DataFrame('NY.GDP.PCAP.CD', 
                       ['CAN', 'FIN', 'ITA', 'KEN', 'NOR', 'SGP'],
                       time=range(2010, 2020, 1), labels=True)
df_gdp


df_gdp_long = df_gdp.reset_index().melt(
    id_vars=['Country'],  
    var_name='Year',      
    value_name='GDP'      
)

df_gdp_long = df_gdp_long.rename(columns = {"Country":"country"})

df_gdp_long = df_gdp_long.iloc[6:]

# Removing YR prefix from Year column values
df_gdp_long['Year'] = [str(year)[2:] for year in df_gdp_long['Year']]

df_gdp_long['Year'] = df_gdp_long['Year'].astype(int)



df_train = df_train.merge(
    df_gdp_long,
    left_on = ['country', 'Year'],
    right_on=['country', 'Year'],   
    how='left'                    
)


df_train['GDP'] = df_train['GDP'].astype('float')

df_train


def feature_encoder(df):
    def sin_transformer(period):
    	return FunctionTransformer(lambda x: np.sin(x* 2 * np.pi / period))

    def cos_transformer(period):
    	return FunctionTransformer(lambda x: np.cos(x * 2 * np.pi / period))


    df['Month Sin'] = sin_transformer(12).fit_transform(df["Month"])
    df['Day Sin'] = sin_transformer(365).fit_transform(df["Day Of Year"])
    df['Month Cos'] = cos_transformer(12).fit_transform(df["Month"])
    df['Day Cos'] = cos_transformer(365).fit_transform(df["Day Of Year"])
    df['Year Sin'] = sin_transformer(1).fit_transform(df["Year"])

    return df


df_train = feature_encoder(df_train)

df_train


# Separating 2016 as the test set
X_train = df_train.query('Year < 2016')
X_test = df_train.query('Year == 2016')

X_test = X_test.dropna()
X_test, y_test = X_test.drop('num_sold', axis = 1), X_test['num_sold']


def feature_normalizer(df):
    origin_df = df.copy()
    
    def poly_fit(x, y, a=1, b=1, deg=1):
        c0 = Polynomial.fit(x, y, deg).convert().coef
        res = minimize(
            lambda c: np.mean(np.abs(Polynomial(c)(x) - y) ** a / y ** b),
            c0, method='Nelder-Mead',
            options={'maxiter': 1000}
        )
        assert res.success
        return Polynomial(res.x)
    
    X = df.groupby(['Year', 'country'])[['num_sold']].sum().reset_index()

    # Merging data with GDP df
    X = X.merge(
        df_gdp_long, 
        on=['country', 'Year'],
        how='left'           
    )
    
    X['GDP'] = X['GDP'].astype('float')
    L = poly_fit(X['GDP'], X['num_sold'])
    X['L(GDP)'] = L(X['GDP'])
    
    
    X = X.join(X.groupby('Year')[['GDP', 'L(GDP)']].sum(), on='Year', rsuffix='_annual')
    for c in ['GDP', 'L(GDP)']:
        X[c] /= X[c+'_annual']
        X.pop(c+'_annual')
    X.pop('num_sold')
    
    df = df.groupby(['date', 'Year', 'country'])[['num_sold']].sum().reset_index().join(
        df.groupby(['date', 'Year'])[['num_sold']].sum(), on=['date', 'Year'],rsuffix='_global')
    
    df['num_sold'] /= df['num_sold_global']
    X = df.reset_index().groupby(['date', 'Year', 'country'])[['num_sold']].sum().reset_index().merge(X, on=['Year', 'country'])
    
    X = X.rename(columns = {"num_sold":"norm_num_sold", "GDP":"norm_GDP", "L(GDP)":"norm_L(GDP)"})

    origin_df = origin_df.merge(
        X,
        left_on = ['country', 'Year', 'date'],
        right_on=['country', 'Year', 'date'],   
        how='left'                    
    )
    
    return(origin_df)


def values_imputer(df, country_ref):
    df_country_ref = df[df.country == country_ref]
    df_country_ref = df_country_ref.rename(columns = {'num_sold':"ref_num_sold",
                                                     "norm_num_sold":"ref_norm_num_sold",
                                                     "norm_GDP":"ref_norm_GDP",
                                                     "norm_L(GDP)":"ref_norm_L(GDP)"})
    df_country_ref = df_country_ref.loc[:, ['date','store', 'product', 'ref_num_sold','ref_norm_num_sold', 'ref_norm_GDP','ref_norm_L(GDP)']]
    
    df = df.merge(
        df_country_ref,
        left_on = ['date', 'store', 'product'],
        right_on = ['date', 'store', 'product']
    )

    df['norm_num_sold_ratios'] = df['norm_num_sold'] / df['ref_norm_num_sold']
    df['norm_GDP_ratios'] = df['norm_GDP'] / df['ref_norm_GDP']
    df['norm_L(GDP)_ratios'] = df['norm_L(GDP)'] / df['ref_norm_L(GDP)']

    miss_mask = df['num_sold'].isna()

    df.loc[miss_mask, 'num_sold'] = (
        df.loc[miss_mask, 'ref_num_sold'] * df.loc[miss_mask, 'norm_num_sold_ratios']
    )

    #return(df.loc[miss_mask, 'num_sold'])
    return(df)


X_italy = X_train.query('country == "Italy" and product == "Holographic Goose" and store == "Discount Stickers"')

X_italy_missing = X_train.copy()

# Filling the data with NaNs
X_italy_missing.loc[X_italy.index,'num_sold'] = float('nan')

# Normalizing data
X_italy_missing = feature_normalizer(X_italy_missing)

# Checking how different countries to impute from affect MAPE
countries_ref = ['Finland', 'Singapore', 'Norway'] # Choosing countries without missing values
for i in countries_ref:

    italy_imputations = values_imputer(X_italy_missing, country_ref = i)

    little_italy = italy_imputations.query('country == "Italy" and product == "Holographic Goose" and store == "Discount Stickers"')
    X_little_italy = X_italy.query('product == "Holographic Goose" and store == "Discount Stickers"')

    print(f"MAPE of imputation using country: {i} as a reference: {mean_absolute_percentage_error(X_little_italy['num_sold'], little_italy['num_sold'])}")


class LightGBMAndOptunaParams:
    def __init__(self,
                 X,
                 y,
                 groups,
                 n_folds=5,
                 hyperparam_bounds=None,
                 random_state=42,
                 verbose=0):
        self.X = X
        self.y = y
        self.groups = groups
        self.n_folds = n_folds
        self.random_state = random_state
        self.verbose = verbose
        self.hyperparam_bounds = hyperparam_bounds or {
            'max_depth': (4, 24),
            'num_leaves': (5, 450),
            'learning_rate': (0.01, 0.1),
            'n_estimators': (700, 1600),
            'lambda_l1': (0.02, 0.1),
            'lambda_l2': (0.02, 0.1),
            'colsample_bytree': (0.1, 1.0),
            'subsample': (0.1, 1.0),
            'min_child_samples': (15, 220)
        }
        
        # Placeholders
        self._oof_predictions = []
        self.hyperparams_history = []
        self.best_score = float('inf')

       
    def _suggest_hyperparams(self, trial):
        return {
            'max_depth': trial.suggest_int('max_depth', *self.hyperparam_bounds['max_depth']),
            'num_leaves': trial.suggest_int('num_leaves', *self.hyperparam_bounds['num_leaves']),
            'learning_rate': trial.suggest_float('learning_rate', *self.hyperparam_bounds['learning_rate'], log=True),
            'n_estimators': trial.suggest_int('n_estimators', *self.hyperparam_bounds['n_estimators']),
            'lambda_l1': trial.suggest_float('lambda_l1', *self.hyperparam_bounds['lambda_l1']),
            'lambda_l2': trial.suggest_float('lambda_l2', *self.hyperparam_bounds['lambda_l2']),
            'colsample_bytree': trial.suggest_float('colsample_bytree', *self.hyperparam_bounds['colsample_bytree']),
            'subsample': trial.suggest_float('subsample', *self.hyperparam_bounds['subsample']),
            'min_child_samples': trial.suggest_int('min_child_samples', *self.hyperparam_bounds['min_child_samples'])
        }

    def objective(self, trial):
        # Assigning variables to cat_features for fitting
        cat_features = [col for col in self.X.select_dtypes(include=['object', 'string', 'category']).columns]

        params = self._suggest_hyperparams(trial)


        folds = GroupKFold(n_splits=self.n_folds)
        fold_mape = []
        oof_preds = np.zeros(len(self.y))

        for train_idx, val_idx in folds.split(self.X, self.y, groups=self.groups):
            X_train, y_train = self.X.iloc[train_idx], self.y.iloc[train_idx]
            X_val, y_val = self.X.iloc[val_idx], self.y.iloc[val_idx]


            model = LGBMRegressor(
                linear_trees=True,
                metric='mae',
                verbosity=-1,
                random_state=self.random_state,
                **params
            )
            model.fit(X_train, y_train, categorical_feature=cat_features)

            # Making predictions and appending MALE
            val_pred = model.predict(X_val)
            val_mape = mean_absolute_percentage_error(y_val, val_pred)
            fold_mape.append(val_mape)
            oof_preds[val_idx] = val_pred

         # Saving oof predictions
        self._oof_predictions.append(oof_preds.copy())

        mean_mape = np.mean(fold_mape)
        self.hyperparams_history.append({**params, 'MALE': mean_mape})
        return mean_mape

    def optimize(self, n_trials=5, direction="minimize"):
        study = optuna.create_study(direction=direction)
        study.optimize(self.objective, n_trials=n_trials)
        self.best_params = study.best_params
        self.best_value = study.best_value
        return study

    @property
    def history(self):
        if not self.hyperparams_history:
            raise ValueError("No hyperparameter history available")
        self.hyperparams_history = sorted(self.hyperparams_history, key=lambda x: x['MALE'])
        return self.hyperparams_history

    @property
    def oof(self):
        if not self._oof_predictions:
            raise ValueError("No OOF predictions available")
        best_trial_index = np.argmin([history['MALE'] for history in self.hyperparams_history])
        chosen_oof = self._oof_predictions[best_trial_index]
        return chosen_oof


italy_imputations = values_imputer(X_italy_missing, country_ref = 'Finland').iloc[:, :18].drop(['id', 'date'], axis = 1)
X_it_imp, y_it_imp = italy_imputations.drop('num_sold', axis = 1), np.log1p(italy_imputations['num_sold'])


X_italy_missing = X_italy_missing.iloc[:, :18].drop(['id', 'date'], axis = 1).dropna()
X_it_miss, y_it_miss = X_italy_missing.drop('num_sold', axis = 1), np.log1p(X_italy_missing['num_sold'])


X_test = X_test.drop(['id', 'date'], axis = 1)


cat_features = ['country', 'store', 'product', 'Day', 'Weekday',
                'Month', 'Year', 'Day Of Year', 'Group',
                'Month Sin', 'Month Cos', 'Year Sin']

for i in cat_features:
    X_it_imp[i] = X_it_imp[i].astype('category')
    X_it_miss[i] = X_it_miss[i].astype('category')
    X_test[i] = X_test[i].astype('category')


#groups = X_it_imp['Year']  
#lgbm_optimizer = LightGBMAndOptunaParams(X_it_imp, y_it_imp, groups=groups, n_folds = 6)

#study = lgbm_optimizer.optimize(n_trials=40)

# Results
#print("Best Parameters:", study.best_params)
#print("Best MALE Score:", study.best_value)


#oof_preds = lgbm_optimizer.oof
#print(f"OOF MAPE SCORE: {mean_absolute_percentage_error(np.expm1(y_it_imp), np.round(np.expm1(oof_preds).astype(float)).astype(int))}")


params = {'max_depth': 11,
          'num_leaves': 19,
          'learning_rate': 0.03501519992214664,
          'n_estimators': 1133,
          'lambda_l1': 0.05232565368692772,
          'lambda_l2': 0.09718396839395343,
          'colsample_bytree': 0.8126741295636939,
          'subsample': 0.8042105378729808,
          'min_child_samples': 180}


# Train LightGBM model
model = LGBMRegressor(
         linear_trees=True,
         metric='mae',
         verbosity=-1,
         **params)
         #**study.best_params)

model.fit(X_it_imp, y_it_imp, categorical_feature = cat_features)

predictions = model.predict(X_test)

print(f"Mape for Year 2016 predictions with imputed training data {mean_absolute_percentage_error(y_test, np.round(np.expm1(predictions).astype(float)).astype(int))}")

italian_mask = X_test.query('country == "Italy" and product == "Holographic Goose" and store == "Discount Stickers"').index
positions = X_test.index.get_indexer(italian_mask)

print(f"Mape for 2016 Italian Holographic Goose from Discount Stickers with imputed training data: {mean_absolute_percentage_error(y_test[italian_mask], np.round(np.exp(predictions[positions]).astype(float)).astype(int))}")
print('~'*111)
model.fit(X_it_miss, y_it_miss, categorical_feature = cat_features)

miss_predictions = model.predict(X_test)

print(f"Mape for Year 2016 predictions with missing training data {mean_absolute_percentage_error(y_test, np.round(np.expm1(miss_predictions).astype(float)).astype(int))}")
print(f"Mape for 2016 Italian Holographic Goose from Discount Stickers with missing training data: {mean_absolute_percentage_error(y_test[italian_mask], np.round(np.exp(miss_predictions[positions]).astype(float)).astype(int))}")


def evaluate_imputation_models(X_train, X_test, y_test, countries_ref, n_splits=6):

    # Model params
    params = {'max_depth': 11,
          'num_leaves': 19,
          'learning_rate': 0.03501519992214664,
          'n_estimators': 1133,
          'lambda_l1': 0.05232565368692772,
          'lambda_l2': 0.09718396839395343,
          'colsample_bytree': 0.8126741295636939,
          'subsample': 0.8042105378729808,
          'min_child_samples': 180}

    
    cat_features = ['country', 'store', 'product', 'Day', 'Weekday',
                    'Month', 'Year', 'Day Of Year', 'Group',
                    'Month Sin', 'Month Cos', 'Year Sin']

    # Converting categorical features
    for col in cat_features:
        if col in X_train.columns:
            X_train[col] = X_train[col].astype('category')
        if col in X_test.columns:
            X_test[col] = X_test[col].astype('category')

    # Placeholders
    oof_preds = []
    test_preds = []

    country_fold = GroupKFold(n_splits=n_splits)

    for ref_country in countries_ref:
        imps = values_imputer(X_train, ref_country)

        # Imputing data
        imps = imps.iloc[:, :18]
        imps = imps.drop(['id', 'date'], axis=1)

        X_imp = imps.drop('num_sold', axis=1)
        y_imp = np.log1p(imps['num_sold'])

        # Converting categorical features
        for col in cat_features:
            if col in X_imp.columns:
                X_imp[col] = X_imp[col].astype('category')

        groups = X_imp['Year']

        oof_predictions = np.zeros(len(X_imp))

        for train_idx, valid_idx in country_fold.split(X_imp, y_imp, groups=groups):
            
            X_train_data, X_valid_data = X_imp.iloc[train_idx], X_imp.iloc[valid_idx]
            y_train_data, y_valid_data = y_imp.iloc[train_idx], y_imp.iloc[valid_idx]

            # Training the LGBM model
            model = LGBMRegressor(linear_trees=True, metric='mae', verbosity=-1, **params)
            model.fit(X_train_data, y_train_data, categorical_feature=cat_features)

            oof_predictions[valid_idx] = model.predict(X_valid_data)

        # Converting OOF predictions back to original scale
        oof_predictions_exp = np.round(np.expm1(oof_predictions).astype(float)).astype(int)


        overall_oof_mape = mean_absolute_percentage_error(np.expm1(y_imp), oof_predictions_exp)

      
        oof_df = pd.DataFrame({
            'country': X_imp['country'],
            'y_true': np.expm1(y_imp),
            'y_pred': oof_predictions_exp
        })
        country_oof_mapes = oof_df.groupby('country', observed = True).apply(
            lambda group: mean_absolute_percentage_error(group['y_true'], group['y_pred'])
        ).to_dict()

        # Training on full data and predicting on test set
        model.fit(X_imp, y_imp)
        test_predictions = model.predict(X_test)

        # Converting test predictions back to original scale
        test_predictions_exp = np.round(np.expm1(test_predictions).astype(float)).astype(int)

        
        overall_test_mape = mean_absolute_percentage_error(y_test, test_predictions_exp)

        
        test_df = pd.DataFrame({
            'country': X_test['country'],
            'y_true': y_test,
            'y_pred': test_predictions_exp
        })
        country_test_mapes = test_df.groupby('country', observed = True).apply(
            lambda group: mean_absolute_percentage_error(group['y_true'], group['y_pred'])
        ).to_dict()

        # Appending results
        oof_preds.append({'Ref Country': ref_country, 'Overall MAPE': overall_oof_mape, **country_oof_mapes})
        test_preds.append({'Ref Country': ref_country, 'Overall MAPE': overall_test_mape, **country_test_mapes})

    oof_preds_df = pd.DataFrame(oof_preds)
    test_preds_df = pd.DataFrame(test_preds)

    # Creating cross tabs
    oof_crosstab = pd.pivot_table(oof_preds_df, index='Ref Country')
    test_crosstab = pd.pivot_table(test_preds_df, index='Ref Country')

    return oof_crosstab, test_crosstab


# Separating 2016 as the test set
X_train = df_train.query('Year < 2016')
X_test = df_train.query('Year == 2016')

X_test = X_test.dropna()
X_test, y_test = X_test.drop(['num_sold', 'id', 'date'], axis = 1), X_test['num_sold']

X_normalized = feature_normalizer(X_train)

countries_ref = ['Finland', 'Singapore', 'Norway', 'Italy']

oof_crosstab, test_crosstab = evaluate_imputation_models(
    X_normalized, X_test, y_test,
    countries_ref = countries_ref, 
    n_splits = 6
)

print("OOF Predictions Cross Tab:")
print(oof_crosstab)
print('~'*78)
print("\nTest Predictions Cross Tab:")
print(test_crosstab)


# Transforming data
df_test = date_extractor(df_test)

# Adding GDP data
df_test = df_test.merge(
    df_gdp_long,
    left_on = ['country', 'Year'],
    right_on=['country', 'Year'],   
    how='left'                    
)

df_test['GDP'] = df_test['GDP'].astype('float')
df_test = df_test.drop(['id', 'date'], axis = 1)

df_test = feature_encoder(df_test)


df_final = feature_normalizer(df_train)
df_final = values_imputer(df_final, country_ref = 'Norway').iloc[:, :18].drop(['id', 'date'], axis = 1)
X, y = df_final.drop('num_sold', axis = 1), np.log1p(df_final['num_sold'])

for i in cat_features:
    X[i] = X[i].astype('category')
    df_test[i] = df_test[i].astype('category')


#groups = X['Year']  
#lgbm_optimizer = LightGBMAndOptunaParams(X, y, groups=groups, n_folds = 7)

#study = lgbm_optimizer.optimize(n_trials=40)

# Results
#print("Best Parameters:", study.best_params)
#print("Best MALE Score:", study.best_value)


#oof_preds = lgbm_optimizer.oof
#print(f"OOF MAPE SCORE: {mean_absolute_percentage_error(np.expm1(y), np.round(np.expm1(oof_preds).astype(float)).astype(int))}")


params =  {'max_depth': 11,
          'num_leaves': 19,
          'learning_rate': 0.03501519992214664,
          'n_estimators': 1133,
          'lambda_l1': 0.05232565368692772,
          'lambda_l2': 0.09718396839395343,
          'colsample_bytree': 0.8126741295636939,
          'subsample': 0.8042105378729808,
          'min_child_samples': 180}


# Train LightGBM model
final_model = LGBMRegressor(
         linear_trees=True,
         metric='mae',
         verbosity=-1,
         **params)

final_model.fit(X, y, categorical_feature = cat_features)

final_predictions = final_model.predict(df_test)

# 1.06 multiplier does wonders
final_predictions =  np.round(np.expm1(final_predictions).astype(float)*1.06).astype(int)


sample_submission = pd.read_csv('/kaggle/input/playground-series-s5e1/sample_submission.csv')

sample_submission['num_sold'] = final_predictions
sample_submission.to_csv("submission.csv", index = False)

