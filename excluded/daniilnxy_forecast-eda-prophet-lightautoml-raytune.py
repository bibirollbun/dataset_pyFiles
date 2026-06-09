import pandas as pd
import numpy as np
import sklearn as sk
import matplotlib.pyplot as plt
import torch as t
import seaborn as sns
from sklearn.metrics import mean_absolute_percentage_error
from lightautoml.automl.presets.tabular_presets import TabularAutoML
from lightautoml.tasks import Task
from sklearn.model_selection import train_test_split
import ray
from ray import tune
from catboost import CatBoostRegressor
from ray.air import session
from ray.tune.schedulers import ASHAScheduler
from dirty_cat import TableVectorizer
from sklearn.pipeline import make_pipeline
import optuna
import lightgbm
import catboost
import logging
import matplotlib.ticker as mticker
import warnings

warnings.filterwarnings("ignore")


train_df = pd.read_csv(r'/kaggle/input/playground-series-s5e1/train.csv')
test_df = pd.read_csv(r'/kaggle/input/playground-series-s5e1/test.csv')
sub = pd.read_csv(r'/kaggle/input/playground-series-s5e1/sample_submission.csv')


train_df.info


train_df.head()


train_df.isna().sum()


train_df.nunique()


train_df.describe()


train_df.dtypes


test_df.info


test_df.head()


test_df.isna().sum()


test_df.nunique()


test_df.describe()


test_df.dtypes


warnings.filterwarnings("ignore", 
                        message="use_inf_as_na option is deprecated and will be removed in a future version. Convert inf values to NaN before operating instead.")
def plot_grouped_sales_sns(df, 
                           group_by, 
                           y, 
                           title, 
                           xlabel, 
                           ylabel, 
                           kind='bar', 
                           palette='viridis', 
                           color=None):

    if isinstance(group_by, str):
        group_by = [group_by]
    
    x_var = group_by[-1].key if isinstance(group_by[-1], pd.Grouper) else group_by[-1]
    hue_var = group_by[0] if len(group_by) > 1 else None
    
    if kind == 'FacetGrid':
        df['month'] = pd.to_datetime(df['date']).dt.month_name()
        grouped_data = df.groupby(group_by + ['month'])[y].sum().reset_index()
        month_order = pd.date_range("2022-01-01", periods=12, freq='M').strftime('%B')

        g = sns.FacetGrid(grouped_data, 
                          col='country', 
                          col_wrap=1, 
                          sharex=False, 
                          sharey=True, 
                          height=4, 
                          aspect=2.5, 
                          palette=palette)
        g.map_dataframe(
            lambda data, color: sns.barplot(
                data=data, 
                x='month', 
                y=y, 
                palette=sns.color_palette(palette, n_colors=len(month_order)), 
                order=month_order
            )
        )
        g.set_titles("{col_name}")
        g.set_axis_labels(xlabel, ylabel)
        for ax in g.axes.flat:
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.yaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=True))
            ax.ticklabel_format(style='plain', axis='y')
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        plt.subplots_adjust(top=0.9)
        g.fig.suptitle(title, fontsize=16)
        plt.show()
        return

    grouped_data = df.groupby(group_by)[y].sum().reset_index()

    plt.figure(figsize=(12, 6))
    match kind:
        case 'bar':
            ax = sns.barplot(data=grouped_data, 
                             x=x_var, 
                             y=y, 
                             hue=hue_var, 
                             palette=palette, 
                             color=color
                            )
            
        case 'lineplot':
            if hue_var:
                grouped_data = grouped_data.sort_values(by=[hue_var, x_var])
            else:
                grouped_data = grouped_data.sort_values(by=x_var)
            ax = sns.lineplot(data=grouped_data, 
                              x=x_var, 
                              y=y, 
                              hue=hue_var, 
                              palette=palette, 
                              linewidth=1, 
                              markers=False)

    
    _apply_plot_template(ax, title, xlabel, ylabel, x_var)

def _apply_plot_template(ax, title, xlabel, ylabel, x_var):
    ax.set_title(title, fontsize=16)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(rotation=45)
    ax.tick_params(axis='x', rotation=45)
    ax.yaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=True))
    ax.ticklabel_format(style='plain', axis='y')
    plt.tight_layout()
    plt.show()


plot_grouped_sales_sns(
    train_df, 
    group_by='country', 
    y='num_sold', 
    title='Total Sales per Country', 
    xlabel='Country', 
    ylabel='Total Sales'
)


plot_grouped_sales_sns(
    train_df, 
    group_by='store', 
    y='num_sold', 
    title='Total Sales per Store', 
    xlabel='Store', 
    ylabel='Total Sales'
)



plot_grouped_sales_sns(
    train_df, 
    group_by='product', 
    y='num_sold', 
    title='Total Sales per product', 
    xlabel='product', 
    ylabel='Total Sales'
)



plot_grouped_sales_sns(
    train_df, 
    group_by=['product', 'country'], 
    y='num_sold', 
    title='Total Sales per Product and Country', 
    xlabel='Product', 
    ylabel='Total Sales'
)


train_df['date'] = pd.to_datetime(train_df['date'])


plot_grouped_sales_sns(
    train_df, 
    group_by=['country', pd.Grouper(key='date', freq='MS')], 
    y='num_sold', 
    title='Total Sales per Country by Month', 
    xlabel='Month', 
    ylabel='Total Sales',
    kind='lineplot'
)


plot_grouped_sales_sns(
    train_df, 
    group_by=['product', pd.Grouper(key='date', freq='MS')], 
    y='num_sold', 
    title='Total Sales per Country by Month', 
    xlabel='Month', 
    ylabel='Total Sales',
    kind='lineplot'
)


plot_grouped_sales_sns(
    train_df, 
    group_by=['store', pd.Grouper(key='date', freq='MS')], 
    y='num_sold', 
    title='Total Sales per Country by Month', 
    xlabel='Month', 
    ylabel='Total Sales',
    kind='lineplot'
)



plot_grouped_sales_sns(
    train_df, 
    group_by=['country', pd.Grouper(key='date', freq='MS')], 
    y='num_sold', 
    title='Total Sales per Country by Month', 
    xlabel='Month', 
    ylabel='Total Sales',
    kind='FacetGrid'
)


train_df.groupby(['country', pd.Grouper(key='date', freq='MS')])['num_sold'].apply(lambda x: x.isna().sum())


train_df.groupby('country')['num_sold'].apply(lambda x: x.isna().sum())



# experiment 1 is to simply drop the missing data
train_data_exp_1 = train_df.copy()

# experiment 2 is to interplotate missing data 
train_data_exp_2 = train_df.copy()


train_data_exp_1.dropna(inplace=True)
train_data_exp_1.reset_index(drop=True, inplace=True)


train_data_exp_1.isna().sum()


train_data_exp_2.isnull().sum()


train_data_exp_2['num_sold'] = train_data_exp_2.groupby(['country', 'store', 'product'])['num_sold'] \
    .transform(lambda x: x.interpolate(method='linear'))



train_data_exp_2.head()


train_data_exp_2.isnull().sum()


train_data_exp_2[train_data_exp_2['num_sold'].isna()]


train_data_exp_2[train_data_exp_2['num_sold'].isna()].nunique()


train_data_exp_2[(train_data_exp_2['num_sold'].notna()) & (train_data_exp_2['country']=='Canada') & (train_data_exp_2['product']=='Holographic Goose') & (train_data_exp_2['store']=='Discount Stickers')]


train_data_exp_2[(train_data_exp_2['num_sold'].notna()) & (train_data_exp_2['country']=='Kenya') & (train_data_exp_2['product']=='Holographic Goose') & (train_data_exp_2['store']=='Discount Stickers')]


train_data_exp_2.groupby('country')['num_sold'].apply(lambda x: x.isna().sum())



train_data_exp_2['num_sold'] = train_data_exp_2.groupby(['country', 'product'])['num_sold'] \
    .transform(lambda x: x.interpolate(method='linear'))


train_data_exp_2.groupby('country')['num_sold'].apply(lambda x: x.isna().sum())



train_data_exp_2.dropna(inplace=True)
train_data_exp_2.reset_index(inplace=True, drop=True)


def sep_data(df):
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['weekday'] = df['date'].dt.weekday
    if 'id' in df.columns:
        df.drop(columns=['id'],inplace=True, axis=1)
        df.reset_index(inplace=True, drop=True)
    return df 


sep_data(train_data_exp_1)


sep_data(train_data_exp_2)


sep_data(test_df)


!pip install prophet -q
from prophet import Prophet


grouped_data_exp_1 = train_data_exp_1.groupby(['country', 'store', 'product'])


train_data_exp_1[(train_data_exp_1['country']=='Canada') & (train_data_exp_1['store']=='Discount Stickers') & (train_data_exp_1['product']=='Kaggle')]


logging.getLogger('cmdstanpy').setLevel(logging.WARNING)
models = {} 
for group, data in grouped_data_exp_1:
    df = data[['date', 'num_sold']].rename(columns={'date': 'ds', 'num_sold': 'y'})
    model = Prophet()
    model.fit(df)
    models[group] = model


predictions = []

for _, row in test_df.iterrows():
    group = (row['country'], row['store'], row['product'])
    if group in models:
        future = pd.DataFrame({'ds': [row['date']]})
        forecast = models[group].predict(future)
        predictions.append(forecast['yhat'].values[0])
    else:
        predictions.append(train_df['num_sold'].mean())


test_df['num_sold'] = predictions


test_df[['id', 'num_sold']].to_csv('prophet_experiment_1.csv', index=False)


grouped_data_exp_2 = train_data_exp_2.groupby(['country', 'store', 'product'])


logging.getLogger('cmdstanpy').setLevel(logging.WARNING)
models = {} 
for group, data in grouped_data_exp_2:
    df = data[['date', 'num_sold']].rename(columns={'date': 'ds', 'num_sold': 'y'})
    model = Prophet()
    model.fit(df)
    models[group] = model


predictions = []

for _, row in test_df.iterrows():
    group = (row['country'], row['store'], row['product'])
    if group in models:
        future = pd.DataFrame({'ds': [row['date']]})
        forecast = models[group].predict(future)
        predictions.append(forecast['yhat'].values[0])
    else:
        predictions.append(train_df['num_sold'].mean())


test_df['num_sold'] = predictions


test_df[['id', 'num_sold']].to_csv('prophet_experiment_2.csv', index=False)


pip install lightautoml -q


auto_ml = TabularAutoML(task = Task(name = 'reg', metric=mean_absolute_percentage_error), timeout = 300)

roles = {'target': 'num_sold', 'drop': 'date'}



train, valid = train_test_split(train_data_exp_1, test_size=0.2, random_state=42)


oof_preds = auto_ml.fit_predict(train, roles=roles)


valid_preds = auto_ml.predict(valid)


valid_true = valid['num_sold'].values


mape = mean_absolute_percentage_error(valid_true, valid_preds.data)
print(f'MAPE on validation set: {mape:.4f}')


train, valid = train_test_split(train_data_exp_2, test_size=0.2, random_state=42)
oof_preds = auto_ml.fit_predict(train, roles=roles)
valid_preds = auto_ml.predict(valid)
valid_true = valid['num_sold'].values
mape = mean_absolute_percentage_error(valid_true, valid_preds.data)
print(f'MAPE on validation set: {mape:.4f}')


pip install dirty_cat -q


pip install -U tensorboardx


to_drop = ['num_sold', 'date']
to_use = ['num_sold']
x_exp_1 = train_data_exp_1.drop(to_drop, axis=1)
y = train_data_exp_1[to_use]


test_df = test_df.drop('date', axis=1)


tv = TableVectorizer(auto_cast=True)

X_train_enc = tv.fit_transform(x_exp_1, y)
X_test_enc = tv.transform(test_df)


X_train, X_val, y_train, y_val = train_test_split(
    X_train_enc, y, test_size=0.15, random_state=42
)



def train_model(config):
    x_train_local = ray.get(x_train_i)
    y_train_local = ray.get(y_train_i)
    x_val_local = ray.get(x_val_i)
    y_val_local = ray.get(y_val_i)
    
    model = CatBoostRegressor(
        iterations=int(config["iterations"]),
        depth=int(config["depth"]),
        learning_rate=float(config["learning_rate"]),
        l2_leaf_reg=float(config["l2_leaf_reg"]),
        border_count = int(config['border_count']),
        subsample = float(config['subsample']),
        random_strength=int(config["random_strength"]),  
        bagging_temperature=float(config["bagging_temperature"]),
        random_seed=42,
        logging_level='Silent', 
    )
    
    
    model.fit(
        x_train_local, y_train_local,
        eval_set=(x_val_local, y_val_local),
        verbose=False,
        use_best_model=True
    )
    
    preds = model.predict(x_val_local)
    mape = mean_absolute_percentage_error(y_val_local, preds)
    
    session.report({"mape": mape})


ray.init(ignore_reinit_error=True)

x_train_i = ray.put(X_train)
y_train_i  = ray.put(y_train)
x_val_i  = ray.put(X_val)
y_val_i  = ray.put(y_val)

search_space = {
    "iterations": tune.randint(700, 800),
    "depth": tune.choice([6,7,8,9,10,12]),
    "learning_rate": tune.loguniform(0.1, 0.5),
    "l2_leaf_reg": tune.loguniform(0.0001, 0.5),
    "border_count": tune.randint(200,400),
    "subsample": tune.uniform(0.6, 1.0),
    "random_strength": tune.randint(3,9),
    "bagging_temperature": tune.uniform(0.7, 1.0)
}

asha_scheduler = ASHAScheduler(
    max_t=150,
    grace_period=25,
    reduction_factor=2
)


tuner = tune.run(
    train_model,
    config=search_space,
    scheduler=asha_scheduler,
    metric="mape",
    mode="min",
    num_samples=10
)

best_trial = tuner.get_best_trial("mape", "min")

# Retrieve and print the best configuration
print("Best hyperparameters:", best_trial.config)

# Retrieve and print the MAPE of the best model
print("best MAPE:", best_trial.last_result["mape"])

ray.shutdown()


best_config  =  best_trial.config


ray.shutdown()


final_model =  CatBoostRegressor(
        iterations=int(best_config["iterations"]),
        depth=int(best_config["depth"]),
        learning_rate=float(best_config["learning_rate"]),
        l2_leaf_reg=float(best_config["l2_leaf_reg"]),
        random_strength=int(best_config["random_strength"]),  
        bagging_temperature=float(best_config["bagging_temperature"]),
        logging_level='Silent', 
        random_seed = 42
    )

final_model.fit(X_train, y_train,
                eval_set=(X_val, y_val))


final_preds = final_model.predict(X_test_enc)





final_preds.shape


sub['num_sold'] = final_preds


sub


sub[['id', 'num_sold']].to_csv('catboost_raytuned.csv', index=False)

