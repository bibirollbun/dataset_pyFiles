import numpy as np
import pandas as pd


train_df = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv')
print(train_df.shape)
train_df.head(2)


test_df = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')
print(test_df.shape)
test_df.head(2)


any(train_df.isna().sum()), any(test_df.isna().sum()), 


from sklearn.model_selection import train_test_split
import pandas as pd

X = train_df[[c for c in train_df if c not in ['Calories', 'id']]]
y = train_df['Calories']

X_modeling, X_test, y_modeling, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_valid, y_train, y_valid = train_test_split(X_modeling, y_modeling, test_size=.125, random_state=42)

X_train = X_train.reset_index(drop=True)
X_valid = X_valid.reset_index(drop=True)
X_test = X_test.reset_index(drop=True)
y_train = y_train.reset_index(drop=True)
y_valid = y_valid.reset_index(drop=True)
y_test = y_test.reset_index(drop=True)

# rescalse target
y_train_tf = np.log1p(y_train)
y_valid_tf = np.log1p(y_valid)
y_test_tf = np.log1p(y_test)


len(X_train), len(X_valid), len(X_test), sum([len(X_train), len(X_valid), len(X_test)])


len(y_train), len(y_valid), len(y_test), sum([len(y_train), len(y_valid), len(y_test)])


import plotly.express as px
from typing import Sequence

def plot_distributions(
    series_list:list[pd.DataFrame],
    series_names:list|None=None,
    title=None
):

    if series_names is None:
        series_names = [f'dataset{i+1}' for i in range(len(series_list))]
        
    if len(series_list) != len(series_names):
        raise Exception(f'df, names length mismatch. {len(series_list), len(series_names)}')

    data = []
    group = []
    for i,series in enumerate(series_list):
        data.extend(series)
        group.extend([series_names[i]]*len(series))

    df = pd.DataFrame({
        'data':data,
        'group':group
    })
    
    fig = px.histogram(
        df, x="data", color="group", nbins=20, opacity=0.5,
        barmode="overlay", histnorm='percent', title=title,
    )
    if title:
        fig.update_layout(
            xaxis_title=title
        )
    fig.show()

def feature_x_target_by_decile(
    feature:Sequence,
    target:Sequence,
    feature_name:str
):
    FEATURE = f'{feature_name}_decile'
    TARGET = 'target'
    df = (
        pd.DataFrame({
            FEATURE:pd.qcut(feature, q=10, labels=False)+1,
            TARGET:target
        })
        .groupby(FEATURE)[TARGET].mean().reset_index()
    )

    px.line(df, x=FEATURE, y=TARGET).show()


continuous_features = ['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp']
for col in continuous_features[:2]:
    plot_distributions(
        [X_train[col], X_valid[col], X_test[col]],
        ['train', 'valid', 'test'],
        col
    )

for col in continuous_features[2:4]:
    print(f'{col} deciles average calories')
    feature_x_target_by_decile(
        X_train[col],
        y_train,
        col
    )


pd.DataFrame([
    dict(df.describe()) for df in [y_train, y_valid, y_test]
])


# probably not useful (and slightly confusing..) but show target before and after transforming)
plot_distributions(
    [y_train, y_valid, y_test],
    ['train', 'valid', 'test'],
    'Calories -- Raw'
)


plot_distributions(
    [y_train_tf, y_valid_tf, y_test_tf],
    ['train', 'valid', 'test'],
    'Calories -- Target'
)


pd.DataFrame(
    [dict(round(df.groupby('Sex')['Age'].count()/len(df),4)) for df in [X_train, X_valid, X_test]]
)


import lightgbm as lgb
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_log_error
from sklearn.model_selection import KFold

def root_mean_squared_log_error(y_true, y_pred):
    return np.sqrt(mean_squared_log_error(y_true, y_pred))


# encode Sex -- doesnt matter to do here or in Pipeline
for df in [X_train, X_valid, X_test]:
    df['Sex'] = df['Sex'].map({'female':0, 'male':1})


column_transformer = ColumnTransformer(
    remainder='passthrough',
    transformers=[
     ("continuous_features", MinMaxScaler(), continuous_features)
    ]
)


X_train_tf = column_transformer.fit_transform(X_train)
X_valid_tf = column_transformer.transform(X_valid)

train_data = lgb.Dataset(X_train_tf, label=y_train_tf)
valid_data = lgb.Dataset(X_valid_tf, label=y_valid_tf)

params = {
    'objective': 'regression',
    'metric': 'rmse',
}

base_model1 = lgb.train(
    params,
    train_data,
    valid_sets=[valid_data],
    num_boost_round=100,
    callbacks=[lgb.early_stopping(stopping_rounds=20)]
)


%%time
base1_preds = base_model1.predict(column_transformer.transform(X_test))


%%time
%%capture

n_folds = 10
kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
params = {
    'objective': 'regression',
    'metric': 'rmse'
}

base2_preds = np.array([0.0]*len(X_test))

X_modeling = pd.concat([X_train, X_valid]).reset_index(drop=True)
y_modeling = pd.concat([y_train_tf, y_valid_tf]).reset_index(drop=True)

for train_idx, valid_idx in kf.split(X_modeling):
    X_train_k, X_valid_k = X_modeling.iloc[train_idx], X_modeling.iloc[valid_idx]
    y_train_k, y_valid_k = y_modeling.iloc[train_idx], y_modeling.iloc[valid_idx]

    min_max_scaler = MinMaxScaler()
    X_train_tf_k = column_transformer.fit_transform(X_train_k)
    X_valid_tf_k = column_transformer.transform(X_valid_k)

    train_data_k = lgb.Dataset(X_train_tf_k, label=y_train_k)
    valid_data_k = lgb.Dataset(X_valid_tf_k, label=y_valid_k)

    model = lgb.train(
        params,
        train_data_k, 
        valid_sets=[valid_data_k],
        num_boost_round=200,
    )
    
    base2_preds += np.array(
        model.predict(
            column_transformer.transform(X_test)
        )
    )

base2_preds = base2_preds/n_folds


root_mean_squared_log_error(y_test, np.exp(base1_preds))


root_mean_squared_log_error(y_test, np.exp(base2_preds))


# Probably wouldve been better (and more consistent) to do math from series not DF or return DF...
    # o well. this is fine albeit halfway between two better (~more elegant~) alternatives

bmi_class_map = {
    0:'Underweight',
    1:'Normal',
    2:'Overweight',
    3:'Obesity_1',
    4:'Obesity_2',
    5:'Obesity_3',        
}

hr_intensity_class_map = {
    0:'Low',
    1:'Moderate',
    2:'High',
    3:'Very High',        
}

def get_bmi_index(
    df:pd.DataFrame,
    wgt_col:str='Weight',
    ht_col:str='Height'
) -> pd.Series:
    return round(df[wgt_col] / (df[ht_col]/100)**2,1)

def get_bmi_class(
    df:pd.DataFrame,
    bmi_idx_col:str='BMI_Index',
    label_map = bmi_class_map # not great practice but ok
) -> pd.Series:
    
    code_labels = pd.cut(
        df[bmi_idx_col],
        bins=[-np.inf,18.5,25,30,35,40,np.inf],
        labels=False
    )

    return code_labels.map(label_map)

def get_max_heart_rate(
    df:pd.DataFrame,
    age_col:str='Age'
) -> pd.Series:
    return 220 - df[age_col]

def get_hr_intensity_pct(
    df:pd.DataFrame,
    hr_col:str = 'Heart_Rate',
    max_hr_col:str='Max_Heart_Rate'
) -> pd.Series:
    return round(df[hr_col] / df[max_hr_col] * 100,1)

def get_hr_intensity_class(
    df:pd.DataFrame,
    hr_pct_max_col = 'HR_Intensity_Pct',
    label_map = hr_intensity_class_map # also not great practice but ok.
):
    code_labels = pd.cut(
        df[hr_pct_max_col],
        bins=[-np.inf,50,70,85,np.inf],
        labels=False
    )

    return code_labels.map(label_map)


# just a little easier to do transformations when data loaded

X = train_df[[c for c in train_df if c not in ['Calories', 'id']]]
y = train_df['Calories']

X['BMI_Index'] = get_bmi_index(X)
X['BMI_Class'] = get_bmi_class(X)
X['Max_Heart_Rate'] = get_max_heart_rate(X)
X['HR_Intensity_Pct'] = get_hr_intensity_pct(X)
X['HR_Intensity_Class'] = get_hr_intensity_class(X)

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

X_train = X_train.reset_index(drop=True)
X_valid = X_valid.reset_index(drop=True)
y_train = y_train.reset_index(drop=True)
y_valid = y_valid.reset_index(drop=True)

# rescalse target
y_train_tf = np.log1p(y_train)
y_valid_tf = np.log1p(y_valid)


X_train.head()


pd.options.mode.chained_assignment = None

def avg_resid_by_variable(df:pd.DataFrame, variable:str, residuals:str):
    df = df[[variable, residuals]]

    if df[variable].dtype == 'float':
        df[variable] = round(df[variable])

    df = df.groupby([variable])[residuals].mean().reset_index()

    px.line(df, x=variable, y=residuals).show()


df = X_valid.copy(deep=True)
df['Calories'] = y_valid
df['Pred'] = np.exp(base2_preds) # created on 'test' in section 1, which is same as 'valid' in section 2 (confirmed below)
df['Resid'] = df.Calories - df.Pred


root_mean_squared_log_error(df.Calories, df.Pred)


df.head(2)


import plotly.graph_objects as go

fig = px.scatter(df, x='Calories', y='Pred')
fig.add_trace(
    go.Scatter(
        x=[df.Calories.min(), df.Calories.max()], y=[df.Pred.min(), df.Pred.max()],
        mode='lines', name='45° Line', line=dict(color='red', dash='dash'))
)

fig.show()


feature_x_target_by_decile(
    df.Calories,
    df.Resid,
    'Overall'
)


feature_x_target_by_decile(
    df.Age,
    df.Resid,
    'Age'
)


df_bad_preds = df[abs(df.Calories-df.Pred)>15]
df_bad_preds.head(2)


for col in ['Age', 'Duration']:
    plot_distributions(
        [df[col], df_bad_preds[col]],
        ['Valid', 'Bad_Preds'],
        col
    )


df[df.Calories<20]


duration_binned_df = (
    pd.DataFrame({
        'Duration_Bins':pd.cut(df.Duration, bins=30, labels=False),
        'Count':df.Calories
    })
    .groupby('Duration_Bins')['Count'].count().reset_index()
)

duration_binned_df['Cum_Pct'] = duration_binned_df['Count'].cumsum() / duration_binned_df['Count'].sum() * 100

fig = px.bar(duration_binned_df, x='Duration_Bins', y='Count')

fig.add_trace(
    go.Scatter(x=duration_binned_df['Duration_Bins'], 
               y=duration_binned_df['Cum_Pct'], 
               mode='lines+markers', name='Cumulative %',
               yaxis="y2"
              )
)

fig.update_layout(
    yaxis=dict(title="Count"),
    yaxis2=dict(title="Cumulative Percentage", overlaying="y", side="right"),
    xaxis_title="Duration_Binned",
    yaxis_title="Count",
    showlegend=True
)

fig.show()


def weighted_average(values, weights):
    
    weighted_sum = sum(v * w for v, w in zip(values, weights))
    
    return weighted_sum / sum(weights)


px.histogram(y_train)


y_train.describe()


# very aggressive weighted
X_train['wgt'] = 1 / (round(y_train,-2) / 100+2)
X_valid['wgt'] = 1 / (round(y_valid,-2) / 100+2)


weighted_average(y_train, X_train.Weight)


X_train['Age_Ln'] = np.log(X_train['Age'])
X_valid['Age_Ln'] = np.log(X_valid['Age'])

X_train['Duration_Ln'] = np.log(X_train['Duration'])
X_valid['Duration_Ln'] = np.log(X_valid['Duration'])


from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder


continuous_features = ['Age_Ln', 'Height', 'Weight', 'Duration_Ln',
                       'Heart_Rate', 'Body_Temp', 'BMI_Index', 'HR_Intensity_Pct'
                      ]

ordinal_features = ['BMI_Class', 'HR_Intensity_Class']
ordinal_features_orders = [list(bmi_class_map.values()), list(hr_intensity_class_map.values())]

column_transformer = ColumnTransformer(
    remainder='passthrough',
    transformers=[
        ('one_hot', OneHotEncoder(),['Sex']),
        ('ordinal', OrdinalEncoder(categories=ordinal_features_orders),ordinal_features),
        ("continuous", MinMaxScaler(), continuous_features)
    ]
)


%%time
%%capture

n_folds = 10
kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
params = {
    'objective': 'regression',
    "learning_rate": 0.03,
    'metric': 'rmse'
}

model3_preds = np.array([0.0]*len(X_test))

for train_idx, valid_idx in kf.split(X_train):
    X_train_k, X_valid_k = X_train.iloc[train_idx], X_train.iloc[valid_idx]
    y_train_k, y_valid_k = y_train_tf.iloc[train_idx], y_train_tf.iloc[valid_idx]

    train_wgt = X_train_k.wgt
    valid_wgt = X_valid_k.wgt

    X_train_k = X_train_k.drop(['Age', 'Duration', 'wgt'], axis=1)
    X_valid_k = X_valid_k.drop(['Age', 'Duration', 'wgt'], axis=1)
    
    train_data_k = lgb.Dataset(
        column_transformer.fit_transform(X_train_k), 
        label=y_train_k,
        weight=train_wgt
    )
    valid_data_k = lgb.Dataset(
        column_transformer.transform(X_valid_k), 
        label=y_valid_k,
        weight=valid_wgt
    )

    model = lgb.train(
        params,
        train_data_k, 
        valid_sets=[valid_data_k],
        num_boost_round=1000,
    )
    
    model3_preds += np.array(
        model.predict(
            column_transformer.transform(X_valid) # valid is holdout
        )
    )

model3_preds = model3_preds/n_folds


root_mean_squared_log_error(y_valid, np.exp(model3_preds))


import xgboost as xgb


%%time
%%capture

# really shouldve just made a function for this... but probably the least concerning improvement to be made :D
n_folds = 10
kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

model4_preds = np.array([0.0]*len(X_test))

for train_idx, valid_idx in kf.split(X_train):
    X_train_k, X_valid_k = X_train.iloc[train_idx], X_train.iloc[valid_idx]
    y_train_k, y_valid_k = y_train_tf.iloc[train_idx], y_train_tf.iloc[valid_idx]

    train_wgt = X_train_k.wgt
    valid_wgt = X_valid_k.wgt

    X_train_k = X_train_k.drop(['Age', 'Duration', 'wgt'], axis=1)
    X_valid_k = X_valid_k.drop(['Age', 'Duration', 'wgt'], axis=1)

    dtrain = xgb.DMatrix(
        column_transformer.fit_transform(X_train_k), 
        label=y_train_k, 
        weight=train_wgt)
    
    dvalid = xgb.DMatrix(
        column_transformer.transform(X_valid_k), 
        label=y_valid_k, 
        weight=valid_wgt
    )
    
    params = {
        'max_depth': 8,
        'learning_rate': 0.03,
        'objective': 'reg:squarederror'
    }
    
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=1000,
        evals=[(dtrain, 'train'), (dvalid, 'valid')],
        verbose_eval=False
    )

    model4_preds += np.array(
        model.predict(
            xgb.DMatrix(
                column_transformer.transform(X_valid)
            )
        )
    )

model4_preds = model4_preds/n_folds


root_mean_squared_log_error(y_valid, np.exp(model4_preds))


import random
import tensorflow as tf
print(tf.__version__)


X_train_tf = X_train.drop(['Age', 'Duration', 'wgt'], axis=1)
n = int(len(X_train_tf)*.9)
X_eval_tf = X_train_tf.iloc[n:]
y_eval_tf = y_train[n:] 
X_train_tf = X_train_tf.iloc[:n]
y_train_tf = y_train[:n]


X_valid_tf = X_valid.drop(['Age', 'Duration', 'wgt'], axis=1)
X_train_tf.head(2), X_eval_tf.head(2)


%%time
random.seed(42)
tf.random.set_seed(42) 
# Did one with and without BatchNorm/LeakyReLU/Dropout and basic one worked best. 
    # Most likely leakyrelu was the big bad here 
    # ... Could've experimented more, but not in scope
model5 = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu'),
    # tf.keras.layers.BatchNormalization(),
    # tf.keras.layers.LeakyReLU(negative_slope=.1),
    # tf.keras.layers.Dropout(.1),
    tf.keras.layers.Dense(64, activation='relu'),
    # tf.keras.layers.BatchNormalization(),
    # tf.keras.layers.LeakyReLU(negative_slope=.1),
    # tf.keras.layers.Dropout(.1),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1)
])

model5.compile(
    loss=tf.keras.losses.mse,
    optimizer=tf.keras.optimizers.Adam(),
    metrics=['root_mean_squared_error']
    
)

model5.fit(
    column_transformer.fit_transform(X_train_tf),
    y_train_tf,
    epochs=10,
    validation_data = (
        column_transformer.transform(X_eval_tf),
        y_eval_tf
    ),
    verbose=0
)


model5_preds = model5.predict(
    column_transformer.transform(X_valid_tf)
)


# lazy but fine
model5_preds = tf.squeeze([.0001 if i<=0 else i for i in model5_preds]).numpy()


model5_preds.min(), model5_preds.max() # (0.87722445, 302.7649)


root_mean_squared_log_error(y_valid, model5_preds)


def summary_table(y_true, y_preds, models):
    d = dict(pd.Series(y_true).describe())
    d.update({'rmsle':0})
    d
    dicts = [d]
    for pred in y_preds:
        d = dict(pd.Series(pred).describe())
        d.update({'rmsle':root_mean_squared_log_error(y_true, pred)})
        dicts.append(d)
    return pd.DataFrame(dicts, index=['Actual']+models)


summary_table(
    y_valid, 
    [np.exp(base2_preds), np.exp(model3_preds), np.exp(model4_preds), model5_preds],
    ['Baseline', 'LightGBM', 'XGBoost', 'TensorFlow']
)




