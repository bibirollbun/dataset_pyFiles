import numpy as np
import matplotlib.pyplot as plt


import pandas as pd

df_train = pd.read_csv(
    "/kaggle/input/playground-series-s5e1/train.csv",
    parse_dates=['date'])

df_train.head(10)


df_test = pd.read_csv(
    "/kaggle/input/playground-series-s5e1/test.csv"
    ,parse_dates=['date'])

df_test.head()


df_train.info()


df_train['num_sold'].fillna(df_train['num_sold'].median(), inplace=True)


df_train.head()



grouped = df_train.groupby(['country','date'])['num_sold'].mean().unstack(0)

grouped.head(10)





group_columns = ['country', 'store', 'product']
plots = {}

for col in group_columns:
    grouped = df_train.groupby([col, 'date'])['num_sold'].sum().unstack(0)
    
    # Plot each category
    grouped.plot(title=f'Date vs Num_Sold for {col}', figsize=(10, 6))
    plt.ylabel('Num Sold')
    plt.xlabel('Date')
    plt.legend(title=col, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()














import matplotlib.pyplot as plt
df_train = df_train.sort_values(by='date')
df_test = df_test.sort_values(by='date')
#ction to plot periodograms for grouped data in one figure
def plot_grouped_periodogram(grouped_data, title):
    """
    Plots periodograms for multiple time series groups in one figure.

    Parameters:
        grouped_data: A DataFrame where each column represents a time series.
        title: Title for the plot.
    """
    top_five_freq={}
    from scipy.signal import periodogram
    fs = pd.Timedelta("365D") / pd.Timedelta("1D")  # Sampling frequency (daily data over a year)
    
    plt.figure(figsize=(12, 8))
    
    for group_name in grouped_data.columns:
        ts = grouped_data[group_name].dropna()  # Drop NaNs for the current group
        if not ts.empty:  # Only plot non-empty time series
            frequencies, spectrum = periodogram(
                ts,
                fs=fs,
                detrend="linear",
                window="boxcar",
                scaling="spectrum",
            )
            plt.step(frequencies, spectrum, label=group_name) # Plot each group
            top_indices = spectrum.argsort()[-5:][::-1]  # Indices of the top 5 spectrum values
            top_five_freq[group_name] =[i for i in top_indices]
    plt.xscale("log")
    plt.xticks(
        [1, 2, 4, 6, 12, 26, 52, 104],
        [
            "Annual (1)",
            "Semiannual (2)",
            "Quarterly (4)",
            "Bimonthly (6)",
            "Monthly (12)",
            "Biweekly (26)",
            "Weekly (52)",
            "Semiweekly (104)",
        ],
        rotation=30,
    )
    plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    plt.ylabel("Variance")
    plt.title(title)
    plt.legend(loc="upper right", bbox_to_anchor=(1.25, 1))
    plt.tight_layout()
    plt.show()
    return top_five_freq


# Assuming df_train is sorted by date and contains the necessary data
top_freq={}
for col in group_columns:
    grouped = df_train.groupby([col, 'date'])['num_sold'].sum().unstack(0)
    top_five_freq=plot_grouped_periodogram(grouped, f"Periodograms for {col} Groups") 
    top_freq[col]=top_five_freq
    # Plot grouped periodograms in one figure for each feature.
    


top_freq


filtered_top_freq = {
    feature: {
        group: [freq for freq in freqs if freq < 200]
        for group, freqs in groups.items()
    }
    for feature, groups in top_freq.items()
}


filtered_top_freq


feature='country'
group_name='Canada'
ts_train = df_train[df_train[feature] == group_name].groupby('date')['num_sold'].mean()
ts_train.head()



from statsmodels.tsa.deterministic import CalendarFourier
top_frequencies=filtered_top_freq

fourier_terms = [
    CalendarFourier(freq="YE", order=freq) for freq in top_frequencies[feature].get(group_name, [])
]


top_frequencies[feature].get(group_name, [])


from statsmodels.tsa.deterministic import CalendarFourier, DeterministicProcess
ts_train = df_train[df_train[feature] == group_name].groupby('date')['num_sold'].mean()
ts_test = df_test[df_test[feature] == group_name].groupby('date').size()
ts_test = ts_test.index  # Get the unique date index

# Create Fourier terms based on top frequencies
fourier_terms = [
    CalendarFourier(freq="YE", order=freq) for freq in top_frequencies[feature].get(group_name, [])
]

# Set up Deterministic Process (use same dp for train and test)
dp = DeterministicProcess(
    index=ts_train.index.union(ts_test),  # Combine train and test indices
    constant=True,
    order=2,  # Linear trend
    seasonal=True,  # Weekly seasonality
    additional_terms=fourier_terms,  # Add filtered Fourier terms
    drop=True  # Drop collinear terms
)

X = dp.in_sample()
X.head()


from statsmodels.tsa.deterministic import CalendarFourier, DeterministicProcess
from sklearn.linear_model import LinearRegression

def train_and_predict(df_train, df_test, features, top_frequencies):
    for feature in features:
        for group_name in df_train[feature].unique():
            # Prepare train data for the current group
            ts_train = df_train[df_train[feature] == group_name].groupby('date')['num_sold'].mean()
            ts_test = df_test[df_test[feature] == group_name].groupby('date').size()
            ts_test = ts_test.index  # Get the unique date index

            # Create Fourier terms based on top frequencies
            fourier_terms = [
                CalendarFourier(freq="YE", order=freq) for freq in top_frequencies[feature].get(group_name, [])
            ]
            
            # Set up Deterministic Process (use same dp for train and test)
            dp = DeterministicProcess(
                index=ts_train.index.union(ts_test),  # Combine train and test indices
                constant=True,
                order=2,  # Linear trend
                seasonal=True,  # Weekly seasonality
                additional_terms=fourier_terms,  # Add filtered Fourier terms
                drop=True  # Drop collinear terms
            )
            
            X = dp.in_sample()
            X_train = X.loc[ts_train.index]
            y_train = ts_train
            X_test = X.loc[ts_test]
            
            # Fit Linear Regression model
            model = LinearRegression()
            model.fit(X_train, y_train)
            train_pred = model.predict(X_train)
            test_pred = model.predict(X_test)

            # Map predictions back to df_train
            id_to_train_pred = dict(zip(ts_train.index, train_pred))
            df_train.loc[df_train[feature] == group_name, f'{feature}_pred'] = df_train.loc[
                df_train[feature] == group_name, 'date'
            ].map(id_to_train_pred)

            # Map predictions back to df_test
            id_to_test_pred = dict(zip(ts_test, test_pred))
            df_test.loc[df_test[feature] == group_name, f'{feature}_pred'] = df_test.loc[
                df_test[feature] == group_name, 'date'
            ].map(id_to_test_pred)

    return df_train, df_test


# Example usage
features = ['country', 'store', 'product']
df_train_pred, df_test_pred= train_and_predict(df_train, df_test, features, filtered_top_freq)



df_train_pred.head()


import matplotlib.pyplot as plt

group_columns = ['country', 'store', 'product']
pred_columns = ['country_pred', 'store_pred', 'product_pred']
plots = {}

# Iterate through each group column and its corresponding prediction column
for col, pred_col in zip(group_columns, pred_columns):
    # Group actual data and predictions by date
    actual_grouped = df_train.groupby(['date', col])['num_sold'].sum().unstack(col)
    pred_grouped = df_train.groupby(['date', col])[pred_col].sum().unstack(col)
    
    # Plot actual data
    actual_grouped.plot(title=f'Actual Data: Date vs Num_Sold for {col}', figsize=(12, 6))
    plt.ylabel('Num Sold')
    plt.xlabel('Date')
    plt.legend(title=f'Actual {col}', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()
    
    # Plot predictions
    pred_grouped.plot(title=f'Predictions: Date vs {pred_col}', figsize=(12, 6), linestyle='--')
    plt.ylabel('Predicted Num Sold')
    plt.xlabel('Date')
    plt.legend(title=f'Predicted {col}', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()



%%time
import holidays
import numpy as np
def get_holidays(df):
    years_list = [2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019]

    holiday_FI = holidays.CountryHoliday('FI', years = years_list)
    holiday_CA = holidays.CountryHoliday('CA', years = years_list)
    holiday_IT = holidays.CountryHoliday('IT', years = years_list)
    holiday_KE = holidays.CountryHoliday('KE', years = years_list)
    holiday_NO = holidays.CountryHoliday('NO', years = years_list)
    holiday_SI = holidays.CountryHoliday('SG', years = years_list)

    holiday_dict = holiday_FI.copy()
    holiday_dict.update(holiday_CA)
    holiday_dict.update(holiday_IT)
    holiday_dict.update(holiday_KE)
    holiday_dict.update(holiday_NO)
    holiday_dict.update(holiday_SI)

    df['holiday_name'] = df['date'].map(holiday_dict)
    df['is_holiday'] = np.where(df['holiday_name'].notnull(), 1, 0)
    df['holiday_name'] = df['holiday_name'].fillna('Not Holiday')
    df.drop(columns='holiday_name', axis=1, inplace=True)
    print(df.columns)
    return df


def feature_engineer(df):
    
    new_df = df.copy()
    new_df['year'] = new_df['date'].dt.year
    new_df['year_sin'] = np.sin(new_df['year'] * (2 * np.pi))
    new_df['year_cos'] = np.cos(new_df['year'] * (2 * np.pi))

    new_df['month'] = new_df['date'].dt.month
    new_df['month_sin'] = np.sin(new_df['month'] * (2 * np.pi / 12))
    new_df['month_cos'] = np.cos(new_df['month'] * (2 * np.pi / 12))
    
    new_df['day'] = new_df['date'].dt.day
    new_df['day_sin'] = np.sin(new_df['day'] * (2 * np.pi / 365))
    new_df['day_cos'] = np.cos(new_df['day'] * (2 * np.pi / 365))
    
    new_df['day_of_week'] = new_df['date'].dt.dayofweek
    new_df['day_of_week'] = new_df['day_of_week'].apply(lambda x: 0 if x<=3 else(1 if x==4 else (2 if x==5 else (3))))
    
    # new_df = pd.get_dummies(new_df, columns=['day_of_week'], drop_first=True, dtype=int)

        
    return new_df.drop(columns=['date', 'month', 'day'], axis=1)

train = get_holidays(df_train_pred)
train = feature_engineer(train)

test = get_holidays(df_test_pred)
test = feature_engineer(test)


#groupby :
# for name, group in grouped:
#     print(f"類別: {name}")
#     print(group)
#     print("-" * 20)








from sklearn.model_selection import KFold, StratifiedKFold
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)


X.shape


train.info()





X = train.drop('num_sold',axis=1)
y = train['num_sold']

df=skf.split(X,y)
print(list(df))


a=[(1,2),(3,4),(5,6)]
for i,(b,c) in enumerate(a):
    print(f'{i}=')
    print(b)
    print(c)


X.select_dtypes(include='object').columns





%%time
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split, GridSearchCV, RepeatedKFold, RepeatedStratifiedKFold, GroupKFold
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_percentage_error

lgb_params = {'learning_rate': 0.07049928250360378,
              'n_estimators': 1000,
              'max_depth': 12,
              'reg_alpha': 0.01260164540047986,
              'reg_lambda': 5.6849501092111305,
              'num_leaves': 82,
              'colsample_bytree': 0.689643373301433,
              'verbose': -1,
              'n_jobs': -1,
              'device': 'gpu'}

test_cv = test.drop(columns='year', axis=1).copy()

X = train.drop('num_sold',axis=1)
y = train['num_sold']

from sklearn.preprocessing import LabelEncoder


X_label_encoded = X.copy()


for col in X_label_encoded.select_dtypes(include='object').columns:
    le = LabelEncoder()
    X_label_encoded[col] = le.fit_transform(X_label_encoded[col])
    test_cv[col]=le.transform(test_cv[col])



scores, test_preds = [], []
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)


for i, (trn_idx, test_idx) in enumerate(skf.split(X_label_encoded, y)):
    
    X_train, X_test = X_label_encoded.iloc[trn_idx], X_label_encoded.iloc[test_idx]
    y_train, y_test = y.iloc[trn_idx], y.iloc[test_idx]

    X_train = X_train.drop(columns=['year'], axis=1)
    X_test = X_test.drop(columns=['year'], axis=1)

    lgb_md = LGBMRegressor(**lgb_params).fit(X_train, y_train)
    lgb_pred = lgb_md.predict(X_test)

    mape_oof = mean_absolute_percentage_error(y_test, lgb_pred)
    scores.append(mape_oof)
    
    print('Fold', i, '==> LGBM oof MAPE is ==>', mape_oof)

    test_preds.append(lgb_md.predict(test_cv))

kf_avg_score = np.mean(scores)
kf_std_score = np.std(scores)
print("\n")
print(f"The average oof MAPE of the LGBM model is {kf_avg_score}")

print(f"The std oof MAPE of the LGBM model is {kf_std_score}")


submission=pd.read_csv('/kaggle/input/playground-series-s5e1/sample_submission.csv')


submission['num_sold'] = np.expm1(np.mean(test_preds, axis=0))
print(submission.head())

submission.to_csv('submission.csv',index=False)




