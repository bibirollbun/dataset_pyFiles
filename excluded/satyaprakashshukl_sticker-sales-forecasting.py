import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from lightgbm import LGBMRegressor
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
from scipy.stats import t
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.metrics import mean_absolute_error,mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import KFold
import warnings
warnings.filterwarnings("ignore")


df_train = pd.read_csv('/kaggle/input/gdp-sticker-sales-fore/train_data.csv')
df_test = pd.read_csv('/kaggle/input/gdp-sticker-sales-fore/test_data.csv')
df_sub = pd.read_csv('/kaggle/input/playground-series-s5e1/sample_submission.csv')


def information(df):    
    
    print("##################### Shape #####################")
    print(df.shape)
    print("##################### Types #####################")    
    print(df.dtypes)
    print("##################### Head #####################")
    print(df.head(4))
    print("##################### Columns #####################")
    print(df.columns)
    print("##################### Null #####################")
    print(df.isnull().sum())   
    print("##################### Describe #####################")
    print(df.describe())
    print("##################### NA #####################")
    
    return df


information(df_train)



#df_train['num_sold'] = df_train['num_sold'].fillna(df_train['num_sold'].median())



df_train.isnull().sum()


df_train.head()


df_test.head()


df_sub.head()


plt.figure(figsize=(10, 6))
sns.histplot(df_train['num_sold'], kde=True, bins=50, color='blue')
plt.title('Distribution of num_sold', fontsize=16)
plt.xlabel('num_sold', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.show()


plt.figure(figsize=(8, 5))
sns.boxplot(df_train['num_sold'], color='green')
plt.title('Boxplot of num_sold', fontsize=16)
plt.xlabel('num_sold', fontsize=14)
plt.show()


def Forecasting_Mini_Course_Sales(df, share_type='num_sold', samples=500, period=24):
    if samples == 'all':
        res = seasonal_decompose(df[share_type].values, period=period)
    else:
        res = seasonal_decompose(df[share_type].values[-samples:], period=period)
    
    observed = res.observed
    trend = res.trend
    seasonal = res.seasonal
    residual = res.resid
    
    fig, axs = plt.subplots(4, figsize=(20,18))
    axs[0].set_title('OBSERVED', fontsize=16)
    axs[0].plot(observed)
    axs[0].grid()
    
    axs[1].set_title('TREND', fontsize=16)
    axs[1].plot(trend)
    axs[1].grid()
    
    axs[2].set_title('SEASONALITY', fontsize=16)
    axs[2].plot(seasonal)
    axs[2].grid()
    
    axs[3].set_title('NOISE', fontsize=10)
    axs[3].plot(residual)
    axs[3].scatter(y=residual, x=range(len(residual)), alpha=0.5)
    axs[3].grid()
    
    plt.show()


#Forecasting_Mini_Course_Sales(df_train, share_type='num_sold', samples=500, period=24)


from statsmodels.tsa.stattools import adfuller

def adfuller_test(sales):
    result=adfuller(sales)
    labels = ['ADF Test Statistic','p-value','#Lags Used','Number of Observations']
    for value,label in zip(result,labels):
        print(label+' : '+str(value) )

    if result[1] <= 0.05:
        print("strong evidence against the null hypothesis(Ho), reject the null hypothesis. Data is stationary")
    else:
        print("weak evidence against null hypothesis,indicating it is non-stationary ")

adfuller_test(df_train['num_sold'])


lag_acf = 48
lag_pacf = 48
height = 4
width = 8

f, ax = plt.subplots(nrows=2, ncols=1, figsize=(width, 2*height))
plot_acf(df_train['num_sold'],lags=lag_acf, ax=ax[0])
plot_pacf(df_train['num_sold'],lags=lag_pacf, ax=ax[1], method='ols')

ax[1].annotate('Strong correlation at lag = 1', xy=(1, 0.25),  xycoords='data',
            xytext=(0.17, 0.75), textcoords='axes fraction',
            arrowprops=dict(color='red', shrink=0.05, width=1))

plt.tight_layout()
plt.show()


df_train = df_train.drop(['id','Unnamed: 0'],axis=1)
df_test = df_test.drop(['id'], axis=1)


df_train.columns


sns.catplot(
    data=df_train, y="store", hue="product", kind="count",
    palette="pastel", edgecolor=".6",
)


sns.catplot(
    data=df_train, y="store", hue="country", kind="count",
    palette="pastel", edgecolor=".6",
)


def extract_date_features(df):
    
    df['date'] = pd.to_datetime(df['date'])    
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['weekday'] = df['date'].dt.weekday 
    df['week'] = df['date'].dt.isocalendar().week
    df['quarter'] = df['date'].dt.quarter
    df['day_of_year'] = df['date'].dt.dayofyear
    df['day_of_week'] = df['date'].dt.dayofweek     
    df['is_weekend'] = (df['weekday'] >= 5).astype(int)    
    df['is_month_start'] = df['date'].dt.is_month_start.astype(int)
    df['is_month_end'] = df['date'].dt.is_month_end.astype(int)    
    df['is_quarter_start'] = df['date'].dt.is_quarter_start.astype(int)
    df['is_quarter_end'] = df['date'].dt.is_quarter_end.astype(int)    
    df['is_year_start'] = df['date'].dt.is_year_start.astype(int)
    df['is_year_end'] = df['date'].dt.is_year_end.astype(int)    
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['weekday_sin'] = np.sin(2 * np.pi * df['weekday'] / 7)
    df['weekday_cos'] = np.cos(2 * np.pi * df['weekday'] / 7)
    df['day_of_year_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
    df['day_of_year_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
    df['week_sin'] = np.sin(2 * np.pi * df['week'] / 52)
    df['week_cos'] = np.cos(2 * np.pi * df['week'] / 52)

    df['Group'] = (df['year'] - 2010) * 48 + df['month'] * 4 + df['day'] // 7

    #if 'hour' not in df.columns:
        #df['hour'] = df['date'].dt.hour
    #df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    #df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

    return df



#df_train = extract_date_features(df_train)
#df_test = extract_date_features(df_test)


#df_train


yearly_sales = df_train.groupby('year')['num_sold'].sum().reset_index()

plt.figure(figsize=(10, 6))
sns.lineplot(data=yearly_sales, x='year', y='num_sold', marker='o', color='red')
plt.title('Yearly Trend of num_sold', fontsize=16)
plt.ylabel('Total num_sold', fontsize=14)
plt.show()


monthly_sales = df_train.groupby('month')['num_sold'].mean().reset_index()

plt.figure(figsize=(10, 6))
sns.barplot(data=monthly_sales, x='month', y='num_sold', palette='viridis')
plt.title('Average num_sold by Month', fontsize=16)
plt.xlabel('Month', fontsize=14)
plt.ylabel('Average num_sold', fontsize=14)
plt.show()


country_sales = df_train.groupby('country')['num_sold'].sum().reset_index().sort_values(by='num_sold', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(data=country_sales, x='num_sold', y='country', palette='coolwarm')
plt.title('Total num_sold by Country', fontsize=16)
plt.xlabel('Total num_sold', fontsize=14)
plt.ylabel('Country', fontsize=14)
plt.show()


store_sales = df_train.groupby('store')['num_sold'].mean().reset_index().sort_values(by='num_sold', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(data=store_sales, x='num_sold', y='store', palette='magma')
plt.title('Average num_sold by Store', fontsize=16)
plt.xlabel('Average num_sold', fontsize=14)
plt.ylabel('Store', fontsize=14)
plt.show()


product_sales = df_train.groupby('product')['num_sold'].sum().reset_index().sort_values(by='num_sold', ascending=False).head(10)

plt.figure(figsize=(12, 6))
sns.barplot(data=product_sales, x='num_sold', y='product', palette='Blues_r')
plt.title('Top 10 Products by Total Sales', fontsize=16)
plt.xlabel('Total num_sold', fontsize=14)
plt.ylabel('Product', fontsize=14)
plt.show()


heatmap_data = df_train.groupby(['year', 'month'])['num_sold'].mean().unstack()

plt.figure(figsize=(12, 8))
sns.heatmap(heatmap_data, annot=True, fmt=".1f", cmap="YlGnBu", linewidths=0.5)
plt.title('Heatmap of Average num_sold by Year and Month', fontsize=16)
plt.xlabel('Month', fontsize=14)
plt.ylabel('Year', fontsize=14)
plt.show()


country_store_sales = df_train.groupby(['country', 'store'])['num_sold'].mean().reset_index()

plt.figure(figsize=(14, 8))
sns.barplot(data=country_store_sales, x='num_sold', y='store', hue='country', palette='Set3')
plt.title('Average num_sold by Store and Country', fontsize=16)
plt.xlabel('Average num_sold', fontsize=14)
plt.ylabel('Store', fontsize=14)
plt.legend(title='Country')
plt.show()


top_products = df_train.groupby('product')['num_sold'].sum().nlargest(5).index
top_product_sales = df_train[df_train['product'].isin(top_products)].groupby(['date', 'product'])['num_sold'].sum().reset_index()

plt.figure(figsize=(16, 8))
sns.lineplot(data=top_product_sales, x='date', y='num_sold', hue='product', marker='o')
plt.title('Sales Trend of Top 5 Products', fontsize=16)
plt.xlabel('Date', fontsize=14)
plt.ylabel('Total num_sold', fontsize=14)
plt.legend(title='Product')
plt.show()


from wordcloud import WordCloud

text = ' '.join(df_train['product'].astype(str).values)
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

plt.figure(figsize=(12, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud of Product Names', fontsize=16)
plt.show()


df_train.drop(columns=['date'], inplace=True)
df_test.drop(columns=['date'], inplace=True)


def label_encode_train_test(df_train, df_test, categorical_columns):
    encoders = {}  
    for col in categorical_columns:
        encoder = LabelEncoder()
        combined_data = pd.concat([df_train[col], df_test[col]], axis=0)
        encoder.fit(combined_data)
        df_train[col] = encoder.transform(df_train[col])
        df_test[col] = encoder.transform(df_test[col]) 
        encoders[col] = encoder    
    return df_train, df_test, encoders

categorical_columns = ['country', 'store', 'product', 'month_country', 'month_store', 'month_product']
df_train_encoded, df_test_encoded, label_encoders = label_encode_train_test(df_train, df_test, categorical_columns)


df_train = df_train_encoded
df_test = df_test_encoded


def target_encoding_for_train_and_test(df_train, df_test, target_column, categorical_columns):
    encoding_map = {}
    
    for col in categorical_columns:
        target_mean = df_train.groupby(col)[target_column].mean()
        encoding_map[col] = target_mean        
        df_train[f'{col}_encoded'] = df_train[col].map(target_mean)
        df_train[f'{col}_encoded'].fillna(df_train[target_column].mean(), inplace=True)    
    for col in categorical_columns:
        df_test[f'{col}_encoded'] = df_test[col].map(encoding_map[col])
        df_test[f'{col}_encoded'].fillna(df_train[target_column].mean(), inplace=True)   
   
    return df_train, df_test
    
#categorical_columns=['country', 'store', 'product']
#df_train, df_test = target_encoding_for_train_and_test(df_train, df_test, target_column='num_sold', categorical_columns=categorical_columns)


print("Training DataFrame with Encoded Columns:")
print(df_train)
print("\nTest DataFrame with Encoded Columns:")
print(df_test)



X = df_train.drop(columns=['num_sold'])
#X = pd.get_dummies(df_train, columns=['country', 'store', 'product'], dtype=int, drop_first=True)
y = np.log1p(df_train['num_sold'])


X_test = df_test


#for col in X_test.columns:
    #if X_test[col].dtype == 'object':  
        #X_test[col] = label_encoder.fit_transform(X_test[col])
#X_test = pd.get_dummies(df_test, drop_first=True,dtype=int)


#X = X.drop(columns=['num_sold'])


scaled_train_data = X
scaled_test_data = X_test



lgb_params={
                'num_leaves': 426,
                 'max_depth': 20,
                 'learning_rate': 0.011353178352988012,
                 'n_estimators': 1000,
                 'metric': 'rmse',
                 'subsample': 0.5772552201954328,
                 'colsample_bytree': 0.9164865430101521,
                 'reg_alpha': 1.48699088003429e-06,
                 'reg_lambda': 0.41539458543414265,
                 'min_data_in_leaf': 73,
                 'feature_fraction': 0.751673655170548,
                 'bagging_fraction': 0.5120415391590843,
                 'bagging_freq': 2,
                 'random_state': 42,
                 'min_child_weight': 0.017236362383443497,
                 'cat_smooth': 54.81317407769262,
                 'verbose' : -1,
                 'early_stopping_rounds': 200,
}





lgbm_predictions = np.zeros(len(scaled_train_data))
lgbm_true_labels = np.zeros(len(scaled_train_data))
lgbm_test_predictions = np.zeros(len(scaled_test_data))

n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(kf.split(scaled_train_data, y)):
    print(f"Training fold {fold + 1}/{n_splits}...")   
    X_train, X_val = scaled_train_data.iloc[train_idx], scaled_train_data.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    lgbm_model = LGBMRegressor(**lgb_params)
    lgbm_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], eval_metric='rmse')
    lgbm_fold_preds = lgbm_model.predict(X_val)
    lgbm_fold_test_preds = lgbm_model.predict(scaled_test_data)
    lgbm_predictions[val_idx] = lgbm_fold_preds
    lgbm_true_labels[val_idx] = y_val 
    lgbm_test_predictions += lgbm_fold_test_preds / n_splits
    fold_mape = mean_absolute_percentage_error(y_val, lgbm_fold_preds)
    print(f"Fold {fold + 1} MAPE: {fold_mape:.4f}")
overall_mape_lgbm = mean_absolute_percentage_error(lgbm_true_labels, lgbm_predictions)
print(f"Overall MAPE (LGBMRegressor): {overall_mape_lgbm:.4f}")


y_pred_original = np.expm1(lgbm_test_predictions)  



lgbm_test_predictions


y_pred_original


df_sub['num_sold'] = y_pred_original
df_sub.to_csv("submission.csv", index=False)
print("submission saved!")


df_sub




