!pip install category_encoders


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import time
sns.set_theme()
warnings.filterwarnings('ignore')

cmap = plt.cm.get_cmap('tab20')
colors = cmap.colors


from sklearn.model_selection import train_test_split, TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from category_encoders import TargetEncoder, CatBoostEncoder
from sklearn.decomposition import PCA
import lightgbm as lgb


train = pd.read_csv('/kaggle/input/playground-series-s5e1/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e1/test.csv')
submission = pd.read_csv('/kaggle/input/playground-series-s5e1/sample_submission.csv')


train.head()


train.info()


# Missing values proportion
print((train.isnull().sum().loc[lambda x: x>0]/len(train)).to_string())


# Missing values imputation
df = train.copy()

for product in df['product'].unique():
  for country in df['country'].unique():
    # Impute missing values using linear interpolation
    sales = df.loc[(df['product'] == product) & (df['country'] == country)]['num_sold']
    sales = sales.interpolate(method='linear')
    sales = sales.fillna(sales.median())
    # Replace original column with imputed values
    df.loc[(df['product'] == product) & (df['country'] == country), 'num_sold'] = sales

train['num_sold'] = df['num_sold']


# Convert date column to datetime format
df = train.copy()
df['date'] = pd.to_datetime(df['date'])

# Set date columns as index
df.set_index('date', inplace=True)
df['year'] = df.index.year
df['quarter'] = df.index.quarter


# Overall daily sales
fig, ax = plt.subplots(figsize=(10,5))

for i, year in enumerate(df.index.year.unique()):
  one_year_sales = df[df.index.year == year]
  one_year_sales = one_year_sales.groupby(one_year_sales.index.dayofyear)['num_sold'].sum()
  ax.plot(one_year_sales.index, one_year_sales.values, label=year, color=colors[i])
  ax.legend()
  ax.set_title('Daily seasonality')
  ax.set_xlabel('Day of the year')
  ax.set_xlim(0,183)
  ax.set_xticks(range(0,183,30))
  ax.set_ylabel('Count')
plt.show()


# Sticker sales per month
fig, ax = plt.subplots(figsize=(10,5))

for i, year in enumerate(df.index.year.unique()):
  _ = df[df.index.year == year]
  _ = _.groupby(_.index.month)['num_sold'].sum()
  ax.plot(_.index, _.values, label=year, color=colors[i])
  ax.legend()
  ax.set_title('Monthly seasonality')
  ax.set_xlabel('Month')
  ax.set_xticks(range(1,13))
  ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
  ax.set_ylabel('Count')
plt.show()


fig, axs = plt.subplots(3, 2, sharey=True, figsize=(15, 10))
fig.suptitle('Quarterly sales')
fig.subplots_adjust(hspace=0.5)
for ax, year in zip(axs.flatten(), df['year'].unique()):
  _ = df[df['year'] == year]
  sns.boxplot(data=_, x='quarter', y='num_sold', ax=ax)
  ax.set_title(f'{year}')
  ax.set_ylabel('Sales volume')
  ax.set_xlabel('Quarters')
plt.tight_layout()


# Weekend sales vs weekdays sales
train.groupby(df.index.dayofweek)['num_sold'].mean().\
  plot(kind='bar', figsize=(6,4),title='Sales are higher on weekend',xlabel='Day of the week', ylabel='Average daily sales')
plt.show()


np.random.seed(42)
train_data = df.loc[df.index.year < 2015].copy()
val_data = df.loc[df.index.year == 2015].copy()
test_data = df.loc[df.index.year == 2016].copy()
y_test = test_data[['id','num_sold']]
test_data = test_data.drop(columns='num_sold')
print((train_data.shape,val_data.shape))


# Transform categorical variables
class categorical_features :
  def __init__(self, train_df, val_df, test_df):
    self.train_df = train_df.copy()
    self.val_df = val_df.copy()
    self.test_df = test_df.copy()
    self.train_encoded = None
    self.val_encoded = None
    self.test_encoded = None
    self.categorical_features = ['country', 'store', 'product']
    self.transformers = []

  def fit(self):
    self.train_cat = self.train_df[self.categorical_features]
    self.val_cat = self.val_df[self.categorical_features]
    self.test_cat = self.test_df[self.categorical_features]
    self.train_df['num_sold'] = np.log(self.train_df['num_sold'])
    self.val_df['num_sold'] = np.log(self.val_df['num_sold'])
    self.train_encoded = pd.DataFrame(self.train_df['id'],index=self.train_cat.index)
    self.val_encoded = pd.DataFrame(self.val_df['id'],index=self.val_cat.index)
    self.test_encoded = pd.DataFrame(self.test_df['id'],index=self.test_cat.index)

  def label_encoding(self):
    label_enc = LabelEncoder()
    for cat in self.train_cat.columns:
      self.train_encoded[cat] = label_enc.fit_transform(self.train_cat[cat])
      self.val_encoded[cat] = label_enc.transform(self.val_cat[cat])
      self.test_encoded[cat] = label_enc.transform(self.test_cat[cat])
    self.transformers.append(label_enc)
    return self.train_encoded, self.val_encoded, self.test_encoded

  def target_encoding(self):
    target_enc = TargetEncoder(smoothing=10)
    for col in self.train_cat.columns:
      self.train_encoded[col] = target_enc.fit_transform(self.train_cat[col], self.train_df['num_sold'])
      self.val_encoded[col] = target_enc.transform(self.val_cat[col])
      self.test_encoded[col] = target_enc.transform(self.test_cat[col])
    self.transformers.append(target_enc)
    return self.train_encoded, self.val_encoded, self.test_encoded

  def one_hot_encoding(self):
    ohe_enc = OneHotEncoder(handle_unknown='ignore')
    pca = PCA(n_components=10)

    self.train_encoded = pd.DataFrame(ohe_enc.fit_transform(self.train_cat).toarray(), columns=ohe_enc.get_feature_names_out(), index=self.train_cat.index)
    self.val_encoded = pd.DataFrame(ohe_enc.transform(self.val_cat).toarray(), columns=ohe_enc.get_feature_names_out(), index=self.val_cat.index)
    self.test_encoded = pd.DataFrame(ohe_enc.transform(self.test_cat).toarray(), columns=ohe_enc.get_feature_names_out(), index=self.test_cat.index)
    self.train_encoded = pd.DataFrame(pca.fit_transform(self.train_encoded), columns=[f'pca_{i}' for i in range(10)], index=self.train_cat.index)
    self.val_encoded = pd.DataFrame(pca.transform(self.val_encoded), columns=[f'pca_{i}' for i in range(10)], index=self.val_cat.index)
    self.test_encoded = pd.DataFrame(pca.transform(self.test_encoded), columns=[f'pca_{i}' for i in range(10)], index=self.test_cat.index)
    self.transformers.append(ohe_enc)
    self.transformers.append(pca)
    self.train_encoded['id'] = self.train_df['id']
    self.val_encoded['id'] = self.val_df['id']
    self.test_encoded['id'] = self.test_df['id']

    return self.train_encoded, self.val_encoded, self.test_encoded

  # Apply transformation
  def categorical_encoding(self, method:str):
    if method == 'label':
      return self.label_encoding()
    elif method == 'target':
      return self.target_encoding()
    elif method == 'one_hot':
      return self.one_hot_encoding()
    else:
      raise ValueError('Invalid encoding method')


# Create datetime features
class datetime_features :
  def __init__(self, train_df, val_df, test_df):
    self.train_df = train_df.copy()
    self.val_df = val_df.copy()
    self.test_df = test_df.copy()
    self.train_calendar_features = None
    self.val_calendar_features = None
    self.test_calendar_features = None
    self.train_agg_features = None
    self.val_agg_features = None
    self.test_agg_features = None
    self.date_col = 'date'
    self.label_encoder = None

  def fit(self):
    # Reset index
    self.train_df = self.train_df.reset_index()
    self.val_df = self.val_df.reset_index()
    self.test_df = self.test_df.reset_index()
    self.date_col = self.train_df.columns[0]
    self.label_encoder = LabelEncoder()
    years = self.label_encoder.fit(range(2010,2017))

  def create_calendar_features(self):
    for df in [self.train_df, self.val_df, self.test_df]:
      date_time_features = {
          'year': df[self.date_col].dt.year,
          'month': df[self.date_col].dt.month,
          'day': df[self.date_col].dt.day,
          'dayofweek': df[self.date_col].dt.dayofweek,
          'quarter': df[self.date_col].dt.quarter,
          'is_weekend': df[self.date_col].dt.dayofweek >= 5
          }
      for col in date_time_features.keys():
        df[col] = date_time_features[col]

      df['year'] = self.label_encoder.transform(df['year'])
      df['is_weekend'] = df['is_weekend'].astype(int)

      # Cyclical encoding
      for col in ['month', 'day', 'dayofweek', 'quarter']:
        max_val = df[col].max()
        df[col + '_sin'] = np.sin(2 * np.pi * df[col]/max_val)
        df[col + '_cos'] = np.cos(2 * np.pi * df[col]/max_val)

    feature_names = ['month_sin', 'month_cos', 'day_sin', 'day_cos', 'dayofweek_sin',
    'dayofweek_cos', 'quarter_sin', 'quarter_cos', 'is_weekend','year','id']
    self.train_calendar_features = self.train_df[feature_names]
    self.val_calendar_features = self.val_df[feature_names]
    self.test_calendar_features = self.test_df[feature_names]

    return self.train_calendar_features, self.val_calendar_features, self.test_calendar_features

  def create_time_agg_features(self,return_roll=False):
    self.train_df = self.train_df.sort_values(by=['product', 'country', 'date'])
    self.val_df = self.val_df.sort_values(by=['product', 'country', 'date'])
    self.test_df = self.test_df.sort_values(by=['product', 'country', 'date'])
    # Set id columns as index
    self.train_df = self.train_df.set_index('id')
    self.val_df = self.val_df.set_index('id')
    self.test_df = self.test_df.set_index('id')

    # Concatenate train, val, test sets and get indices
    all_data = pd.concat([self.train_df, self.val_df, self.test_df])
    train_indices = self.train_df.index
    val_indices = self.val_df.index
    test_indices = self.test_df.index

    # Create lag features
    for product in all_data['product'].unique():
      for country in all_data['country'].unique():
        mask = (all_data['product'] == product) & (all_data['country'] == country)
        yearly_lags = [1,2,3]
        monthly_lags = [1,2,3]
        for year in yearly_lags:
          all_data.loc[mask, f'sales_lag_year_{year}'] = all_data.loc[mask, 'num_sold'].shift(year * 365)
          all_data.loc[mask, f'sales_lag_year_{year}'] = all_data.loc[mask, f'sales_lag_year_{year}'].fillna(all_data.loc[mask, 'num_sold'].median())

        for month in monthly_lags:
          all_data.loc[mask, f'sales_lag_month_{month}'] = all_data.loc[mask, 'num_sold'].shift(month*30) 
          all_data.loc[mask, f'sales_lag_month_{month}'] = all_data.loc[mask, f'sales_lag_month_{month}'].fillna(all_data.loc[mask, 'num_sold'].median())

        # Same month last year +/- 1 month
        year, month = 1, 1
        all_data.loc[mask, f'sales_lag_{year}+{month}'] = all_data.loc[mask, 'num_sold'].shift(year*365+month*30)
        all_data.loc[mask, f'sales_lag_{year}-{month}'] = all_data.loc[mask, f'num_sold'].shift(year*365-month*30)
        all_data.loc[mask, f'sales_lag_{year}+{month}'] = all_data.loc[mask, f'sales_lag_{year}+{month}'].\
          fillna(all_data.loc[mask, 'num_sold'].median())
        all_data.loc[mask, f'sales_lag_{year}-{month}'] = all_data.loc[mask, f'sales_lag_{year}-{month}'].\
          fillna(all_data.loc[mask, 'num_sold'].median())

    # Create rolling features
    stats = {
        'mean': np.mean,
        'std': np.std,
        'min': np.min,
        'max': np.max,
        'median': np.median
      }
    for product in all_data['product'].unique():
      for country in all_data['country'].unique():
        mask = (all_data['product'] == product) & (all_data['country'] == country)
        short_windows = [3,6] # Monthly data
        long_windows = [12,24] # Yearly data
        all_windows = short_windows + long_windows

        for window in all_windows:
          roll = all_data.loc[mask, 'num_sold'].rolling(window=window, min_periods=1)
          # Calculate statistics for each window
          for stat_name, stat_func in stats.items():
            all_data.loc[mask, f'num_sold_rolling_{stat_name}_{window}'] = roll.apply(stat_func)

    drop_columns = ['country', 'store', 'product', 'num_sold', 'year', 'quarter','date', 'month', 'day', 'dayofweek', 'is_weekend', 'month_sin',
    'month_cos', 'day_sin', 'day_cos', 'dayofweek_sin', 'dayofweek_cos','quarter_sin', 'quarter_cos']
    lag_features = [col for col in all_data.columns if 'lag' in col]
    roll_features = [col for col in all_data.columns if 'rolling' in col]

    self.train_agg_features = all_data.loc[train_indices].drop(columns=drop_columns)
    self.val_agg_features = all_data.loc[val_indices].drop(columns=drop_columns)
    self.test_agg_features = all_data.loc[test_indices].drop(columns=drop_columns)
    # Feature available in the training set
    train_lag_features = all_data[lag_features].loc[train_indices]
    val_lag_features = all_data[lag_features].loc[val_indices]
    test_lag_features = all_data[lag_features].loc[test_indices]
    # Feature computed with expanding windows
    train_roll_features = all_data[roll_features].loc[train_indices]
    val_roll_features = all_data[roll_features].loc[val_indices]
    test_roll_features = all_data[roll_features].loc[test_indices]

    if return_roll :
      return self.train_agg_features, self.val_agg_features, self.test_agg_features
    else :
      return train_lag_features, val_lag_features, test_lag_features


# Model architecture, training and optimization
class modelization :
  def __init__(self, train_calendar_features,val_calendar_features,test_calendar_features,
               train_cat_features, val_cat_features, test_cat_features, test_df, train_df, val_df,
               train_agg_features,val_agg_features,test_agg_features
               ):
    self.train_df = train_df
    self.val_df = val_df
    self.test_df = test_df
    # Input features
    self.train_calendar_features = train_calendar_features
    self.val_calendar_features = val_calendar_features
    self.test_calendar_features = test_calendar_features
    self.train_agg_features = train_agg_features
    self.val_agg_features = val_agg_features
    self.test_agg_features = test_agg_features
    self.train_cat_features = train_cat_features
    self.val_cat_features = val_cat_features
    self.test_cat_features = test_cat_features
    self.train_date_col = self.train_df.index
    self.val_date_col = self.val_df.index
    self.test_date_col = self.test_df.index
    self.train_features = None
    self.val_features = None
    self.test_features = None
    # Model architecture
    self.model = lgb.LGBMRegressor()
    self.baseline_score = None
    self.default_params = self.model.get_params()
    self.rng = np.random.RandomState(1338)
    self.default_params['random_state'] = self.rng
    self.y_train = self.train_df[['id','num_sold']]
    self.y_val = self.val_df[['id','num_sold']]
    self.y_pred = None
    self.best_params = None
    self.best_score = None
    self.best_model = None
    self.submission = None

  # Concatenate input features
  def fit(self):
    self.y_train = self.y_train.set_index('id')
    self.y_val = self.y_val.set_index('id')
    self.train_cat_features = self.train_cat_features.set_index('id')
    self.val_cat_features = self.val_cat_features.set_index('id')
    self.train_calendar_features = self.train_calendar_features.set_index('id')
    self.val_calendar_features = self.val_calendar_features.set_index('id')
    self.test_cat_features = self.test_cat_features.set_index('id')
    self.test_calendar_features = self.test_calendar_features.set_index('id')
    # Concatenate input features
    self.train_features = pd.concat([self.train_cat_features, self.train_calendar_features, self.train_agg_features], axis=1) # Remove .reset_index() calls
    self.val_features = pd.concat([self.val_cat_features, self.val_calendar_features, self.val_agg_features], axis=1) # Remove .reset_index() calls
    self.test_features = pd.concat([self.test_cat_features, self.test_calendar_features, self.test_agg_features], axis=1)
    # Convert dtypes to float32
    self.train_features = self.train_features.astype('float32')
    self.val_features = self.val_features.astype('float32')
    self.test_features = self.test_features.astype('float32')


  # Define a baseline model
  def baseline_model(self):
    baseline_scores = []
    for feature in self.train_agg_features.columns :
      self.y_train_pred = self.train_features[feature]
      mape = mean_absolute_percentage_error(self.y_train, self.y_train_pred)
      baseline_scores.append(mape)
    self.baseline_score =  np.min(baseline_scores)
    baseline_features = self.train_agg_features.columns[np.argmin(baseline_scores)]
    print(f'Baseline feature : {baseline_features}')
    print(f'Baseline model MAPE: {self.baseline_score:.5f}')

    return self.baseline_score

  # Annual time series C.V
  def time_series_cv(self):
    """Implement annual cross validation framework."""
    years = self.train_df.index.year.unique()
    years = np.sort(years)
    folds = []
    for i in range(len(years) - 1):
      train_indices = self.train_df.loc[self.train_df.index.year.isin(years[:i+1])]['id']
      # Filter the index based on the next year
      test_indices = self.train_df.loc[self.train_df.index.year == years[i+1]]['id']

      folds.append((train_indices, test_indices))
    return folds

  def train_and_evaluate(self, folds):
    train_scores = []
    val_scores = []
    for train_index, val_index in folds:
      X_train_fold, X_val_fold = self.train_features.loc[train_index], self.train_features.loc[val_index]
      y_train_fold, y_val_fold = self.y_train.loc[train_index], self.y_train.loc[val_index]
      self.model.fit(X_train_fold, y_train_fold)
      y_train_pred = self.model.predict(X_train_fold)
      y_val_pred = self.model.predict(X_val_fold)
      train_scores.append(mean_absolute_percentage_error(y_train_fold, y_train_pred))
      val_scores.append(mean_absolute_percentage_error(y_val_fold, y_val_pred))
    return train_scores, val_scores

  # Feature importance analysis
  def feature_importance(self):
    feature_importance = pd.DataFrame({'feature': self.train_features.columns, 'importance': self.model.feature_importances_})
    feature_importance = feature_importance.sort_values(by='importance', ascending=False)
    return feature_importance


# Datetime feature extraction : calendar features + time aggregation features
datetime_columns = datetime_features(train_data, val_data,test_data)
datetime_columns.fit()
train_calendar_features, val_calendar_features, test_calendar_features = datetime_columns.create_calendar_features()
train_agg_features, val_agg_features, test_agg_features = datetime_columns.create_time_agg_features(return_roll=True)


print(train_calendar_features.shape, val_calendar_features.shape, test_calendar_features.shape)
print(train_agg_features.shape, val_agg_features.shape, test_agg_features.shape)


# Visualize datetime encoding
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.scatter(train_calendar_features['month_sin'], train_calendar_features['month_cos'])
plt.title('Month Cyclical Encoding')
plt.xlabel('sin(month)')
plt.ylabel('cos(month)')
plt.tight_layout()

plt.subplot(1, 2, 2)
plt.scatter(train_calendar_features['day_sin'], train_calendar_features['day_cos'])
plt.title('Day Cyclical Encoding')
plt.xlabel('sin(dayofweek)')
plt.ylabel('cos(dayofweek)')

plt.tight_layout()


to = time.time()
train_scores = {'one_hot':[],'label': [], 'target': []}
val_scores = {'one_hot':[],'label': [], 'target': []}
for method in ['one_hot','label', 'target']:
  # Categorical encoding
  cat_data = categorical_features(train_data, val_data, test_data)
  cat_data.fit()
  train_cat_cols, val_cat_cols, test_cat_cols = cat_data.categorical_encoding(method)
  model = modelization(
      train_calendar_features,  val_calendar_features,test_calendar_features,
      train_cat_cols, val_cat_cols, test_cat_cols, test_data, train_data, val_data,
      train_agg_features,val_agg_features,test_agg_features
  )
  model.fit()
  folds = model.time_series_cv()
  train_cv_scores, val_cv_scores = model.train_and_evaluate(folds)
  train_scores[method] = train_cv_scores
  val_scores[method] = val_cv_scores
t1 = time.time()


# Input features dimensionality
model.train_features.shape


baseline_score = model.baseline_model()


# Improvement over naive approach
print(f'Training time: {t1-to:.2f}s')
for method,score in val_scores.items():
  score = np.mean(score)/baseline_score
  print(f'{method.capitalize()}: {score:.5f}')


# Plot cross validation results
fig, (ax1,ax2) =plt.subplots(1, 2, figsize=(15, 5), sharex=True)

ax1 = sns.lineplot(data=train_scores, ax=ax1, legend=True)
ax1.set_title('Train scores')
ax1.set_xlabel('CV iteration')
ax1.set_xticks(range(4))
ax1.set_ylabel('MAPE')

ax2 = sns.lineplot(data=val_scores, ax=ax2, legend=True)
ax2.set_title('Validation scores')
ax2.set_xlabel('CV iteration')
ax2.set_xticks(range(4))
ax2.set_ylabel('MAPE')
plt.show()


# Check if the models generalize well
data = {k: [np.mean(train_scores[k]), np.mean(val_scores[k])] for k in train_scores.keys()}

fig, ax = plt.subplots(figsize=(5, 3))
ax = sns.barplot(x=list(data.keys()), y=[data[k][0] for k in data.keys()], label='Train', color=colors[0])
ax = sns.barplot(x=list(data.keys()), y=[data[k][1] for k in data.keys()], label='Validation', color=colors[1])
ax.set_title('Model generalization capability')
ax.set_xlabel('Method')
ax.set_ylabel('MAPE')
plt.legend(loc='lower right')
plt.show()


# Feature importance analysis
feature_importance = model.feature_importance()
feature_importance.head(10).plot(kind='barh', x='feature', y='importance', figsize=(6, 4))
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Feature Importance')
plt.show()


cat_data = categorical_features(train_data, val_data, test_data)
cat_data.fit()
train_cat_cols, val_cat_cols, test_cat_cols = cat_data.categorical_encoding('target') # Best encoding method


model = modelization(
    train_calendar_features=train_calendar_features, val_calendar_features=val_calendar_features,test_calendar_features=test_calendar_features,
    train_cat_features=train_cat_cols, val_cat_features=val_cat_cols, test_cat_features=test_cat_cols,
    test_df=test_data, train_df=train_data, val_df=val_data,
    train_agg_features=train_agg_features, val_agg_features=val_agg_features, test_agg_features=test_agg_features
)
model.fit()
# Apply feature selection
important_features = feature_importance.head(5)['feature'].values
model.train_features = model.train_features[important_features]
model.val_features = model.val_features[important_features]
model.test_features = model.test_features[important_features]
train_cv_scores, val_cv_scores = model.train_and_evaluate(folds)


# Optimized C.V score
model.y_pred = model.model.predict(model.val_features)
score = mean_absolute_percentage_error(model.y_val, model.y_pred)
print(f'Optimized CV MAPE: {score:.5f}')
print(f'Improvement over baseline model : {score/baseline_score:.5f}')

