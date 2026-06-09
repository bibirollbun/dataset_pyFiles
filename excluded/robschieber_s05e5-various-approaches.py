import os
import pandas as pd
import numpy as np
import kagglehub
from matplotlib import pyplot as plt
from sklearn.utils import resample
from google.colab import data_table

try:
  import catboost
except Exception:
  !pip install catboost
  import catboost

VERSION = 'v10'

playground_series_s5e5_path = kagglehub.competition_download('playground-series-s5e5')
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 1000)

def calculate_bmi(height_cm, weight_kg):
  height_m = height_cm / 100
  bmi = weight_kg / (height_m ** 2)
  return bmi

def categorize_bmi(bmi):
  if bmi < 24.4:
      return 'Normal'
  else:
      return 'Overweight'

def get_bmr(row):
  if row['Sex_Male']:  # Male
    return 66.5 + (13.75 * row['Weight']) + (5.003 * row['Height']) - (6.75 * row['Age'])
  else:  # Female
    return 655.1 + (9.563 * row['Weight']) + (1.850 * row['Height']) - (4.676 * row['Age'])

def get_bmr_miff(row): # Mifflin-St Jeor Equation
  if row['Sex_Male']:  # Male
      return (10 * row['Weight']) + (6.25 * row['Height']) - (5 * row['Age'])
  else:  # Female
      return (10 * row['Weight']) + (9.563 * row['Height']) - (4.676 * row['Age'])

def encode_categorical(data):
  data['Body_Temp_Diff'] = (data['Body_Temp'] - 37).astype('float32')
  #data['Body_Temp_Class'] = pd.cut(data['Body_Temp_Diff'], bins=[1,3,5])

  data['Age_Group'] = pd.cut(data['Age'], bins=[19.99, 30, 40, 60, 70, 80],
                             labels=['20-30', '31-40', '41-60', '61-70', '71-80'])
  data['Sex'] = (data['Sex'] == 'female').astype('int8')

  data['BMI'] = data.apply(lambda x: calculate_bmi(height_cm=x['Height'], weight_kg=x['Weight']), axis=1)
  data['BMI_Class'] = data['BMI'].apply(categorize_bmi)
  data['BMI_Class'] = pd.Categorical(data['BMI_Class'], categories=[
      'Normal', 'Overweight'
  ])

  return data
    
def build_cal_per_min_table():
    """Build a calories per min table"""
    path = os.path.join(playground_series_s5e5_path, 'train.csv')
    data = pd.read_csv(path)
    
    data['Duration'] = data['Duration'].astype('int8')
    data = encode_categorical(data)
    
    data['Calories_per_Minute'] = (data['Calories'] / data['Duration']).astype('float16')
    
    data = data[data['Calories_per_Minute'] <= 10]
    
    data = data.drop(['id','Age', 'BMI_Class', 'Heart_Rate', 'Body_Temp', 'BMI', "Height", "Weight", 'Calories' ], axis=1)
    
    data = data.groupby(['Duration', 'Age_Group', 'Sex' ])['Calories_per_Minute']\
    .agg( ['mean','std','median','var']).rename(columns={
        #'count': 'Calories_per_Minute_count',
        #'max': 'Calories_per_Minute_max',
        #'min': 'Calories_per_Minute_min',
        'mean': 'Calories_per_Minute_mean',
        'std': 'Calories_per_Minute_std',
        'median': 'Calories_per_Minute_median',
        'var': 'Calories_per_Minute_var'
    })
    return data

def get_data(path, lookup_table, do_resample=False):
    is_train = False
    
    data = pd.read_csv(path)
    
    if 'Calories' in data.columns:
        is_train = True
    
    data['Duration'] = data['Duration'].astype('int8')
    
    if do_resample:
        data = resample(train_data, n_samples=5000, replace=False, random_state=42)
    
    len_before = len(data)
    data = encode_categorical(data)
    data = pd.merge(data, lookup_table, how='left',
                  left_on=['Duration','Sex', 'Age_Group' ],
                  right_on=['Duration','Sex', 'Age_Group' ])
    
    assert(len(data) == len_before)
  
    #data['Estimated_Calories'] = data['Calories_per_Minute'] * data['Duration']
    
    # Sex
    data = pd.get_dummies(data, columns=['Age_Group'], dtype='int8')
    #data['Estimated_Calories'] = (data['Duration'] * np.round(data['Calories_per_Minute_mean'],0)).astype('float32')
    data['Sex_Male'] = (data['Sex'] == 0).astype('int8')
    data['Sex_Female'] = (data['Sex'] == 1).astype('int8')
    #data['BMI_Normalweight'] = (data['BMI_Class'] == 'Normal').astype('int8')
    #data['BMI_Overweight'] = (data['BMI_Class'] == 'Overweight').astype('int8')

    # HR, Body Temp
    data['HR_Max'] = 220 - data['Age']
    data['HR_Ratio'] = (data['Heart_Rate'] / data['HR_Max']).astype('float32')
    data['Body_Temp_Ratio'] = (data['Body_Temp'] / data['Body_Temp'].max()).astype('float16')
    data['Intensity'] = (data['HR_Ratio'] * (data['Body_Temp_Diff'])**2).astype('float32')
    data['BMR'] = data.apply(lambda x: get_bmr(x), axis=1).astype('float32')
    data['BMR_per_Minute'] = ((data['BMR'] / 24) / 60).astype('float32')
    data['Base_Calories'] = data['BMR_per_Minute'] * data['Duration']
    data['Duration_x_Body_Temp_Diff'] = (data['Duration'] * data['Body_Temp_Diff']).astype('float32')
    data['Duration_x_Heart_Rate'] = (data['Duration'] * data['Heart_Rate']).astype('float32')
    data['Duration_x_HR_Ratio'] = (data['Duration'] * data['HR_Ratio']).astype('float32')
    #data['Duration_x_HR_Ratio_Recip'] = np.reciprocal(data['Duration_x_HR_Ratio']).astype('float32')
    data['HR_Ratio_x_Body_Temp_Diff'] = data['HR_Ratio'] * data['Body_Temp_Diff']
    #data['Duration_Recip'] = (1 / data['Duration']).astype('float32')
    data['Log_Duration'] = np.log1p(data['Duration']).astype('float32')
    data['Very_Short_Duration'] = (data['Duration'] == 1).astype('int8')
    data['Short_Duration'] = ((data['Duration'] > 1) & (data['Duration'] < 5)).astype('int8')
    return data


df_cal_per_min = build_cal_per_min_table()
train_data = get_data(os.path.join(playground_series_s5e5_path, 'train.csv'), df_cal_per_min, False)
test_data = get_data(os.path.join(playground_series_s5e5_path, 'test.csv'), df_cal_per_min, False)

train_data.head()


import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.metrics import make_scorer
import matplotlib.pyplot as plt


def rmsle(y_true, y_pred):
    y_pred = np.clip(y_pred, a_min=0, a_max=None)
    y_true = np.clip(y_true, a_min=0, a_max=None)
    return np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_true))**2))


X = train_data.drop(['id',  'Calories'], axis=1)
X_test = test_data.drop(['id'], axis=1)
y = train_data['Calories']
y_log = np.log1p(y)  # Log-transform target for training


numeric_features = X.select_dtypes(include=np.number).columns.tolist()
categorical_features = X.select_dtypes(exclude=np.number).columns.tolist()
features = numeric_features + categorical_features


X = X[numeric_features]
X_test = X_test[numeric_features]


model = LinearRegression()

# Set up 3-fold cross-validation
cv = KFold(n_splits=3, shuffle=True, random_state=42)

# Store scores and test predictions
scores = []
fold = 1
predictions = np.zeros(len(X_test))

# Perform cross-validation
for train_idx, test_idx in cv.split(X):
    X_train, X_val = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_val = y_log.iloc[train_idx], y_log.iloc[test_idx]

    # Train model on training fold
    model.fit(X_train, y_train)

    # Predict on validation fold (log scale)
    y_pred_val = model.predict(X_val)

    # Calculate RMSLE on original scale
    y_val_orig = np.expm1(y_val)  # Convert back to original scale
    y_pred_val_orig = np.expm1(y_pred_val)
    fold_score = rmsle(y_val_orig, y_pred_val_orig)
    scores.append(fold_score)

    print(f'Fold {fold}:')
    print(f'  Training indices: {train_idx}')
    print(f'  Test indices: {test_idx}')
    print(f'  RMSLE: {fold_score:.4f}')

    # Test predictions (in log scale, then convert back)
    predictions += np.expm1(model.predict(X_test)) / 3  # Use n_splits=3
    fold += 1

# Print average and standard deviation of RMSLE
print(f'\nAverage RMSLE: {np.mean(scores):.4f} (±{np.std(scores):.4f})')

# Train the model on all data
model.fit(X, y_log)  # Train on log-transformed target

# Make predictions
y_pred = np.expm1(model.predict(X))  # Convert back to original scale
y_pred_test = np.expm1(model.predict(X_test))

# Print coefficients
print(f'Slope (for first feature): {model.coef_[0]:.2f}')
print(f'Intercept: {model.intercept_:.2f}')


plt.scatter(X['Duration'][:1000], y[:1000], color='blue', label='Data points', s=2)
plt.scatter(X['Duration'][:1000], y_pred[:1000], color='red', label='Linear regression', s=2)
plt.xlabel('Duration')
plt.ylabel('Calories')
plt.title('Linear Regression with RMSLE Cross-Validation')
plt.legend()
plt.show()

# Save submission
submission = pd.DataFrame({'id': test_data['id'], 'Calories': predictions})
submission_file = 'submission_linear_regression_v1.csv'  # Define VERSION explicitly
submission.to_csv(submission_file, index=False)
print(f"Submission file '{submission_file}' created!")


import tqdm
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_log_error
import os
import datetime

drop_columns = ['Height', 'Sex']
# Prepare data
X = train_data.drop(['id', 'Calories'] + drop_columns, axis=1)
X_test = test_data.drop(['id'] + drop_columns, axis=1)
y = train_data['Calories']
y_log = np.log1p(y)
num_folds = 10

# Define features
numeric_features = X.select_dtypes(include=np.number).columns.tolist()
categorical_features = X.select_dtypes(exclude=np.number).columns.tolist()
features = numeric_features + categorical_features

X = X[numeric_features]
X_test = X_test[numeric_features]

skf = KFold(n_splits=num_folds, shuffle=True, random_state=42)

cb_params = {
    'iterations': 3000,
    'depth': 10,
    'learning_rate': 0.03,
    'l2_leaf_reg': 2,
    'bagging_temperature': .01,
    'random_seed': 42,
    'task_type': 'GPU',
    'devices': '0',
    'verbose': 100,
    'early_stopping_rounds':50,
    'task_type': 'GPU',
    'devices': '0',
}

predictions = np.zeros(len(X_test))
cv_scores = []

catboost_models = []
X_val_splits = []
y_val_splits = []

# Train and predict with CV
# Train and predict with CV
for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y_log.iloc[train_idx], y_log.iloc[val_idx]
    #catboost = CatBoostRegressor(**cb_params)
    model = CatBoostRegressor(**cb_params)
    # Initialize Pool

    weights = np.ones(len(X_train))
    weights[X_train['Duration'] <= 4] *= 3.5
    weights[X_train['Age'] <= 30] *= 2
    weights[X_train['Sex_Male'] == 1] *= 1.5  

    train_pool = Pool(X_train, y_train, weight=weights)
    test_pool = Pool(X_val)

    model.fit(train_pool, eval_set=(X_val, y_val), use_best_model=True)

    # Compute validation RMSLE
    val_pred = np.expm1(model.predict(test_pool))
    val_pred = np.clip(val_pred, 1, 500)

    rmsle = np.sqrt(mean_squared_log_error(np.expm1(y_val), val_pred))
    cv_scores.append(rmsle)
    cb_importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)

    print(f"Fold {fold} RMSLE: {rmsle:.6f}\n")
    print(f"Fold {fold} Top 10 Feature Importances:\n{cb_importances}\n")
    print(f"Fold {fold} Validation Set Calories:")
    print(f"{np.expm1(y_val).describe()}\n")
    print(f'CV distribution:\n{pd.cut(np.expm1(y_val), [0, 10, 50, 100, 200], right=False).value_counts()}\n')

    # Test predictions using X_test (not test_pool)
    test_pool_full = Pool(X_test)  # Create a Pool for the full test set
    test_pred = np.expm1(model.predict(test_pool_full))  # Predict on X_test
    predictions += test_pred / num_folds

    # Collect fold artifacts for SHAP
    catboost_models.append(model)
    X_val_splits.append(X_val)
    y_val_splits.append(y_val)

# Print CV results
mean_cv_score = np.mean(cv_scores)
std_cv_score = np.std(cv_scores)
print(f"CatBoost CV RMSLE: {mean_cv_score:.6f} ± {std_cv_score:.6f}")

# Save submission
submission = pd.DataFrame({'id': test_data['id'], 'Calories': predictions})
submission_file = f'submission.csv'
submission.to_csv(submission_file, index=False)
print(f"Submission file '{submission_file}' created!")

plt.scatter(X['Duration'][:1000], y[:1000], color='blue', marker='x', label='Data points', s=2)
plt.scatter(X_test['Duration'][:1000], predictions[:1000], color='red', label='Linear regression', s=2, alpha=0.5)
plt.xlabel('Duration')
plt.ylabel('Calories')
plt.title('Catboost with RMSLE Cross-Validation')
plt.legend()
plt.show()


import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_log_error
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor, DMatrix, train
import os
import datetime

# Custom RMSLE evaluation metric for XGBoost
def rmsle_metric(y_true, y_pred):
    y_true_exp = np.expm1(y_true)
    y_pred_exp = np.clip(np.expm1(y_pred), a_min=0, a_max=None)  # Avoid negative values
    log_true = np.log1p(y_true_exp)
    log_pred = np.log1p(y_pred_exp)
    rmsle = np.sqrt(np.mean((log_true - log_pred) ** 2))
    return 'rmsle', rmsle, False  # False: lower is better

# Define RMSLE computation functions (no clipping, as requested)
def compute_rmsle(y_true, y_pred):
    y_true_exp = np.expm1(y_true)
    y_pred_exp = np.expm1(y_pred)
    log_true = np.log1p(y_true_exp)
    log_pred = np.log1p(y_pred_exp)
    rmsle = np.sqrt(np.mean((log_true - log_pred) ** 2))
    assert np.isfinite(rmsle), "Non-finite RMSLE detected"
    return rmsle

def compute_rmsle_per_sample(y_true, y_pred):
    y_true_exp = np.expm1(y_true)
    y_pred_exp = np.expm1(y_pred)
    log_true = np.log1p(y_true_exp)
    log_pred = np.log1p(y_pred_exp)
    return np.sqrt((log_true - log_pred) ** 2)


X = train_data.drop(['id', 'Calories'], axis=1)
numeric_features = X.select_dtypes(include=np.number).columns
print(f'numeric features: {numeric_features}')

# Prepare data

X = X[numeric_features]
X_test = test_data[numeric_features]

y = train_data['Calories']
y_log = np.log1p(y)

# XGBoost parameters
xgb_params = {
    'n_estimators': 3000,
    'learning_rate': 0.02,
    'max_depth': 10,
    'reg_lambda': 1.0,
    'subsample': 0.9,
    'colsample_bytree':0.7,
    'gamma': .01,
    'max_delta_step':2,
    'tree_method': 'gpu_hist',
    'device': 'cuda',
    'early_stopping_rounds': 100,
    'random_state': 42,
    'verbosity': 2
}

# K-fold CV
num_folds = 10
skf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
fold_rmsles = []
oof_xgboost = np.zeros(len(X))
test_predictions = np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
    print(f"\nTraining Fold {fold}...")

    # Split data
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y_log.iloc[train_idx], y_log.iloc[val_idx]

    # Train XGBoost
    dtrain = DMatrix(X_train, label=y_train)
    dval = DMatrix(X_val, label=y_val)

    xgb_model = train(
        xgb_params,
        dtrain,
        num_boost_round=1500,
        evals=[(dval, 'eval')],
        early_stopping_rounds=50,
        verbose_eval=100
    )

    # OOF predictions
    xgb_pred = xgb_model.predict(DMatrix(X_val))
    oof_xgboost[val_idx] = xgb_pred

    fold_rmsle = compute_rmsle(y_val, xgb_pred)
    fold_rmsles.append(fold_rmsle)
    print(f"Fold {fold} RMSLE: {fold_rmsle:.6f}")

    # Test predictions
    dtest = DMatrix(X_test)
    test_pred = xgb_model.predict(DMatrix(X_test))
    test_predictions += test_pred / num_folds

# Compute and log overall RMSLE
mean_rmsle = np.mean(fold_rmsles)
std_rmsle = np.std(fold_rmsles)
print(f"\nMean RMSLE: {mean_rmsle:.6f} ± {std_rmsle:.6f}")

# Save submission
submission = pd.DataFrame({
    'id': test_data['id'],
    'Calories': np.expm1(test_predictions)  # Convert back from log
})
submission_file = f'submission_xgboost_{VERSION}.csv'
submission.to_csv(submission_file, index=False)
print(f"Submission file '{submission_file}' created!")


from matplotlib import pyplot as plt
import seaborn as sns

numerical_features = ["Duration"]

for feature in numerical_features:
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    sns.histplot(train_data[feature], kde=True, bins=30)
    plt.title(f"Histogram of {feature}")
    plt.xlabel(feature)
    plt.ylabel("Frequency")

    plt.subplot(1, 2, 2)
    sns.boxplot(x=train_data[feature])
    plt.title(f"Box Plot of {feature}")

    plt.tight_layout()
    plt.show()

    print(f"\nStatistics for {feature}:")



from sklearn.utils import resample
import seaborn as sns
sns.set_theme()
data = resample(train_data, n_samples=500)
data = data[data['Duration'] < 25]
# Plot sepal width as a function of sepal_length across days
g = sns.catplot(
    data=data, hue='Age',
    x="Duration", y="Calories"
)
sns.regplot(
    data=data, x="Duration", y="Calories",
    scatter=False, truncate=False, order=2, color="green",
)


# Use more informative axis labels than are provided by default


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


#  Calculate calories per minute
train_data['Calories_per_minute'] = train_data['Calories'] / train_data['Duration']

# Bin Duration into 30 bins
num_bins = 30
train_data['Duration_bin'] = pd.cut(
    train_data['Duration'],
    bins=num_bins,
    labels=[f'{i+1}' for i in range(num_bins)],
    include_lowest=True
)

# Compute mean calories per minute by Duration bin
mean_by_duration = train_data.groupby('Duration_bin', observed=True)['Calories_per_minute'].mean()

# Calculate slope (first derivative)
slopes = mean_by_duration.diff().fillna(0)  # Difference between consecutive bins
slopes.name = 'Slope'

# Calculate change in slope (second derivative)
slope_changes = slopes.diff().fillna(0)  # Difference of slopes
slope_changes.name = 'Slope Change'

result_df = pd.DataFrame({
    'Mean Calories per Minute': mean_by_duration,
    'Slope': slopes,
    'Slope Change': slope_changes
})

# Plot 1: Mean Calories per Minute and Slope
fig, ax1 = plt.subplots(figsize=(12, 6))
ax1.plot(result_df.index, result_df['Mean Calories per Minute'], marker='o', color='blue', label='Mean Calories per Minute')
ax1.set_xlabel('Duration Bin (1-minute intervals)')
ax1.set_ylabel('Mean Calories per Minute', color='blue')
ax1.tick_params(axis='y', labelcolor='blue')
ax1.set_xticks(range(0, len(result_df.index), 2))
ax1.set_xticklabels(result_df.index[::2], rotation=45)
ax1.grid(True)

ax2 = ax1.twinx()
ax2.plot(result_df.index, result_df['Slope'], marker='s', color='red', label='Slope')
ax2.set_ylabel('Slope (Change in Calories per Minute)', color='red')
ax2.tick_params(axis='y', labelcolor='red')

fig.suptitle('Mean Calories per Minute and Slope over Duration Bins')
fig.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=2)
plt.tight_layout()
plt.show()

# Plot 2: Slope Change
plt.figure(figsize=(12, 6))
plt.plot(result_df.index, result_df['Slope Change'], marker='^', color='green')
plt.title('Change in Slope of Mean Calories per Minute over Duration Bins')
plt.xlabel('Duration Bin (1-minute intervals)')
plt.ylabel('Slope Change')
plt.xticks(range(0, len(result_df.index), 2), result_df.index[::2], rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()

# Save results
result_df.to_csv('calories_per_duration_slope.csv')
print("Results saved to 'calories_per_duration_slope.csv'")
print("\nResult DataFrame:")
print(result_df)

if 'Calories_per_minute' in train_data.columns:
    train_data.drop('Calories_per_minute', axis=1)


import seaborn as sns

numerical_features = ["HR_Ratio", "Age","Height","Weight",
                      "Heart_Rate","Body_Temp","Calories"]

for feature in numerical_features:
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    sns.histplot(train_data[feature], kde=True, bins=30)
    plt.title(f"Histogram of {feature}")
    plt.xlabel(feature)
    plt.ylabel("Frequency")

    plt.subplot(1, 2, 2)
    sns.boxplot(x=train_data[feature])
    plt.title(f"Box Plot of {feature}")

    plt.tight_layout()
    plt.show()

    print(f"\nStatistics for {feature}:")

