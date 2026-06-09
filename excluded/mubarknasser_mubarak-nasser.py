import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import PowerTransformer, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, mean_squared_log_error
from sklearn.compose import TransformedTargetRegressor
from xgboost import XGBRegressor
import matplotlib.pyplot as pl


train = pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv", index_col = "id")
test = pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv", index_col = "id")


def safe_msle(y_true, y_pred):
    # Convert inputs to numpy arrays
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Clip to ensure non-negativity
    y_pred = np.clip(y_pred, 0, None)
    y_true = np.clip(y_true, 0, None)

    return -math.sqrt(mean_squared_log_error(y_true, y_pred))

safe_msle_scorer = make_scorer(safe_msle, greater_is_better=False)


def get_avg_cv_score(pipeline, features, folds=5, verbose=True, scoring=safe_msle_scorer, data=None, target_column='Calories'):
    """
    Evaluates a pipeline using cross-validation with a custom scorer.

    Parameters:
    - pipeline: sklearn pipeline
    - features: list of feature column names
    - folds: number of CV folds (default 5)
    - verbose: if True, prints per-fold and summary scores
    - scoring: a sklearn-compatible scorer (default: safe_msle_scorer)
    - data: optional DataFrame to use instead of global `train`
    - target_column: name of the target column in the DataFrame

    Returns:
    - dict with individual scores, average, and std deviation
    """
    df = data if data is not None else train

    X = df[features]
    y = df[target_column]

    scores = cross_val_score(pipeline, X, y, cv=folds, scoring=scoring)

    if verbose:
        for i, score in enumerate(scores, start=1):
            print(f"Fold {i}: Score = {score:.5f}")
        print("#####")
        print(f"Average Score: {np.mean(scores):.5f}")
        print(f"Standard Deviation: {np.std(scores):.5f}")

    return {
        'scores': scores,
        'average': np.mean(scores),
        'std_dev': np.std(scores)
    }


data = train.copy()

sns.boxplot(x='Sex', y='Calories', data=data)
plt.title("Calories vs Sex")
plt.show()



print(f"train set shape: {data.shape}")
print(f"test set shape: {test.shape}")

print('\n')
data.info()

print('\n')
data.describe()


ax = data.hist(bins=50, figsize=(15, 12), edgecolor='black')
plt.tight_layout()
plt.show()


print(data['Calories'].skew())


data['Calories'].hist(bins=100, color='skyblue', edgecolor='black')


# Filter only numeric columns
numeric_data = data.select_dtypes(include=[np.number])


# Compute correlation matrix and clean it
corr = numeric_data.corr().replace([np.inf, -np.inf], np.nan).fillna(0)

# Convert correlation matrix to NumPy arrays
corr_values = corr.values
colors = np.where(corr_values < 0, 'red', 'blue')
sizes = np.abs(corr_values) * 2000  # scale for visibility

# Plot the correlogram
plt.figure(figsize=(10, 8))
plt.title("Correlogram with Circles")

for i in range(len(corr)):
    for j in range(len(corr)):
        plt.scatter(i, j, s=sizes[i, j], color=colors[i, j], alpha=0.6)

plt.xticks(range(len(corr)), corr.columns, rotation=90)
plt.yticks(range(len(corr)), corr.columns)
plt.grid(False)
plt.show()


print(data['Body_Temp'].skew())
data['Body_Temp'].hist(bins=100)


# Initialize PowerTransformer without standardizing (only apply power transform)
transformer = PowerTransformer(standardize=False, copy=False)

# Reshape the data to 2D array and apply the power transform to 'Body_Temp'
body_temp_transformed = transformer.fit_transform(data['Body_Temp'].values.reshape(-1, 1))

# Replace the original 'Body_Temp' with transformed values in the dataframe
data['Body_Temp'] = body_temp_transformed.flatten()

# Check and print the skewness after transformation
print(f"Skewness of 'Body_Temp' after Power Transformation: {data['Body_Temp'].skew():.4f}")

# Plot histogram of transformed 'Body_Temp'
plt.figure(figsize=(10, 6))
plt.hist(data['Body_Temp'], bins=100, color='mediumslateblue', edgecolor='black', alpha=0.8)
plt.title("Distribution of 'Body_Temp' After Power Transformation")
plt.xlabel("Transformed Body Temperature")
plt.ylabel("Frequency")
plt.grid(axis='y', alpha=0.7)
plt.tight_layout()
plt.show()


# Drop non-numeric columns (add any other categorical columns here if needed)
numeric_data = data.drop(columns=['Sex'])  

# Calculate correlation matrix with numeric data only
corr_matrix = numeric_data.corr()

# Extract and sort correlations with target 'Calories'
calories_corr = corr_matrix["Calories"].sort_values(ascending=False)

# Print the correlations in a clean format
print("Features correlation with 'Calories':\n")
for feature, corr_value in calories_corr.items():
    print(f"{feature:15} : {corr_value:.4f}")



# Assuming numeric_data is already defined
corr = numeric_data.corr()
plt.figure(figsize=(10, 8))
plt.title("Correlogram with Circles")
# Sizes based on absolute correlation values
sizes = np.abs(corr.values) * 2000  # Convert to NumPy array for indexing
# Colors based on sign of correlation values
colors = np.where(corr.values < 0, 'red', 'blue')
# Plot circles
for i in range(len(corr)):
    for j in range(len(corr)):
        plt.scatter(i, j, s=sizes[i, j], color=colors[i, j], alpha=0.6)

plt.xticks(range(len(corr)), corr.columns, rotation=90)
plt.yticks(range(len(corr)), corr.columns)
plt.grid(False)
plt.show()


# Drop non-numeric columns (e.g., 'Sex') before correlation
numeric_data = data.select_dtypes(include=[np.number])  # only numeric columns

# Now calculate Spearman correlation on numeric data
p_corr_matrix = numeric_data.corr(method='spearman')

# Sort correlations with target 'Calories'
spearman_corr = p_corr_matrix['Calories'].sort_values(ascending=False)

print("Spearman correlation of features with 'Calories':\n")
for feature, corr_value in spearman_corr.items():
    print(f"{feature:15} : {corr_value:.4f}")


sns.heatmap(p_corr_matrix, annot=True)


data["BMI"] = data["Weight"]/((data["Height"]/100) ** 2)
data["BSA"] = ((data["Weight"]* data["Height"])/3600) ** 0.5

# checking correlation
cols_to_drop = ['Age','Duration', 'Body_Temp', 'Heart_Rate', 'Sex']
df = data.drop(cols_to_drop, axis=1)

corr_matrix = df.corr(method='spearman')
corr_matrix["Calories"].sort_values(ascending=False)


strong_corr = ["Duration", "Body_Temp", "Heart_Rate"]
pairs = []


for col1 in strong_corr:
    for col2 in strong_corr:

        cur_pair = [col1, col2]
        cur_pair.sort()
        
        if col1 == col2 or cur_pair in pairs: continue
            
        new_name = col1 + "x" + col2

        data[new_name] = data[col1] * data[col2]

        pairs.append(cur_pair) # Avoids unecessary columns ex: ( AxB e BxA)


# checking correlation
cols_to_drop = ['Age', 'Height', 'Weight', 'Sex', 'BMI', 'BSA']
df = data.drop(cols_to_drop, axis=1)

corr_matrix = df.corr(method='spearman')
corr_matrix["Calories"].sort_values(ascending = False)


# Create new feature: Heart Rate per Minute (rounded to 6 decimals)
data['Heart_Rate/Minute'] = (data['Heart_Rate'] / data['Duration']).round(6)

# Create interaction feature: Duration x Body Temperature x Heart Rate
data['DurationxBody_TempxHeart_Rate'] = data['Duration'] * data['Body_Temp'] * data['Heart_Rate']

# Select columns to exclude before correlation (mostly demographic or less relevant features)
cols_to_drop = ['Age', 'Height', 'Weight', 'Sex', 'BMI', 'BSA']

# Create a dataframe with only numeric & relevant features
df_numeric = data.drop(columns=cols_to_drop)

# Calculate Spearman correlation matrix on filtered dataframe
corr_matrix = df_numeric.corr(method='spearman')

# Sort correlations with 'Calories' descending
calories_corr = corr_matrix['Calories'].sort_values(ascending=False)

# Display correlations in a clear format
print("Spearman Correlation of Features with 'Calories':\n")
for feature, corr_val in calories_corr.items():
    print(f"{feature:30} : {corr_val:.4f}")



cols =  [i for i in train.columns if i != "Calories"]

train_duplicates = train.duplicated(subset=cols, keep= False)
test_duplicates = test.duplicated(subset=cols, keep= False)

print(f"duplicates on train set: {train_duplicates.sum()}\nduplicates on test set: {test_duplicates.sum()}")


train = train.groupby(cols, as_index=False, sort=False)["Calories"].min()

print(f"training set shape after dropping duplicates: {train.shape}")


class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self):
        return
        
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):        
        bmi = X['Weight'] / ((X['Height'] / 100) ** 2)
        duration_x_heart_rate = X['Duration'] * X['Heart_Rate']
        duration_x_body_temp = X['Duration'] * X['Body_Temp']
        body_temp_x_heart_rate = X['Body_Temp'] * X['Heart_Rate']
        heart_rate_per_min = X['Heart_Rate']/X['Duration']

        X_new = X.copy()
        
        # X_new['BMI'] = bmi
        # X_new['DurationxBody_TempxHeart_Rate'] = X['Duration'] * X['Body_Temp'] * X['Heart_Rate']
        X_new['Duration*HeartRate'] = duration_x_heart_rate
        X_new['Duration*BodyTemp'] = duration_x_body_temp
        X_new['BodyTemp*HeartRate'] = body_temp_x_heart_rate
        X_new['Heart_Rate/Min'] = heart_rate_per_min

        return X_new


num_cols = ['Age','Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp']
cat_cols = ['Sex']


# Defining transformers
one_hot_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output= False) # Encode categorical attribute 'Sex'
power_tr = PowerTransformer(standardize = True) # Make 'Body_Temp' distribution more normal-like
attr_adder = CombinedAttributesAdder() # Add combined attributes
std_scaler = StandardScaler() # Adjusting parameter scales

# Preprocessor
preprocessor = ColumnTransformer(transformers=[
    ('one_hot_encoder', one_hot_encoder, cat_cols),
    ('power_tr', power_tr, ['Body_Temp']),
    ('attr_adder', attr_adder, num_cols),
    ('scaler', std_scaler, num_cols),
])

# Model
xgb = XGBRegressor(n_estimators =2000, learning_rate= 0.02, random_state = 1)


# Full pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', xgb)
])


num_cols = ['Age','Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp']
cat_cols = ['Sex']


# Defining transformers
one_hot_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output= False) # Encode categorical attribute 'Sex'
power_tr = PowerTransformer(standardize = True) # Make 'Body_Temp' distribution more normal-like
attr_adder = CombinedAttributesAdder() # Add combined attributes
std_scaler = StandardScaler() # Adjusting parameter scales

# Preprocessor
preprocessor = ColumnTransformer(transformers=[
    ('one_hot_encoder', one_hot_encoder, cat_cols),
    ('power_tr', power_tr, ['Body_Temp']),
    ('attr_adder', attr_adder, num_cols),
    ('scaler', std_scaler, num_cols),
])

# Model
xgb = TransformedTargetRegressor(
        regressor= XGBRegressor(n_estimators=2000, learning_rate=0.02, random_state = 1),
        func=np.sqrt,
        inverse_func=np.square
)


# Full pipeline
pipeline_tt = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', xgb)
])


ftrs = ['Sex', 'Age','Weight', 'Height', 'Duration', 'Heart_Rate', 'Body_Temp']

print('Score WITHOUT target transformation:')
get_avg_cv_score(pipeline, ftrs, 5)


print('Score WITH target transformation:')
get_avg_cv_score(pipeline_tt, ftrs, 5)


X, X_test= train[ftrs], test[ftrs]
y = train["Calories"]


pipeline_tt.fit(X, y)
preds = pipeline_tt.predict(X_test)


# Sometimes XGB predicts negative values
for i in range(len(preds)):
    preds[i] = abs(preds[i])


# Adpating prediction to submission template
sub = pd.DataFrame()
sub["id"] = test.index
sub["Calories"] = preds
sub.to_csv("/kaggle/working/submission.csv", index = False)

