import os
os.listdir('/kaggle/input/bluebook-for-bulldozers')


import matplotlib.pyplot as plt
import seaborn as sns

from fastai.tabular.all import * # Import all necessary fastai tabular components
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split # If you use custom train_test_split

# Helper function for displaying full DataFrame content
def display_all(df):
    with pd.option_context("display.max_rows", 1000, "display.max_columns", 1000):
        display(df)

print("Initial imports and helper functions loaded.")


# Make output easier to read
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 100)


df = pd.read_csv('/kaggle/input/bluebook-for-bulldozers/TrainAndValid.csv', low_memory=False)
df.head()


#checking the shape

df.shape


df['SalePrice'].describe()


# Re-read the original CSV
df = pd.read_csv('/kaggle/input/bluebook-for-bulldozers/TrainAndValid.csv', low_memory=False)

# Convert 'saledate' to datetime
df['saledate'] = pd.to_datetime(df['saledate'])

# Extract date parts
df['saleYear'] = df['saledate'].dt.year
df['saleMonth'] = df['saledate'].dt.month
df['saleDay'] = df['saledate'].dt.day
df['saleDayOfWeek'] = df['saledate'].dt.dayofweek
df['saleDayOfYear'] = df['saledate'].dt.dayofyear

# Then drop the original 'saledate'
df.drop('saledate', axis=1, inplace=True)


#checking how many missing values 
df.isnull().sum().sort_values(ascending=False).head(20)


# continous missing values 

for col in df.columns:
    if df[col].dtype != 'object':
        df[col] = df[col].fillna(df[col].median())



#Categorical missing values
for colname in df.select_dtypes(include='object').columns:
    df[colname] = df[colname].astype('category')
    df[colname] = df[colname].cat.codes


# A quick check

df.shape


df.dtypes


df.isnull().sum().sort_values(ascending=False).head(10)


for col in df.columns:
    if df[col].dtype == 'int64' and df[col].nunique() < 25:
        print(f"{col}: {df[col].nunique()} unique values â†’ {df[col].unique()}")



# Our target variable
y = df['SalePrice']

# Drop target from features
X = df.drop('SalePrice', axis=1)


from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42)



from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(n_jobs=-1, random_state=42, n_estimators=100)
model.fit(X_train, y_train)


# predict and evaluate

from sklearn.metrics import mean_squared_log_error
from numpy import sqrt

preds = model.predict(X_valid)

# Clip predictions to avoid log(negative)
preds = np.maximum(0, preds)

rmsle = sqrt(mean_squared_log_error(y_valid, preds))
print(f"RMSLE: {rmsle:.4f}")



importances = model.feature_importances_


feature_importances = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)



# Let us view the obtained datafram

feature_importances.head(20)


import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 8))
sns.barplot(
    x='Importance', 
    y='Feature', 
    data=feature_importances.head(20),
    palette='viridis'
)
plt.title("Top 20 Feature Importances")
plt.tight_layout()
plt.show()



from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_log_error
from math import sqrt
import numpy as np

model_final = RandomForestRegressor(
    n_estimators=120,        # not too many trees
    max_depth=30,            # not too deep
    min_samples_leaf=5,      # prevent overfitting on noise
    max_features='sqrt',     # more randomness = less correlation
    n_jobs=-1,
    random_state=42
)

model_final.fit(X_train, y_train)

# Predict and ensure no negative predictions
preds_final = model_final.predict(X_valid)
preds_final = np.maximum(0, preds_final)

# Evaluate
rmsle_final = sqrt(mean_squared_log_error(y_valid, preds_final))
print(f"ğŸŸ¢ Final RMSLE: {rmsle_final:.4f}")



from sklearn.ensemble import ExtraTreesRegressor

model_et = ExtraTreesRegressor(
    n_estimators=100,
    max_depth=30,
    min_samples_leaf=5,
    max_features='sqrt',
    n_jobs=-1,
    random_state=42
)

model_et.fit(X_train, y_train)
preds_et = model_et.predict(X_valid)
preds_et = np.maximum(0, preds_et)
rmsle_et = sqrt(mean_squared_log_error(y_valid, preds_et))
print(f"ğŸŒ² ExtraTrees RMSLE: {rmsle_et:.4f}")



import joblib
joblib.dump(model, 'best_model_rf_02124.pkl')

#To always reload the best model;
#model = joblib.load('best_model_rf_02124.pkl')



#To check for overfitting 
# both rmsle have to be almost the same
from sklearn.metrics import mean_squared_log_error
from math import sqrt

train_preds = model.predict(X_train)
valid_preds = model.predict(X_valid)

train_rmsle = sqrt(mean_squared_log_error(y_train, np.maximum(0, train_preds)))
valid_rmsle = sqrt(mean_squared_log_error(y_valid, np.maximum(0, valid_preds)))

print(f"Train RMSLE: {train_rmsle:.4f}")
print(f"Valid RMSLE: {valid_rmsle:.4f}")


import matplotlib.pyplot as plt

plt.figure(figsize=(6,6))
plt.scatter(y_valid, valid_preds, alpha=0.3)
plt.plot([y_valid.min(), y_valid.max()], [y_valid.min(), y_valid.max()], 'r--')
plt.xlabel('Actual Sale Price')
plt.ylabel('Predicted Sale Price')
plt.title('Actual vs. Predicted Sale Price')
plt.grid(True)
plt.show()


from sklearn.model_selection import cross_val_score

rmsle_scores = cross_val_score(model, X_train, y_train,
                               scoring='neg_mean_squared_log_error',
                               cv=5)

rmsle_scores = [sqrt(-s) for s in rmsle_scores]
print(f"Cross-Validation RMSLE Scores: {rmsle_scores}")
print(f"Mean CV RMSLE: {np.mean(rmsle_scores):.4f}")


# check distribution errors

errors = y_valid - valid_preds
plt.hist(errors, bins=50)
plt.title("Distribution of Prediction Errors")
plt.xlabel("Prediction Error")
plt.ylabel("Count")
plt.grid(True)
plt.show()

