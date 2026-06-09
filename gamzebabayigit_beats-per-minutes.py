import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Machine learning libraries
import joblib  # for saving the model
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
# Ignore warnings for cleaner output
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler


train=pd.read_csv('/kaggle/input/playground-series-s5e9/train.csv')
test=pd.read_csv('/kaggle/input/playground-series-s5e9/test.csv')


train.head()


train.shape


train.info(())


train.describe()


train.isnull().sum()


num_cols = train.select_dtypes(include=['int64', 'float64']).columns

plt.figure(figsize=(15, 10))

for i, col in enumerate(num_cols, 1):
    plt.subplot(3, 4, i)  # Adjust grid size based on number of features
    sns.boxplot(y=train[col], color="skyblue")
    plt.title(f"Boxplot of {col}", fontsize=10)

plt.tight_layout()
plt.show()


num_cols = train.select_dtypes(include=['int64', 'float64']).columns

for col in num_cols:
    Q1 = train[col].quantile(0.25)
    Q3 = train[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    train = train[(train[col] >= lower_bound) & (train[col] <= upper_bound)]


x = train.drop(columns=['BeatsPerMinute', 'id']) 
y = train['BeatsPerMinute']                       

x_test = test.drop(columns=['id'])


test_ids = test['id'].copy()



x_test = test.drop(columns=['id'])
x_test_submission = test.drop(columns=['id'])



scaler = StandardScaler()
x = scaler.fit_transform(x)


x_test_scaled = scaler.transform(x_test)
x_test_submission_scaled = scaler.transform(x_test_submission)



x_train, x_test, y_train, y_test=train_test_split(x,y, test_size=0.20, random_state=42)


lgb_model = lgb.LGBMRegressor(random_state=42)
lgb_model.fit(x_train, y_train)


y_pred_submission = lgb_model.predict(x_test_submission_scaled)

y_pred_validation = lgb_model.predict(x_test)   
rmse = np.sqrt(mean_squared_error(y_test, y_pred_validation))
r2 = r2_score(y_test, y_pred_validation)


rmse


r2


 
submission = pd.DataFrame({
    'id': test_ids,   
    'BeatsPerMinute': y_pred_submission
})


submission


submission.to_csv('submission.csv', index=False)
print("\n✓ Submission saved to 'submission.csv'")

