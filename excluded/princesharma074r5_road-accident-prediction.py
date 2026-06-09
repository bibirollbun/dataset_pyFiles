import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


train_df = pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv')


train_df.head()


test_df.head()


train_df.describe()


train_df.shape


train_df.info()


# Distribution of accident_risk
plt.figure(figsize=(8, 5))
sns.histplot(train_df['accident_risk'], kde=True, bins=30)
plt.title('Distribution of Accident Risk')
plt.xlabel('Accident Risk')
plt.ylabel('Frequency')
plt.show()


# Distribution of numerical features
numerical_features = ['num_lanes', 'curvature', 'speed_limit', 'num_reported_accidents']
train_df[numerical_features].hist(figsize=(10, 8), bins=20)
plt.suptitle('Distribution of Numerical Features', y=1.02)
plt.tight_layout()
plt.show()



subset_cols = [
    'curvature',
    'speed_limit',
    'public_road',
    'num_reported_accidents',
    'accident_risk'
]

sns.pairplot(train_df.sample(1000, random_state=42)[subset_cols])


# Calculate the correlation matrix for numerical features
correlation_matrix = train_df[numerical_features + ['accident_risk']].corr()

# Create a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap of Numerical Features and Accident Risk')
plt.show()


# Distribution of categorical features
categorical_features = ['road_type', 'lighting', 'weather', 'time_of_day']
plt.figure(figsize=(12, 10))
for i, col in enumerate(categorical_features):
    plt.subplot(2, 2, i+1)
    sns.countplot(data=train_df, x=col, palette='viridis')
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Count')
    plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()


train_df.head()


from sklearn.preprocessing import LabelEncoder
columns = ['road_type','lighting','weather','road_signs_present','public_road','time_of_day','holiday','school_season']
le = LabelEncoder()
train_df[columns]= le.fit_transform(columns)
test_df[columns]= le.fit_transform(columns)


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
train_df[['speed_limit']] = scaler.fit_transform(train_df[['speed_limit']]) 
test_df[['speed_limit']] = scaler.fit_transform(test_df[['speed_limit']])


train_df.head()


X = train_df.drop(columns='accident_risk')
y = train_df['accident_risk']


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
model_lr = LinearRegression()
model_xgb = XGBRegressor(n_estimators=200,
                        learning_rate=0.05,      
                        max_depth=8,   
                        subsample=1,    
                        colsample_bytree=0.7,    
                        reg_lambda=1,       
                        reg_alpha=0.1,           
                        random_state=42,
                        eval_metric='rmse')
model_lr.fit(X_train,y_train)
model_xgb.fit(X_train,y_train)


y_pred_lr = model_lr.predict(X_test)
y_pred_xgb = model_xgb.predict(X_test)



from sklearn.metrics import mean_squared_error
print("\nLinearReg: ",np.sqrt(mean_squared_error(y_test,y_pred_lr)),"\nXGBoost: ",np.sqrt(mean_squared_error(y_test,y_pred_xgb)))


test_pred = model_xgb.predict(test_df)


submission = pd.DataFrame({
    'id': test_df['id'],
    'accident_risk': test_pred
})

submission.to_csv('submission.csv',index=False)
print('File saved...✅')




