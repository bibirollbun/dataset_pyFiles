#Data and libraries Import
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split,cross_val_score, KFold
from sklearn.preprocessing import LabelEncoder, OneHotEncoder,StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix, r2_score, mean_squared_error, mean_absolute_error)
import warnings
warnings.filterwarnings('ignore')
np.random.seed(42)
print("Necessary libraries imported successfully")


train = pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv')

print(f"DataSet Shapes:\t Train:{train.shape} - \t Test:{test.shape}")
print(f"\nData Info: {train.info()}")
print(f"\nMissing values: {train.isnull().sum()}")
print("\nTarget Distribution")

print(f"\nBasic stats: {train.describe()}")


#Normal Distribution
plt.figure(figsize=(10,5))
sns.histplot(train['accident_risk'],bins=20,kde=True)
plt.title('Distribution of Risk')
plt.xlabel("accident_risk")
plt.ylabel('frequency')
plt.tight_layout()
plt.show()


#Correlation Analysis
numericols= train.select_dtypes(include=[np.number])
corr_mat = numericols.corr()
plt.figure(figsize=(10,5))
sns.heatmap(corr_mat,cmap="coolwarm",center=0,annot=True,fmt='.2f')
plt.title('Correlation Matrix of Numeric features')
plt.tight_layout()
plt.show()

#Distribution of Numerical Features
num_cols = ["num_lanes","curvature","speed_limit","num_reported_accidents","accident_risk"] # numerical features
cat_cols = ["road_type", "lighting", "weather", "road_signs_present", "public_road", "time_of_day", "holiday", "school_season"]
for col in num_cols:
    plt.figure(figsize=(8,4))
    sns.histplot(train[col],bins=20,kde=True)
    plt.title(f"Measurement of {col}")
    plt.xlabel(f"{col}")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()


combined = pd.concat([train.drop(train.columns[-1],axis=1),test],ignore_index=True)
target = train.iloc[:, -1]
print(f"Combined Shape: {combined.shape}")

#Missing Data Cleaning
numericols_missing = combined.select_dtypes(include=[np.number]).columns
for col in numericols_missing:
    if combined[col].isnull().sum()>0:
        median_val=train[col].median()
        combined[col].fillna(median_val,inplace=True)

categorical = combined.select_dtypes(include=['object']).columns
for col in categorical:
    if combined[col].isnull().sum()>0:
        mode_val=train[col].mode()
        combined[col].fillna(mod_val,inplace=True)

print("Successfully handled Missing Values")

#Handling Outliners
numeric_combined = combined.select_dtypes(include=[np.number])
for col in numeric_combined.columns:
    Q1 = numeric_combined[col].quantile(0.25)
    Q3 = numeric_combined[col].quantile(0.75)
    IQR = Q3 -Q1
    lb = Q1-1.5*IQR
    ub = Q3+1.5*IQR
    combined[col]=combined[col].clip(lb,ub)
print("Successfully handled outliners")




categorical_cols = combined.select_dtypes(include=['object']).columns
combined = pd.get_dummies(combined,columns=categorical_cols, drop_first=True)


X_train = combined.iloc[:len(train)]
X_test = combined.iloc[len(train):]
y_train=target

from sklearn.feature_selection import VarianceThreshold
selector=VarianceThreshold(threshold=0.01)
X_train_selected=selector.fit_transform(X_train)
X_test_selected=selector.transform(X_test)

selected_features = X_train.columns[selector.get_support()]
print(f"Features after variance filtering: {X_train_selected.shape[1]} (removed {X_train.shape[1]-X_train_selected.shape[1]})")

X_train = pd.DataFrame(X_train_selected, columns=selected_features)
X_test = pd.DataFrame(X_test_selected, columns=selected_features)


print("Scaling Features.....")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled= scaler.transform(X_test)

X_train=pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test=pd.DataFrame(X_test_scaled, columns=X_test.columns)
print("Successfully scaled the features")


print(X_train.head())
print(y_train.head())
print(X_test.head())


X,X_val,y,y_val=train_test_split(X_train,y_train,test_size=0.2,random_state=42)
print("X_train shape:", X.shape)
print("X_val shape:", X_val.shape)
print("y_train shape:", y.shape)
print("y_val shape:", y_val.shape)


from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

model = LinearRegression()
kf = KFold(n_splits=5,shuffle=True,random_state=42)

cv_scores=cross_val_score(model,X,y,cv=kf,scoring='r2')
cv_scorem=cross_val_score(model,X,y,cv=kf,scoring='neg_mean_squared_error')
print(f"CV r2 mean:{cv_scores.mean():.4f}(+/-{cv_scores.std():.4f})")
print(f"CV MSE mean:{cv_scorem.mean():.4f}(+/-{cv_scorem.std():.4f})")

#Training the model on linear regression
model.fit(X,y)


y_pred=model.predict(X_val)
rmse=mean_squared_error(y_val,y_pred)**0.5
print(f"LinearRegression RMSE on validation data:{rmse}")
print(X_val.shape,X_test.shape)
test_pred=model.predict(X_test)


submission_df=pd.DataFrame({'id':test['id'],'accident_risk':test_pred})
submission_df.to_csv('submission.csv',index=False)
print("Successfully created submission")

