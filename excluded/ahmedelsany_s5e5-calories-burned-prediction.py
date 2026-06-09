import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_squared_log_error
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor



train = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv')
train.head()


test = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')
test_ids = test["id"]


train.columns


train.info()


train.shape


train.describe().T


train.isna().sum()


train.duplicated().sum()


train.drop(columns=['id'],inplace=True,axis=1)


test.drop(columns=['id'],inplace=True,axis=1)


num_cols = train.select_dtypes(include=["number"]).columns
num_cols


train["BMI"] = train["Weight"] / ((train["Height"] / 100) ** 2)
train.head()


train["Duration_HeartRate"] = train["Duration"] * train["Heart_Rate"]  # Interaction between duration and heart rate
train.head()


num_cols_test = test.select_dtypes(include=["number"]).columns
num_cols_test


test["BMI"] = test["Weight"] / ((test["Height"] / 100) ** 2)
test.head()


test["Duration_HeartRate"] = test["Duration"] * test["Heart_Rate"]  # Interaction between duration and heart rate
test.head()


plt.figure(figsize=(15,8))
sns.heatmap(train[num_cols].corr(), cmap='rocket', annot=True, annot_kws={"size": 16}, fmt=".2f",)
plt.show()


# Check for outliers using boxplots
for col in num_cols:
    plt.figure(figsize=(6, 4))
    sns.boxplot(x=train[col])
    plt.title(f'Boxplot of {col}')
    plt.show()



def handle_outliers_iqr(train, columns=None, method='remove'):

    train_out = train.copy()

    if columns is None:
        columns = train_out.select_dtypes(include=['number']).columns.tolist()

    for col in columns:
        Q1 = train_out[col].quantile(0.25)
        Q3 = train_out[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        if method == 'remove':
            train_out = train_out[(train_out[col] >= lower_bound) & (train_out[col] <= upper_bound)]

        elif method == 'cap':
            train_out[col] = train_out[col].clip(lower=lower_bound, upper=upper_bound)

        else:
            raise ValueError("method must be either 'remove' or 'cap'")

    return train_out



train_cleaned = handle_outliers_iqr(train, method='cap')
print(f"Rows before: {len(train)}, Rows after removing outliers: {len(train_cleaned)}")


# Check for outliers using boxplots
for col in num_cols:
    plt.figure(figsize=(6, 4))
    sns.boxplot(x=train_cleaned[col])
    plt.title(f'Boxplot of {col}')
    plt.show()



plt.figure(figsize=(8,5))
plt.scatter(train["Heart_Rate"], train["Calories"], alpha=0.5, color='green')
plt.xlabel("Heart Rate (BPM)")
plt.ylabel("Calories Burned")
plt.title("Heart Rate vs Calories Burned")
plt.show()


# Boxplot: Calories by Gender
plt.figure(figsize=(8,5))
train.boxplot(column="Calories", by="Sex", grid=False)
plt.xlabel("Gender")
plt.ylabel("Calories Burned")
plt.title("Calories Burned by Gender")
plt.show()


plt.figure(figsize=(20, 15))

for i, column in enumerate(train.select_dtypes(include=['float64', 'int64']).columns):
    plt.subplot(3, 3, i+1)
    sns.histplot(train[column], kde=True)
    plt.title(f'Distribution {column}')
    plt.tight_layout()

plt.show()



plt.figure(figsize=(20, 15))

for i, column in enumerate(train.select_dtypes(include=['float64', 'int64']).columns):
    plt.subplot(3, 3, i+1)
    sns.boxplot(x='Sex', y=column, data=train)
    plt.title(f'{column} by gender')   
    plt.tight_layout()

plt.show()



plt.figure(figsize=(20, 15))

numeric_cols = ['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp', 'BMI']
for i, column in enumerate(numeric_cols):
    plt.subplot(3, 3, i+1)
    sns.scatterplot(x=column, y='Calories', data=train, hue='Gender' if 'Gender' in train.columns else None, alpha=0.6)
    plt.title(f'Relationship between {column} and calories burned')
    plt.tight_layout()

plt.show()



train['Age_Group'] = pd.cut(train['Age'], bins=[0, 20, 30, 40, 50, 60, 100], 
                         labels=['<20', '20-30', '30-40', '40-50', '50-60', '>60'])

plt.figure(figsize=(12, 6))
sns.boxplot(x='Age_Group', y='BMI', data=train)
plt.title('BMI distribution by age group')
plt.xlabel('Age group')
plt.ylabel('BMI')
plt.show()



from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
object_columns = train_cleaned.select_dtypes(include='object').columns
for col in object_columns:
    train_cleaned[col] = le.fit_transform(train_cleaned[col])

# train_encoded = pd.get_dummies(train_cleaned, drop_first=True)

train_cleaned.info()


object_columns_test = test.select_dtypes(include='object').columns
for col in object_columns_test:
    test[col] = le.fit_transform(test[col])

# train_encoded = pd.get_dummies(train_cleaned, drop_first=True)

test.info()


X = train_cleaned.drop('Calories', axis=1)
y = train_cleaned['Calories']


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


X_train ,X_test,y_train ,y_test = train_test_split(X_scaled,y,random_state = 42 , test_size=250000 , shuffle=True)


print("Shape of Features Train :" , X_train.shape,"\n")
print("Shape of Target Train :" , y_train.shape,"\n")
print("Shape of Features Test :" , X_test.shape,"\n")
print("Shape of Target Test :" , y_test.shape)


def rmsle(y_true, y_pred):
    y_pred = np.maximum(0, y_pred) 
    return np.sqrt(mean_squared_log_error(y_true, y_pred))

 
models = {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
        "XGBoost": XGBRegressor(n_estimators=100, random_state=42, verbosity=0),
        "LightGBM": LGBMRegressor(n_estimators=100, random_state=42),
        "KNN": KNeighborsRegressor(n_neighbors=5,weights='uniform',algorithm='auto'),
        'Lasso Regression': make_pipeline(
        PolynomialFeatures(degree=2, include_bias=False),
        Lasso(alpha=0.1, random_state=42)),
        'Ridge Regression': make_pipeline(
        PolynomialFeatures(degree=2, include_bias=False),
        Ridge(alpha=1.0, random_state=42))

}


def rmsle(y_true, y_pred):
    y_pred = np.maximum(0, y_pred) 
    return np.sqrt(mean_squared_log_error(y_true, y_pred))


from sklearn.metrics import r2_score

def train_and_evaluate(X_train, y_train, X_val, y_val):
    best_model = None
    best_score = float("inf")

    for name, model in models.items():
        model.fit(X_train, y_train)
        val_preds = model.predict(X_val)
        score = rmsle(y_val, val_preds)
        r2 = r2_score(y_val, val_preds)
        print(f"{name} RMSLE: {score:.4f}")
        print(f"{name} Score : {r2:.4f} ")

        if score < best_score:
            best_score = score
            best_model = model

    print(f"\nBest model: {type(best_model).__name__}, RMSLE: {best_score:.4f}")
    print(f"\nBest model: {type(best_model).__name__}, Score: {r2:.4f}")

    print("*"*50)
    return best_model



X_train.shape, y_train.shape, X_test.shape, y_test.shape


# Train and evaluate
best_model = train_and_evaluate(X_train, y_train, X_test, y_test)


def make_submission(model, X_test, test_ids, filename="submission.csv"):
    preds = model.predict(X_test)
    preds = np.maximum(0, preds)
    submission = pd.DataFrame({
        "id": test_ids,
        "Calories": preds
    })
    submission.to_csv(filename, index=False)
    print(f"Submission file saved as '{filename}'")


# Save predictions
make_submission(best_model, X_test, test_ids)


submission = pd.read_csv("/kaggle/working/submission.csv")
submission.head()




