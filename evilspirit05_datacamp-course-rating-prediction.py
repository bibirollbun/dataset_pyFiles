import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from catboost import CatBoostRegressor

from sklearn.metrics import mean_squared_error


df=pd.read_csv("/kaggle/input/datacamp-cources-rating/train.csv")


df.head()


df.info()


df.drop(columns=["id"],axis=1,inplace=True)


df.shape


df.isnull().sum()


mode=df["topic_id"].mode()[0]
df["topic_id"]=df["topic_id"].fillna(mode)


categorical_cols = ['programming_language','content_area']

label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col]) 
    label_encoders[col] = le  


df.info()


# Select only numeric columns for boxplot visualization
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns

# Set the style and figure size
sns.set(style="whitegrid")
plt.figure(figsize=(20, 7))

# Create a boxplot for each numeric column
df_melted = df.melt(value_vars=numeric_cols)
sns.boxplot(x='variable', y='value', data=df_melted)

plt.title("Boxplot of Numeric Features (Outlier Detection)", fontsize=14)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


X=df.drop(columns=["course_rating"],axis=1)
y=df["course_rating"]
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train["nb_of_subscriptions"]=scaler.fit_transform(X_train["nb_of_subscriptions"].values.reshape(-1,1))
X_val["nb_of_subscriptions"]=scaler.transform(X_val["nb_of_subscriptions"].values.reshape(-1,1))

# TF-IDF vectorizer
tfidf = TfidfVectorizer(max_features=200, ngram_range=(1,2))
tfidf_train = tfidf.fit_transform(X_train['title']).toarray()
tfidf_val = tfidf.transform(X_val['title']).toarray()

tfidf_columns = [f'tfidf_{i}' for i in range(tfidf_train.shape[1])]

tfidf_train_df = pd.DataFrame(tfidf_train, columns=tfidf_columns, index=X_train.index)
tfidf_val_df = pd.DataFrame(tfidf_val, columns=tfidf_columns, index=X_val.index)
X_train_final = pd.concat([X_train.drop('title', axis=1), tfidf_train_df], axis=1)
X_val_final = pd.concat([X_val.drop('title', axis=1), tfidf_val_df], axis=1)


from sklearn.linear_model import Ridge

ridge_model = CatBoostRegressor(
    iterations=5000,       # number of trees
    learning_rate=0.1,
    depth=8,
    eval_metric='RMSE',    # you can also use MAE
    random_seed=42,
    verbose=100
)


ridge_model.fit(X_train_final, y_train)


y_pred_val = ridge_model.predict(X_val_final)
y_pred_val = np.clip(y_pred_val, 0.0, 5.0)
mse = mean_squared_error(y_val, y_pred_val)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_val, y_pred_val)
r2 = r2_score(y_val, y_pred_val)
# Print all metrics
print(f"Validation Metrics for Random Forest:")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"R² Score: {r2:.4f}")


test=pd.read_csv("/kaggle/input/datacamp-cources-rating/test.csv")
Id=test.id
test.drop(columns=["id"],axis=1,inplace=True)

for col in categorical_cols:
    le = label_encoders[col]
    # Map unseen labels to 'unknown'
    test[col] = test[col].apply(lambda x: x if x in le.classes_ else 'unknown')
    # Add 'unknown' to encoder classes
    if 'unknown' not in le.classes_:
        le.classes_ = np.append(le.classes_, 'unknown')
    # Transform
    test[col] = le.transform(test[col])


test["nb_of_subscriptions"]=scaler.transform(test["nb_of_subscriptions"].values.reshape(-1,1))
tfidf_test = tfidf.transform(test['title']).toarray()

tfidf_test_df = pd.DataFrame(tfidf_test, columns=tfidf_columns, index=test.index)
test_final = pd.concat([test.drop('title', axis=1), tfidf_test_df], axis=1)
y_pred=ridge_model.predict(test_final)
y_pred = np.clip(y_pred, 0.0, 5.0)
submission = pd.DataFrame({'id': Id,'course_rating': y_pred})
submission.to_csv('new_submission.csv', index=False)
submission.head()


df=pd.read_csv("/kaggle/input/datacamp-cources-rating/train.csv")
print(f"Data Shape: {df.shape}")
print(f"Data info: {df.info()}")
print(f"Check Null Values: {df.isnull().sum()}")
df.drop(columns=["id","title"],axis=1,inplace=True)



from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
for col in ['programming_language', 'content_area']:
    df[col] = le.fit_transform(df[col])
df.head()


X=df.drop(columns=["course_rating"],axis=1)
y=df["course_rating"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
cat_features = ['programming_language', 'content_area']

model = CatBoostRegressor(
    iterations=1000,
    learning_rate=0.05,
    depth=6,
    loss_function='RMSE',
    eval_metric='RMSE',
    random_seed=42,
    verbose=100
)
model.fit(
    X_train, y_train,
    cat_features=cat_features,
    eval_set=(X_test, y_test),
    use_best_model=True
)
y_pred_val = model.predict(X_val_final)
y_pred_val = np.clip(y_pred_val, 0.0, 5.0)
mse = mean_squared_error(y_val, y_pred_val)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_val, y_pred_val)
r2 = r2_score(y_val, y_pred_val)
# Print all metrics
print(f"Validation Metrics for Random Forest:")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"R² Score: {r2:.4f}")


test=pd.read_csv("/kaggle/input/datacamp-cources-rating/test.csv")
Id=test.id
test.drop(columns=["id","title"],axis=1,inplace=True)
le = LabelEncoder()
for col in ['programming_language', 'content_area']:
    test[col] = le.fit_transform(test[col])
    
model.fit(X,y)

y_pred=model.predict(test)
y_pred = np.clip(y_pred, 0.0, 5.0)
submission = pd.DataFrame({'id': Id,'course_rating': y_pred})
submission.to_csv('cat_submission.csv', index=False)
submission.head()




