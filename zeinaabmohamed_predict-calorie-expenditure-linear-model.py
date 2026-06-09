import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


train_df = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')
submission_df = pd.read_csv('/kaggle/input/playground-series-s5e5/sample_submission.csv')


train_df


train_df.info()


train_df.isna().sum()


train_df.shape


train_df.describe()


train_df['Sex'] = train_df['Sex'].map({'male':0,'female':1})


plt.figure(figsize=(15,10))
sns.heatmap(train_df.corr(),annot=True,cmap='Greens')
plt.show()


test_df


test_df['Sex'] = test_df['Sex'].map({'male':0,'female':1})


plt.figure(figsize=(15,10))
for col in train_df.columns:
  plt.subplot(3,3,train_df.columns.get_loc(col)+1)
  sns.kdeplot(train_df[col])
  plt.tight_layout()
  plt.xlabel(col)
  plt.ylabel('Frequency')
  plt.title(col)
plt.show()


train_df['BMI'] = train_df['Weight'] / ((train_df['Height']/100)**2)


test_df['BMI'] = test_df['Weight'] / ((test_df['Height']/100)**2)





from sklearn.model_selection import train_test_split

# Separate features and target
X = train_df.drop(['Calories', 'id'], axis=1)
y = train_df['Calories']
X_test = test_df.drop(['id'], axis=1)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


from sklearn.preprocessing import StandardScaler
# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)


from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(X_train_scaled, y_train)


reg.score(X_train_scaled, y_train)


X_train_scaled


X_test_scaled


from sklearn.metrics import mean_squared_log_error

# Predict on validation set
# Predict on validation set
y_val_pred = np.maximum(0, reg.predict(X_val_scaled))

# Calculate RMSLE
rmsle = np.sqrt(mean_squared_log_error(y_val, y_val_pred))
print("RMSLE on validation set:", rmsle)


# Predict on test set
y_test_pred = np.maximum(0, reg.predict(X_test_scaled))


# Save submission
submission = pd.DataFrame({
    'id': test_df['id'],
    'Calories': y_test_pred
})
submission.to_csv('submission.csv', index=False)


submission

