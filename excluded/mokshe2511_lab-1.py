import numpy as np
import pandas as pd


train = pd.read_csv("/kaggle/input/iiita-iml-fall-2025-lab-1/train.csv")


X_train = train[['lotsize', 'bedrooms', 'bathrms']].values
y_train = train['price'].values.reshape(-1, 1)


X_b = np.c_[np.ones((X_train.shape[0], 1)), X_train]


theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y_train)
print("Learned parameters (theta):")
print(theta.flatten())


test = pd.read_csv("/kaggle/input/iiita-iml-fall-2025-lab-1/test.csv")
X_test = test[['lotsize', 'bedrooms', 'bathrms']].values
X_test_b = np.c_[np.ones((X_test.shape[0], 1)), X_test]


y_pred = X_test_b.dot(theta)


submission = pd.DataFrame({
    'ID': test['ID'],
    'Predicted_price': y_pred.flatten()
})

submission.to_csv("submission.csv", index=False)
print("file saved\n", submission.head())

