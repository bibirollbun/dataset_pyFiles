import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def theta(X,Y):
    # Convert to numpy arrays if they aren't already
    X = np.array(X)
    Y = np.array(Y)
    # Ensure Y is properly shaped
    if len(Y.shape) == 1:
        Y = Y.reshape(-1, 1)
    return np.linalg.inv(X.T @ X) @ X.T @ Y 
    
def y_hat(X,theta):
    X = np.array(X)
    return X@theta 

# Load and prepare data
df = pd.read_csv('/kaggle/input/black-jack-smart-effect-of-removal-ml/train.csv')
y = np.array(df.iloc[:,-1]).reshape(-1, 1)
X = np.array(df.iloc[:,1:-1])
X=np.c_[np.ones((X.shape[0], 1)),X]

weights = theta(X,y)

test_df=pd.read_csv('/kaggle/input/black-jack-smart-effect-of-removal-ml/test.csv')
X_test = np.array(test_df.iloc[:,1:])
X_test=np.c_[np.ones((X_test.shape[0], 1)),X_test]

predictions = y_hat(X_test,weights)


submission = pd.DataFrame({
    'id': test_df['id'],
    'target': predictions.flatten()  # flatten() in case predictions is 2D
})
submission.to_csv('submission.csv',index=False)
print('Submission created')


import matplotlib.pyplot as plt
predictions = y_hat(X,weights)
mse=sum(np.square(predictions-y))/len(predictions)
plt.scatter(predictions,y)
plt.title(f'MSE:{mse}')
import pandas as pd
index_=['ev_base','A',2,3,4,5,6,7,8,9,'JQK']
print('Linear Effect of Removal')
print(pd.Series(weights.flatten(), index=index_))
residuals = y - predictions
plt.figure()
plt.scatter(predictions, residuals)
plt.axhline(y=0, color='r', linestyle='-')
plt.title('Residual Plot')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')

