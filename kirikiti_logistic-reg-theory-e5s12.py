import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

submit_test = pd.read_csv("/kaggle/input/playground-series-s5e12/test.csv")
train = pd.read_csv("/kaggle/input/playground-series-s5e12/train.csv", index_col='id')
sample = pd.read_csv('/kaggle/input/playground-series-s5e12/sample_submission.csv')


train.head()


feature = 'bmi'
target = 'diagnosed_diabetes'

train[[feature,target]].head()


train[[feature,target]].plot.scatter(x=feature,y=target)


# Parameter Tests
w = 0.08 
b = -1.5


# After building the model (explained later)
# Intercept (b): [-1.48711166]
# Slope (w): [[0.07716485]]


# Points of the line
x = np.linspace(0, train[feature].max(), 100)
y = 1 / (1 + np.exp(-(w * x + b)))

# Plot of the line
train.plot.scatter(x=feature, y=target)
plt.plot(x, y, '-r')
plt.ylim(0, train[target].max() * 1.1)
# plt.grid()
plt.show()


# Calculation of predictions
train['sigmoid'] = 1 / (1 + np.exp(-(train[feature] * w + b)))

# Calculation of the error function
train['loss_xi'] = -train[target] * np.log(train['sigmoid']) - (1 - train[target]) * np.log(1 - train['sigmoid'])
cost_j = train['loss_xi'].mean()
cost_j


# We create a DataFrame to calculate the error as a function of the parameters w, b


#array = np.mgrid[0.05:0.15:0.01, -4:-3:0.01].reshape(2, -1).T
array = np.mgrid[0.01:0.25:0.01, -3.8:-2.8:0.01].reshape(2, -1).T
df = pd.DataFrame(data=array, 
                  columns=['w', 'b'])

# Rounding to solve the issue with too many decimals
df['w'] = np.round(df['w'], 6)
df['b'] = np.round(df['b'], 6)


df


def sum_error_df(df):
    train['sigmoid'] = 1/(1+np.exp(-(train[feature]*df['w']+df['b'])))
    train['loss_xi'] = -train[target]*np.log(train['sigmoid'])-(1-train[target])*np.log(1-train['sigmoid'])
    j_cost = train['loss_xi'].mean()
    return(j_cost)


df['error'] = df.apply(sum_error_df, axis=1)


df.sort_values(by=['error']).head()


df_3d = df.pivot(index='w', columns='b', values='error')


df_3d.head()


import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


x = df_3d.columns
y = df_3d.index
X,Y = np.meshgrid(x,y)
Z = df_3d

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z)


x = df_3d.columns
y = df_3d.index
X,Y = np.meshgrid(x,y)
Z = df_3d
plt.contourf(Y, X, Z, alpha=0.7, cmap=plt.cm.jet)


def delta_j_w(w, b):
    train['sigmoid'] = 1/(1+np.exp(-(train[feature]*w+b)))
    train['partial_loss'] = (train['sigmoid']-train[target])*train[feature]
    derivative = train['partial_loss'].mean()
    return(derivative) 

def delta_j_b(w, b):
    train['sigmoid'] = 1/(1+np.exp(-(train[feature]*w+b)))
    train['partial_loss'] = (train['sigmoid']-train[target])
    derivative = train['partial_loss'].mean()
    return(derivative) 


w_0 = 0.08
b_0 = -1.5


alpha_w = 0.001
alpha_b = 0.1

# Number of iterations
num_iterations = 10000

for _ in range(num_iterations):
    # Update the parameters
    w_new = w_0 - alpha_w * delta_j_w(w_0, b_0)
    b_new = b_0 - alpha_b * delta_j_b(w_0, b_0)
    
    # Update the values
    w_0 = w_new
    b_0 = b_new

# Printing the final values
print(f"Optimal value of w: {w_0:.8f}")
print(f"Optimal value of b: {b_0:.8f}")


# Optimal values from sklearn (explanation below)
# w = 0.07708645
# b = -1.48507329


from sklearn.linear_model import LogisticRegression

# Defining input and output
X_train = np.array(train[feature]).reshape((-1, 1))
Y_train = np.array(train[target])

# Creating the model
model = LogisticRegression()
model.fit(X_train, Y_train)

# Printing parameters
print(f"Intercept (b): {model.intercept_}")
print(f"Slope (w): {model.coef_}")

