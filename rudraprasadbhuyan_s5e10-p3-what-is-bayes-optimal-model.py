"""
Goal: Understand the Bayes' theorem.

Author: Rudra Prasad Bhuyan
V1: 21-10-2025 12:26 IST
"""
print("")








import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import scipy.stats as stats

import warnings
warnings.filterwarnings('ignore')


sub_path = '/kaggle/input/playground-series-s5e10/sample_submission.csv'
train_path = '/kaggle/input/playground-series-s5e10/train.csv'
test_path = '/kaggle/input/playground-series-s5e10/test.csv'

sub_df = pd.read_csv(sub_path)
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)


train_df


train_df['accident_risk'].describe()


train_df['accident_risk'].skew()


sns.kdeplot(train_df['accident_risk'])
plt.show()


# Code Source:https://www.kaggle.com/competitions/playground-series-s5e10/discussion/609994#3296622
# --------------------------------------------------------------------------------------------------

import scipy

def clip(f):
    def clip_f(X):
        sigma = 0.05
        mu = f(X)
        a, b = -mu/sigma, (1-mu)/sigma
        Phi_a, Phi_b = scipy.stats.norm.cdf(a), scipy.stats.norm.cdf(b)
        phi_a, phi_b = scipy.stats.norm.pdf(a), scipy.stats.norm.pdf(b)
        return mu*(Phi_b-Phi_a)+sigma*(phi_a-phi_b)+1-Phi_b
    return clip_f

























































# The Bayes Optimal Clipping Function (adapted from your image)
def clip(f):
    def clip_f(X):
        sigma = 0.05
        mu = f(X) # The mean is the raw model output f(x)
        
        # 1. Calculate standardized bounds (a and b)
        a, b = -mu/sigma, (1-mu)/sigma 
        
        # 2. Get CDF (Phi) and PDF (phi) values from standard normal distribution
        Phi_a, Phi_b = stats.norm.cdf(a), stats.norm.cdf(b)
        phi_a, phi_b = stats.norm.pdf(a), stats.norm.pdf(b)
        
        # 3. Apply the Bayes Optimal Formula
        # E[Y|x] = mu*(Phi(b)-Phi(a)) + sigma*(phi(a)-phi(b)) + 1-Phi(b)
        return mu * (Phi_b - Phi_a) + sigma * (phi_a - phi_b) + 1 - Phi_b
    return clip_f



# Step 1: Define a simple base model f(x)
def dummy_model(x):
    # Let's use a simple linear function: f(x) = 0.5x + 0.2
    return 0.5 * x + 0.2

# Step 2: Generate the Bayes Optimal function
bayes_optimal_predictor = clip(dummy_model)

# Step 3: Test with an input x
x_input = 1.2
prediction = bayes_optimal_predictor(x_input)

# Compare results:
raw_output = dummy_model(x_input)
simple_clip = np.clip(raw_output, 0, 1) # What a simple clip would give

print(f"Input x: {x_input}")
print(f"Raw Model Output f(x): {raw_output:.4f}")
print(f"Simple Clipped Output (min(1, max(0, f(x)))): {simple_clip:.4f}")
print(f"Bayes Optimal Expectation E[Y|x]: {prediction:.4f}")























# Code Source: https://www.kaggle.com/competitions/playground-series-s5e10/discussion/609994#3296513
# --------------------------------------------------------------------------------------------------

"""
from sklearn.base import BaseEstimator, RegressorMixin

class FunctionRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, func):
        self.func = func
    def fit(self, X, y=None):
        return self
    def predict(self, X):
        return self.func(X)

# data generation model from https://www.kaggle.com/code/ianktoo/simulated-road-accident-data-generator
def f(X):
    return \
    0.3 * X["curvature"] + \
    0.2 * (X["lighting"] == "night").astype(int) + \
    0.1 * (X["weather"] != "clear").astype(int) + \
    0.2 * (X["speed_limit"] >= 60).astype(int) + \
    0.1 * (X["num_reported_accidents"] > 2).astype(int)

X = pd.read_csv('/kaggle/input/simulated-roads-accident-data/synthetic_road_accidents_100k.csv')
y = X.pop('accident_risk')

from sklearn.model_selection import cross_val_score, KFold

kfold = KFold(10, shuffle=True, random_state=0)
model = FunctionRegressor(f)

scores = cross_val_score(
    model, X, y, cv=kfold,
    scoring='r2', n_jobs=1
)
print(F'{scores.mean():.5f} Â± {scores.std():.5f}')
"""
print('')


def f(X):
    return \
    0.3 * X["curvature"] + \
    0.2 * (X["lighting"] == "night").astype(int) + \
    0.1 * (X["weather"] != "clear").astype(int) + \
    0.2 * (X["speed_limit"] >= 60).astype(int) + \
    0.1 * (X["num_reported_accidents"] > 2).astype(int)





# Code Source:https://www.kaggle.com/competitions/playground-series-s5e10/discussion/609994#3296622
# --------------------------------------------------------------------------------------------------

import scipy

def clip(f):
    def clip_f(X):
        sigma = 0.05
        mu = f(X)
        a, b = -mu/sigma, (1-mu)/sigma
        Phi_a, Phi_b = scipy.stats.norm.cdf(a), scipy.stats.norm.cdf(b)
        phi_a, phi_b = scipy.stats.norm.pdf(a), scipy.stats.norm.pdf(b)
        return mu*(Phi_b-Phi_a)+sigma*(phi_a-phi_b)+1-Phi_b
    return clip_f

