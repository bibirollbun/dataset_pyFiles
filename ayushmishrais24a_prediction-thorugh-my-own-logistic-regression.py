# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import numpy as np  # (it is like a calculator on steroids!)
import pandas as pd   # (it is like Excel but better)
import matplotlib.pyplot as plt  # For creating charts and graphs

# Note: We're NOT using scikit-learn's LogisticRegression class - we'll build our own!
# But we CAN use their helpful tools for encoding, metrics, etc.
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Let's set a random seed so our results are consistent
np.random.seed(42)

print(" Note: We'll build our own Logistic Regression class, but use scikit-learn for encoding and metrics!")



# Let's visualize the sigmoid function to understand it better
def sigmoid(z):
    """
    Args:
        z: Input value (can be any number)
    
    Returns:
        Probability between 0 and 1
    """
    return 1 / (1 + np.exp(-z))



# Let's see what sigmoid does to different numbers
x = np.linspace(-10, 10, 100)  # Create 100 values from -10 to 10
y = sigmoid(x)

plt.figure(figsize=(10, 6))
plt.plot(x, y, linewidth=2, color='blue')
plt.axhline(y=0.5, color='red', linestyle='--', label='Decision Boundary (0.5)')
plt.xlabel('Input Value (z)', fontsize=12)
plt.ylabel('Probability (sigmoid(z))', fontsize=12)
plt.title('Sigmoid Function', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()


import numpy as np

class MyLogisticRegression:
    """
    We will first write a method to initiate our settings 
    
    Parameters:
    -----------
    learning_rate (float): How fast the model learns -- it is a hyperparameter
        - Too high: Model learns too fast and makes mistakes
        - Too low: Model learns too slowly
        
    max_iterations (int): How many times to practice learning
        - More iterations = more practice = better learning
        - But sometimes too high value is not required as it wastes time after plateau
        
    lambda_reg (float): Regularization strength (prevents overfitting)
        - Helps model not memorize training data too much
        - Makes model work better on new data
    """

    def __init__(self, learning_rate=0.01, max_iterations=2000, lambda_reg=0.01):
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.lambda_reg = lambda_reg
        # These we will be learning during training
        self.weights = None
        self.bias = None 

    def sigmoid(self, z):
        """
        Now we will write a sigmoid function - convert any num to probabilities (b/w 0 & 1)
        
        Args:
            z: Input values (can be a single number or array)
        
        Returns:
            Probability between 0 and 1
            
        Why Sigmoid ??
        - Our calculation might give us any number (-1000, 500, etc.)
        - But we need probabilities (between 0 and 1) Hence Sigmoid.
        """
        # If input is sparse matrix (special data format), convert to regular array
        if hasattr(z, "toarray"):
            z = z.toarray().ravel()

        # Clip values to prevent overflow (math errors with very large numbers)
        z = np.clip(z, -500, 500)  

        # Apply sigmoid formula
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        """
        Now we will TRAIN THE MODEL
        
        Think of this like teaching a small kid or a student:
        - Show them examples (X = features, y = correct answers)
        - Let them practice and learn from mistakes
        - Adjust their understanding each time
        
        Args:
            X: Feature data (like income, credit score, etc.)
            y: Target data (0 = didn't pay back, 1 = paid back)
        
        Returns:
            self (the trained model)
        """

        n_samples, n_features = X.shape # Get the Dimension of the data (How many rows and columns)

        # STEP 1 - Initialize parameters
        self.weights = np.random.randn(n_features) * 0.01 # Start with a small random Weight and bias = 0
        self.bias = 0

        #STEP 2 -TRAINING LOOP 
        for iteration in range(self.max_iterations):
            # STEP 2.1: MAKE PREDICTIONS :-Formula: z = (weight1 Ã— feature1) + (weight2 Ã— feature2) + ... + bias
            linear = X.dot(self.weights) + self.bias

            # STEP 2.2: CONVERT TO PROBABILITIES :-# Use sigmoid to convert z to probability
            predictions = self.sigmoid(linear)
            
            # STEP 2.3: CALCULATE ERROR
            error = predictions - y

            # STEP 2.4: CALCULATE GRADIENTS (Direction to improve)
            # Think of gradient as "which way should we adjust our weights?"
            # L2 Regularization prevents weights from getting too large
            dw = (1 / n_samples) * (X.T.dot(error)) + (self.lambda_reg / n_samples) * self.weights
            db = (1 / n_samples) * np.sum(error)

            # STEP 2.5: UPDATE WEIGHTS (Learn from mistakes!)
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

        return self

    def predict_proba(self, X):
        """
        Predict PROBABILITIES for each class
        
        Args:
            X: Feature data for new examples
        
        Returns:
            2D array with probabilities for each class
            - Column 0: Probability of NOT paying back (class 0)
            - Column 1: Probability of paying back (class 1)
        
        Example:
            [[0.2, 0.8],   # 20% chance of default, 80% chance of payback
             [0.9, 0.1]]   # 90% chance of default, 10% chance of payback
        """
        linear = X.dot(self.weights) + self.bias
        p1 = self.sigmoid(linear)
        p0 = 1 - p1
        return np.vstack([p0, p1]).T

    def predict(self, X, threshold=0.5):
        """
        Predict CLASS LABELS (0 or 1)
        
        Args:
            X: Feature data for new examples
            threshold: Decision boundary (default 0.5)
                - If probability >= 0.5, predict 1 (will pay back)
                - If probability < 0.5, predict 0 (won't pay back)
        
        Returns:
            Array of predictions (0s and 1s)
        """
        probabilities = self.predict_proba(X)
        return (probabilities[:, 1] >= threshold).astype(int)


train_df = pd.read_csv('/kaggle/input/playground-series-s5e11/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e11/test.csv')
print("Training Data Shape:", train_df.shape)
print("Test Data Shape:", test_df.shape)
train_df.head()


print(train_df.columns.tolist())
print("\n")
train_df.info()


# Check the target variable (what we want to predict)
print("Target variable distribution:")
print(train_df['loan_paid_back'].value_counts())
print("\nPercentage:")
print(train_df['loan_paid_back'].value_counts(normalize=True) * 100)


y = train_df["loan_paid_back"]
X = train_df.drop("loan_paid_back", axis=1)

categorical_cols = X.select_dtypes(include="object").columns
numeric_cols = X.select_dtypes(exclude="object").columns


preprocess = ColumnTransformer([
    ("num", StandardScaler(), numeric_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
])


model = MyLogisticRegression()

pipe = Pipeline([
    ("prep", preprocess),
    ("clf", model)
])

pipe.fit(X, y)

test_pred = pipe.predict(test_df)

test_pred[:10]


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=42, stratify=y)

from sklearn.metrics import roc_auc_score
test_pred_proba = pipe.predict_proba(X_test)[:, 1]  # Get probability of class 1

# Calculate ROC-AUC score
roc_auc = roc_auc_score(y_test, test_pred_proba)

print(f"\n MODEL PERFORMANCE:")
print(f"   ROC-AUC Score: {roc_auc:.4f}")



# Calculate ROC curve points
fpr, tpr, thresholds = roc_curve(y_test, test_pred_proba)
# Plot ROC curve
plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, linewidth=2, label=f'Our Model (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Guessing (AUC = 0.5)')
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curve - Model Performance Visualization', fontsize=14)
plt.legend(loc='lower right')
plt.grid(True, alpha=0.3)
plt.show()


print(" CONGRATULATIONS! YOU'VE BUILT A LOGISTIC REGRESSION MODEL!")
print("="*70)

print("\n WHAT YOU LEARNED:")
print("    1.How logistic regression works (sigmoid function)")
print("    2.How to build a model from scratch")
print("    3.How to preprocess data (scaling, encoding)")
print("    4.How to train and evaluate a model")
print("    5.What ROC-AUC score means")
print("    6.How to interpret model performance")

print(f"\n YOUR MODEL'S PERFORMANCE:")
print(f"   ROC-AUC Score: {roc_auc:.4f}")
print(f"   This is a {'very good' if roc_auc >= 0.8 else 'good'} model!")

print("\n NEXT STEPS:")
print("   1. Try tuning hyperparameters (learning_rate, max_iterations)")
print("   2. Try feature engineering (creating new features)")
print("   3. Compare with sklearn's LogisticRegression")
print("   4. Make predictions on the test set for submission!")

