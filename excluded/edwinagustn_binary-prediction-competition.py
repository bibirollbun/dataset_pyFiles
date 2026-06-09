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


#Estyle
import matplotlib.pyplot as plt 
import seaborn as sns 
from tqdm import tqdm


#Models
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.feature_selection import RFECV
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

#metrics
from sklearn.metrics import roc_curve, roc_auc_score, auc



"""
Realized by Edwin Agustin
2025-3-25

Classify as: academic project
"""

train = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv')

print(f"train_info:\n{train.info()}\n\n{train.describe()}")
print(f"test_info:\n{test.info()}\n\n{test.describe()}")



test.fillna(test.mean(),inplace=True)
test.info()


#Target variable
Target_Classes = train['rainfall'].value_counts()

#do groupby
plt.figure(figsize = (20,10))
Target_Classes.plot(kind = 'bar',color =['blue','orange'])
plt.title('Target Clases Distribution')
plt.xlabel("Category")
plt.ylabel("Count")
plt.xticks(rotation=0)
plt.show()


#corr plot 
plt.figure(figsize=(10,6))
sns.heatmap(train.corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)

# TÃ­tulo del grÃ¡fico
plt.title("Heatmap - Pearson Correlation")
plt.show()


#preparing features 
X = train.drop(columns=['rainfall','id'])  
y = train['rainfall'].drop(columns =['id'])




#Models Diccionary 
models = {
    'XGBClassifier': XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1),
    'CatBoost': CatBoostClassifier(iterations=100, depth=3, learning_rate=0.1, verbose=0),
    'RandomForest': RandomForestClassifier(n_estimators=100, max_depth=3)
}


   
class Create_and_test:
    """
    DESCRIPTION:
    This class trains models, selects features using Recursive Feature Elimination (RFE),
    and evaluates model performance using cross-validation.

    PARAMETERS:
    - `X`: DataFrame, feature matrix.
    - `y`: Series or array, target variable.
    - `model_name`: str, name of the model to use from `models`.
    - `cv_folds`: int, number of folds for cross-validation (default: 5).

    METHODS:
    - `show_results()`: Prints the number of selected features and cross-validation score.
    - `plot_model()`: Evaluates the trained model using plots and descriptive statistics 
    """

    def __init__(self, X, y, model_name, cv_folds=5):
        """
        Initializes the model, applies Recursive Feature Elimination (RFE) with cross-validation, 
        and selects the optimal features.
        
        Parameters:
        - `X`: Feature matrix (DataFrame).
        - `y`: Target variable (Series or array).
        - `model_name`: Model name (string), must be in `models` dictionary.
        - `cv_folds`: Number of cross-validation folds (default: 5).

        Return:
        - model name
        - the optimal features
        - the name of these features
        - the mean by cros validation AUC_ROC metric and general
        
        
        """
        self.X = X
        self.y = y
        self.model_name = model_name 
        self.model = models[model_name]  # Model selection
        self.test  = test
        
        # Cros-validation stratified
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    
        # RFE with cv
        with tqdm(desc=f"Selecting features with {model_name}..."):
            self.selector = RFECV(self.model, step=1, cv=cv, scoring='roc_auc', n_jobs=-1)
            self.selector.fit(X, y)
        
        # Results from cros-validation
        self.optimal_features = self.selector.n_features_
        self.selected_features = X.columns[self.selector.support_].tolist
        self.all_results = self.selector.cv_results_['mean_test_score'] #extract the mean of cv by model
        self.mean_score = np.mean(self.selector.cv_results_["mean_test_score"]) #extract the mean of all_results variable
        
    def show_results(self):
        """
        Displays the results of the Recursive Feature Elimination process.

        Prints:
        - The optimal number of selected features.
        - The names of the selected features.
        - The average cross-validation accuracy score.
        """
        print(f"\nğŸ”¹ Model: {self.model_name}")
        print(f"ğŸ“Œ Optimal number of features: {self.optimal_features}")
        print(f"âœ… Selected features: {self.selected_features}")
        print(f'âš™ï¸� Results: {self.all_results}')
        print(f"ğŸ�¯ ROC_AUC_MEAN_CV: {self.mean_score:.4f}")
        
    def testing_model(self):
        """
        Display analysis of the model's ROC-AUC score using selected features.
        """

        # Conditional features
        if self.model_name == 'XGBClassifier_F':
            self.X = train[['day', 'pressure', 'maxtemp', 'temparature', 'mintemp',
                            'dewpoint', 'humidity', 'cloud', 'sunshine', 'winddirection',
                            'windspeed']]
            self.y = train['rainfall'] 
        
        elif self.model_name == 'CatBoost_M':
            self.X = train[['pressure', 'maxtemp', 'mintemp', 'dewpoint', 'humidity', 'cloud',
                            'sunshine', 'windspeed']]
            self.y = train['rainfall']  
        
        else:
            self.X = train[['temparature', 'dewpoint', 'humidity', 'cloud', 'sunshine']]
            self.y = train['rainfall']  

        
        # Splitting data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
        
        # Train the selected model with the RFE-selected features
        model = self.model
        model.fit(X_train, y_train)
    
        # Make predictions and calculate ROC-AUC score
        y_probs = model.predict_proba(X_test)[:, 1]  # Get probabilities for class 1
        
        roc_auc = roc_auc_score(y_test, y_probs)

        

        print(f"âš™ï¸� {self.model_name} ROC-AUC on Test Set : {roc_auc:.4f}") #showing results
        print(f'ğŸ“ˆ Plot of ROC_AUC of {self.model_name}: ')
        
        #Plotting ROC_AUC
        fpr, tpr, thresholds = roc_curve(y_test, y_probs) 
        roc_auc = auc(fpr, tpr)
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.show()
        
    def submission(self):
        
        test_features = self.test[self.X.columns]
        
        #Models training
        predictions = self.model.predict_proba(test_features)[:, 1]
        predictions = np.round(predictions, 1) 
        
        # Create submission file
        submission_df = pd.DataFrame({"id": self.test['id'], "rainfall": predictions})
        submission_df.to_csv('submission.csv', index=False)
        print(submission_df.head(10))
                


#Training Models
XGBClassifier= Create_and_test(X,y,'XGBClassifier')
CatBoost = Create_and_test(X,y,'CatBoost')
RandomForest = Create_and_test(X,y,'RandomForest')

#show results
XGBClassifier.show_results()
CatBoost.show_results()
RandomForest.show_results()


#Testing models
XGBClassifier.testing_model()
CatBoost.testing_model()
RandomForest.testing_model()


CatBoost.submission()

