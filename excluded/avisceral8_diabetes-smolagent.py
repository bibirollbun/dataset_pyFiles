!pip install category-encoders -q
!pip install smolagents -q
!pip install xgboost lightgbm catboost optuna -q
!pip install -U scikit-learn -q
!pip install -U scipy -q


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import json
import joblib
import logging
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()
OPENROUTER_API_KEY = user_secrets.get_secret("ORKey")


from smolagents import (
    CodeAgent,
    ToolCallingAgent,
    InferenceClientModel,
    WebSearchTool,
    OpenAIModel,
    tool,
)

#Initate the model
model = OpenAIModel(
    # You can use any model ID available on OpenRouter
    model_id="kwaipilot/kat-coder-pro:free",
    # OpenRouter API base URL
    api_base="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
    temperature = 0.1
)

#Test Agent
agent = CodeAgent(tools=[], model=model)
result = agent.run("Calculate the sum of numbers from 1 to 10")
print(result)


# Imports List
imports=[
    "pandas",
    "numpy",
    "category_encoders",
    "sklearn",
    "xgboost",
    "lightgbm",
    "catboost",
    "optuna",
    "sklearn.model_selection",
    "sklearn.metrics",
    "sklearn.ensemble",
    "sklearn.preprocessing",
    "sklearn.linear_model",
    'sklearn.base', 'sklearn.pipeline',
    "os",'io', 'glob']

# Create Agent with Corrected Imports
simple_agent = CodeAgent(
    tools=[],
    model=model,
    additional_authorized_imports=imports,
)

# Revised Task Using scikit-learn's Metrics
task = f"""
1. Use only the following imports:
   {imports}
   
2. Load data with:
   train = pd.read_csv('/kaggle/input/playground-series-s5e12/train.csv')
   test = pd.read_csv('/kaggle/input/playground-series-s5e12/test.csv')
   cat_cols = ['gender', 'ethnicity', 'education_level', 'income_level', 
    'smoking_status', 'employment_status', 
    'family_history_diabetes', 'hypertension_history', 'cardiovascular_history']
   
3. Split the training data into training and validation sets (70/30 split) using train_test_split. Set random_state=42 for reproducibility.

4. Use scikit-learn's `cross_val_score` with `roc_auc_score` to create classification models ['XGBoost', 'CatBoost', 'LightGBM', keeping in mind the categorical columns ]
   - Use best practices
   - Use 5-fold cross-validation
   - Limit hyperparameter tuning to 10 iterations per model
   
5. Perform ensemble modeling:
   - Create an ensemble of the top 3 classifiers using VotingClassifieer 
   - Train the ensemble and indiviudal models on the entire training dataset (including validation folds)
   
6. Apply the final models on test data and create 4 outputs as csv, one for ensemble model and one each of the best models

7. Return the AUC scores for all the of the final models
"""


result = simple_agent.run(task,max_steps=15)
print("Agent execution completed!")
print(result)

