!pip install flaml
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time,os
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import lightgbm as lgb
from flaml import AutoML
import polars as pl
import nltk
from nltk.corpus import stopwords
import warnings
warnings.filterwarnings('ignore')



# Read train and test datasets
train = pl.read_csv('/kaggle/input/learning-agency-lab-automated-essay-scoring-2/train.csv')
test = pl.read_csv('/kaggle/input/learning-agency-lab-automated-essay-scoring-2/test.csv')
submission=pl.read_csv('/kaggle/input/learning-agency-lab-automated-essay-scoring-2/sample_submission.csv')



# Display the first few rows of each dataset
print("train:")
print(train.head())
print("\ntest :")
print(test.head())



print(len(train))
print(len(test))


# Summary statistics for numerical columns
print("\nSummary statistics for train dataset:")
print(train.describe())



print("Summary statistics for test dataset :")
print(test.describe())


%%time
#This code hrlp from that notebook-->[https://www.kaggle.com/code/davidjlochner/base-tfidf-lgbm/notebook]
# TF-IDF Vectorization
vectorizer = TfidfVectorizer(min_df=.05)
train_tfid = vectorizer.fit_transform(train['full_text'])
test_tfid = vectorizer.transform(test['full_text'])

train_y = np.array(train['score'])

# Initialize AutoML for hyperparameter optimization
aml = AutoML()

# Fit AutoML to find the best hyperparameters
aml.fit(train_tfid, train_y, estimator_list=['lgbm'], task='classification', metric='macro_f1', time_budget=600)

# Retrieve the best hyperparameters found by AutoML
best_config = aml.best_config

# Initialize LGBMClassifier with the best hyperparameters found
model = lgb.LGBMClassifier(**best_config)

# Train the model
model.fit(train_tfid, train_y)


# Predict scores for test data using the model
submission = test.select('essay_id').with_columns(score=model.predict(test_tfid))

# Display the submission data
display(submission)

# Write the submission to a CSV file
submission.write_csv('submission.csv')

# Add insights
print("Submission generated successfully.")


