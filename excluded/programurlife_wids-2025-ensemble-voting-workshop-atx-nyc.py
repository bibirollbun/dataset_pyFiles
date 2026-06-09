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


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder


kaggle = True
main_dir = '/kaggle/input/widsdatathon2025/' if kaggle else '../widsdatathon2025/'

train_connectome = pd.read_csv(f"{main_dir}/TRAIN/TRAIN_FUNCTIONAL_CONNECTOME_MATRICES.csv")
test_connectome = pd.read_csv(f"{main_dir}/TEST/TEST_FUNCTIONAL_CONNECTOME_MATRICES.csv")
train_quantitative = pd.read_excel(f"{main_dir}/TRAIN/TRAIN_QUANTITATIVE_METADATA.xlsx")
train_categorical = pd.read_excel(f"{main_dir}/TRAIN/TRAIN_CATEGORICAL_METADATA.xlsx")
test_quantitative = pd.read_excel(f"{main_dir}/TEST/TEST_QUANTITATIVE_METADATA.xlsx")
test_categorical = pd.read_excel(f"{main_dir}/TEST/TEST_CATEGORICAL.xlsx")
train_solutions = pd.read_excel(f"{main_dir}/TRAIN/TRAINING_SOLUTIONS.xlsx")
sample_submission = pd.read_excel(f'{main_dir}/SAMPLE_SUBMISSION.xlsx')


X_solutions = pd.merge(train_solutions, train_connectome, on='participant_id')
X_solutions.set_index('participant_id', inplace=True)
train_connectome.set_index('participant_id', inplace=True)

X_adhd_outcome = X_solutions.drop(['ADHD_Outcome', 'Sex_F'], axis=1)
y_adhd_outcome = X_solutions['ADHD_Outcome']

X_sex_f = X_solutions.drop(['Sex_F', 'ADHD_Outcome'], axis=1)
y_sex_f = X_solutions['Sex_F']


X_train, X_test, y_train_adhd, y_test_adhd, y_train_s, y_test_s = train_test_split(
    train_connectome, y_adhd_outcome, y_sex_f, test_size=0.2, random_state=42)


rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
gb_classifier = GradientBoostingClassifier(n_estimators=100, random_state=42)


# Create an ensemble model using VotingClassifier (stacking)
ensemble_model= VotingClassifier(
    estimators=[('rf', rf_classifier), ('gb', gb_classifier)],
    voting='hard'  # 'hard' for majority class voting
)


# Train the ensemble model
ensemble_model.fit(X_train, y_train_adhd) 


# Make predictions for the first target
y_pred= ensemble_model.predict(X_test)
print(f"Accuracy for target1 (classification): {accuracy_score(y_test_adhd, y_pred):.4f}")


# Train the ensemble model for the second target
ensemble_model.fit(X_train, y_train_s)  # Train for the second target (y_train2)


# Make predictions for the second target
y_pred_s = ensemble_model.predict(X_test)
print(f"Accuracy for target2 (classification): {accuracy_score(y_test_s, y_pred_s):.4f}")


test_connectome.set_index('participant_id', inplace=True)


# Predict the final test data using the trained ensemble model for target1
final_pred1 = ensemble_model.predict(test_connectome)

# Predict the final test data using the trained ensemble model 
final_pred2 = ensemble_model.predict(test_connectome)


submission = pd.DataFrame({
    'participant_id': test_connectome.index,
    'ADHD_Outcome': final_pred1,
    'Sex_F': final_pred2,
})

submission['ADHD_Outcome'] = submission['ADHD_Outcome'].astype(int)
submission['Sex_F'] = submission['Sex_F'].astype(int)

submission.to_csv('submission.csv', index=False)

