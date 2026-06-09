import pandas as pd
import numpy as np


train = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv')
train.head()


test = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')
test.head()


def feature_engineering(df):
    # Body Mass Index (BMI)
    df["BMI"] = df["Weight"] / ((df["Height"] / 100) ** 2)
    
    # Intensity index
    df['Intensity_Index'] = df['Heart_Rate'] / df['Duration']
    
    # Log transformations 
    df['Age'] = np.log1p(df['Age'])
    df['Body_Temp'] = np.log1p(df['Body_Temp'])

    # BMR
    df['BMR'] = (
        10 * df['Weight'] + 
        6.25 * df['Height'] - 
        5 * df['Age'] + 
        np.where(df['Sex'] == 'male', 5, -161)
    )

    # Interaction features
    df['HR_Temp_Interaction'] = df['Heart_Rate'] * df['Body_Temp']
    df['HR_Duration_Interaction'] = df['Heart_Rate'] * df['Duration']
    df['Metabolic_Load'] = df['Heart_Rate'] * df['Body_Temp'] * df['Duration']  # Same as HR_Duration_Temp
    df['Age_Duration'] = df['Age'] * df['Duration']
    df['BMI_HR'] = df['BMI'] * df['Heart_Rate']
    df['Age_Body_Temp'] = df['Age'] * df['Body_Temp']
    df['Duration_Body_Temp'] = df['Duration'] * df['Body_Temp']
    df['BMI_Body_Temp'] = df['BMI'] * df['Body_Temp']
    df['Age_Duration_Temp'] = df['Age'] * df['Duration'] * df['Body_Temp']

    # Log transform Calories only for training set
    if 'Calories' in df.columns:
        df['Calories'] = np.log1p(df['Calories'])

    return df
    
# Apply to both datasets
train = feature_engineering(train)
test = feature_engineering(test)


# Encode 'Sex' to numeric values
train['Sex'] = train['Sex'].map({'male': 1, 'female': 0}).astype(int)
test['Sex'] = test['Sex'].map({'male': 1, 'female': 0}).astype(int)


test_df = test.drop(['id'], axis = 1)

X = train.drop(['Calories', 'id'], axis = 1)
y = train['Calories']


from sklearn.model_selection import train_test_split

# Train-test split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


!pip install ray==2.10.0


!pip install autogluon.tabular --no-cache-dir -q
!pip install -U ipywidgets


!pip uninstall -y scikit-learn
!pip install scikit-learn==1.2.2
!pip install autogluon --no-deps


import numpy as np
from autogluon.core.metrics import make_scorer
from autogluon.tabular import TabularPredictor
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Define RMSLE custom evaluation metric
def rmsle_func(y_true, y_pred):
    y_true = np.maximum(0, y_true)
    y_pred = np.maximum(0, y_pred)
    return np.sqrt(np.mean(np.square(np.log1p(y_pred) - np.log1p(y_true))))

# Wrap the custom metric using AutoGluon
rmsle = make_scorer(
    name='RMSLE',
    score_func=rmsle_func,
    optimum=0,
    greater_is_better=False
)

# Define label column
label = 'Calories'

# Initialize TabularPredictor
predictor = TabularPredictor(
    label=label,
    path='/kaggle/working/AutogluonCalories',
    problem_type='regression',
    eval_metric=rmsle,
    verbosity=2
)

# Fit the predictor
predictor.fit(
    train_data=train,
    time_limit=3600*2,  # 30 minutes
    presets='best_quality',
    excluded_model_types=['KNN'],
    ag_args_fit={'num_cpus': 4}
)



predictor.leaderboard(silent=True)


test_ids = test['id']

# Predict
preds = predictor.predict(test)

# Create submission file
submission = pd.DataFrame({
    'id': test_ids,
    'Calories': preds
})
submission.to_csv('submission.csv', index=False)


