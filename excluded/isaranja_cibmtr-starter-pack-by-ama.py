
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

trn = pd.read_csv('/kaggle/input/equity-post-HCT-survival-predictions/train.csv')
tst = pd.read_csv('/kaggle/input/equity-post-HCT-survival-predictions/test.csv')
sub = pd.read_csv('/kaggle/input/equity-post-HCT-survival-predictions/sample_submission.csv')
des = pd.read_csv('/kaggle/input/equity-post-HCT-survival-predictions/data_dictionary.csv')

with pd.option_context('display.max_columns', None): # setting the max rows
    display(tst.head())



# risk_score suggested by gpt
def gpt_risk_score(efs: pd.Series, efs_time: pd.Series, alpha=1, epsilon=1e-6) -> pd.Series:
    """
    Calculate risk scores for survival data.

    Args:
        efs (pd.Series): Event occurrence indicator (1 if event occurred, 0 if censored).
        efs_time (pd.Series): Time to event or censoring.
        alpha (float): Weight for the event indicator in the risk score.
        epsilon (float): Small constant to avoid division by zero.

    Returns:
        pd.Series: Risk scores for each individual.
    """
    # Ensure no zero values in efs_time to avoid division by zero
    efs_time = efs_time + epsilon

    # Calculate risk score
    risk_score = (1 / efs_time) * (efs + alpha)
    return risk_score

# calculating the ris score
trn['risk_gpt'] = gpt_risk_score(trn.efs, trn.efs_time)



# installing lifelines library
!pip install -q /kaggle/input/pip-install-lifelines/autograd-1.7.0-py3-none-any.whl
!pip install -q /kaggle/input/pip-install-lifelines/autograd-gamma-0.5.0.tar.gz
!pip install -q /kaggle/input/pip-install-lifelines/interface_meta-1.3.0-py3-none-any.whl
!pip install -q /kaggle/input/pip-install-lifelines/formulaic-1.0.2-py3-none-any.whl
!pip install -q /kaggle/input/pip-install-lifelines/lifelines-0.30.0-py3-none-any.whl

# importing evaluation metric
from metric import score



# Splitting data
from sklearn.model_selection import train_test_split

X_trn, X_val = train_test_split(trn, test_size=0.2, random_state=42, stratify=trn['race_group'])



import h2o
from h2o.automl import H2OAutoML

# Initialize H2O cluster
h2o.init(verbose=False)

# Convert pandas DataFrame to H2OFrame
trn_hf = h2o.H2OFrame(trn)

# Define the target and feature columns
y = 'risk_gpt'
X = trn_hf.columns
X.remove('risk_gpt')
X.remove('efs')
X.remove('efs_time')
X.remove('ID')


# Split the data into training and testing sets
#trn_hf, val_hf = trn_hf.split_frame(ratios=[.8], seed=42)

# Initialize H2O AutoML model
aml = H2OAutoML(max_models=32, seed=1, max_runtime_secs=3600)

# Train the model
_ = aml.train(x=X, y=y, training_frame=trn_hf)

leaderboard_df = aml.leaderboard.as_data_frame(use_multi_thread=True)
display(leaderboard_df)



# Get the leader model (best model from AutoML run)
leader_model = aml.leader

# Make predictions on the validation set
# Convert pandas DataFrame to H2OFrame
val_hf = h2o.H2OFrame(X_val)
val_hf['prediction'] = leader_model.predict(val_hf)

# Convert the predictions to pandas DataFrame for easier inspection
val_df = val_hf.as_data_frame(use_multi_thread=True)
y_pred = val_df.prediction
y_true = val_df['risk_gpt'].values

# evaluate model performance (e.g., Root Mean Squared Log Error)

sci = score(val_df[['ID','efs','efs_time','race_group']],val_df[['ID','prediction']],'ID')

print('\n')
print(f'Stratified Concordance Index: {sci:.3f}')



# Submission file generation.
tst_hf= h2o.H2OFrame(tst)
tst_hf['donor_age'] = tst_hf['donor_age'].asnumeric()
tst_hf['prediction'] = leader_model.predict(tst_hf)

tst_df = tst_hf.as_data_frame(use_multi_thread=True)
submission = tst_df[['ID','prediction']]
submission.to_csv('submission.csv')

print("Submission file generated")

