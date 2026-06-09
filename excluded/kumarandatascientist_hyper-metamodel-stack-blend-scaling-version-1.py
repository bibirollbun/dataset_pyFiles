# you can use your own diverse ML model 
import pandas as pd
no_model = pd.read_csv('/kaggle/input/private-1/submission (19).csv') # 0.05415
linear_reg=pd.read_csv('/kaggle/input/blending-forecast/linear_reg_predict.csv') 
lgbm_reg = pd.read_csv('/kaggle/input/blending-forecast/lgbm_predict.csv') 


blended = no_model.copy()

blended['num_sold'] = (
    (0.11) * lgbm_reg['num_sold'] +
    (0.09) * linear_reg['num_sold'] +
    (0.80) * no_model['num_sold'] 
)
# Save the blended results
blended.to_csv('submission.csv', index=False)

blended.head(10)


from sklearn.linear_model import Ridge
import pandas as pd

# Base model predictions
X = pd.DataFrame({
    'lgbm': lgbm_reg['num_sold'],
    'linear': linear_reg['num_sold'],
    'no_model': blended['num_sold']
})

# Ground truth (actual values)
y = blended['num_sold']  # Replace with actual target values

# Train the meta-model on the entire dataset
meta_model = Ridge(alpha=1.0)
meta_model.fit(X, y)

# Make predictions for the entire dataset
blended_predictions = meta_model.predict(X)
blended_predictions_1 = blended_predictions * 0.98
# Create a DataFrame for submission
submission1 = pd.DataFrame({
    'id': no_model['id'],  # Use the IDs from one of your models
    'num_sold': blended_predictions_1
})

# Save the submission file
submission1.to_csv('submission1.csv', index=False)

# Print first few rows to confirm
print(submission1.head())



from sklearn.linear_model import Ridge
import pandas as pd

# Base model predictions
X = pd.DataFrame({
    'lgbm': lgbm_reg['num_sold'],
    'linear': linear_reg['num_sold'],
    'no_model': blended['num_sold']
})

# Ground truth (actual values)
y = blended['num_sold']  # Replace with actual target values

# Train the meta-model on the entire dataset
meta_model = Ridge(alpha=1.0)
meta_model.fit(X, y)

# Make predictions for the entire dataset
blended_predictions = meta_model.predict(X)
blended_predictions_2 = blended_predictions * 0.9
# Create a DataFrame for submission
submission2 = pd.DataFrame({
    'id': no_model['id'],  # Use the IDs from one of your models
    'num_sold': blended_predictions_2
})

# Save the submission file
submission2.to_csv('submission2.csv', index=False)

# Print first few rows to confirm
print(submission2.head())



final_blended = no_model.copy()

final_blended['num_sold'] = (
    (0.11) * submission1['num_sold'] +
    (0.09) * submission2['num_sold'] +
    (0.80) * no_model['num_sold'] 
)
# Save the blended results
final_blended.to_csv('hyper_blend_submission.csv', index=False)

final_blended.head(10)


#Hyper Stacking and Hyper scaling with Meta model
from sklearn.linear_model import Ridge
import pandas as pd

# Base model predictions
X = pd.DataFrame({
    'submission1': submission1['num_sold'],
    'submission2': submission2['num_sold'],
    'blended': final_blended['num_sold']
})

# Ground truth (actual values)
y = final_blended['num_sold']  # Replace with actual target values

# Train the meta-model on the entire dataset
meta_model = Ridge(alpha=1.0)
meta_model.fit(X, y)

# Make predictions for the entire dataset
blended_predictions = meta_model.predict(X)
blended_predictions_2 = blended_predictions * 0.95
# Create a DataFrame for submission
final_submission = pd.DataFrame({
    'id': no_model['id'],  # Use the IDs from one of your models
    'num_sold': blended_predictions_2
})

# Save the submission file
final_submission.to_csv('submission.csv', index=False)

# Print first few rows to confirm
print(final_submission.head())





