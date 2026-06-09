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
    'no_model': no_model['num_sold']
})

# Ground truth (actual values)
y = blended['num_sold']  # Replace with actual target values

# Train the meta-model on the entire dataset
meta_model = Ridge(alpha=1.0)
meta_model.fit(X, y)

# Make predictions for the entire dataset
blended_predictions = meta_model.predict(X)
blended_predictions = blended_predictions * 0.98
# Create a DataFrame for submission
submission = pd.DataFrame({
    'id': no_model['id'],  # Use the IDs from one of your models
    'num_sold': blended_predictions
})

# Save the submission file
submission.to_csv('submission.csv', index=False)

# Print first few rows to confirm
print(submission.head())





