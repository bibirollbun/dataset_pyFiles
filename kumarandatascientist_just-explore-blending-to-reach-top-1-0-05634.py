import pandas as pd
no_model = pd.read_csv('/kaggle/input/blending-forecast/just_forecast_no_model.csv') # 0.05663
linear_reg=pd.read_csv('/kaggle/input/blending-forecast/linear_reg_predict.csv') 
lgbm_reg = pd.read_csv('/kaggle/input/blending-forecast/lgbm_predict.csv') 


blended = no_model.copy()

blended['num_sold'] = (
    (0.39) * lgbm_reg['num_sold'] +
    (0.53) * linear_reg['num_sold'] +
    (0.08) * no_model['num_sold'] 
)
# Save the blended results
blended.to_csv('submission.csv', index=False)

blended.head(10)




