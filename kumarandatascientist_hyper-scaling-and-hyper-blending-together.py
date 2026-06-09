import pandas as pd
no_model = pd.read_csv('/kaggle/input/74747363737/submission (4).csv') # 0.05415
linear_reg=pd.read_csv('/kaggle/input/blending-forecast/linear_reg_predict.csv') 
lgbm_reg = pd.read_csv('/kaggle/input/231232434324/submission (29).csv') 


blended = no_model.copy()

blended['num_sold'] = (
    (0.11) * lgbm_reg['num_sold'] +
    (0.09) * linear_reg['num_sold'] +
    (0.80) * no_model['num_sold'] 
) 
# Save the blended results
#blended.to_csv('submission.csv', index=False)

blended.head(10)


blended = no_model.copy()

blended['num_sold'] = (
    (0.11) * lgbm_reg['num_sold'] +
    (0.09) * linear_reg['num_sold'] +
    (0.80) * no_model['num_sold'] 
)*1.00123
# Save the blended results
blended.to_csv('submission.csv', index=False)

blended.head(10)




