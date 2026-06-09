import pandas as pd
rs_1960 = pd.read_csv('/kaggle/input/rohlik-sfc-quacking-the-forecast/submission.csv') 
rs_1962=pd.read_csv('/kaggle/input/rohit-sales-better-public-dataset/rs_1962.csv') 
rs_1968 = pd.read_csv('/kaggle/input/rohit-sales-better-public-dataset/rs_1968.csv') 


blended = rs_1960.copy()

blended['sales_hat'] = (
    (0.002) * rs_1968['sales_hat'] +
    (0.990) * rs_1960['sales_hat'] +
    (0.008) * rs_1962['sales_hat']
)

# Save the blended results
blended.to_csv('submission.csv', index=False)

blended.head(10)




