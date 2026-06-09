!pip install -f https://h2o-release.s3.amazonaws.com/h2o/latest_stable_Py.html h2o


import h2o
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from h2o.automl import H2OAutoML
import warnings
warnings.filterwarnings("ignore")


h2o.init()



train = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')



for df in [train, test]:
    df['Sex'] = df['Sex'].astype('category')
   # df['Height_m'] = df['Height'] / 100
   # df['BMI'] = df['Weight'] / (df['Height_m'] ** 2)
   # df['Duration_per_kg'] = df['Duration'] / df['Weight']
   # df['Heart_Temp_Interaction'] = df['Heart_Rate'] * df['Body_Temp']
   # df.drop('Height_m', axis=1, inplace=True)  # Drop temp feature
    


hf_train = h2o.H2OFrame(train)
hf_test = h2o.H2OFrame(test)



x = [col for col in train.columns if col not in ['id', 'Calories']]
y = 'Calories'


aml = H2OAutoML(
    max_models=5,
    seed=42,
    sort_metric="RMSE",
    include_algos=["GBM", "XGBoost", "DeepLearning", "GLM", "DRF", "StackedEnsemble"]
)
aml.train(x=x, y=y, training_frame=hf_train)


lb = aml.leaderboard
print(lb.head(rows=5))



best_model = aml.leader
print("\nBest model used:", best_model.algo)


hf_preds = aml.predict(hf_test)
preds = hf_preds.as_data_frame().values.flatten()



submission = pd.DataFrame({
    'id': test['id'],
    'Calories': np.maximum(0, preds)
})
submission.to_csv('submission.csv', index=False)
print("Saved: submission.csv")



h2o.shutdown(prompt=False)




