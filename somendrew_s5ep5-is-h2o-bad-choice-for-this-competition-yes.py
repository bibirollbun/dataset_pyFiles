import pandas as pd
import h2o
from h2o.automl import H2OAutoML
from itertools import combinations
from scipy.stats import gmean, hmean
from scipy import stats
import matplotlib.pyplot as plt
import numpy as np
from autogluon.tabular import TabularPredictor
from sklearn.metrics import mean_squared_error

h2o.init()


df_train = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv')
df_test  = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')

df_train.drop(columns=['id'], inplace=True)
df_test.drop(columns=['id'], inplace=True)


numerical_features = ['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp']

def add_feature_cross_terms(df, numerical_features):
    df_new = df.copy()
    for i in range(len(numerical_features)):
        for j in range(i + 1, len(numerical_features)):
            feature1 = numerical_features[i]
            feature2 = numerical_features[j]
            cross_term_name = f"{feature1}_x_{feature2}"
            df_new[cross_term_name] = df_new[feature1] * df_new[feature2]
    return df_new

df_train = add_feature_cross_terms(df_train, numerical_features)
df_test = add_feature_cross_terms(df_test, numerical_features)


df_train.head()


train_data = h2o.H2OFrame(df_train)


from h2o.frame import H2OFrame
with h2o.utils.threading.local_context(polars_enabled=True, datatable_enabled=True):
    pandas_df = train_data.as_data_frame()


train_data = h2o.H2OFrame(df_train)


test_data = h2o.H2OFrame(df_test)


test_data


aml = H2OAutoML(max_runtime_secs=100, seed=42, sort_metric="RMSLE",distribution="AUTO",nfolds=5)



aml.train(y='Calories', training_frame=train_data)


leaderboard = aml.leaderboard
print(leaderboard)
best_model = aml.leader
print(best_model)


df_test = h2o.H2OFrame(df_test)


h2o_preds = best_model.predict(df_test)


# df_sub = pd.read_csv('/kaggle/input/playground-series-s5e5/sample_submission.csv')
# df_sub['Calories'] =(predictions_df['predict'].values)
# df_sub.to_csv('submission.csv', index=False)
# df_sub['Calories'].hist()


# glu_preds = predictor.predict(df_test)

# glu_pred_clipped = glu_preds.clip(0)
# h2o_preds_clipped = h2o_preds.clip(0)





# final_preds = np.mean(glu_preds_clipped,h2o_preds_clipped)

# final_preds


# final_preds2 = np.mean(glu_pred,h2o)
# final_preds2

