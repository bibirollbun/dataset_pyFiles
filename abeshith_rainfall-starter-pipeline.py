import h2o
import pandas as pd
from h2o.automl import H2OAutoML


# Initialize H2O
h2o.init()


submission_data = pd.read_csv("/kaggle/input/playground-series-s5e3/sample_submission.csv")
train = pd.read_csv("/kaggle/input/playground-series-s5e3/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e3/test.csv")


train.describe().T


train.info()


train['rainfall'].value_counts()


train.isnull().sum()


# Convert Pandas to H2O Frame
train_h2o = h2o.H2OFrame(train)
test_h2o = h2o.H2OFrame(test)


target = "rainfall"  
features = [col for col in train.columns if col != target and col != "id"]


# Convert Target to Categorical (for Classification)
train_h2o[target] = train_h2o[target].asfactor()


aml = H2OAutoML(max_models=50, 
                seed=42, 
                balance_classes=True, 
                sort_metric="logloss",
                stopping_metric="AUC",
                stopping_rounds=5,
                verbosity="info",
                nfolds=5, 
               )  
aml.train(x=features, y=target, training_frame=train_h2o)


# Get Leader Model
best_model = aml.leader
print(best_model)


# Make Predictions
test_preds = best_model.predict(test_h2o).as_data_frame()["predict"]


submission = pd.DataFrame({"id": submission_data["id"], "rainfall": test_preds})
submission.to_csv("submission.csv", index=False)
print("✅ Submission file saved as submission.csv")


submission


submission_data




