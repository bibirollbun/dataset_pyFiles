import pandas as pd
# Ignore warnings
import warnings
warnings.filterwarnings('ignore')

# load data
train = pd.read_csv("/kaggle/input/playground-series-s5e2/train.csv")
train_extra = pd.read_csv("/kaggle/input/playground-series-s5e2/training_extra.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e2/test.csv")

# Merge the train and train_extra datasets
df = pd.concat([train, train_extra], axis=0, ignore_index=True)


# Install H2O
!pip install h2o

# Initialize the H2O cluster
import h2o
h2o.init()


# Convert the training set and the test set
#If you are using the combined data from train and train_extra datasets, uncomment the following
train_h2o = h2o.H2OFrame(df)
#train_h2o = h2o.H2OFrame(train)
test_h2o = h2o.H2OFrame(test)


categorical_cols = ["Brand", "Material", "Size", "Laptop Compartment", 
                    "Waterproof", "Style", "Color"]
for col in categorical_cols:
    train_h2o[col] = train_h2o[col].asfactor()
    test_h2o[col] = test_h2o[col].asfactor()


# No use train_extra, LB score 39.12863

from h2o.automl import H2OAutoML

features = train_h2o.columns
features.remove("Price")
features.remove("id") 

aml = H2OAutoML(
    max_models=20,     # maximum number of models
    seed=42,          
    include_algos=["XGBoost", "GBM", "DRF"]  # Specifies the algorithm to use
)

aml.train(
    x=features,
    y="Price",
    training_frame=train_h2o
)

# View model ranking
lb = aml.leaderboard
print(lb)


preds = aml.predict(test_h2o)
submission = test_h2o["id"].as_data_frame()
submission["Price"] = preds.as_data_frame()
submission.to_csv("submission.csv", index=False)


submission.head()

