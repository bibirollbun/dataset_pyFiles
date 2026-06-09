import h2o
from h2o.automl import H2OAutoML


h2o.init()


train = h2o.import_file("/kaggle/input/playground-series-s5e3/train.csv")


train


y = "rainfall"
x = train.columns
x.remove(y)


train[y] = train[y].asfactor()


aml = H2OAutoML(max_runtime_secs=3600, seed=1)
aml.train(x=x, y=y, training_frame=train)


test = h2o.import_file("/kaggle/input/playground-series-s5e3/test.csv")


test


predictions = aml.leader.predict(test)


predictions


h2o.download_csv(predictions, "predictions.csv")

