!pip install ydf -U


import ydf
import pandas as pd


train = pd.read_csv("/kaggle/input/playground-series-s5e4/train.csv", index_col='id')
test = pd.read_csv("/kaggle/input/playground-series-s5e4/test.csv")


train.head()


tuner = ydf.RandomSearchTuner(num_trials=100)


model = ydf.GradientBoostedTreesLearner(label="Listening_Time_minutes",
                                        tuner=tuner,
                                        task=ydf.Task.REGRESSION).train(train)


model.describe()


model.hyperparameter_optimizer_logs()


evaluation = model.predict(test)

print(evaluation)


submission = pd.DataFrame({
    "id": test["id"],  # Replace with your test set's ID column name
    "Prediction": evaluation  # Replace with your prediction column name if needed
})
submission.to_csv("submission.csv", index=False)




