!pip install ydf -q


import pandas as pd 
import ydf


train = pd.read_csv(r'/kaggle/input/playground-series-s5e5/train.csv')
test = pd.read_csv(r'/kaggle/input/playground-series-s5e5/test.csv')
submission = pd.read_csv(r'/kaggle/input/playground-series-s5e5/sample_submission.csv')
target = "Calories"


# Train a Gradient Boosted Trees model
model = ydf.GradientBoostedTreesLearner(label=target, task=ydf.Task.REGRESSION).train(train)

# Look at a model (input features, training logs, structure, etc.)
model.describe()

# Evaluate a model (e.g. roc, accuracy, confusion matrix, confidence intervals)
#model.evaluate(test) # requires labels

# Generate predictions
y_preds = model.predict(test)

# Analyse a model (e.g. partial dependence plot, variable importance)
# model.analyze(test) # requires labels

# Benchmark the inference speed of a model
model.benchmark(test)
submission[target] = y_preds
submission[target] = submission[target].clip(0)
submission.to_csv("submission.csv", index=False)
submission

