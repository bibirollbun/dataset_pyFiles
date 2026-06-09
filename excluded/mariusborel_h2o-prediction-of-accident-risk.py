!pip install --upgrade h2o


import h2o
from h2o.automl import H2OAutoML

# Start H2O cluster
h2o.init()

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set Seaborn theme with dark grid and brighter palette
my_palette = "tab20"
sns.set_theme(style="darkgrid", palette=my_palette, font_scale=0.9)

# Update matplotlib parameters for brighter dark theme
plt.rcParams.update({
    'axes.facecolor': '#333333',       # Slightly lighter than #222222
    'figure.facecolor': '#333333',
    'text.color': '#ffd700',           # Bright gold for better contrast
    'axes.labelcolor': '#ffd700',      # Softer mint green
    'xtick.color': '#ffd700',
    'ytick.color': '#ffd700',
    'grid.color': '#666666',           # Lighter grid lines
    'axes.edgecolor': '#dddddd'        # Light gray edges
})
# verify the versions
print(f'h2oautml version: {h2o.__version__}')

import warnings
warnings.filterwarnings('ignore')


# Load your dataset
train = h2o.import_file("/kaggle/input/playground-series-s5e10/train.csv", skipped_columns=[0])
test = h2o.import_file("/kaggle/input/playground-series-s5e10/test.csv", skipped_columns=[0])
orig = h2o.import_file("/kaggle/input/simulated-roads-accident-data/synthetic_road_accidents_10k.csv")

# Set the target
target = 'accident_risk'


train_comb = train.rbind(orig)
train_comb


# Define target and features
features = [col for col in train.columns if col != target]

# Run AutoML for regression
aml = H2OAutoML(max_models=10, seed=28, sort_metric='RMSE', nfolds=4)
aml.train(x=features, y=target, training_frame=train_comb)

# View leaderboard
lb = aml.leaderboard
print(lb)


# Predict on test set
preds = aml.leader.predict(test)

preds.as_data_frame()


submission = pd.read_csv('/kaggle/input/playground-series-s5e10/sample_submission.csv')


submission[target] = preds.as_data_frame(use_multi_thread=True)

submission.head()


sns.histplot(submission, x=target, kde=True, bins=50, color='gold')
plt.title('Distribution of the predictions on test data')
plt.show()


submission.to_csv('submission.csv', index=False)
print('ðŸ¥‚The submission file is ready!')

