# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler

from sklearn.cluster         import KMeans, \
                                    AgglomerativeClustering, \
                                    Birch, \
                                    MiniBatchKMeans, \
                                    SpectralClustering, \
                                    AffinityPropagation, \
                                    MeanShift, \
                                    OPTICS, \
                                    DBSCAN, \
                                    BisectingKMeans
from sklearn.neighbors       import NearestNeighbors, \
                                    LocalOutlierFactor
from sklearn.ensemble        import IsolationForest, \
                                    RandomTreesEmbedding

from sklearn.model_selection import RandomizedSearchCV

from sklearn.metrics         import silhouette_score, \
                                    calinski_harabasz_score, \
                                    davies_bouldin_score, \
                                    make_scorer



dataset = pd.read_csv('/kaggle/input/physical-activity-clustering/Physical_Activity_Monitoring_unlabeled.csv')

dataset.head()


print("The shape of the dataset is : ", dataset.shape)


dataset.info()


# Check is there any NaN values in our dataset

dataset.isnull().sum()


# for col in dataset.columns:

#   plt.figure(figsize=(5,5))
#   sns.displot(dataset[col])
#   plt.show()



# Let's start handling our NaN values

for col in dataset.columns:
  dataset[col] = dataset[col].fillna(dataset[col].mode()[0])


# Let's handle the ranges of the variables

for col in dataset.columns:

  scaler = MinMaxScaler(
      feature_range=(-1,1)
  )

  dataset[col] = scaler.fit_transform(dataset[[col]])



# Let's remove the outliers

def handle_outliers(df, col):

  Q1 = df[col].quantile(0.25)
  Q3 = df[col].quantile(0.75)

  IQR = Q3 - Q1

  lower_bound = Q1 - 1.5 * IQR
  upper_bound = Q3 + 1.5 * IQR

  df[col] = np.where(df[col] < lower_bound, lower_bound, df[col])
  df[col] = np.where(df[col] > upper_bound, upper_bound, df[col])

  return df


cols_has_outliers=['handAcc16_1', 'handAcc16_2',
       'handAcc16_3', 'handAcc6_1', 'handAcc6_2', 'handAcc6_3', 'handGyro1',
       'handGyro2', 'handGyro3', 'handMagne1', 'handMagne2', 'handMagne3',
       'chestAcc16_1', 'chestAcc16_2',
       'chestAcc16_3', 'chestAcc6_1', 'chestAcc6_2', 'chestAcc6_3',
       'chestGyro1', 'chestGyro2', 'chestGyro3', 'chestMagne1', 'chestMagne2',
       'chestMagne3',
       'ankleAcc16_1', 'ankleAcc16_2', 'ankleAcc16_3', 'ankleAcc6_1',
       'ankleAcc6_2', 'ankleAcc6_3', 'ankleGyro1', 'ankleGyro2', 'ankleGyro3',
       'ankleMagne1', 'ankleMagne2', 'ankleMagne3'
       ]

original_dataset = dataset.copy()
for col in cols_has_outliers:
    dataset = handle_outliers(dataset, col)


# Let's define all our clustering models

clustering_models = {
        # cluster
        'KMeans':                       KMeans(n_clusters=3),
        'MiniBatchKMeans':              MiniBatchKMeans(n_clusters=3),

    }



param_grids = {
    'KMeans': {
        'n_clusters': [2, 3, 4, 5, 6, 7, 8, 9, 10],
        'init': ['k-means++', 'random'],
        'n_init': [10, 20, 30, 40, 50],
        'max_iter': [200, 400, 600, 800, 1000],
        'algorithm': ['lloyd', 'elkan'],
    },
    'MiniBatchKMeans': {
        'n_clusters': [2, 3, 4, 5, 6, 7, 8, 9, 10],
        'init': ['k-means++', 'random'],
        'n_init': [10, 20, 30, 40, 50],
        'max_iter': [200, 400, 600, 800, 1000],
        'batch_size': [100, 200, 300, 400, 500],
    },


}

def silhouette_scorer(estimator, X):
    estimator.fit(X)
    labels = estimator.labels_
    return silhouette_score(X, labels)


def tune_hyperparams(model, params, X, scoring, cv):

  halving_search = RandomizedSearchCV(
      estimator=model,
      param_distributions=params,
      scoring=scoring,
      n_jobs=-1,
      random_state=42,
      cv=cv,
      n_iter=5
  )

  halving_search.fit(X)

  best_params = halving_search.best_params_
  best_estimator = halving_search.best_estimator_
  best_score = halving_search.best_score_

  return best_estimator, best_params, best_score

def tune_models(models, models_params, X, scoring, cv=3):

  for model_name, model in models.items():

    print(f'Tuning {model_name}...')

    best_estimator, best_params, best_score = tune_hyperparams(model=model,
                                                               params=models_params[model_name],
                                                               X=X,
                                                               scoring=scoring,
                                                               cv=cv
                                                               )

    models[model_name] = best_estimator

    print(f'Best parameters: {best_params}')
    print(f'Best score: {best_score}')
    print()

  return models



# tune_models(
#     models=clustering_models,
#     models_params=param_grids,
#     X=dataset,
#     scoring=silhouette_scorer
# )


kmeans_best_params={'n_init': 10, 'n_clusters': 2, 'max_iter': 400, 'init': 'random', 'algorithm': 'lloyd'}

kmeans = KMeans(**kmeans_best_params)
kmeans.fit(dataset)


the_original_dataset = pd.read_csv('/kaggle/input/physical-activity-clustering/Physical_Activity_Monitoring_unlabeled.csv')

predictions = kmeans.predict(original_dataset)

predictions


# print(f"The final silhouette_score : {silhouette_score(original_dataset,predictions)}")


submission_df = pd.DataFrame({
    'Index': range(0, len(predictions)),  
    'activityID': predictions 
})

submission_df.head()


submission_df.to_csv('submission.csv', index=False)

