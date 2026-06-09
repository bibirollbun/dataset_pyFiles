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


men_team = pd.read_csv('/kaggle/input/march-machine-learning-mania-2025/MTeams.csv')
men_team.head()


women_team = pd.read_csv('/kaggle/input/march-machine-learning-mania-2025/WTeams.csv')
women_team.head()


men_season = pd.read_csv('/kaggle/input/march-machine-learning-mania-2025/MSeasons.csv')
men_season.head()


women_season = pd.read_csv('/kaggle/input/march-machine-learning-mania-2025/MSeasons.csv')
women_season.head()


seeds_men = pd.read_csv('/kaggle/input/march-machine-learning-mania-2025/MNCAATourneySeeds.csv')
seeds_men.head()


new_men_data = pd.merge(men_team, seeds_men, on=['TeamID'], how='outer')
new_men_data.head()


women_seeds = pd.read_csv('/kaggle/input/march-machine-learning-mania-2025/MNCAATourneySeeds.csv')
women_seeds.head()


new_women_data = pd.merge(women_team, women_seeds, on=['TeamID'], how='right')
new_women_data.head()
new_women_data.drop(columns = 'TeamName')


men_compact_results = pd.read_csv('/kaggle/input/march-machine-learning-mania-2025/MNCAATourneyCompactResults.csv')
men_compact_results.head()


new_men_data = pd.merge(new_men_data, men_compact_results, on=['Season'], how='right')
new_men_data


#y and y_2 are our prediction targets
y = new_men_data.WTeamID
y_2 = new_men_data.LTeamID
#men_features are the columns given to our model for prediction purposes
men_features = ['DayNum', 'WScore', 'LScore', 'TeamID', 'NumOT']
X = new_men_data[men_features]


X.head()


X.describe()


from sklearn.tree import DecisionTreeRegressor


#define the model
model = DecisionTreeRegressor(random_state = 1)
#fit the model
model1 = model.fit(X,y)


print("Making winning predictions of the following")
print(X.head())
print("The predictions are:")
print(model.predict(X.head(10)))


model_2 = model.fit(X,y_2)


print('Making the losing predictions of the following:')
print(X.head())
print("And the predictions are:")
print(model.predict(X.head()))
submission1 = model1.predict(X)
submission2 = model_2.predict(X)


submission_1 = submission1.to_csv(submission1, index=False)
submission_2 = submission2.to_csv(submission2, index=False)




