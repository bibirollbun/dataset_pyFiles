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


train_data = pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv")
train_data.head()


test_data = pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv")
test_data.head()


def introvertsRatioBreakdown(category, greaterThan=True):
    """
    Gives printout of percentage of people at each value of a category are introverts.
    Category must be of ratio data type.
    category = 'train_data.category' where category is a column name.
    greaterThan = boolean that determines if each iteration includes all data >= the current number or just the current number.
        eg. greaterThan=True for Time_spent_Alone = 0 includes all people
        whereas greaterThan=False would only include people who sppent 0 hours alone
    """
    #find max of category in train_data
    max_in_data = category.max()
    if greaterThan:
        for i in range(int(max_in_data) +1):
            #list of people above a certain threshold in the category and their personality type
            cat_value = train_data.loc[category >= i]['Personality']
        
            #percentage of people in that threshold of the category that are introverts
            percent_above = sum([1 for item in cat_value if item == 'Introvert'])/len(cat_value)
            print(f"% of people >= {i} units per day who are introverts: ",percent_above)

    #selects just the population from each threshold, not including everyone above that threshold. 
    elif not greaterThan:
        for i in range(int(max_in_data) +1):
            #list of people above a certain threshold in the category and their personality type
            cat_value = train_data.loc[category == i]['Personality']
        
            #percentage of people in that threshold of the category that are introverts
            percent_above = sum([1 for item in cat_value if item == 'Introvert'])/len(cat_value)
            print(f"% of people = {i} units who are introverts: ",percent_above)


print("Percentage of people spending at least x number of hours alone who are introverts")
introvertsRatioBreakdown(train_data.Time_spent_Alone)


print("Percentage of people spending x number of hours alone who are introverts")
introvertsRatioBreakdown(train_data.Time_spent_Alone, greaterThan=False)


print("percentage of people attending at least x frequency of social event attendance who are introverts")
introvertsRatioBreakdown(train_data.Social_event_attendance)


print("percentage of people with x frequency of social event attendance who are introverts")
introvertsRatioBreakdown(train_data.Social_event_attendance, greaterThan=False)


print("percentage of people who go outside x often")
introvertsRatioBreakdown(train_data.Going_outside, greaterThan=False)


print("percentage of people with x friends circle size who are introverts")
introvertsRatioBreakdown(train_data.Friends_circle_size, greaterThan=False)
print('\n')
print("percentage of people who post with x frequency who are introverts")
introvertsRatioBreakdown(train_data.Post_frequency, greaterThan=False)


#make list of everyone that is drained after socializing
drained = train_data.loc[train_data.Drained_after_socializing == 'Yes']['Personality']

#find the percentage of these people who are Introverts
drained_introverts = sum([1 for person in drained if person == 'Introvert'])

#percentage of the drained who are introverts
drained_introverts/len(drained)


#lets do the same with people who are NOT drained
not_drained = train_data.loc[train_data.Drained_after_socializing == 'No']['Personality']
undrained_introverts = sum([1 for person in drained if person == 'Introvert'])
undrained_introverts/len(not_drained)


train_data = pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv")
test_data = pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv")


# Make copies
copy_train = train_data.copy()
copy_test = test_data.copy()


# Clean code
def cleanCode(df):
    # Imputing median values for numeric columns
    df['Time_spent_Alone'] = df['Time_spent_Alone'].fillna(df['Time_spent_Alone'].median())
    df['Social_event_attendance'] = df['Social_event_attendance'].fillna(df['Social_event_attendance'].median())
    df['Going_outside'] = df['Going_outside'].fillna(df['Going_outside'].median())
    df['Post_frequency'] = df['Post_frequency'].fillna(df['Post_frequency'].median())

    # Replace NaN data for Drained_after_socializing based on Going_outside value
    # Using 0, 1, or 2 hours a day outside as qualifying someone as probably being drained after socializing
    df['Drained_after_socializing'] = df.apply(
    lambda row: 'Yes' if pd.isna(row['Drained_after_socializing']) and row['Going_outside'] <= 2
    else ('No' if pd.isna(row['Drained_after_socializing']) and row['Going_outside'] > 2
          else row['Drained_after_socializing']),
    axis=1
    )


cleanCode(copy_test)
cleanCode(copy_train)


copy_test.head(25)


# Encoding 'Personality' as 1 (Extrovert) & 0 (Introvert) for training data
#copy_train['Personality'].replace({'Extrovert': 1, 'Introvert': 0}, inplace=True)

# Encoding 'Drained_after_socializing' as 1 (Yes) & 0 (No) for training data
copy_train['Drained_after_socializing'].replace({'Yes': 1, 'No': 0}, inplace=True)
copy_test['Drained_after_socializing'].replace({'Yes': 1, 'No': 0}, inplace=True)


copy_train.head(25)


from sklearn.ensemble import RandomForestClassifier

y = train_data["Personality"]

features = ["Going_outside",
            "Drained_after_socializing"]
X = copy_train[features]
X_test = copy_test[features]



model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=49)
model.fit(X,y)
predictions = model.predict(X_test)
ID = test_data.id


submission = pd.DataFrame({'id': ID, "Personality": predictions})


submission.to_csv('submission.csv', index=False)
print("Your submission was successfully saved!")


submission.head()

