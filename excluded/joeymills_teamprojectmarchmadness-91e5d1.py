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


reg_df = pd.read_csv("/kaggle/input/march-machine-learning-mania-2025/MRegularSeasonDetailedResults.csv")
tourney_df = pd.read_csv("/kaggle/input/march-machine-learning-mania-2025/MNCAATourneyCompactResults.csv")
# add files for seeds below
seeds_df = pd.read_csv("/kaggle/input/march-machine-learning-mania-2025/MNCAATourneySeeds.csv")


def build_team_stats(df):
    win_stats = df[['Season', 'WTeamID', 'WScore', 'WFGM', 'WFGA', 'WFGM3', 'WFGA3',
                    'WFTM', 'WFTA', 'WOR', 'WDR', 'WAst', 'WTO', 'WStl', 'WBlk']].copy()
    lose_stats = df[['Season', 'LTeamID', 'LScore', 'LFGM', 'LFGA', 'LFGM3', 'LFGA3',
                     'LFTM', 'LFTA', 'LOR', 'LDR', 'LAst', 'LTO', 'LStl', 'LBlk']].copy()

    win_stats.columns = ['Season', 'TeamID', 'Score', 'FGM', 'FGA', 'FGM3', 'FGA3',
                         'FTM', 'FTA', 'OR', 'DR', 'Ast', 'TO', 'Stl', 'Blk']
    lose_stats.columns = win_stats.columns

    all_stats = pd.concat([win_stats, lose_stats], axis=0)
    team_stats = all_stats.groupby(['Season', 'TeamID']).mean().reset_index()
    return team_stats

team_stats = build_team_stats(reg_df)


def create_matchups(tourney_df, team_stats, seeds_df):
    matchups = tourney_df.copy()
    matchups['TeamA'] = matchups[['WTeamID', 'LTeamID']].min(axis=1)
    matchups['TeamB'] = matchups[['WTeamID', 'LTeamID']].max(axis=1)
    matchups['Result'] = (matchups['WTeamID'] == matchups['TeamA']).astype(int)

    merged = matchups[['Season', 'TeamA', 'TeamB', 'Result']].copy()

    # Merge stats for TeamA
    merged = merged.merge(team_stats, left_on=['Season', 'TeamA'], right_on=['Season', 'TeamID'], how='left')
    merged = merged.drop(columns='TeamID')
    merged = merged.rename(columns={col: f"A_{col}" for col in team_stats.columns if col not in ['Season', 'TeamID']})

    # Merge stats for TeamB
    merged = merged.merge(team_stats, left_on=['Season', 'TeamB'], right_on=['Season', 'TeamID'], how='left')
    merged = merged.drop(columns='TeamID')
    merged = merged.rename(columns={col: f"B_{col}" for col in team_stats.columns if col not in ['Season', 'TeamID']})

    # Merge seed info
    seeds_df = seeds_df.copy()
    seeds_df['SeedInt'] = seeds_df['Seed'].str.extract(r'(\d+)').astype(int)

    merged = merged.merge(seeds_df[['Season', 'TeamID', 'SeedInt']], left_on=['Season', 'TeamA'], right_on=['Season', 'TeamID'], how='left')
    merged = merged.rename(columns={'SeedInt': 'SeedA'}).drop(columns='TeamID')

    merged = merged.merge(seeds_df[['Season', 'TeamID', 'SeedInt']], left_on=['Season', 'TeamB'], right_on=['Season', 'TeamID'], how='left')
    merged = merged.rename(columns={'SeedInt': 'SeedB'}).drop(columns='TeamID')

    # Seed diff
    merged['SeedDiff'] = merged['SeedA'] - merged['SeedB']

    # Create feature diffs
    stat_cols = [col for col in merged.columns if col.startswith("A_")]
    for col in stat_cols:
        base_stat = col[2:]
        if f"B_{base_stat}" in merged.columns:
            merged[f"Diff_{base_stat}"] = merged[col] - merged[f"B_{base_stat}"]

    feature_cols = [col for col in merged.columns if col.startswith("Diff_")] + ['SeedDiff']
    merged = merged[feature_cols + ['Result']]

    # Final cleanup
    merged = merged.dropna()
    return merged



matchup_df = create_matchups(tourney_df, team_stats, seeds_df)
x = matchup_df.drop(columns='Result')
y = matchup_df['Result']
print("x shape:", x.shape)



# Imports
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier


print(x.shape, y.shape)



def regression_(x, y):
   
    L = LinearRegression()
    R = Ridge()
    Lass = Lasso()
    E = ElasticNet()
    ETR = ExtraTreeRegressor()
    GBR = GradientBoostingRegressor()
    XGBC = XGBRegressor()
    dt = DecisionTreeRegressor()
    kn = KNeighborsRegressor()
    RF = RandomForestRegressor()
    

  
    log_reg = LogisticRegression()
    rf_clf = RandomForestClassifier()
    gbr_clf = GradientBoostingClassifier()
    xgb_clf = XGBClassifier()

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

   
    algos = [L, R, Lass, E, ETR, GBR, XGBC, dt, kn, RF]
    
    classifiers = [log_reg, rf_clf, gbr_clf, xgb_clf]
    
  
    algo_names = ['Linear', 'Ridge', 'Lasso', 'ElasticNet', 'Extra Tree', 'Gradient Boosting',
                  'XGBoost', 'DecisionTree', 'KNeighbors', 'Random Forest']

    
    classifier_names = ['Logistic Regression', 'Random Forest Classifier', 'Gradient Boosting Classifier', 'XGBoost Classifier']

 
    r2Score, rmse, mae = [], [], []
    result = pd.DataFrame(columns=['R2_score', 'RMSE', 'MAE'], index=algo_names)

    
    for i, algo in enumerate(algos):
        p = algo.fit(x_train, y_train).predict(x_test)
        r2Score.append(r2_score(y_test, p))
        rmse.append(mean_squared_error(y_test, p) ** 0.5)
        mae.append(mean_absolute_error(y_test, p))

    result['R2_score'] = r2Score
    result['RMSE'] = rmse
    result['MAE'] = mae

    
    for i, clf in enumerate(classifiers):
        clf.fit(x_train, y_train)
        preds = clf.predict(x_test)
        accuracy = clf.score(x_test, y_test)
        result.loc[classifier_names[i], 'R2_score'] = accuracy
        result.loc[classifier_names[i], 'RMSE'] = mean_squared_error(y_test, preds) ** 0.5
        result.loc[classifier_names[i], 'MAE'] = mean_absolute_error(y_test, preds)
        
          

 
    return result.sort_values('R2_score', ascending=False)


# x = matchup_df.drop(columns='Result')  
# y = matchup_df['Result']

# # Run regression and print results
# results = regression_(x, y)
# print(results)



from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier

from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
from sklearn.metrics import confusion_matrix,classification_report

# defining the function that calls all classification algorithms

def classification_(x,y):
    
    k=KNeighborsClassifier()
    svc=SVC(kernel='linear')
    d=DecisionTreeClassifier()
    log=LogisticRegression()
    gbc=GradientBoostingClassifier()
   
    rf=RandomForestClassifier()
    ab=AdaBoostClassifier()
    
    algos=[k,svc,d,log,gbc,rf,ab]
    algos_name=['KNeigbors','SVC','DecisionTree','LogisticRegr','GradientBoosting','RandomForest','AdaBoost']
    
    accuracy = []
    precision = []
    recall = []
    f1 = []
   
    result=pd.DataFrame(columns=['AccuracyScore','PrecisionScore','RecallScore','f1_Score'],index=algos_name)

    x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=42)

    for i in algos:
        
        predict=i.fit(x_train,y_train).predict(x_test)
        
        accuracy.append(accuracy_score(y_test,predict))
        precision.append(precision_score(y_test,predict,average='weighted'))
        recall.append(recall_score(y_test,predict,average='weighted'))
        f1.append(f1_score(y_test,predict,average='weighted'))
      
    result.AccuracyScore=accuracy
    result.PrecisionScore=precision
    result.RecallScore=recall
    result.f1_Score=f1
    
    
    return result.sort_values('f1_Score',ascending=False)
results = classification_(x, y)
print(results)


# Plotting feature importance for the men's dataset. 

# Plotting the importance involves uign the coef_ property of SVC. It gives a numpy array
# of the weights for each feature. For more information, see here:
# https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC.coef_

import matplotlib.pyplot as plt

def mensDataShow(x,y):

    # Designating the model and fitting it. 
    svc=SVC(kernel='linear')
    x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=42)
    model = svc.fit(x_train, y_train)
    prediction = model.predict(x_train)

    # Get feature importances from coef_ to display which features are more important.
    mensImportantValues = np.abs(svc.coef_[0])
    featureNames = x.columns
    
    # Plotting the data. 
    plt.figure(figsize=(10, 10))
    plt.barh(featureNames, mensImportantValues)
    plt.xlabel("Importance Value")
    plt.title("Men's Dataset - Feature Importance from SVC")
    plt.show()
    
    return 

mensDataShow(x, y)


# Assuming you have already defined x and y earlier in the code
# x = features, y = target variable (e.g., your 'Result' column)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

logR = LogisticRegression()
logR.fit(x_train, y_train)
y_pred = logR.predict(x_test)


clf = LogisticRegression(random_state=0, max_iter=1000).fit(x_train, y_train)
predictions = clf.predict(x_test)
accuracy_score(y_test, predictions)





# Working with Women's Basketball Data
wreg_df = pd.read_csv("/kaggle/input/march-machine-learning-mania-2025/WRegularSeasonDetailedResults.csv")
wtourney_df = pd.read_csv("/kaggle/input/march-machine-learning-mania-2025/WNCAATourneyCompactResults.csv")
# add files for seeds below
wseeds_df = pd.read_csv("/kaggle/input/march-machine-learning-mania-2025/WNCAATourneySeeds.csv")

# Team stat building (calling the above declared function)
wteam_stats = build_team_stats(wreg_df)

# Creating the Women's matchup dataframe
# Notice the 
wmatchup_df = create_matchups(wtourney_df, wteam_stats, wseeds_df)
x_w = wmatchup_df.drop(columns='Result')
y_w = wmatchup_df['Result']
print("x shape:", x.shape, "\n\n")

# Getting the results for various classification algorithms
results = classification_(x_w, y_w)
print(results)

"""
Women's data appears to do much better on the classification data. Possibly due to the fewer data points.
"""


# Plotting feature importance for the women's dataset. 

# Plotting the importance involves uign the coef_ property of SVC. It gives a numpy array
# of the weights for each feature. For more information, see here:
# https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC.coef_

import matplotlib.pyplot as plt

def womensDataShow(x,y):

    # Designating the model and fitting it. 
    svc=SVC(kernel='linear')
    x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=42)
    model = svc.fit(x_train, y_train)
    prediction = model.predict(x_train)

    # Get feature importances from coef_ to display which features are more important.
    womensImportantValues = np.abs(svc.coef_[0])
    featureNames = x.columns
    
    # Plotting the data. 
    plt.figure(figsize=(10, 10))
    plt.barh(featureNames, womensImportantValues)
    plt.xlabel("Importance Value")
    plt.title("Women's Dataset - Feature Importance from SVC")
    plt.show()
    
    return 

womensDataShow(x_w, y_w)

print(wmatchup_df.head)


from sklearn.preprocessing import MinMaxScaler

# Working on data with scaling applied. 
def scaledSVCrunner(x, y):
    
    # Designating the model and fitting it. 
    svc=SVC(kernel='linear')
    x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=42)
    model = svc.fit(x_train, y_train)
    print("Pre-Scaling Score: " + str(svc.score(x_test, y_test)))

    # Performing 0 to 1 scaling
    scaler = MinMaxScaler()
    scaler.fit(x_train)
    x_train_scaled = scaler.transform(x_train)
    x_test_scaled = scaler.transform(x_test)
    svc.fit(x_train_scaled, y_train)
    print("Post-Scaling Score: " + str(svc.score(x_test_scaled, y_test)))
    
    return 

print("Men's Scaling Performance")
scaledSVCrunner(x, y)
print("\nWomen's Scaling Performance")
scaledSVCrunner(x_w, y_w)

# Scaling doesn't appear to have much performance impact. 


# Trying different regularization params. 
def regularParam(x, y):
    
    # Designating the model and fitting it. 
    svc=SVC(kernel='linear', C=1)
    x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=42)
    model = svc.fit(x_train, y_train)
    print("C=1 Score: " + str(svc.score(x_test, y_test)))

    svc=SVC(kernel='linear', C=10)
    x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=42)
    model = svc.fit(x_train, y_train)
    print("C=10 Score: " + str(svc.score(x_test, y_test)))

    svc=SVC(kernel='linear', C=20)
    x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=42)
    model = svc.fit(x_train, y_train)
    print("C=20 Score: " + str(svc.score(x_test, y_test)))



print("Men's Reg Performance")
regularParam(x, y)
print("\nWomen's Reg Performance")
regularParam(x_w, y_w)

# Scaling doesn't appear to have much performance impact. 


# Testing different kernels for SVC. 
def kernelTesting(x, y):
    
    # Designating the model and fitting it. 
    svc=SVC(kernel='linear')
    x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=42)
    model = svc.fit(x_train, y_train)
    print("Linear Score: " + str(svc.score(x_test, y_test)))

    svc=SVC(kernel='poly')
    x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=42)
    model = svc.fit(x_train, y_train)
    print("Polynomial Score: " + str(svc.score(x_test, y_test)))

    svc=SVC(kernel='sigmoid')
    x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=42)
    model = svc.fit(x_train, y_train)
    print("Sigmoid Score: " + str(svc.score(x_test, y_test)))

print("Men's Kernel Performance")
kernelTesting(x, y)
print("\nWomen's Kernel Performance")
kernelTesting(x_w, y_w)

# Sigmoid performs worse in both cases, but polynomial and linear systems are very close, 
# with linear better for Men and polynomial better for Women. More testing done here. 

def polynomialTest(x, y, maxPoly):
    
    # Checking to see how each degree of polynomial performs. 
    for degreeCount in range(1, maxPoly+1):
        svc=SVC(kernel='poly', degree=degreeCount)
        x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=42)
        model = svc.fit(x_train, y_train)
        print("Degree " + str(degreeCount) + " Polynomial Score: " + str(svc.score(x_test, y_test)))

    return

print("\nMen's Polynomial Performance")
polynomialTest(x, y, 10)
print("\nWomen's Polynomial Performance")
polynomialTest(x_w, y_w, 10)


from itertools import combinations
from tqdm import tqdm
import pandas as pd
import numpy as np

def generate_submission_file(team_stats, seeds_df, model, feature_columns, output_name="submission.csv"):
    # Load team IDs from 2025
    team_ids = team_stats[team_stats['Season'] == 2025]['TeamID'].unique()
    matchups = list(combinations(sorted(team_ids), 2))

    records = []

    for team1, team2 in tqdm(matchups, desc="Generating predictions"):
        teamA, teamB = team1, team2

        # Extract stats
        a_stats = team_stats[(team_stats['Season'] == 2025) & (team_stats['TeamID'] == teamA)].drop(columns=['Season', 'TeamID']).values.flatten()
        b_stats = team_stats[(team_stats['Season'] == 2025) & (team_stats['TeamID'] == teamB)].drop(columns=['Season', 'TeamID']).values.flatten()

        if a_stats.size == 0 or b_stats.size == 0:
            continue  # Skip missing

        # Seeds
        seedA = seeds_df[(seeds_df['Season'] == 2025) & (seeds_df['TeamID'] == teamA)]['Seed'].str.extract(r'(\d+)').astype(float).fillna(18).values.flatten()
        seedB = seeds_df[(seeds_df['Season'] == 2025) & (seeds_df['TeamID'] == teamB)]['Seed'].str.extract(r'(\d+)').astype(float).fillna(18).values.flatten()
        seed_diff = seedA[0] - seedB[0] if seedA.size > 0 and seedB.size > 0 else 0

        # Feature diff
        feature_diff = a_stats - b_stats
        features = np.append(feature_diff, seed_diff)

        # Wrap in DataFrame with proper feature names
        features_df = pd.DataFrame([features], columns=feature_columns)

        # Predict probability TeamA wins
        prob = model.predict_proba(features_df)[0][1]

        records.append({
            "ID": f"2025_{teamA}_{teamB}",
            "Pred": prob
        })

    sub_df = pd.DataFrame(records)
    sub_df.to_csv(output_name, index=False)
    print(f" {output_name} created with {len(sub_df)} rows.")
    print(sub_df.head())
    return sub_df



clf_w = LogisticRegression(random_state=0, max_iter=1000)
clf_w.fit(x_w, y_w)


feature_columns = x.columns
generate_submission_file(team_stats, seeds_df, clf, feature_columns, output_name="mens_submission.csv")
generate_submission_file(wteam_stats, wseeds_df, clf_w, feature_columns=x_w.columns, output_name="womens_submission.csv")

mens = pd.read_csv("mens_submission.csv")
womens = pd.read_csv("womens_submission.csv")
combined = pd.concat([mens, womens])
combined.to_csv("submission.csv", index=False)
print("Combined submission.csv is printed")

