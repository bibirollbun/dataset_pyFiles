#Import libraries

import numpy as np
import scipy as sp
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import zipfile
import scipy.stats



# for train and test data set split
from sklearn.model_selection import train_test_split

# for grid search
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

# for evaluation metric
# accuracy
from sklearn.metrics import accuracy_score
# Report
from sklearn.metrics import classification_report
# AUC
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from pylab import rcParams

# for model comparision
from sklearn import metrics


# for decision tree model
from sklearn import tree

# for decision tree visualizaiton
from six import StringIO
from IPython.display import Image
from sklearn.tree import export_graphviz
import pydotplus

# for gradient boosting
from sklearn.ensemble import GradientBoostingClassifier

# for random search
from sklearn.ensemble import RandomForestClassifier


def load_csv_from_zip(zip_path):
    """
    Extracts a ZIP file and loads the first CSV file into a Pandas DataFrame.

    Parameters:
        zip_path (str): Path to the ZIP file.

    Returns:
        pd.DataFrame: Loaded DataFrame.
    """
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall("/kaggle/working")  # Extract ZIP contents
        csv_files = [f for f in z.namelist() if f.endswith(".csv")]  # Find CSV files
        
        if not csv_files:
            raise ValueError("No CSV files found in the ZIP archive.")
        
        csv_path = f"/kaggle/working/{csv_files[0]}"  # Select first CSV
        return pd.read_csv(csv_path)  # Load CSV into DataFrame


train = load_csv_from_zip('/kaggle/input/facebook-recruiting-iv-human-or-bot/train.csv.zip')
bids = load_csv_from_zip('/kaggle/input/facebook-recruiting-iv-human-or-bot/bids.csv.zip')


train


bids


# Merge the datasets on the common column 'bidder_id'
merged_df = pd.merge(train, bids, on='bidder_id', how='left')


missing_values_count = merged_df.isnull().sum()
missing_values_count


# Calculer le ratio de valeurs manquantes par colonne
missing_values_ratio = merged_df.isnull().sum() / len(merged_df) * 100

# Afficher le ratio de valeurs manquantes
print(missing_values_ratio)



# for all the missing value observation, drop it
merged_df = merged_df.dropna()


missing_values_count_after_removal = merged_df.isnull().sum()
print(missing_values_count_after_removal)


# Distribution of auction outcomes
outcome_counts = merged_df['outcome'].value_counts()

# Visualizing the distribution of auction outcomes
plt.figure(figsize=(8, 6))
outcome_counts.plot(kind='bar', color=['skyblue', 'salmon'])
plt.title('Distribution of Auction Outcomes')
plt.xlabel('Outcome')
plt.ylabel('Number of Bidders')
plt.show()


# Distribution of Auctions by Merchandise Type
plt.figure(figsize=(12, 6))
sns.countplot(x='merchandise', data=merged_df, order=merged_df['merchandise'].value_counts().index)
plt.title('Distribution of Auctions by Merchandise Type')
plt.show()


# Visualizing the top 10 merchandise types with the highest number of bids per label (outcome)
plt.figure(figsize=(14, 8))
top_merchandise_by_label = merged_df.groupby(['outcome', 'merchandise']).size().nlargest(30).reset_index(name='count')

sns.barplot(x='merchandise', y='count', hue='outcome', data=top_merchandise_by_label)
plt.title('Top 30 Merchandise Types with the Highest Number of Bids by Label')
plt.xlabel('Merchandise Type')
plt.ylabel('Number of Bids')
plt.legend(title='Label (Outcome)')
plt.show()


# Visualizing the top 10 countries with the highest number of bids
plt.figure(figsize=(14, 8))
top_countries = merged_df['country'].value_counts().nlargest(10)
sns.barplot(x=top_countries.index, y=top_countries.values)
plt.title('Top 10 Countries with the Highest Number of Bids')
plt.xlabel('Country')
plt.ylabel('Number of Bids')
plt.show()


# Visualizing the top 10 countries with the highest number of bids by label (outcome)
plt.figure(figsize=(14, 8))
top_countries_by_label = merged_df.groupby(['outcome', 'country']).size().nlargest(40).reset_index(name='count')

sns.barplot(x='country', y='count', hue='outcome', data=top_countries_by_label)
plt.title('Top 40 Countries with the Highest Number of Bids by Label')
plt.xlabel('Country')
plt.ylabel('Number of Bids')
plt.legend(title='Label (Outcome)')
plt.show()



plt.figure(figsize=(14, 8))
top_devices_by_label = merged_df.groupby(['outcome', 'device']).size().nlargest(50).reset_index(name='count')

# Create the bar plot
ax = sns.barplot(x='device', y='count', hue='outcome', data=top_devices_by_label)

# Rotate the x-axis labels
ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha='right')

plt.title('Top 50 Device Types with the Highest Number of Bids by Label')
plt.xlabel('Device Type')
plt.ylabel('Number of Bids')
plt.legend(title='Label (Outcome)')
plt.show()


plt.figure(figsize=(14, 8))

# Convert the 'time' column to DateTime objects if it is not already done
merged_df['time'] = pd.to_datetime(merged_df['time'])

# Create two subsets for each label
human_bids = merged_df[merged_df['outcome'] == 0]
robot_bids = merged_df[merged_df['outcome'] == 1]

# Plot line for human bids
human_bids.set_index('time').resample('H').size().plot(label='Human Bids', color='skyblue', linewidth=2)

# Plot line for robot bids
robot_bids.set_index('time').resample('H').size().plot(label='Robot Bids', color='orange', linewidth=2)

plt.title('Number of Bids per Hour Over Time by Label')
plt.xlabel('Time')
plt.ylabel('Number of Bids')
plt.legend()
plt.show()



def ent(data):

    p_data = data.value_counts()/len(data)  # calculates the probabilities
    # input probabilities to get the entropy
    entropy = scipy.stats.entropy(p_data)
    return entropy


# Convert the 'time' column to datetime 
merged_df['time'] = pd.to_datetime(merged_df['time'], errors='coerce')

# Now convert the datetime to Unix timestamp
merged_df['time'] = merged_df['time'].apply(lambda x: int(x.timestamp()) if pd.notna(x) else None)

# Check the result
print(merged_df['time'])



# bidding time difference per user (bidder_id)
merged_df = merged_df.sort_values(by=['time'])
merged_df['timediffs'] = merged_df.groupby('bidder_id')['time'].transform(pd.Series.diff)


merged_df['timediffs']


# Count the number of bids per user per auction
bids_per_auction = merged_df.groupby(['auction', 'bidder_id']).size()

# Convert to DataFrame and rename the column
bids_per_auction = bids_per_auction.to_frame(name='bids_count')


# Proportion of bots for each country
pbots_country = merged_df[merged_df['outcome'] == 1].groupby('country').size() / merged_df.groupby('country').size()

# Fill NaN values with 0
pbots_country = pbots_country.fillna(0)

# Convert the result to a DataFrame and explicitly name the column
pbots_country = pbots_country.to_frame(name='proportion_bots')


# Proportion of bots per device
pbots_device = merged_df[merged_df['outcome'] == 1].groupby('device').size() / merged_df.groupby('device').size()

# Fill NaN values with 0
pbots_device = pbots_device.fillna(0)

# Convert the result to a DataFrame and explicitly name the column
pbots_device = pbots_device.to_frame(name='proportion_bots')



# Number of unique IP to number of bids ratio
ip_bids_ratio = merged_df.groupby('bidder_id')['ip'].nunique() / merged_df.groupby('bidder_id')['bid_id'].nunique()

# Convert the result to a DataFrame and explicitly name the column
ip_bids_ratio = ip_bids_ratio.to_frame(name='ip_bids_ratio')


# Mean per auction URL entropy for each user
# Input a pandas series
auction_url_entropy = merged_df.groupby(['auction', 'bidder_id'])['url'].apply(ent)

# Group by bidder_id and calculate the mean entropy per user
auction_url_entropy = auction_url_entropy.groupby('bidder_id').mean().reset_index()

# Rename the resulting column to something meaningful
auction_url_entropy = auction_url_entropy.rename(columns={0: 'mean_url_entropy'})


# Merge the features back
m1 = pd.merge(merged_df, bids_per_auction, on=['auction', 'bidder_id'], how='left')
m2 = pd.merge(m1, pbots_country, on='country', how='left')
m3 = pd.merge(m2, pbots_device, on='device', how='left')
m4 = pd.merge(m3, ip_bids_ratio, on='bidder_id', how='left')
merged_df_f = pd.merge(m4, auction_url_entropy, on='bidder_id', how='left')


# Set column names
merged_df_f.columns = ['bidder_id', 'payment_account', 'address', 'outcome',
                     'bid_id', 'auction', 'merchandise', 'device', 'time', 'country',
                     'ip', 'url', 'timediffs', 'bids_per_auction', 'pbots_country', 'pbots_device',
                     'ip_bids_ratio', 'auction_url_entropy']


merged_df_f


df = pd.concat([merged_df_f.iloc[:, 3], merged_df_f.iloc[:, -6:]], axis=1)
df


df.to_csv("featured.csv")


df = pd.read_csv("featured.csv")
df = df.iloc[:, 1:8]


df


bots = df.loc[df.outcome == 1]
human = df.loc[df.outcome == 0]

fig, axes = plt.subplots(3, 2, figsize=(14, 12), sharex=False)
sns.distplot(bots['bids_per_auction'], hist=False, kde=True,
             bins=int(180/5), color='darkblue',
             kde_kws={'linewidth': 1.5}, ax=axes[0, 0])
sns.distplot(human['bids_per_auction'], hist=False, kde=True,
             bins=int(180/5), color='darkred',
             kde_kws={'linewidth': 1.5}, ax=axes[0, 0])

sns.distplot(bots['pbots_country'], hist=False, kde=True,
             bins=int(180/5), color='darkblue',
             kde_kws={'linewidth': 1.5}, ax=axes[0, 1])
sns.distplot(human['pbots_country'], hist=False, kde=True,
             bins=int(180/5), color='darkred',
             kde_kws={'linewidth': 1.5}, ax=axes[0, 1])

sns.distplot(bots['pbots_device'], hist=False, kde=True,
             bins=int(180/5), color='darkblue', label='bots',
             kde_kws={'linewidth': 1.5}, ax=axes[1, 0])
sns.distplot(human['pbots_device'], hist=False, kde=True,
             bins=int(180/5), color='darkred', label='human',
             kde_kws={'linewidth': 1.5}, ax=axes[1, 0])

sns.distplot(bots['ip_bids_ratio'], hist=False, kde=True,
             bins=int(180/5), color='darkblue',
             kde_kws={'linewidth': 1.5}, ax=axes[1, 1])
sns.distplot(human['ip_bids_ratio'], hist=False, kde=True,
             bins=int(180/5), color='darkred',
             kde_kws={'linewidth': 1.5}, ax=axes[1, 1])

sns.distplot(bots['auction_url_entropy'], hist=False, kde=True,
             bins=int(180/5), color='darkblue',
             kde_kws={'linewidth': 1.5}, ax=axes[2, 0])
sns.distplot(human['auction_url_entropy'], hist=False, kde=True,
             bins=int(180/5), color='darkred',
             kde_kws={'linewidth': 1.5}, ax=axes[2, 0])


# Plot the correlation matrix for the numerical values
corr_matrix = df.corr()
sns.heatmap(corr_matrix.corr(),
            xticklabels=corr_matrix.corr().columns,
            yticklabels=corr_matrix.corr().columns,
            cmap="Blues",
            fmt='d')


#Split
df_train, df_test = train_test_split(df, test_size=0.2)


features = ['bids_per_auction',	'pbots_country',	'pbots_device',	'ip_bids_ratio',	'auction_url_entropy']
target = ['outcome']


X = np.array(df[features])
y = np.array(df[target]).ravel()


bots_train = df_train.loc[df_train.outcome == 1]
human_train = df_train.loc[df_train.outcome == 0]
human_sample = human_train.sample(n=len(bots_train))
df_train_balance = pd.concat([df_train, human_sample])

y_train = df_train_balance['outcome']
X_train = df_train_balance.iloc[:, -5:]
y_test = df_test['outcome']
X_test = df_test.iloc[:, -5:]


# base model accuracy
print(f"base model accuracy: {len(df[df['outcome'] == 0]) / (len(df[df['outcome'] == 0]) + len(df[df['outcome'] == 1]))}")


# for decision tree model
from sklearn import tree

# for decision tree visualization
from six import StringIO
from IPython.display import Image
from sklearn.tree import export_graphviz
import pydotplus


# hyperparameter tuning
dt = tree.DecisionTreeClassifier()
param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': range(3, 6),
    'max_leaf_nodes': range(10, 15),
    'min_samples_split': range(2, 6)
}

dt_cv = GridSearchCV(estimator=dt,
                     param_grid=param_grid,
                     cv=5)
dt_cv.fit(X_train, y_train)
print(dt_cv.best_params_)


kwargs_regularize = dict(criterion='gini',
                         max_depth=5,
                         max_leaf_nodes=14,
                         min_samples_split=2)
dt = tree.DecisionTreeClassifier(**kwargs_regularize)


dt.fit(X_train, y_train)


dot_data = StringIO()
export_graphviz(dt, out_file=dot_data,
                filled=True, rounded=True,
                feature_names=X_train.columns.values,
                class_names=['human', 'bot'],
                special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
Image(graph.create_png())


# Check feature importance and display in bar plot.
print('Feature importance of Decision Tree Model')
plt.style.use('ggplot')
fig = plt.figure(figsize=(5, 5))
feat_importances = pd.Series(dt.feature_importances_, index=X_train.columns)
feat_importances.nsmallest(5).plot(kind='barh', alpha=0.7)
fig.savefig('dt_feature.png')


# predict
y_dt_pred = dt.predict(X_test)


# accuracy score
print(f"Decision Tree Accuracy: {accuracy_score(y_dt_pred, y_test):.3f}")


# Plot ROC in one graph
y_dt_score = dt.predict_proba(X_test)[:, 1]
fpr_dt, tpr_dt, _dt = roc_curve(y_test, y_dt_score)
roc_dt_auc = auc(fpr_dt, tpr_dt)

fig = plt.figure(figsize=(5, 5))
plt.plot(fpr_dt, tpr_dt, label='DT ROC curve (area = %0.2f)' % roc_dt_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.005])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()
fig.savefig('roc_dt_auc.png')


print('Classification Report of Decision Tree Model')
print(classification_report(y_test, y_dt_pred))


from sklearn.ensemble import RandomForestClassifier


rf = RandomForestClassifier(n_estimators=20)  # Nombre rÃ©duit d'estimateurs
param_grid = {
    'max_depth': [3],
    'max_leaf_nodes': range(8, 12),
    'max_features': ['sqrt', 'auto', 'log2']
}

rf_cv = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5)
rf_cv.fit(X_train, y_train)
print(rf_cv.best_params_)


rf = RandomForestClassifier(n_estimators=30, max_depth=5,
                            max_leaf_nodes=11, max_features='log2',
                            bootstrap=True, oob_score=True)
rf.fit(X_train, y_train)


# Check feature importance and display in bar plot.
print('Feature importance of Random Forest Model')
plt.style.use('ggplot')
fig = plt.figure(figsize=(5, 5))
feat_importances = pd.Series(rf.feature_importances_, index=X_train.columns)
feat_importances.nsmallest(5).plot(kind='barh', alpha=0.7)
fig.savefig('rf_feature.png')


y_rf_pred = rf.predict(X_test)
print(f"Random Forest Accuracy: {accuracy_score(y_rf_pred, y_test):.3f}")


# Plot ROC in one graph
y_rf_score = rf.predict_proba(X_test)[:, 1]
fpr_rf, tpr_rf, _rf = roc_curve(y_test, y_rf_score)
roc_rf_auc = auc(fpr_rf, tpr_rf)

plt.figure(figsize=(5, 5))
plt.plot(fpr_rf, tpr_rf, label='RF ROC curve (area = %0.2f)' % roc_rf_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.005])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()
fig.savefig('roc_rf_auc.png')


print('Classification Report of Random Forest Model')
print(classification_report(y_test, y_rf_pred))


gb = GradientBoostingClassifier(n_estimators=10)
param_grid = {
    'max_depth': range(3, 6),
    'max_leaf_nodes': range(8, 11)
}

gb_cv = GridSearchCV(estimator=gb,
                     param_grid=param_grid,
                     cv=5)
gb_cv.fit(X_train, y_train)
print(gb_cv.best_params_)


gb = GradientBoostingClassifier(n_estimators=30, max_depth=5, max_features='sqrt',
                                max_leaf_nodes=9)
gb.fit(X_train, y_train)


# Check feature importance and display in bar plot.
print('Feature importance of Gradient Boosting Model')
plt.style.use('ggplot')
fig = plt.figure(figsize=(5, 5))
feat_importances = pd.Series(gb.feature_importances_, index=X_train.columns)
feat_importances.nsmallest(5).plot(kind='barh', alpha=0.7)
fig.savefig('gb_feature.png')


y_gb_pred = gb.predict(X_test)
print(f"Gradient Boosting Accuracy: {accuracy_score(y_gb_pred, y_test):.3f}")


# Plot ROC in one graph
y_gb_score = gb.predict_proba(X_test)[:, 1]
fpr_gb, tpr_gb, _gb = roc_curve(y_test, y_gb_score)
roc_gb_auc = auc(fpr_gb, tpr_gb)

plt.figure(figsize=(5, 5))
plt.plot(fpr_gb, tpr_gb, label='GB ROC curve (area = %0.2f)' % roc_gb_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.005])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()
fig.savefig('roc_gb_auc.png')


print('Classification Report of Gradient Boosting Model')
print(classification_report(y_test, y_gb_pred))


dt_metrics = [metrics.accuracy_score(y_test, y_dt_pred), metrics.precision_score(y_test, y_dt_pred),
              metrics.recall_score(y_test, y_dt_pred), metrics.f1_score(
                  y_test, y_dt_pred),
              metrics.roc_auc_score(y_test, y_dt_pred)]
rf_metrics = [metrics.accuracy_score(y_test, y_rf_pred), metrics.precision_score(y_test, y_rf_pred),
              metrics.recall_score(y_test, y_rf_pred), metrics.f1_score(
                  y_test, y_rf_pred),
              metrics.roc_auc_score(y_test, y_rf_pred)]
gb_metrics = [metrics.accuracy_score(y_test, y_gb_pred), metrics.precision_score(y_test, y_gb_pred),
              metrics.recall_score(y_test, y_gb_pred), metrics.f1_score(
                  y_test, y_gb_pred),
              metrics.roc_auc_score(y_test, y_gb_pred)]


fig, ax = plt.subplots(figsize=(10, 8))
index = np.arange(5)
width = 0.2
b1 = plt.bar(index, dt_metrics[0:5], width,
             alpha=0.4, color='grey', label='decision tree')
b2 = plt.bar(index+width, rf_metrics[0:5], width,
             alpha=0.8, color='powderblue', label='random forest')
b3 = plt.bar(index+2*width, gb_metrics[0:5], width,
             alpha=0.8, color='pink', label='gradient boosting')
plt.title('Model Comparison')
plt.ylabel('score')
plt.xticks(index+width, ('accuracy', 'precision', 'recall', 'F1', 'ROC AUC'))
plt.legend(loc=8, ncol=3, mode="expand", borderaxespad=0.)
plt.show()
fig.savefig('model_comparison.png')



label = ["Accuracy_score", "Precision_score",
         "Recall_Score", "F1_score", "ROC_AUC_score"]
table = pd.DataFrame({'Decision Tree': dt_metrics,
                      'Random Forest': rf_metrics, 'Gradient Boosting': gb_metrics})
table = table.transpose()
table.columns = label
table.transpose().round(3)

