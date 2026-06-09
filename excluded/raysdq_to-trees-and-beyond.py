import pandas as pd

train_dataset = pd.read_csv( "/kaggle/input/playground-series-s5e7/train.csv", index_col="id" )
test_dataset = pd.read_csv( "/kaggle/input/playground-series-s5e7/test.csv", index_col="id" )


## Encoder

def ordinal_encoder( data ):
    yes_no_dict = {"Yes": 1, "No": 0}
    per_dict = {"Introvert": 0, "Extrovert": 1}

    data.Stage_fear = data.Stage_fear.map( yes_no_dict )
    data.Drained_after_socializing = data.Drained_after_socializing.map( yes_no_dict )

    if( data.columns == "Personality" ).any():
        data.Personality = data.Personality.map( per_dict )

    return data

train_dataset = ordinal_encoder( train_dataset )
test_dataset = ordinal_encoder( test_dataset )


## Impute

from sklearn.impute import SimpleImputer

imp = SimpleImputer( strategy="mean" )
train_dataset = pd.DataFrame( imp.fit_transform(train_dataset), columns=train_dataset.columns )
test_dataset = pd.DataFrame( imp.fit_transform(test_dataset), columns=test_dataset.columns )


## Clustering

from sklearn.cluster import KMeans

kmeans = KMeans( n_clusters=4, n_init=1, random_state=1 )
train_dataset['C1'] = kmeans.fit_predict( train_dataset[ ["Time_spent_Alone", "Going_outside"] ] )
test_dataset['C1'] = kmeans.predict( test_dataset[ ["Time_spent_Alone", "Going_outside"] ] )


## Plot scatterplot of C1 feature

import seaborn as sns, matplotlib.pyplot as plt

plt.figure( figsize=(10,4) )
plt.subplot(1, 2, 1)
sns.scatterplot( x="Going_outside", y="Time_spent_Alone", data=train_dataset, hue="Personality" )
plt.subplot(1, 2, 2)
sns.scatterplot( x="Going_outside", y="Time_spent_Alone", data=train_dataset, hue="C1", palette="Set1" )
plt.tight_layout()
plt.show()


## Train test split

y = train_dataset.Personality
x = train_dataset.drop( columns="Personality" )
x_test = test_dataset

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score

x_train, x_valid = train_test_split( x, train_size=0.8, random_state=1 )
y_train, y_valid = train_test_split( y, train_size=0.8, random_state=1 )


## RandomForest model

from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier( 
    n_estimators=1000,
    n_jobs=-1,
    random_state=1
)

rf_model.fit( x_train, y_train )
preds = rf_model.predict( x_valid )


auc = roc_auc_score( y_valid, preds )
acc = accuracy_score( y_valid, preds )

print( f"acc: {acc}\nauc: {auc}" )


model = rf_model
pred = model.predict( x_test )

out = pd.read_csv( "/kaggle/input/playground-series-s5e7/sample_submission.csv" )
out.Personality = pd.Series( pred ).map( {0: "Introvert", 1: "Extrovert"} )

out.to_csv( "submission.csv", index=False )

