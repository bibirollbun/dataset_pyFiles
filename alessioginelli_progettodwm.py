import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


dictionary = pd.read_csv("/kaggle/input/child-mind-institute-problematic-internet-use/data_dictionary.csv")
dataset = pd.read_csv("/kaggle/input/child-mind-institute-problematic-internet-use/train.csv")
dataset_test = pd.read_csv("/kaggle/input/child-mind-institute-problematic-internet-use/test.csv")



# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        # print(os.path.join(dirname, filename))
        pass
        
        
# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


dictionary


#definizione costanti

LABEL = 'PreInt_EduHx-computerinternet_hoursday'


from sklearn.model_selection import train_test_split

train = None
test = None

train, test = train_test_split(dataset, train_size=0.8, random_state=0)


normalizable_types = set(["int", "float"])


dictionary[dictionary["Instrument"] == "Children's Global Assessment Scale"] 

#DATI SPORCHI: CGAS-Season	e CGAS-CGAS_Score  sono due colonne collegate, capita che la prima sia valorizzata mentre la seconda no.
train[~train["CGAS-Season"].isnull() & train["CGAS-CGAS_Score"].isnull()][["CGAS-Season", "CGAS-CGAS_Score"]]


both_nan = train[~train["CGAS-Season"].isnull() & train["CGAS-CGAS_Score"].isnull()][["CGAS-Season", "CGAS-CGAS_Score"]]
both_nan["CGAS-Season"] = np.nan
both_nan


physical_measures_fields =  dictionary[dictionary["Instrument"] == "Physical Measures"]["Field"]

train[train[physical_measures_fields.iloc[0]].isnull() &
(        
    ~train[physical_measures_fields.iloc[1]].isnull() |
    ~train[physical_measures_fields.iloc[2]].isnull() |
    ~train[physical_measures_fields.iloc[3]].isnull() |
    ~train[physical_measures_fields.iloc[4]].isnull() |
    ~train[physical_measures_fields.iloc[5]].isnull() |
    ~train[physical_measures_fields.iloc[6]].isnull() |
    ~train[physical_measures_fields.iloc[7]].isnull() 
)
][physical_measures_fields].head(50)


FitnessGram_Vitals_and_Treadmill_cols = dictionary[dictionary["Instrument"] == "FitnessGram Vitals and Treadmill"]["Field"]
train[
train["Fitness_Endurance-Season"].isnull() &
(
    ~train["Fitness_Endurance-Max_Stage"].isnull() &
    ~train["Fitness_Endurance-Time_Mins"].isnull()
)]
[FitnessGram_Vitals_and_Treadmill_cols]
#le colonne Fitness_Endurance-Time_Mins e	Fitness_Endurance-Time_Sec possono essere condensate in una unica colonna dei secondi


#questa categoria ha 15 colonne, non posso effettuare lo stesso controllo di prima
print(f'N colonne: {dictionary[dictionary["Instrument"] == "FitnessGram Child"].shape[0]}')


#nuovo controllo: prendo le colonne numeriche e ne estraggo la media il minimo e il massimo, eventualmente le normalizzo

def print_col_stats(dataframe):
    copy = dataframe[
        dictionary[
            dictionary["Field"].isin(dataframe.columns) &
            dictionary["Type"].isin(normalizable_types)]
        ["Field"]
    ]

    for i in copy.columns:
        print(f'Colonna: {i:<15} media:{dataframe[i].mean():<8.2f} std deviation: {dataframe[i].std():<8.2f} massimo: {dataframe[i].max():<8.2f} minimo: {dataframe[i].min():<8.2f}')


fitnessgram_child_cols =  dictionary[dictionary["Instrument"] == "FitnessGram Child"]["Field"]
print_col_stats(train[fitnessgram_child_cols])



train[dictionary[dictionary["Instrument"] == "Bio-electric Impedance Analysis"]["Field"]]

bio_electric_cols = dictionary[dictionary["Instrument"] == "Bio-electric Impedance Analysis"]["Field"]

print_col_stats(train[bio_electric_cols])


PCIAT_cols = dictionary[dictionary["Instrument"] == "Parent-Child Internet Addiction Test"]["Field"]


#alcune istanze hanno la label non valorizzata, non potendo essere d'aiuto le elimineró
train[train["sii"].isnull()][["id", "sii"]]


#definisco tutte le correzioni che vanno fatte ai dati

def clear_CGAS_cols(dataset):
    copy = dataset.copy()
    condition = copy["CGAS-Season"].isnull() | copy["CGAS-CGAS_Score"].isnull()
    copy.loc[condition, "CGAS-Season"] = np.nan
    copy.loc[condition, "CGAS-CGAS_Score"] = np.nan
    return copy

def drop_PCIAT_cols(dataset):
    return dataset.drop(columns=PCIAT_cols)

def drop_null_label_rows(dataset):     
    return dataset.drop(dataset[dataset["sii"].isnull()].index)

def drop_id_column(dataset):
    return dataset.drop("id", axis=1)
    

def merge_min_sec_cols(dataset):
    copy = dataset.copy()
    copy["Fitness_Endurance-Time_Sec"] += copy["Fitness_Endurance-Time_Mins"]*60
    return copy.drop(columns=["Fitness_Endurance-Time_Mins"])

def to_float(dataset):
    numerical = dataset.select_dtypes(include="number")
    classified = dataset.select_dtypes(exclude="number")
    # result = pd.concat([numerical.astype("float"), classified], axis=1)
    return pd.concat([numerical.astype("float"), classified], axis=1)



from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer





from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer


#creato un OneHotEncoder che agisce solo sulle colonne necessarie
season_columns = dictionary[
    (dictionary["Field"].str.contains("Season")) &
    (dictionary["Instrument"] != "Parent-Child Internet Addiction Test") #PCIAT columns
]["Field"]


from sklearn.base import BaseEstimator, TransformerMixin

#classe che implementa fit e transform senza perdere il pandas.DataFrame, cosi da poterlo mantenere in pipeline e poter usare le colonne
class DataFrameTransformer(
    BaseEstimator, #per avere get_params()/set_params() 
):

    def __init__(self, preprocessor, columns=None):
        self.preprocessor = preprocessor
        self.columns = columns
        
    def fit(self, X, y=None):
        self.columns_ = X.columns if (self.columns is None) else self.columns
        copy = pd.DataFrame(data=X, columns=self.columns_, index=X.index)
        return self.preprocessor.fit(copy[self.columns_])

    def transform(self, X):
        np_array = self.preprocessor.transform(X[self.columns_])
        new_df = pd.DataFrame(np_array, columns=self.preprocessor.get_feature_names_out(), index=X.index)
        result =  pd.concat([X.drop(columns=self.columns_), new_df], axis=1)

        return result

    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)


season_encoder = DataFrameTransformer(
    OneHotEncoder(sparse_output=False),
    columns=season_columns.tolist()
)

normalizer = DataFrameTransformer(
    StandardScaler(),
    columns=dictionary[dictionary["Type"].isin(["int", "float"])]["Field"].tolist()
)


from sklearn.impute import SimpleImputer

nan_filler = DataFrameTransformer(
    SimpleImputer(strategy="median")
)

cleaning_pipeline = Pipeline([
    ('normalize', normalizer),
    ('clear CGAS columns', FunctionTransformer(clear_CGAS_cols)),
    ('drop PCIAT cols', FunctionTransformer(drop_PCIAT_cols)),
    ('drop null label rows',FunctionTransformer(drop_null_label_rows)),
    ('drop id col', FunctionTransformer(drop_id_column)),
    ('merge_mins_sec_cols', FunctionTransformer(merge_min_sec_cols)),
    ('convert to float', FunctionTransformer(to_float)),
    ('encode seasons', season_encoder) ,
    ('remove NaN', nan_filler)
])



train, test = train_test_split(dataset, train_size=0.8, random_state=0)
train = cleaning_pipeline.fit_transform(train)


#inizio scelta e fine-tuning dei modelli

valid_X, train_X, valid_y, train_y  = train_test_split(train.drop("sii", axis=1), train["sii"], train_size=0.25, random_state=0)
X = pd.concat([train_X, valid_X], axis=0)
y = pd.concat([train_y, valid_y], axis=0)

def accuracy(y_true, y_pred):
    matches=0
    for t,p in zip(y_true, y_pred):
        if (t == p):
            matches+=1
    return matches/len(y_true)


print(f'baseline accuracy: {(train["sii"].value_counts().max())/train["sii"].value_counts().sum():.3f}')


from sklearn.tree import DecisionTreeClassifier

decision_tree_classifier = DecisionTreeClassifier()
decision_tree_classifier.fit(train_X, train_y)
pred_y = decision_tree_classifier.predict(valid_X)
print(f'accuracy: {accuracy(valid_y, pred_y):.3f}')




#modello 1:

max_depth=[3,5,7,10,15,20, 99]
min_samples_split=[32,16,8,4,2]



for d in max_depth:
    for msp in min_samples_split:
        decision_tree_classifier = DecisionTreeClassifier(
            splitter='random', #per alleviare il bias
            max_depth=d,
            random_state=0,
            min_samples_split=msp
        )
        decision_tree_classifier.fit(train_X, train_y)
        pred_y = decision_tree_classifier.predict(valid_X)
        print(f' max_depth: {d:>2}, min_samples_split: {msp:>2} accuracy: {accuracy(valid_y, pred_y):>.3f}, depth reached: {decision_tree_classifier.get_depth():>2}, n leaves: {decision_tree_classifier.get_n_leaves()}')  
    print()


from catboost import CatBoostClassifier

import xgboost as xgb
from sklearn.metrics import accuracy_score

cbc = CatBoostClassifier()

cbc.fit(train_X, train_y)
y_pred = cbc.predict(valid_X)
print(f'accuracy di CatBoostClassifier: {accuracy_score(valid_y, pred_y):>.3f}')  


from lightgbm import LGBMClassifier

lgbmc = LGBMClassifier()
lgbmc.fit(train_X, train_y)
y_pred = lgbmc.predict(valid_X)
print(f'accuracy di LGBMClassifier: {accuracy_score(valid_y, pred_y):>.3f}')  


import xgboost as xgb
from xgboost import XGBClassifier, plot_tree

xgbmodel = XGBClassifier(random_state=0)
xgbmodel.fit(train_X,train_y)
pred_y = xgbmodel.predict(valid_X)
print(f'accuracy: {accuracy_score(pred_y, valid_y)}')


from sklearn.ensemble import RandomForestClassifier as RandomForest

n_estimator = [10,25,50,100,125]
max_features = ['log2', 'sqrt', 15, 50] # ~60 feature totali,  log2=6, sqrt=8
max_depth = [3,5,8,12,15,20]

class RandomForestConfig:
    def __init__(self, n_estimators, max_depth, max_features, accuracy):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.accuracy = accuracy
        

rf_config_list = []

for est in n_estimator:
    for mf in max_features:
        for md in max_depth:
            model = RandomForest(
                est,
                max_features=mf,
                max_depth=md,
                random_state=0
            )
            model.fit(train_X, train_y)
            pred_y = model.predict(valid_X)
            acc = accuracy_score(pred_y, valid_y)
            rf_config_list.append(RandomForestConfig(est, md, mf, accuracy=acc))


#migliori scelte di parametri
rf_config_list.sort(key = lambda conf: -conf.accuracy)
for c in rf_config_list[:20]: 
    print(f'  accuracy: {c.accuracy:>.3f} n_estimators: {c.n_estimators:>3} max_depth: {c.max_depth if (c.max_depth is not None) else "None":>4}, max_features: {c.max_features if (c.max_features is not None) else "None":>4}') 


model1 = RandomForest(
    100,
    max_depth=8,
    min_samples_leaf=10
)
model1.fit(train_X, train_y)
pred_y = model1.predict(valid_X)

print(f'accuracy: {accuracy_score(pred_y, valid_y):>.3f}')


class XGBConfig:
    def __init__(self, gamma, subsample, accuracy, max_depth):
        self.gamma = gamma
        self.subsample = subsample
        self.accuracy = accuracy
        self.max_depth = max_depth

    def __repr__(self):
        return f"accuracy: {self.accuracy: <3.2f} gamma: {self.gamma: <2} subsample: {self.subsample: <2} max_depth: {self.max_depth: <2}"


xgb_config_list = []

#loss minima necessaria per eseguire uno split
gamma = [0, 5, 10, 20]
#quanta porzione di dataset prende ogni tree
subsample = [1, .66, .5]
#max depth
max_depth = [3,5,8,12,15,20]

for g in gamma:
    for s in subsample:
        for md in max_depth:
            model = XGBClassifier(
                subsample=s,
                gamma=g,
                max_depth = md,
                random_state=0
            )
            model.fit(train_X, train_y)
            pred_y = model.predict(valid_X)
            acc = accuracy_score(pred_y, valid_y)
            xgb_config_list.append(XGBConfig(gamma=g, subsample=s, accuracy=acc, max_depth=md))


def printXgbScores(XGBConfigArray):
    pass

xgb_config_list.sort(key= lambda el : el.accuracy, reverse=True)
for el in xgb_config_list:
    print(el)


model2 = XGBClassifier(
    subsamples=0.66,
    gamma=8
)

model2.fit(train_X, train_y)
pred_y2 = model2.predict(valid_X)


print(f"""
    accuracy Random Forest: {accuracy_score(valid_y, pred_y):<3.3f}
    accuracy XGB classifier: {accuracy_score(valid_y, pred_y2):<3.3f}
""")


import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

MAX_CLUSTERS = 15
previousGain = 0
gain = 0
gains = []
clusters_range = range(1, MAX_CLUSTERS + 1)

for c in clusters_range:
    k_means = KMeans(
        n_clusters=c,
        max_iter=10,
        init='k-means++',
        random_state=0,
        n_init='auto'
    )
    k_means.fit(train_X)
    gain = previousGain - k_means.inertia_ if previousGain > 0 else k_means.inertia_
    print(f"clusters: {c:<2} SSE = {k_means.inertia_:<.2f} gain = {gain:<.2f}")
    gains.append( k_means.inertia_)
    previousGain = k_means.inertia_

# Plot gain curve
plt.plot(clusters_range, gains, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('Gain in SSE')
plt.title('Gain Curve of KMeans Clustering')
plt.grid(True)
plt.show()


MAX_CLUSTERS = 15
previousSSE = 0

for c in range(1, MAX_CLUSTERS+1):
    k_means = KMeans(
        n_clusters=c,
        max_iter=10,
        init='random',
        random_state=0,
        n_init='auto'
    )
    k_means.fit(train_X)
    print(f"clusters: {c}, SSE = {k_means.inertia_:<.2f}, gain = {previousSSE - k_means.inertia_:<.2f}")
    previousSSE = k_means.inertia_


def print_clustering_stats(labels):
    value, count = np.unique(labels, return_counts=True)
    for i in value:
        print(f"value: {value[i]} instances: {count[i]}")


def do_kmeans(n_clusters = 1):
    k_means = KMeans(
        n_clusters=n_clusters,
        max_iter=10,
        init='k-means++',
        random_state=0,
        n_init='auto'
    )
    
    k_means.fit(train_X)
    train_X_with_cluster = train_X.copy()
    train_X_with_cluster['cluster'] = k_means.labels_
    print_clustering_stats(k_means.labels_)
    
    return train_X_with_cluster


do_kmeans(6)
print() #evita di printare il dataset ritornato da do_kmeans


do_kmeans(4)
print()


from sklearn.cluster import AgglomerativeClustering

linkage_types = ['ward', 'complete', 'average', 'single']

for l in linkage_types:
    clustering = AgglomerativeClustering(
        n_clusters=4,
        linkage=l,
        memory='/kaggle/tmp'
    )
    clustering.fit(X)

    print(f"\nlinkage: {l}")
    print_clustering_stats(clustering.labels_)


from sklearn.cluster import DBSCAN

dbscan = DBSCAN(
    eps=15,
    min_samples = 30
)
dbscan.fit(X)

print_clustering_stats(dbscan.labels_)


from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=50)
rf.fit(X,y)

from sklearn.metrics import pairwise_distances

# pairwise distance
leaves = rf.apply(X)
pair_wise_dist = pairwise_distances( 
    X=leaves, 
    metric= lambda oi, oj: 1.0-np.mean(oi==oj)
)


dbscan.fit(pair_wise_dist)
print_clustering_stats(dbscan.labels_)


dbscan2 = DBSCAN(
    eps=1,
    min_samples = 30
)

dbscan2.fit(pair_wise_dist)
print_clustering_stats(dbscan2.labels_)


def find_decent_esp(dataset, min_samples, logging=True, custom_esp_max=None):

    esp_min = 0
    esp_max = custom_esp_max if custom_esp_max != None else 100 #custom_esp_max aggiunto perche se un esp decente fosse >100 l'algoritmo non sarebbe accurato
    esp = esp_max
    
    test_dbscan = DBSCAN(
            eps=esp,
            min_samples = min_samples
    )
    test_dbscan.fit(dataset)

    iterationsCount = 0
    
    while(not esp_is_decent(test_dbscan)):

        iterationsCount+=1
        if (iterationsCount > 20):
            print("no decent esp value found")
            return 0
        
        esp = (esp_max+esp_min)/2
        
        test_dbscan = DBSCAN(
            eps=esp,
            min_samples = min_samples
        )
        test_dbscan.fit(dataset)

        if (esp_is_too_big(test_dbscan)):
            esp_max = esp

        if (esp_is_too_small(test_dbscan)):
            esp_min = esp     

        if logging:
            print(f"[find_decent_esp] - current esp: {esp}")
        
    return esp


def esp_is_too_big(clustering_model):
    labels = clustering_model.labels_

    n_outliers = np.sum(labels == -1)
    n_instances_first_cluster = np.sum(labels == 0)
    n_instances_other_clusters = np.sum(labels > 0)

    return n_instances_other_clusters == 0 and n_instances_first_cluster > n_outliers


def esp_is_too_small(clustering_model):
    labels = clustering_model.labels_

    n_outliers = np.sum(labels == -1)
    n_instances_clusters = np.sum(labels != -1)

    return n_outliers > n_instances_clusters

def esp_is_decent(clustering_model):
    labels = clustering_model.labels_

    n_outliers = np.sum(labels == -1)
    n_instances_first_cluster = np.sum(labels == 0)
    n_instances_other_clusters = np.sum(labels > 0)

    return n_instances_other_clusters > 1


print(find_decent_esp(X, 30))


def find_esp_range(dataset, min_samples, step_size=0.05, logging=True):
    decent_esp = find_decent_esp(dataset, min_samples, logging=logging)

    step = decent_esp*step_size
    
    #test limite superiore
    test_dbscan = DBSCAN(
        eps=decent_esp,
        min_samples = min_samples
    )
    test_dbscan.fit(dataset)
    
    max_decent_esp = decent_esp
    while(not esp_is_too_big(test_dbscan)):

        test_dbscan = DBSCAN(
            eps=max_decent_esp+step,
            min_samples = min_samples
        )        
        test_dbscan.fit(dataset)

        if not esp_is_too_big(test_dbscan):
            max_decent_esp+=step

        if logging:
            print(f"[find_esp_range] - current max: {max_decent_esp}")
    
    #test limite inferiore
    test_dbscan = DBSCAN(
        eps=decent_esp,
        min_samples = min_samples
    )
    test_dbscan.fit(dataset)
    
    min_decent_esp = decent_esp
    while(not esp_is_too_small(test_dbscan)):

        test_dbscan = DBSCAN(
            eps=min_decent_esp-step,
            min_samples = min_samples
        )        
        test_dbscan.fit(dataset)

        if not esp_is_too_small(test_dbscan):
            min_decent_esp-=step

        if logging:
            print(f"[find_esp_range] - current min: {min_decent_esp}")
    
    return min_decent_esp, max_decent_esp


find_esp_range(X, 30, step_size=0.05)


#nota: ho avuto problemi a usare il dbscan per pair_wise_dist, find_decent_esp potrebbe non riuscire a trovare un parametro, dipende dalla randomness del random forest. 
find_esp_range(pair_wise_dist, 30)


from sklearn.metrics import silhouette_score

dbscan = DBSCAN(
    eps = 1.46,
    min_samples = 20
)
dbscan.fit(pair_wise_dist)
print_clustering_stats(dbscan.labels_)

labels = dbscan.labels_ != -1
X_no_noise = X[labels]
labels_no_noise = dbscan.labels_[labels]


dbscan = DBSCAN(
    eps = 16,
    min_samples = 30
)
dbscan.fit(X)
print_clustering_stats(dbscan.labels_)

labels = dbscan.labels_ != -1
X_no_noise = X[labels]
labels_no_noise = dbscan.labels_[labels]
print(f"Silhouette Score: { silhouette_score(X_no_noise, labels_no_noise):.3f}")


print(y.value_counts())


features_with_score = []

scores = rf.feature_importances_
cols = train_X.columns
for i in range(len(cols)):
    features_with_score.append((cols[i], scores[i]))

features_with_score.sort(key=lambda it : it[1], reverse=True)
print(f"feature importance of [{len(features_with_score)}] features:\n")
for item in features_with_score:
    print(f"{item[0]}, score: {item[1]}")
    



feature_names = [f[0] for f in features_with_score]
feature_scores = [f[1] for f in features_with_score]

plt.figure(figsize=(10, 20))
plt.barh(feature_names, feature_scores)
plt.xlabel("Importance Score")
plt.title(f"Feature Importance of {len(feature_names)} Features")
plt.gca().invert_yaxis()
plt.grid(axis='x')
plt.show()


important_features = [f for f in features_with_score if f[1] > 0.01]
important_features.sort()
important_cols = [f[0] for f in important_features]

N_IMPORTANT_FEATURES = len(important_features)

X[important_cols]

rf.fit(train_X[important_cols], train_y)
new_pred_y = rf.predict(valid_X[important_cols])

accuracy_score(new_pred_y, valid_y)


# print(tuned_model.predict_proba(train_X))
from sklearn.metrics import ConfusionMatrixDisplay
ConfusionMatrixDisplay.from_estimator(
    estimator = model1,
    X=valid_X, y=valid_y
)


class Prediction_Info:
    def __init__(self, idx, predicted_value, predicted_prob, correct_value, correct_prob):
        self.idx = idx
        self.predicted_value = predicted_value
        self.predicted_prob = predicted_prob
        self.correct_value = correct_value
        self.correct_prob = correct_prob
        self.is_correct = predicted_value == correct_value

    def __repr__(self):
        return f"idx: {self.idx:<4.0f} | value predicted: {self.predicted_value:<1.0f} | predicted prob: {self.predicted_prob:<1.2f} | correct value: {self.correct_value:<1.0f} | correct prob: {self.correct_prob:<1.2f} | {'RIGHT' if self.is_correct else 'WRONG'}"



from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    
predictions = []
for train_idx, valid_idx in skf.split(X, y):
    
    rf.fit(X.iloc[train_idx], y.iloc[train_idx])
    preds = rf.predict(X.iloc[valid_idx])
    probs = rf.predict_proba(X.iloc[valid_idx])

    for i, idx in enumerate(valid_idx):
        prediction = Prediction_Info(
            idx,
            preds[i],
            probs[i].max(),
            y.iloc[idx],
            probs[i, int(y.iloc[idx])]
        )
        predictions.append(prediction)

predictions.sort(key=lambda item:item.idx)


for p in predictions[:5]:
    print(p)


right_pred = [p for p in predictions if p.is_correct]
wrong_pred = [p for p in predictions if not p.is_correct]


importan_cols_set = set(important_cols)


class FeaturesVarianceRank:
    def __init__(self, name, rank, localRank):
        self.name = name
        self.rank = rank
        self.localRank = localRank

    def __repr__(self):
        return f"nome: {self.name:<30} | rank (global): {self.rank:<3} | rank (local): {self.localRank:<3}"

#sortate per importanza decrescente
features_with_score.sort(key=lambda it : it[1], reverse=True)

ranked_features = {
    feature: FeaturesVarianceRank(feature, i + 1, None)
    for i, (feature, score) in enumerate(features_with_score)
}

wrong_pred.sort(key=lambda el: el.correct_prob)

#estraiamo le 50 istanze peggiori e vediamo come si comportano le features

wrong_idx = [el.idx for el in wrong_pred[:50]]
feature_variances = X.iloc[wrong_idx].var()

sorted_features = feature_variances.sort_values(ascending=False)

for rank, feature_name in enumerate(sorted_features.index, start=1):
    ranked_features[feature_name].localRank = rank


l = list(ranked_features.values())
l = [el for el in l if el.name in importan_cols_set]
l.sort(key = lambda el: el.rank, reverse=False)
l


l2 =[(el.name, el.rank-el.localRank) for el in l]
l2.sort(key = lambda el: el[1], reverse=True)
for el in l2:
    string = "variance increased by" if el[1]>=0 else "variance decreased by"
    print(f"nome: {el[0]:<35} {string}: {el[1]}")



#sortate per importanza decrescente
features_with_score.sort(key=lambda it : it[1], reverse=True)

ranked_features = {
    feature: FeaturesVarianceRank(feature, i + 1, None)
    for i, (feature, score) in enumerate(features_with_score)
}

right_pred.sort(key=lambda el: el.predicted_prob, reverse=True)

right_idx = [el.idx for el in right_pred[:100]]
feature_variances = X.iloc[right_idx].var()

sorted_features = feature_variances.sort_values(ascending=False)

for rank, feature_name in enumerate(sorted_features.index, start=1):
    ranked_features[feature_name].localRank = rank


l = list(ranked_features.values())
l = [el for el in l if el.name in importan_cols_set]
l.sort(key = lambda el: el.rank, reverse=False)
l


l2 =[(el.name, el.rank-el.localRank) for el in l]
l2.sort(key = lambda el: el[1], reverse=True)
for el in l2:
    string = "variance increased by" if el[1]>=0 else "variance decreased by"
    print(f"nome: {el[0]:<35} {string}: {el[1]}")

