pip install -qq scikit-learn==1.6.1


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
import os 
from tqdm import tqdm
from sklearn.preprocessing import TargetEncoder
import warnings
warnings.simplefilter('ignore')



plt.style.use("seaborn-v0_8")
sns.set_palette("husl")
pd.option_context('mode.use_inf_as_na', True)


def numerical_distrib_analysis(data, numerical_features):
    """
    Analyse la distribution des variables numÃ©riques avec histogrammes et boxplots.
    
    :param data: DataFrame Pandas contenant les donnÃ©es
    :param numerical_features: Liste des noms de colonnes numÃ©riques
    """
    
    for feature in numerical_features:
        plt.figure(figsize=(12, 5))

        # Histogramme avec KDE
        plt.subplot(1, 2, 1)
        sns.histplot(data[feature], kde=True, bins=30)
        plt.title(f"Histogram of {feature}")
        plt.xlabel(feature)
        plt.ylabel("Frequency")

        # Box plot pour dÃ©tecter les outliers
        plt.subplot(1, 2, 2)
        sns.boxplot(x=data[feature])
        plt.title(f"Box Plot of {feature}")

        plt.tight_layout()
        plt.show()

        # Statistiques supplÃ©mentaires
        print(f"\nStatistics for {feature}:")
        print(f"Skewness: {data[feature].skew():.2f}")
        print(f"Number of Missing Values: {data[feature].isnull().sum()}")

def categorical_distrib_analysis(data, categorical_features, top_n=10):
    """
    Analyse et visualisation des variables catÃ©gorielles.

    :param data: DataFrame Pandas contenant les donnÃ©es.
    :param categorical_features: Liste des colonnes catÃ©gorielles Ã  analyser.
    :param top_n: Nombre de catÃ©gories les plus frÃ©quentes Ã  afficher pour les variables avec beaucoup de valeurs uniques.
    """
    
    for feature in categorical_features:
        plt.figure(figsize=(10, 6))

        # VÃ©rifier le nombre de catÃ©gories uniques
        unique_count = data[feature].nunique()

        if unique_count > top_n:  
            # Si beaucoup de valeurs uniques, afficher uniquement les top_n catÃ©gories les plus frÃ©quentes
            top_categories = data[feature].value_counts().nlargest(top_n)
            sns.barplot(x=top_categories.index, y=top_categories.values, palette="pastel")
            plt.title(f"Top {top_n} {feature} Categories")
        else:
            # Sinon, afficher toutes les catÃ©gories
            sns.countplot(x=data[feature], order=data[feature].value_counts().index, palette="pastel")
            plt.title(f"Distribution of {feature}")

        plt.xlabel(feature)
        plt.ylabel("Count")
        plt.xticks(rotation=45)
        plt.show()

        # Affichage des statistiques
        print(f"Feature: {feature}")
        print(f"Number of Unique Values: {unique_count}")
        print(f"Missing Values: {data[feature].isnull().sum()}\n")

def numerical_correlation_analysis(data, numerical_features, target):
    """
    Analyse et visualisation des relations entre les variables numÃ©riques et la variable cible.

    :param data: DataFrame Pandas contenant les donnÃ©es.
    :param numerical_features: Liste des colonnes numÃ©riques Ã  analyser.
    :param target: Nom de la variable cible.
    """
    
    # Scatter plots pour chaque variable numÃ©rique (sauf la cible)
    for feature in numerical_features:
        if feature != target:  # Exclure la variable cible
            plt.figure(figsize=(8, 6))
            sns.scatterplot(x=data[feature], y=data[target], alpha=0.5)
            plt.title(f"{feature} vs. {target}")
            plt.xlabel(feature)
            plt.ylabel(target)
            plt.show()

    # Matrice de corrÃ©lation
    correlation_matrix = data[numerical_features].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Matrix of Numerical Features")
    plt.show()

def categorical_correlation_analysis(data, categorical_features, target, high_cardinality_threshold=10):
    """
    Visualisation des variables catÃ©gorielles par rapport Ã  la variable cible avec des box plots.

    :param data: DataFrame Pandas contenant les donnÃ©es.
    :param categorical_features: Liste des colonnes catÃ©gorielles Ã  analyser.
    :param target: Nom de la variable cible.
    :param high_cardinality_threshold: Seuil de cardinalitÃ© pour ignorer les variables ayant trop de catÃ©gories.
    """

    for feature in categorical_features:
        if data[feature].nunique() <= high_cardinality_threshold:  # Ignorer les variables Ã  haute cardinalitÃ©
            plt.figure(figsize=(10, 6))
            sns.boxplot(x=data[feature], y=data[target], palette='husl')
            plt.title(f"{feature} vs. {target}")
            plt.xlabel(feature)
            plt.ylabel(target)
            plt.xticks(rotation=45)
            plt.show()
        else:
            print(f"Skipping {feature}: too many unique values ({data[feature].nunique()})\n")


original_df = pd.read_csv("/kaggle/input/podcast-listening-time-prediction-dataset/podcast_dataset.csv")
train_df = pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv', index_col='id')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv', index_col='id')
sample_submission = pd.read_csv('/kaggle/input/playground-series-s5e4/sample_submission.csv')

# Concatenate original data with synthetics ones
train_df = pd.concat([train_df, original_df], axis=0, ignore_index=True)
train_df.drop_duplicates()

# Drop index columns
# train_df = train_df.drop(columns=['id'])
# test_df = test_df.drop(columns=['id'])


print("\nData Info:")
train_df.info()

print("\nNumerical Features Summary:")
display(train_df.describe())

print("\nFirst 10 rows of Dataset:")
train_df.head(10)


# Analysing distributions of numerical features
numerical_features = [
    'Episode_Length_minutes', 
    'Host_Popularity_percentage',
    'Guest_Popularity_percentage', 
    'Number_of_Ads',
    'Listening_Time_minutes',
]

numerical_distrib_analysis(train_df, numerical_features)



categorical_features = [
    'Podcast_Name', 
    'Genre', 
    'Publication_Day',
    'Publication_Time', 
    'Episode_Sentiment'
]

categorical_distrib_analysis(train_df, categorical_features)


numerical_correlation_analysis(train_df, numerical_features, "Listening_Time_minutes")


categorical_correlation_analysis(train_df, categorical_features, 'Listening_Time_minutes')


print("Missing Values per Column:")
print(train_df.isnull().sum())

print("Missing Values per Column:")
print(test_df.isnull().sum())


test_df.isna().sum()


# Replacing null values by median
train_df['Episode_Length_minutes'].fillna(train_df['Episode_Length_minutes'].median(), inplace=True)
test_df['Episode_Length_minutes'].fillna(train_df['Episode_Length_minutes'].median(), inplace=True)


# Null values could mean no guest 
train_df['Guest_Popularity_percentage'].fillna(train_df['Guest_Popularity_percentage'].median(), inplace=True)
train_df.dropna(inplace=True)

test_df['Guest_Popularity_percentage'].fillna(test_df['Guest_Popularity_percentage'].median(), inplace=True)

# Deleting outliers 
train_df = train_df[train_df['Number_of_Ads']<10]



from sklearn.preprocessing import LabelEncoder

# Encoder for categorical data
label_encoders = {col: LabelEncoder() for col in categorical_features}

# Apply LabelEncoder to each categorical column
for col in categorical_features:
    train_df[col] = label_encoders[col].fit_transform(train_df[col])
    test_df[col] = label_encoders[col].transform(test_df[col])

    # Converting to category type 
    train_df[col] = train_df[col].astype('category')
    test_df[col] = test_df[col].astype('category')



train_df.head(10)


import gc
gc.collect()


# creating most relevant feature 
train_df['Episode_Num'] = train_df['Episode_Title'].str[8:].astype('category')
test_df['Episode_Num'] = test_df['Episode_Title'].str[8:].astype('category')

train_df = train_df.drop(columns=['Episode_Title'])
test_df = test_df.drop(columns=['Episode_Title'])


from tqdm import tqdm
from itertools import combinations

columns_to_encode = ['Episode_Length_minutes', 
                     'Episode_Num', 
                     'Host_Popularity_percentage', 
                     'Number_of_Ads', 
                     'Episode_Sentiment', 
                     'Publication_Day', 
                     'Publication_Time']

pair_size = [2, 3, 4]

for r in pair_size: 
    for cols in tqdm(list(combinations(columns_to_encode, r))): 
        new_col_name = '_'.join(cols)

        train_df[new_col_name] = train_df[list(cols)].astype(str).agg('_'.join, axis=1) 
        train_df[new_col_name] = train_df[new_col_name].astype('category')

        test_df[new_col_name] = test_df[list(cols)].astype(str).agg('_'.join, axis=1) 
        test_df[new_col_name] = test_df[new_col_name].astype('category')
        


X = train_df.drop(columns=['Listening_Time_minutes'])
y = train_df['Listening_Time_minutes']


from sklearn.model_selection import KFold
from sklearn.preprocessing import TargetEncoder
import lightgbm as lgb

warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=FutureWarning)

cv = KFold(5, random_state=42, shuffle=True)
y_pred = np.zeros(len(sample_submission))

for idx_train, idx_valid in cv.split(X, y):
    X_train, y_train = X.iloc[idx_train], y.iloc[idx_train]
    X_valid, y_valid = X.iloc[idx_valid], y.iloc[idx_valid]
    X_test = test_df[X.columns].copy()

    encoded_columns = train_df.columns[11:]
    encoder = TargetEncoder(random_state=42)

    X_train.loc[:, encoded_columns] = encoder.fit_transform(X_train[encoded_columns], y_train)
    X_valid.loc[:, encoded_columns] = encoder.transform(X_valid[encoded_columns])
    X_test.loc[:, encoded_columns] = encoder.transform(X_test[encoded_columns])



    model = lgb.LGBMRegressor(
        n_iter=1000,
        max_depth=-1,
        num_leaves=1024,
        colsample_bytree=0.7,
        learning_rate=0.03,
        objective='l2',
        metric='rmse', 
        verbosity=-1,
        max_bin=1024,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        callbacks=[
            lgb.log_evaluation(100),
            lgb.early_stopping(stopping_rounds=100)
            ],
    )

    y_pred += model.predict(X_test)


pred_lgbm = y_pred /5
#sample_submission = pd.read_csv('/kaggle/input/playground-series-s5e4/sample_submission.csv')Â  
submission_lgbm = pd.DataFrame({'id': sample_submission.id, 'Listening_Time_minutes' : pred_lgbm})
submission_lgbm.to_csv('submissions.csv', index=False)


sample_submission.head()

