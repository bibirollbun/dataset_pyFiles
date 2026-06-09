import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


xtrain=pd.read_csv('/kaggle/input/home-credit-default-risk/application_train.csv')
xtrain.head(10)


xtest=pd.read_csv('/kaggle/input/home-credit-default-risk/application_test.csv')
xtest.head(10)


xtrain.info()


xtest.info()


print("The number of duplicate clients in application_train is : ",xtrain.duplicated(subset=['SK_ID_CURR']).sum())


print("The number of duplicate clients in application_test is : ",xtest.duplicated(subset=['SK_ID_CURR']).sum())


pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)


print(xtrain.isna().sum())


print(xtest.isna().sum())


PourcentxtrainNaN=((xtrain.isna().sum()*100)/len(xtrain))
PourcentxtrainNaN


PourcentxtestNaN=((xtest.isna().sum()*100)/len(xtest))
PourcentxtestNaN


# Affiche seulement les colonnes avec plus de 40% de valeurs manquantes
columns_Train_over_50 = PourcentxtrainNaN[PourcentxtrainNaN > 40]
print(columns_Train_over_50)
print(len(columns_Train_over_50))


columns_Test_over_50 = PourcentxtestNaN[PourcentxtestNaN > 40]
print(columns_Test_over_50)
print(len(columns_Test_over_50))


xtrain.drop(columns_Train_over_50.index,axis=1,inplace=True)


xtrain.head()


print("The dimension of the application_train table after deleting columns containing more than 40% missing values is:",xtrain.shape)


xtest.drop(columns_Test_over_50.index,axis=1,inplace=True)


print("The dimension of the application_test table after deleting columns containing more than 40% missing values is:",xtest.shape)


columns_Train_less_40 = PourcentxtrainNaN[(PourcentxtrainNaN <= 40) & (PourcentxtrainNaN != 0)]
print(columns_Train_less_40)
print(len(columns_Train_less_40))


columns_Test_less_40 = PourcentxtestNaN[(PourcentxtestNaN <= 40) & (PourcentxtestNaN != 0)]
print(columns_Test_less_40)
print(len(columns_Test_less_40))


xtrain.shape


def missing_value_Preprocessing(df, colonnes=None):
     
    # Si aucune colonne n'est spécifiée, utiliser toutes les colonnes
    if colonnes == None:
        colonnes = df.columns
    
    # Dictionnaire pour stocker les statistiques
    stats = {}
    for colonne in colonnes:
        if colonne not in df.columns:
            print(f"Attention: La colonne '{colonne}' n'existe pas dans le DataFrame")
            continue
            
        # Ignorer les colonnes sans valeurs manquantes
        if not df[colonne].isna().any():
            stats[colonne] = {
                "nb_remplacements": 0,
                "valeur_remplacement": None,
                "pourcentage_remplacement": 0
            }
            continue
            
        # Trouver la valeur la plus fréquente
        valeur_frequente = df[colonne].mode()[0]
        
        # Compter le nombre de valeurs manquantes avant remplacement
        nb_manquants = df[colonne].isna().sum()
        
        # Remplir les valeurs manquantes
        df[colonne] = df[colonne].fillna(valeur_frequente)
        
        # Calculer le pourcentage de remplacement
        pourcentage = (nb_manquants / len(df)) * 100
        
        # Stocker les statistiques
        stats[colonne] = {
            "nb_remplacements": nb_manquants,
            "valeur_remplacement": valeur_frequente,
            "pourcentage_remplacement": round(pourcentage, 2)
        }
        
        # Afficher un résumé pour cette colonne
        print(f"\nColonne: {colonne}")
        print(f"- Nombre de valeurs manquantes remplacées: {nb_manquants}")
        print(f"- Valeur utilisée pour le remplacement: {valeur_frequente}")
        print(f"- Pourcentage de valeurs remplacées: {pourcentage:.2f}%")
    
    return df, stats


missing_value_Preprocessing(xtrain,['AMT_ANNUITY','AMT_GOODS_PRICE','NAME_TYPE_SUITE','OCCUPATION_TYPE','EXT_SOURCE_2','EXT_SOURCE_3','OBS_30_CNT_SOCIAL_CIRCLE','DEF_30_CNT_SOCIAL_CIRCLE','OBS_60_CNT_SOCIAL_CIRCLE','DEF_60_CNT_SOCIAL_CIRCLE','DAYS_LAST_PHONE_CHANGE','AMT_REQ_CREDIT_BUREAU_HOUR','AMT_REQ_CREDIT_BUREAU_DAY','AMT_REQ_CREDIT_BUREAU_WEEK','AMT_REQ_CREDIT_BUREAU_MON','AMT_REQ_CREDIT_BUREAU_QRT','AMT_REQ_CREDIT_BUREAU_YEAR'])


xtrain.shape


xtrain.isnull().sum()


new_application_test , select =missing_value_Preprocessing(xtest,['AMT_ANNUITY','NAME_TYPE_SUITE','OCCUPATION_TYPE','EXT_SOURCE_2','EXT_SOURCE_3','OBS_30_CNT_SOCIAL_CIRCLE','DEF_30_CNT_SOCIAL_CIRCLE','OBS_60_CNT_SOCIAL_CIRCLE','DEF_60_CNT_SOCIAL_CIRCLE','AMT_REQ_CREDIT_BUREAU_HOUR','AMT_REQ_CREDIT_BUREAU_DAY','AMT_REQ_CREDIT_BUREAU_WEEK','AMT_REQ_CREDIT_BUREAU_MON','AMT_REQ_CREDIT_BUREAU_QRT','AMT_REQ_CREDIT_BUREAU_YEAR'])


def afficher_stats(stats):
    
    for colonne, info in stats.items():
        print(f"\nColonne: {colonne}")
        print(f"- Nombre de valeurs manquantes remplacées: {info['nb_remplacements']}")
        if info['valeur_remplacement'] is not None:
            print(f"- Valeur utilisée pour le remplacement: {info['valeur_remplacement']}")
        else:
            print("- Aucune valeur manquante à remplacer")
        print(f"- Pourcentage de valeurs remplacées: {info['pourcentage_remplacement']:.2f}%")
afficher_stats(select)   


new_application_test.isnull().sum()


xtest.shape


def plot_boxplots(df):
    num_cols = len(df.columns)
    fig, axes = plt.subplots(nrows=(num_cols // 5) + 1, ncols=5, figsize=(40, 30))
    axes = axes.flatten()
    
    for i, col in enumerate(df.columns):
        if df[col].dtype in ['int64', 'float64']:  # Variables numériques
            sns.boxplot(y=df[col], ax=axes[i])
        else:  # Variables catégorielles
            sns.countplot(y=df[col], ax=axes[i])
        axes[i].set_title(col)
        
    plt.tight_layout()
    plt.show()
plot_boxplots(xtrain)


plot_boxplots(new_application_test)


xbureau=pd.read_csv('/kaggle/input/home-credit-default-risk/bureau.csv',low_memory=False)
xbureau.head(10)


xbureau.isnull().sum()


PourcentxbureauNaN=((xbureau.isna().sum()*100)/len(xbureau))
PourcentxbureauNaN


# Affiche seulement les colonnes avec plus de 40% de valeurs manquantes
columns_xbureau_over_50 = PourcentxbureauNaN[PourcentxbureauNaN > 50]
print(columns_xbureau_over_50)
print(len(columns_xbureau_over_50))


xbureau.drop(columns_xbureau_over_50.index,axis=1,inplace=True)


xbureau.shape


xbureau = xbureau.dropna()


print("the new shape of this dataset is :",xbureau.shape)


plot_boxplots(xbureau)


def agreger_par_type(df, id_colonnes):
    colonnes_numeriques = df.select_dtypes(include=['number']).columns.tolist()
    colonnes_categorielles = df.select_dtypes(exclude=['number']).columns.tolist()
    
    # Exclure la colonne d'identifiant des listes
    for id_col in id_colonnes:
        if id_col in colonnes_numeriques:
            colonnes_numeriques.remove(id_col)
    for id_col in id_colonnes:
        if id_col in colonnes_categorielles:
            colonnes_categorielles.remove(id_col)
    
    agg_dict = {}
    
    for col in colonnes_numeriques:
        agg_dict[col] = ['sum', 'mean']
    
    for col in colonnes_categorielles:
        agg_dict[col] = [
            ('valeur_frequente', lambda x: x.mode()[0] if not x.mode().empty else None),
            ('nb_valeurs_uniques', pd.Series.nunique)
        ]
    
    if agg_dict:
        return df.groupby(id_colonnes).agg(agg_dict)
    else:
        return pd.DataFrame()  


xbureau=agreger_par_type(xbureau,['SK_ID_CURR','SK_ID_BUREAU']).head(10)


xbureau.head(10)


xprevious_application=pd.read_csv('/kaggle/input/home-credit-default-risk/previous_application.csv')
xprevious_application.head(10)


xprevious_application.isnull().sum()


Pourcentxprevious_applicationNaN=((xprevious_application.isna().sum()*100)/len(xprevious_application))
Pourcentxprevious_applicationNaN


columns_xprevious_application_over_50 = Pourcentxprevious_applicationNaN[Pourcentxprevious_applicationNaN > 40]
print(columns_xprevious_application_over_50)
print(len(columns_xprevious_application_over_50))


xprevious_application.drop(columns_xprevious_application_over_50.index,axis=1,inplace=True)


xprevious_application = xprevious_application.dropna()


xprevious_application.shape


plot_boxplots(xprevious_application)


def agreger_par_type(df, id_colonnes):
    colonnes_numeriques = df.select_dtypes(include=['number']).columns.tolist()
    colonnes_categorielles = df.select_dtypes(exclude=['number']).columns.tolist()
    
    # Exclure la colonne d'identifiant des listes
    for id_col in id_colonnes:
        if id_col in colonnes_numeriques:
            colonnes_numeriques.remove(id_col)
    for id_col in id_colonnes:
        if id_col in colonnes_categorielles:
            colonnes_categorielles.remove(id_col)
    
    agg_dict = {}
    
    for col in colonnes_numeriques:
        agg_dict[col] = ['sum', 'mean']
    
    for col in colonnes_categorielles:
        agg_dict[col] = [
            ('valeur_frequente', lambda x: x.mode()[0] if not x.mode().empty else None),
            ('nb_valeurs_uniques', pd.Series.nunique)
        ]
    
    if agg_dict:
        return df.groupby(id_colonnes).agg(agg_dict)
    else:
        return pd.DataFrame()  


xprevious_application=agreger_par_type(xprevious_application,['SK_ID_PREV','SK_ID_CURR'])


xinstallments_payments=pd.read_csv('/kaggle/input/home-credit-default-risk/installments_payments.csv', low_memory=False)
xinstallments_payments.head(10)


xinstallments_payments.isnull().sum()


Pourcentxinstallments_paymentsNaN=((xinstallments_payments.isna().sum()*100)/len(xinstallments_payments))
Pourcentxinstallments_paymentsNaN





xinstallments_payments = xinstallments_payments.dropna()


print("the new shape of this dataset is :",xinstallments_payments.shape)


plot_boxplots(xinstallments_payments)


def agreger_par_type(df, id_colonnes):
    agg_dict = {}
    for id_col in id_colonnes:
        if id_col in id_colonnes:
            df.drop(id_col,axis=1)
    for col in df.columns:
        agg_dict[col] = ['sum', 'mean']
    if agg_dict:
        return df.groupby(id_colonnes).agg(agg_dict)
    else:
        return pd.DataFrame()  


xinstallments_payments=agreger_par_type(xinstallments_payments,['SK_ID_PREV','SK_ID_CURR'])


xinstallments_payments.shape


xinstallments_payments.head(10)


fusion_left = pd.merge(xtrain,xbureau,xprevious_application,xinstallments_payments, on='id', how='left')


fusion_left = pd.merge(xtest,xbureau,xprevious_application,xinstallments_payments, on='id', how='left')


from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model

# Dimensions des données
input_dim = xtrain.shape[1]  # Nombre de features d'entrée
latent_dim = 8  # Taille de la représentation latente (feature engineering)

# Définition de l'encodeur
input_layer = Input(shape=(input_dim,))
encoded = Dense(16, activation='relu')(input_layer)
encoded = Dense(latent_dim, activation='relu')(encoded)  # Couche latente

# Définition du décodeur
decoded = Dense(16, activation='relu')(encoded)
decoded = Dense(input_dim, activation='sigmoid')(decoded)  # Reconstruction

# Création du modèle autoencodeur
autoencoder = Model(input_layer, decoded)

# Extraction de l'encodeur (pour le feature engineering)
encoder = Model(input_layer, encoded)

# Compilation du modèle
autoencoder.compile(optimizer='adam', loss='mse')

# Affichage du résumé du modèle
autoencoder.summary()



autoencoder.fit(X_train_tf, X_train_tf,
                epochs=50,
                batch_size=32,
                shuffle=True,
                validation_data=(X_test_tf, X_test_tf))


# Extraction des nouvelles features
X_train_encoded = encoder.predict(X_train)
X_test_encoded = encoder.predict(X_test)

print("Ancienne dimension des features :", X_train.shape)
print("Nouvelle dimension des features :", X_train_encoded.shape) 

