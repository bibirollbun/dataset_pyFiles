import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Charger les datasets
transactions = pd.read_csv('/kaggle/input/h-and-m-personalized-fashion-recommendations/transactions_train.csv')
articles = pd.read_csv('/kaggle/input/h-and-m-personalized-fashion-recommendations/articles.csv')
customers = pd.read_csv('/kaggle/input/h-and-m-personalized-fashion-recommendations/customers.csv')

# Aperçu des premières lignes des datasets
print("Transactions Data:")
print(transactions.head())
print("\nArticles Data:")
print(articles.head())
print("\nCustomers Data:")
print(customers.head())


# Afficher les 5 premières lignes de chaque DataFrame
print("Aperçu des articles:")
print(articles.head(), "\n")

print("Aperçu des clients:")
print(customers.head(), "\n")

print("Aperçu des transactions:")
print(transactions.head(), "\n")


# Statistiques descriptives pour les colonnes numériques
print("Statistiques descriptives des articles:")
print(df_articles.describe(), "\n")

print("Statistiques descriptives des clients:")
print(df_customers.describe(), "\n")

print("Statistiques descriptives des transactions:")



# Vérifier les valeurs manquantes
print("Valeurs manquantes dans les articles:")
print(articles.isnull().sum(), "\n")

print("Valeurs manquantes dans les clients:")
print(customers.isnull().sum(), "\n")

print("Valeurs manquantes dans les transactions:")
print(transactions.isnull().sum(), "\n")


# Remplacer les valeurs manquantes dans df_articles
articles['detail_desc'] = articles['detail_desc'].fillna('Valeur Manquante')


# Remplacer les valeurs manquantes dans df_customers
customers['FN'] = df_customers['FN'].fillna('Valeur Manquante')
customers['Active'] = df_customers['Active'].fillna('Valeur Manquante')
customers['club_member_status'] = df_customers['club_member_status'].fillna('Valeur Manquante')
customers['fashion_news_frequency'] = df_customers['fashion_news_frequency'].fillna('Valeur Manquante')
customers['age'] = df_customers['age'].fillna('Valeur Manquante')


# Remplacer les valeurs manquantes dans df_transactions (bien que toutes les colonnes aient des valeurs non manquantes)
transactions = transactions.fillna('Valeur Manquante')


# Vérifier les valeurs manquantes après remplacement
print(articles.isnull().sum())
print(customers.isnull().sum())
print(transactions.isnull().sum())


# Histogramme de la distribution des prix des articles
plt.figure(figsize=(10, 6))
sns.histplot(transactions['price'], kde=True, bins=50)
plt.title('Distribution des prix des articles')
plt.xlabel('Prix')
plt.ylabel('Fréquence')
plt.show()

# Histogramme du nombre d'articles achetés par transaction
plt.figure(figsize=(10, 6))
sns.histplot(transactions.groupby('customer_id')['article_id'].count(), kde=True, bins=50)
plt.title('Distribution du nombre d\'articles achetés par transaction')
plt.xlabel('Nombre d\'articles achetés')
plt.ylabel('Fréquence')
plt.show()



# Analyse du nombre total de transactions
total_transactions = transactions.shape[0]
print(f"Nombre total de transactions: {total_transactions}")

# Nombre d'articles uniques dans les transactions
unique_articles = transactions['article_id'].nunique()
print(f"Nombre d'articles uniques dans les transactions: {unique_articles}")

# Nombre de clients uniques
unique_customers = transactions['customer_id'].nunique()
print(f"Nombre de clients uniques: {unique_customers}")


# Comptage des achats par couleur
color_counts = transactions_with_articles['colour_group_name'].value_counts()

# Visualisation de la répartition des achats par couleur
plt.figure(figsize=(10,6))
sns.barplot(x=color_counts.index, y=color_counts.values, palette='coolwarm')
plt.title("Répartition des achats par couleur")
plt.xlabel('Couleur')
plt.ylabel('Nombre d\'achats')
plt.show()

# Conclusion: Afficher la couleur la plus populaire
print(f"La couleur la plus populaire est : {color_counts.index[0]} avec {color_counts.values[0]} achats.")


# Nombre d'achats par client
customer_purchase_counts = transactions['customer_id'].value_counts()

# Visualisation de la répartition des achats par client
plt.figure(figsize=(10,6))
sns.histplot(customer_purchase_counts, kde=True, color='blue')
plt.title("Répartition du nombre d'achats par client")
plt.xlabel('Nombre d\'achats')
plt.ylabel('Fréquence')
plt.show()

# Conclusion: Analyser les clients avec le plus grand nombre d'achats
top_customers = customer_purchase_counts.head(10)
print("Top 10 des clients avec le plus grand nombre d'achats:")
print(top_customers)


# Comptage des articles les plus populaires
article_counts = transactions['article_id'].value_counts()

# Visualisation des 10 articles les plus populaires
top_articles = article_counts.head(10)



plt.figure(figsize=(10,6))
sns.barplot(x=top_articles.index, y=top_articles.values, palette='plasma')
plt.title("Top 10 des articles les plus populaires")
plt.xlabel('Article ID')
plt.ylabel('Nombre d\'achats')
plt.show()

# Conclusion: Afficher les IDs des 10 articles les plus populaires
print("Les 10 articles les plus populaires sont :")
print(top_articles)


print(articles.columns)



# Conclusion: Afficher les noms des 10 articles les plus populaires
print("Les 10 articles les plus populaires sont :")
print(top_articles_names_sorted[['prod_name']])


# Convertir la date de transaction en format datetime
transactions['t_dat'] = pd.to_datetime(transactions['t_dat'])

# Trouver la date la plus récente dans les transactions
recent_date = transactions['t_dat'].max()

# Calculer le nombre de clients ayant effectué un achat après une certaine période
recent_transactions = transactions[transactions['t_dat'] == recent_date]
recent_customers = recent_transactions['customer_id'].nunique()

print(f"Nombre de clients ayant effectué des achats récemment ({recent_date}): {recent_customers}")


print("\nConclusion générale :")
print(f"- La catégorie la plus populaire est {category_counts.index[0]} avec {category_counts.values[0]} achats.")
print(f"- La couleur la plus populaire est {color_counts.index[0]} avec {color_counts.values[0]} achats.")
print(f"- Les 10 clients les plus actifs sont : {top_customers.index.tolist()}.")
print(f"- Les 10 articles les plus populaires sont : {top_articles.index.tolist()}.")
print(f"- Il y a {recent_customers} clients ayant effectué un achat récent le {recent_date}.")


# 1. Distribution des ventes (histogramme)
plt.figure(figsize=(10,6))
sns.histplot(article_counts.values, kde=True, color='teal')
plt.title("Distribution des ventes d'articles")
plt.xlabel("Nombre de ventes")
plt.ylabel("Fréquence")
plt.show()

