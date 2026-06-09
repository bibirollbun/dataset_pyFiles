import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # graph visualization
import matplotlib.image as mpimg # image visualizations 
from IPython.display import Image, display # image visualizations
import seaborn as sns # graph visualizations
from sklearn.metrics.pairwise import cosine_similarity 


articles = pd.read_csv('/kaggle/input/h-and-m-personalized-fashion-recommendations/articles.csv')
customers = pd.read_csv('/kaggle/input/h-and-m-personalized-fashion-recommendations/customers.csv')
sample_submission = pd.read_csv('/kaggle/input/h-and-m-personalized-fashion-recommendations/sample_submission.csv')
transactions_train = pd.read_csv('/kaggle/input/h-and-m-personalized-fashion-recommendations/transactions_train.csv')


articles.head() # Affiche les 5 premières lignes par défaut


customers.head()


sample_submission.head()


transactions_train.head()


articles.info() #Pour obtenir des informations générales


customers.info()


transactions_train.info()


articles.shape #dimensions of dataframe


customers.shape


transactions_train.shape


articles.isnull().sum() #count the number of missing values


customers.isnull().sum()


transactions_train.isnull().sum()


articles.nunique() #number of unique values for each column


customers.nunique()


def tranche_age(age):
    if age < 20:
        return 'Moins de 20 ans'
    elif 20 <= age < 30:
        return '20-29 ans'
    elif 30 <= age < 40:
        return '30-39 ans'
    elif 40 <= age < 50:
        return '40-49 ans'
    elif 50 <= age < 60:
        return '50-59 ans'
    else:
        return '60 ans et plus'

customers['tranche_age'] = customers['age'].apply(tranche_age)

# Compter le nombre d'achats par tranche d'âge
achats_par_tranche = customers['tranche_age'].value_counts()

# Afficher les résultats
print("Nombre d'achats par tranche d'âge :")
print(achats_par_tranche)

# Visualiser les résultats
achats_par_tranche.plot(kind='bar', color='blue', title='Nombre d\'achats par tranche d\'âge')
plt.xlabel('Tranche d\'âge')
plt.ylabel('Nombre d\'achats')
plt.show()


articles['product_type_name'].value_counts()[:20].plot(kind='barh')


articles['product_group_name'].value_counts()[:20].plot(kind='barh')


articles['section_name'].value_counts()[:20].plot(kind='barh')  


max_price_ids = transactions_train[transactions_train.t_dat==transactions_train.t_dat.max()].sort_values('price', ascending=False).iloc[:5][['article_id', 'price']]
min_price_ids = transactions_train[transactions_train.t_dat==transactions_train.t_dat.min()].sort_values('price', ascending=True).iloc[:5][['article_id', 'price']]

f, ax = plt.subplots(1, 5, figsize=(20,10))
i = 0
for _, data in max_price_ids.iterrows():
    desc = articles[articles['article_id'] == data['article_id']]['detail_desc'].iloc[0]
    desc_list = desc.split(' ')
    for j, elem in enumerate(desc_list):
        if j > 0 and j % 5 == 0:
            desc_list[j] = desc_list[j] + '\n'
    desc = ' '.join(desc_list)
    img = mpimg.imread(f'../input/h-and-m-personalized-fashion-recommendations/images/0{str(data.article_id)[:2]}/0{int(data.article_id)}.jpg')
    ax[i].imshow(img)
    ax[i].set_title(f'price: {data.price:.2f}')
    ax[i].set_xticks([], [])
    ax[i].set_yticks([], [])
    ax[i].grid(False)
    ax[i].set_xlabel(desc, fontsize=10)
    i += 1
plt.show()



f, ax = plt.subplots(1, 5, figsize=(20,10))
i = 0
for _, data in min_price_ids.iterrows():
    desc = articles[articles['article_id'] == data['article_id']]['detail_desc'].iloc[0]
    desc_list = desc.split(' ')
    for j, elem in enumerate(desc_list):
        if j > 0 and j % 4 == 0:
            desc_list[j] = desc_list[j] + '\n'
    desc = ' '.join(desc_list)
    img = mpimg.imread(f'../input/h-and-m-personalized-fashion-recommendations/images/0{str(data.article_id)[:2]}/0{int(data.article_id)}.jpg')
    ax[i].imshow(img)
    ax[i].set_title(f'price: {data.price:.4f}')
    ax[i].set_xlabel(desc, fontsize=10)
    ax[i].set_xticks([], [])
    ax[i].set_yticks([], [])
    ax[i].grid(False)
    i += 1
plt.axis('off')
plt.show()



# Chargement des données
data = transactions_train.head(10000)  
purchase_matrix = data.pivot_table(index='customer_id', columns='article_id', aggfunc='size', fill_value=0)  # Supposons que les colonnes soient 'customer_id' et 'product_id' 
similarity_matrix = cosine_similarity(purchase_matrix)  # Calcul de la similarité cosinus 
similarity_df = pd.DataFrame(similarity_matrix, index=purchase_matrix.index, columns=purchase_matrix.index)  # Création d'un DataFrame pour la similarité 

def recommend_products(customer_id, n_recommendations=5): 
    if customer_id not in similarity_df.index:
        return []  # Retourne une liste vide si le client n'existe pas
    
    similar_scores = similarity_df[customer_id]  # Obtenir les scores de similarité pour le client donné 
    similar_customers = similar_scores.sort_values(ascending=False).index[1:]  # Trier les clients par score de similarité 
    recommended_products = [] 
    
    for similar_customer in similar_customers: 
        products = purchase_matrix.loc[similar_customer][purchase_matrix.loc[similar_customer] > 0].index.tolist() 
        recommended_products.extend(products) 
        if len(recommended_products) >= n_recommendations: 
            break 
    
    return list(set(recommended_products))[:n_recommendations]  # Retourner les produits recommandés sans doublons 

customer_id_example = "001ea4e9c54f7e9c88811260d954edc059d596147e1cf8adc73323aebf571fd8"
recommended_items = recommend_products(customer_id_example) 

print(f"Produits recommandés pour le client n° {customer_id_example}: {recommended_items}")


img_1 = mpimg.imread('/kaggle/input/h-and-m-personalized-fashion-recommendations/images/055/0557248001.jpg') 
img_2 = mpimg.imread('/kaggle/input/h-and-m-personalized-fashion-recommendations/images/055/0557248009.jpg') 
img_3 = mpimg.imread('/kaggle/input/h-and-m-personalized-fashion-recommendations/images/053/0534181007.jpg') 
img_4 = mpimg.imread('/kaggle/input/h-and-m-personalized-fashion-recommendations/images/062/0624006001.jpg') 
img_5 = mpimg.imread('/kaggle/input/h-and-m-personalized-fashion-recommendations/images/038/0387843036.jpg') 
# Titres pour chaque image
titles = ['Id = 557248001', 'Id = 557248009', 'Id = 534181007', 'Id = 624006001', 'Id = 387843036']
# Création de la figure et des axes
fig, axs = plt.subplots(1, 5, figsize=(15, 5))

# Affichage des images
for ax, img, title in zip(axs, [img_1, img_2, img_3, img_4, img_5],titles):
    ax.imshow(img)
    ax.set_title(title)  # Ajout du titre à chaque image
    ax.axis('off')  # Masquer les axes

plt.tight_layout()  # Ajuste l'espacement
plt.show()  # Afficher les images


from sklearn.metrics import precision_score, recall_score, f1_score

# Simulated test dataset with customer purchase history (ground truth)
# Format: {customer_id: [list_of_actual_purchases]}
ground_truth = {
    "001ea4e9c54f7e9c88811260d954edc059d596147e1cf8adc73323aebf571fd8": ["P1", "P2", "P3"],
    "002f4e6c51d5f8c123456789abcdef9876543210fedcba0987654321fedcba00": ["P4", "P5"],
    "003abcde000123456789fedcba9876543210fedcba9876543210fedcba987654": ["P6", "P7", "P8", "P9"]
}

# Recommendations generated by the system
# Format: {customer_id: [list_of_recommended_products]}
recommendations = {
    "001ea4e9c54f7e9c88811260d954edc059d596147e1cf8adc73323aebf571fd8": ["P2", "P4", "P6"],
    "002f4e6c51d5f8c123456789abcdef9876543210fedcba0987654321fedcba00": ["P5", "P8"],
    "003abcde000123456789fedcba9876543210fedcba9876543210fedcba987654": ["P10", "P8", "P9"]
}

# Function to calculate precision, recall, and F1-score
def evaluate_recommendations(ground_truth, recommendations):
    all_precisions = []
    all_recalls = []
    all_f1s = []

    for customer_id, actual_purchases in ground_truth.items():
        recommended_products = recommendations.get(customer_id, [])

        # Convert to sets for easier comparison
        actual_set = set(actual_purchases)
        recommended_set = set(recommended_products)

        # Calculate precision, recall, and F1 for each customer
        true_positives = len(actual_set & recommended_set)  # Intersection of actual and recommended
        precision = true_positives / len(recommended_set) if recommended_set else 0
        recall = true_positives / len(actual_set) if actual_set else 0
        f1 = (2 * precision * recall) / (precision + recall) if precision + recall > 0 else 0

        all_precisions.append(precision)
        all_recalls.append(recall)
        all_f1s.append(f1)

    # Calculate averages across all customers
    avg_precision = sum(all_precisions) / len(all_precisions)
    avg_recall = sum(all_recalls) / len(all_recalls)
    avg_f1 = sum(all_f1s) / len(all_f1s)

    return avg_precision, avg_recall, avg_f1

# Evaluate the recommendations
avg_precision, avg_recall, avg_f1 = evaluate_recommendations(ground_truth, recommendations)

print(f"Average Precision: {avg_precision:.2f}")
print(f"Average Recall: {avg_recall:.2f}")
print(f"Average F1-Score: {avg_f1:.2f}")



import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Simulated test dataset with customer purchase history (ground truth)
ground_truth = {
    "001ea4e9c54f7e9c88811260d954edc059d596147e1cf8adc73323aebf571fd8": ["P1", "P2", "P3"],
    "002f4e6c51d5f8c123456789abcdef9876543210fedcba0987654321fedcba00": ["P4", "P5"],
    "003abcde000123456789fedcba9876543210fedcba9876543210fedcba987654": ["P6", "P7", "P8", "P9"]
}

# Recommendations generated by the system
recommendations = {
    "001ea4e9c54f7e9c88811260d954edc059d596147e1cf8adc73323aebf571fd8": ["P2", "P4", "P6"],
    "002f4e6c51d5f8c123456789abcdef9876543210fedcba0987654321fedcba00": ["P5", "P8"],
    "003abcde000123456789fedcba9876543210fedcba9876543210fedcba987654": ["P10", "P8", "P9"]
}

# Generate the binary vectors for confusion matrix
all_items = set()  # Collect all unique items across ground truth and recommendations
for purchases in ground_truth.values():
    all_items.update(purchases)
for recs in recommendations.values():
    all_items.update(recs)

all_items = sorted(all_items)  # Sorted list of all unique items
item_index = {item: idx for idx, item in enumerate(all_items)}

# Prepare y_true (actual purchases) and y_pred (recommended items)
y_true = np.zeros(len(all_items))
y_pred = np.zeros(len(all_items))

for customer_id, purchases in ground_truth.items():
    for item in purchases:
        y_true[item_index[item]] = 1  # Mark actual purchases as 1
    recommended_items = recommendations.get(customer_id, [])
    for item in recommended_items:
        y_pred[item_index[item]] = 1  # Mark recommended items as 1

# Compute confusion matrix
cm = confusion_matrix(y_true, y_pred)

# Plot confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Not Recommended", "Recommended"])
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix for Recommendation System")
plt.show()



# Function to calculate precision, recall, and F1-score for each customer
def evaluate_recommendations_per_customer(ground_truth, recommendations):
    results = {}

    for customer_id, actual_purchases in ground_truth.items():
        recommended_products = recommendations.get(customer_id, [])

        # Convert to sets for easier comparison
        actual_set = set(actual_purchases)
        recommended_set = set(recommended_products)

        # Calculate precision, recall, and F1 for each customer
        true_positives = len(actual_set & recommended_set)  # Intersection of actual and recommended
        precision = true_positives / len(recommended_set) if recommended_set else 0
        recall = true_positives / len(actual_set) if actual_set else 0
        f1 = (2 * precision * recall) / (precision + recall) if precision + recall > 0 else 0

        results[customer_id] = {
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        }

    return results

# Evaluate the recommendations per customer
customer_results = evaluate_recommendations_per_customer(ground_truth, recommendations)

# Display the results for each customer
for customer_id, metrics in customer_results.items():
    print(f"Customer ID: {customer_id}")
    print(f"  Precision: {metrics['precision']:.2f}")
    print(f"  Recall: {metrics['recall']:.2f}")
    print(f"  F1-Score: {metrics['f1_score']:.2f}")
    print()



import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Combine all actual and recommended products into a single list
all_items = set()
for purchases in ground_truth.values():
    all_items.update(purchases)
for recs in recommendations.values():
    all_items.update(recs)

all_items = sorted(all_items)  # Unique list of all products
item_index = {item: idx for idx, item in enumerate(all_items)}

# Prepare binary ground truth (y_true) and predictions (y_pred) for the confusion matrix
y_true = []
y_pred = []

for customer_id, actual_purchases in ground_truth.items():
    recommended_products = recommendations.get(customer_id, [])
    actual_set = set(actual_purchases)
    recommended_set = set(recommended_products)

    for item in all_items:
        y_true.append(1 if item in actual_set else 0)
        y_pred.append(1 if item in recommended_set else 0)

# Compute confusion matrix
cm = confusion_matrix(y_true, y_pred)

# Plot confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Not Purchased", "Purchased"])
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix for Recommendation System")
plt.show()


