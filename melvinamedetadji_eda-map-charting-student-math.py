%pip install wordcloud

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from collections import Counter
from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# TÃ©lÃ©charger les ressources NLTK nÃ©cessaires (si ce n'est pas dÃ©jÃ  fait)
# Ces tÃ©lÃ©chargements sont nÃ©cessaires pour la tokenisation et la suppression des mots vides.
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

# Chemins d'accÃ¨s aux donnÃ©es sur Kaggle
TRAIN_PATH = '/kaggle/input/map-charting-student-math-misunderstandings/train.csv'
TEST_PATH = '/kaggle/input/map-charting-student-math-misunderstandings/test.csv'
SAMPLE_SUBMISSION_PATH = '/kaggle/input/map-charting-student-math-misunderstandings/sample_submission.csv'

# Chargement des donnÃ©es
train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)
sample_submission_df = pd.read_csv(SAMPLE_SUBMISSION_PATH)

print(f"DonnÃ©es d'entraÃ®nement chargÃ©es. Forme : {train_df.shape}")
print(f"DonnÃ©es de test chargÃ©es. Forme : {test_df.shape}")
print(f"Exemple de soumission chargÃ©. Forme : {sample_submission_df.shape}")


print("Informations sur train_df:")
train_df.info()
print("Informations sur test_df:")
test_df.info()
print("Informations sur sample_submission_df:")
sample_submission_df.info()

print("PremiÃ¨res lignes de train_df:")
print(train_df.head())
print("PremiÃ¨res lignes de test_df:")
print(test_df.head())
print("PremiÃ¨res lignes de sample_submission_df:")
print(sample_submission_df.head())


print("Valeurs manquantes dans train_df:")
print(train_df.isnull().sum())
print("\nValeurs manquantes dans test_df:")
print(test_df.isnull().sum())


print("\nDistribution de 'Category':")
print(train_df['Category'].value_counts())
plt.figure(figsize=(8, 6))
sns.countplot(data=train_df, x='Category', palette='viridis')
plt.title('Distribution des CatÃ©gories')
plt.savefig('category_distribution.png')
plt.close()

print("\nDistribution de 'Misconception':")
print(train_df['Misconception'].value_counts())
plt.figure(figsize=(12, 6))
sns.countplot(data=train_df, y='Misconception', order=train_df['Misconception'].value_counts().index, palette='magma')
plt.title('Distribution des Misconceptions')
plt.tight_layout()
plt.savefig('misconception_distribution.png')
plt.close()

# Combinaison Category:Misconception
train_df['Category_Misconception'] = train_df['Category'] + ':' + train_df['Misconception'].fillna('NA')
print("\nDistribution de 'Category:Misconception':")
print(train_df['Category_Misconception'].value_counts().head(20))


# Longueur des explications des Ã©tudiants
train_df['explanation_length'] = train_df['StudentExplanation'].apply(
    lambda x: len(str(x).split()) if pd.notna(x) else 0
)
print("\nStatistiques de longueur des explications des Ã©tudiants (train_df):")
print(train_df['explanation_length'].describe())

plt.figure(figsize=(10, 6))
sns.histplot(train_df['explanation_length'], bins=50, kde=True)
plt.title('Distribution de la longueur des explications des Ã©tudiants')
plt.xlabel('Nombre de mots')
plt.ylabel('FrÃ©quence')
plt.savefig('explanation_length_distribution.png')
plt.show()  # Remplace plt.close() pour afficher le graphique

# Nuage de mots pour StudentExplanation
print("\nGÃ©nÃ©ration du nuage de mots pour StudentExplanation...")

# VÃ©rification que les modules nÃ©cessaires sont disponibles
try:
    all_explanations = ' '.join(train_df['StudentExplanation'].dropna().astype(str).tolist())
    
    # Nettoyage du texte pour le nuage de mots
    # Supprimer les caractÃ¨res spÃ©ciaux, chiffres et convertir en minuscules
    all_explanations = re.sub(r'[^a-zA-Z\s]', '', all_explanations).lower()
    
    # Tokenisation et suppression des stopwords
    words = word_tokenize(all_explanations)
    stop_words = set(stopwords.words('english'))  # Les explications sont en anglais
    filtered_words = [word for word in words if word.isalpha() and len(word) > 1 and word not in stop_words]
    
    if filtered_words:  # VÃ©rifier qu'il y a des mots Ã  afficher
        wordcloud = WordCloud(
            width=800, 
            height=400, 
            background_color='white',
            max_words=200
        ).generate(' '.join(filtered_words))
        
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title('Nuage de mots des explications des Ã©tudiants')
        plt.savefig('student_explanation_wordcloud.png')
        plt.show()  # Remplace plt.close() pour afficher le graphique
    else:
        print("Aucun mot valide trouvÃ© pour crÃ©er le nuage de mots")
        
except Exception as e:
    print(f"Erreur lors de la gÃ©nÃ©ration du nuage de mots: {e}")
    print("VÃ©rifiez que NLTK et WordCloud sont correctement installÃ©s")


# Analyse des questions (QuestionText)
print("Analyse des questions (QuestionText)...")
# Extraire les informations d'image si prÃ©sentes
train_df['has_image'] = train_df['QuestionText'].apply(lambda x: '[Image:' in str(x))
print(f"Questions avec image : {train_df['has_image'].sum()}")
print(f"Questions sans image : {len(train_df) - train_df['has_image'].sum()}")

# Nettoyage de QuestionText pour l'analyse de contenu
def clean_question_text(text):
    text = re.sub(r'\[Image:.*?\]', '', str(text)) # Supprimer la balise image
    text = re.sub(r'\\\(.*?\\\)', '', text) # Supprimer les expressions LaTeX
    text = re.sub(r'[^a-zA-Z\s]', '', text) # Supprimer les caractÃ¨res spÃ©ciaux
    return text.lower().strip()

train_df['CleanedQuestionText'] = train_df['QuestionText'].apply(clean_question_text)

# FrÃ©quence des mots dans les questions
all_questions_words = ' '.join(train_df['CleanedQuestionText'].dropna().tolist())
question_words = word_tokenize(all_questions_words)
filtered_question_words = [word for word in question_words if word.isalpha() and word not in stop_words]

question_word_freq = Counter(filtered_question_words)
print("Top 20 des mots les plus frÃ©quents dans QuestionText (aprÃ¨s nettoyage et stopwords):")
print(question_word_freq.most_common(20))

# Analyse de MC_Answer (rÃ©ponse correcte ou attendue)
print("Analyse de MC_Answer (rÃ©ponse correcte ou attendue)...")
print(train_df['MC_Answer'].value_counts().head(20))


# Relation entre explanation_length et Category
plt.figure(figsize=(10, 6))
sns.boxplot(data=train_df, x='Category', y='explanation_length', palette='coolwarm')
plt.title('Longueur des explications par CatÃ©gorie')
plt.xlabel('CatÃ©gorie')
plt.ylabel("Nombre de mots dans l'explication")
plt.show()

# Relation entre has_image et Category
plt.figure(figsize=(8, 6))
sns.countplot(data=train_df, x='has_image', hue='Category', palette='pastel')
plt.title("Distribution des CatÃ©gories selon la prÃ©sence d'image dans la question")
plt.xlabel('Question contient une image')
plt.ylabel("Nombre d'occurrences")
plt.show()


# 7.1 Extraction des nombres et opÃ©rations mathÃ©matiques dans les explications
def extract_mathematical_elements(text):
    """Extrait les Ã©lÃ©ments mathÃ©matiques des explications"""
    text = str(text).lower()
    
    # Extraction des nombres (entiers, dÃ©cimaux, fractions)
    numbers = re.findall(r'\b\d+(?:\.\d+)?\b', text)
    
    # Extraction des fractions (format a/b)
    fractions = re.findall(r'\b\d+/\d+\b', text)
    
    # Extraction des opÃ©rations mathÃ©matiques
    operations = []
    if 'add' in text or '+' in text or 'plus' in text or 'sum' in text:
        operations.append('addition')
    if 'subtract' in text or '-' in text or 'minus' in text or 'take away' in text:
        operations.append('subtraction')
    if 'multiply' in text or '*' in text or 'times' in text or 'x' in text:
        operations.append('multiplication')
    if 'divide' in text or '/' in text or 'divided by' in text:
        operations.append('division')
    
    # Mots-clÃ©s mathÃ©matiques
    math_keywords = []
    math_terms = ['fraction', 'decimal', 'percent', 'whole', 'part', 'equal', 
                  'greater', 'less', 'same', 'different', 'simplify', 'reduce']
    for term in math_terms:
        if term in text:
            math_keywords.append(term)
    
    return {
        'numbers': numbers,
        'fractions': fractions,
        'operations': operations,
        'math_keywords': math_keywords,
        'num_count': len(numbers),
        'fraction_count': len(fractions),
        'operation_count': len(operations),
        'math_keyword_count': len(math_keywords)
    }

# Application de l'extraction sur le dataset d'entraÃ®nement
print("Extraction des Ã©lÃ©ments mathÃ©matiques des explications...")
math_elements = train_df['StudentExplanation'].apply(extract_mathematical_elements)

# CrÃ©ation de nouvelles colonnes
train_df['numbers_mentioned'] = [elem['num_count'] for elem in math_elements]
train_df['fractions_mentioned'] = [elem['fraction_count'] for elem in math_elements]
train_df['operations_mentioned'] = [elem['operation_count'] for elem in math_elements]
train_df['math_keywords_count'] = [elem['math_keyword_count'] for elem in math_elements]

print("Statistiques des Ã©lÃ©ments mathÃ©matiques dans les explications:")
print(f"Moyenne de nombres mentionnÃ©s: {train_df['numbers_mentioned'].mean():.2f}")
print(f"Moyenne de fractions mentionnÃ©es: {train_df['fractions_mentioned'].mean():.2f}")
print(f"Moyenne d'opÃ©rations mentionnÃ©es: {train_df['operations_mentioned'].mean():.2f}")
print(f"Moyenne de mots-clÃ©s mathÃ©matiques: {train_df['math_keywords_count'].mean():.2f}")

# 7.2 Analyse des patterns par type de misconception
plt.figure(figsize=(15, 10))

# Sous-graphique 1: Nombres mentionnÃ©s par catÃ©gorie
plt.subplot(2, 2, 1)
sns.boxplot(data=train_df, x='Category', y='numbers_mentioned', palette='Set2')
plt.title('Nombres mentionnÃ©s par CatÃ©gorie')
plt.xticks(rotation=45)

# Sous-graphique 2: Fractions mentionnÃ©es par catÃ©gorie
plt.subplot(2, 2, 2)
sns.boxplot(data=train_df, x='Category', y='fractions_mentioned', palette='Set2')
plt.title('Fractions mentionnÃ©es par CatÃ©gorie')
plt.xticks(rotation=45)

# Sous-graphique 3: OpÃ©rations mentionnÃ©es par catÃ©gorie
plt.subplot(2, 2, 3)
sns.boxplot(data=train_df, x='Category', y='operations_mentioned', palette='Set2')
plt.title('OpÃ©rations mentionnÃ©es par CatÃ©gorie')
plt.xticks(rotation=45)

# Sous-graphique 4: Mots-clÃ©s mathÃ©matiques par catÃ©gorie
plt.subplot(2, 2, 4)
sns.boxplot(data=train_df, x='Category', y='math_keywords_count', palette='Set2')
plt.title('Mots-clÃ©s mathÃ©matiques par CatÃ©gorie')
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig('mathematical_patterns_by_category.png', dpi=300, bbox_inches='tight')
plt.show()

# 7.3 Analyse spÃ©cifique aux misconceptions les plus frÃ©quentes
top_misconceptions = train_df['Misconception'].value_counts().head(10).index.tolist()
misconception_data = train_df[train_df['Misconception'].isin(top_misconceptions)]

plt.figure(figsize=(12, 8))
sns.boxplot(data=misconception_data, x='Misconception', y='numbers_mentioned', palette='viridis')
plt.title('Nombres mentionnÃ©s par type de Misconception (Top 10)')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('numbers_by_misconception.png', dpi=300, bbox_inches='tight')
plt.show()

# 7.4 Analyse des patterns d'erreurs numÃ©riques
def analyze_numerical_consistency(row):
    """Analyse la cohÃ©rence numÃ©rique entre la rÃ©ponse correcte et l'explication"""
    correct_answer = str(row['MC_Answer'])
    explanation = str(row['StudentExplanation']).lower()
    
    # Extraction des nombres de la rÃ©ponse correcte
    correct_numbers = re.findall(r'\d+(?:\.\d+)?', correct_answer)
    
    # Extraction des nombres de l'explication
    explanation_numbers = re.findall(r'\d+(?:\.\d+)?', explanation)
    
    # VÃ©rification de la cohÃ©rence
    consistency_score = 0
    if correct_numbers and explanation_numbers:
        common_numbers = set(correct_numbers) & set(explanation_numbers)
        consistency_score = len(common_numbers) / len(set(correct_numbers))
    
    return {
        'correct_numbers': correct_numbers,
        'explanation_numbers': explanation_numbers,
        'numerical_consistency': consistency_score,
        'has_numerical_reference': len(explanation_numbers) > 0
    }

print("\nAnalyse de la cohÃ©rence numÃ©rique...")
numerical_analysis = train_df.apply(analyze_numerical_consistency, axis=1)
train_df['numerical_consistency'] = [analysis['numerical_consistency'] for analysis in numerical_analysis]
train_df['has_numerical_reference'] = [analysis['has_numerical_reference'] for analysis in numerical_analysis]

# Distribution de la cohÃ©rence numÃ©rique par catÃ©gorie
plt.figure(figsize=(10, 6))
sns.boxplot(data=train_df, x='Category', y='numerical_consistency', palette='coolwarm')
plt.title('CohÃ©rence numÃ©rique entre rÃ©ponse correcte et explication')
plt.xticks(rotation=45)
plt.ylabel('Score de cohÃ©rence (0-1)')
plt.tight_layout()
plt.savefig('numerical_consistency_by_category.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"Pourcentage d'explications avec rÃ©fÃ©rences numÃ©riques: {train_df['has_numerical_reference'].mean()*100:.1f}%")
print(f"Score moyen de cohÃ©rence numÃ©rique: {train_df['numerical_consistency'].mean():.3f}")

# 7.5 Identification des patterns d'erreurs communes
def identify_common_error_patterns(text, misconception):
    """Identifie les patterns d'erreurs communes selon le type de misconception"""
    text = str(text).lower()
    error_patterns = []
    
    if misconception == 'Additive':
        if 'add' in text and ('fraction' in text or '/' in text):
            error_patterns.append('additive_fraction_error')
    
    elif misconception == 'Subtraction':
        if any(word in text for word in ['take away', 'minus', 'subtract']):
            if 'wrong' in text or 'mistake' in text:
                error_patterns.append('subtraction_confusion')
    
    elif misconception == 'Inversion':
        if any(word in text for word in ['flip', 'reverse', 'upside', 'backwards']):
            error_patterns.append('inversion_attempt')
    
    elif misconception == 'Wrong_fraction':
        if 'fraction' in text and any(num in text for num in ['wrong', 'incorrect', 'mixed']):
            error_patterns.append('fraction_misconception')
    
    return error_patterns

# Application de l'analyse des patterns d'erreurs
print("\nIdentification des patterns d'erreurs communes...")
error_patterns_data = []
for idx, row in train_df.iterrows():
    if pd.notna(row['Misconception']):
        patterns = identify_common_error_patterns(row['StudentExplanation'], row['Misconception'])
        for pattern in patterns:
            error_patterns_data.append({
                'misconception': row['Misconception'],
                'error_pattern': pattern,
                'category': row['Category']
            })

if error_patterns_data:
    error_patterns_df = pd.DataFrame(error_patterns_data)
    print("Patterns d'erreurs identifiÃ©s:")
    print(error_patterns_df.groupby(['misconception', 'error_pattern']).size().sort_values(ascending=False))
else:
    print("Aucun pattern d'erreur spÃ©cifique identifiÃ© avec cette mÃ©thode.")

print("\n=== RÃ©sumÃ© de l'analyse des patterns mathÃ©matiques ===")
print(f"â€¢ {train_df['has_numerical_reference'].sum()} explications contiennent des rÃ©fÃ©rences numÃ©riques")
print(f"â€¢ Score moyen de cohÃ©rence numÃ©rique: {train_df['numerical_consistency'].mean():.3f}")
print(f"â€¢ Moyenne d'Ã©lÃ©ments mathÃ©matiques par explication: {(train_df['numbers_mentioned'] + train_df['fractions_mentioned'] + train_df['operations_mentioned']).mean():.2f}")


# 8.1 PrÃ©paration des donnÃ©es pour l'analyse des corrÃ©lations
print("Analyse des corrÃ©lations entre misconceptions...")

# Filtrer les donnÃ©es avec misconceptions (exclure les NaN)
misconception_data = train_df[train_df['Misconception'].notna()].copy()
print(f"Nombre d'Ã©chantillons avec misconceptions: {len(misconception_data)}")

# 8.2 Analyse de la co-occurrence par QuestionId
print("\nAnalyse de la co-occurrence des misconceptions par question...")

# Grouper par QuestionId pour voir quelles misconceptions apparaissent ensemble
question_misconceptions = misconception_data.groupby('QuestionId')['Misconception'].apply(list).reset_index()
question_misconceptions['unique_misconceptions'] = question_misconceptions['Misconception'].apply(lambda x: list(set(x)))
question_misconceptions['misconception_count'] = question_misconceptions['unique_misconceptions'].apply(len)

print("Distribution du nombre de misconceptions diffÃ©rentes par question:")
print(question_misconceptions['misconception_count'].value_counts().sort_index())

# Questions avec multiple misconceptions
multi_misconception_questions = question_misconceptions[question_misconceptions['misconception_count'] > 1]
print(f"\nNombre de questions avec plusieurs types de misconceptions: {len(multi_misconception_questions)}")

# 8.3 Matrice de co-occurrence des misconceptions
from itertools import combinations
import numpy as np

# CrÃ©er une matrice de co-occurrence
unique_misconceptions = sorted(misconception_data['Misconception'].unique())
n_misconceptions = len(unique_misconceptions)
cooccurrence_matrix = np.zeros((n_misconceptions, n_misconceptions))

# Remplir la matrice de co-occurrence
misconception_to_idx = {misc: idx for idx, misc in enumerate(unique_misconceptions)}

for misconceptions_list in multi_misconception_questions['unique_misconceptions']:
    if len(misconceptions_list) > 1:
        # Pour chaque paire de misconceptions dans la mÃªme question
        for misc1, misc2 in combinations(misconceptions_list, 2):
            idx1, idx2 = misconception_to_idx[misc1], misconception_to_idx[misc2]
            cooccurrence_matrix[idx1][idx2] += 1
            cooccurrence_matrix[idx2][idx1] += 1  # Matrice symÃ©trique

# Conversion en DataFrame pour faciliter l'analyse
cooccurrence_df = pd.DataFrame(cooccurrence_matrix, 
                              index=unique_misconceptions, 
                              columns=unique_misconceptions)

print("\nTop 10 des paires de misconceptions qui co-occurent le plus:")
# Extraire les valeurs de co-occurrence (partie supÃ©rieure de la matrice)
cooccurrence_pairs = []
for i in range(len(unique_misconceptions)):
    for j in range(i+1, len(unique_misconceptions)):
        if cooccurrence_matrix[i][j] > 0:
            cooccurrence_pairs.append({
                'misconception_1': unique_misconceptions[i],
                'misconception_2': unique_misconceptions[j],
                'cooccurrence_count': int(cooccurrence_matrix[i][j])
            })

cooccurrence_pairs_df = pd.DataFrame(cooccurrence_pairs)
if not cooccurrence_pairs_df.empty:
    top_cooccurrences = cooccurrence_pairs_df.sort_values('cooccurrence_count', ascending=False).head(10)
    print(top_cooccurrences)
else:
    print("Aucune co-occurrence significative trouvÃ©e.")

# 8.4 Visualisation de la matrice de corrÃ©lation des misconceptions
# Calculer la corrÃ©lation basÃ©e sur les caractÃ©ristiques textuelles
print("\nCalcul des corrÃ©lations basÃ©es sur les caractÃ©ristiques textuelles...")

# CrÃ©er des features pour chaque misconception
misconception_features = []
for misconception in unique_misconceptions:
    misc_data = misconception_data[misconception_data['Misconception'] == misconception]
    
    features = {
        'misconception': misconception,
        'avg_explanation_length': misc_data['explanation_length'].mean(),
        'avg_numbers_mentioned': misc_data['numbers_mentioned'].mean(),
        'avg_fractions_mentioned': misc_data['fractions_mentioned'].mean(),
        'avg_operations_mentioned': misc_data['operations_mentioned'].mean(),
        'avg_math_keywords': misc_data['math_keywords_count'].mean(),
        'avg_numerical_consistency': misc_data['numerical_consistency'].mean(),
        'sample_count': len(misc_data)
    }
    misconception_features.append(features)

misconception_features_df = pd.DataFrame(misconception_features)
misconception_features_df = misconception_features_df.set_index('misconception')

# Calculer la matrice de corrÃ©lation
correlation_matrix = misconception_features_df.drop('sample_count', axis=1).T.corr()

# Visualisation de la matrice de corrÃ©lation (top 15 misconceptions pour la lisibilitÃ©)
top_misconceptions = misconception_data['Misconception'].value_counts().head(15).index
correlation_subset = correlation_matrix.loc[top_misconceptions, top_misconceptions]

plt.figure(figsize=(12, 10))
sns.heatmap(correlation_subset, 
            annot=False, 
            cmap='coolwarm', 
            center=0,
            square=True,
            cbar_kws={'label': 'CorrÃ©lation'})
plt.title('Matrice de CorrÃ©lation entre Misconceptions\n(basÃ©e sur les caractÃ©ristiques textuelles)')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('misconception_correlation_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

# 8.5 Clustering des misconceptions similaires
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist

# Clustering hiÃ©rarchique des misconceptions
print("\nClustering hiÃ©rarchique des misconceptions...")

# Utiliser les features pour le clustering (top 20 misconceptions)
top_20_misconceptions = misconception_data['Misconception'].value_counts().head(20).index
features_for_clustering = misconception_features_df.loc[top_20_misconceptions].drop('sample_count', axis=1)

# Normalisation des features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
features_normalized = scaler.fit_transform(features_for_clustering)

# Clustering hiÃ©rarchique
linkage_matrix = linkage(features_normalized, method='ward')

plt.figure(figsize=(12, 8))
dendrogram(linkage_matrix, 
           labels=top_20_misconceptions.tolist(),
           leaf_rotation=45,
           leaf_font_size=10)
plt.title('Dendrogramme - Clustering des Misconceptions\n(basÃ© sur les caractÃ©ristiques textuelles)')
plt.xlabel('Misconceptions')
plt.ylabel('Distance')
plt.tight_layout()
plt.savefig('misconception_clustering_dendrogram.png', dpi=300, bbox_inches='tight')
plt.show()

# 8.6 Analyse des transitions entre catÃ©gories et misconceptions
print("\nAnalyse des patterns de transition Category â†’ Misconception...")

# CrÃ©er une matrice de transition
categories = ['True_Correct', 'False_Misconception', 'False_Neither', 'True_Neither', 
              'True_Misconception', 'False_Correct']

transition_analysis = []
for category in categories:
    cat_data = train_df[train_df['Category'] == category]
    misconception_dist = cat_data['Misconception'].value_counts(normalize=True)
    
    transition_analysis.append({
        'category': category,
        'total_samples': len(cat_data),
        'misconception_diversity': len(misconception_dist),
        'top_misconception': misconception_dist.index[0] if len(misconception_dist) > 0 else 'None',
        'top_misconception_prob': misconception_dist.iloc[0] if len(misconception_dist) > 0 else 0
    })

transition_df = pd.DataFrame(transition_analysis)
print("\nAnalyse des transitions Category â†’ Misconception:")
print(transition_df)

# 8.7 Identification des groupes de misconceptions
print("\n=== Groupes de Misconceptions IdentifiÃ©s ===")

# Groupe 1: Misconceptions liÃ©es aux fractions
fraction_related = ['Wrong_fraction', 'Wrong_Fraction', 'Additive', 'Denominator-only_change', 
                   'Incorrect_equivalent_fraction_addition']
fraction_misconceptions = [m for m in fraction_related if m in unique_misconceptions]
print(f"Groupe Fractions: {fraction_misconceptions}")

# Groupe 2: Misconceptions liÃ©es aux opÃ©rations
operation_related = ['Subtraction', 'Mult', 'Division', 'Adding_across', 'Inverse_operation']
operation_misconceptions = [m for m in operation_related if m in unique_misconceptions]
print(f"Groupe OpÃ©rations: {operation_misconceptions}")

# Groupe 3: Misconceptions liÃ©es Ã  la comprÃ©hension des nombres
number_related = ['Whole_numbers_larger', 'Positive', 'Longer_is_bigger', 'Shorter_is_bigger']
number_misconceptions = [m for m in number_related if m in unique_misconceptions]
print(f"Groupe Nombres: {number_misconceptions}")

print(f"\n=== RÃ©sumÃ© de l'analyse des corrÃ©lations ===")
print(f"â€¢ {len(unique_misconceptions)} types de misconceptions uniques")
print(f"â€¢ {len(multi_misconception_questions)} questions avec misconceptions multiples")
print(f"â€¢ {len(cooccurrence_pairs_df) if not cooccurrence_pairs_df.empty else 0} paires de co-occurrence identifiÃ©es")
print(f"â€¢ 3 groupes thÃ©matiques principaux identifiÃ©s")


# 9.1 Installation et importation des outils de vÃ©rification orthographique
try:
    from textblob import TextBlob
except ImportError:
    print("Installation de TextBlob en cours...")
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "textblob"])
    from textblob import TextBlob

import string
from collections import defaultdict

print("Analyse de la qualitÃ© linguistique des explications...")

# 9.2 Fonctions d'analyse linguistique
def analyze_spelling_grammar(text):
    """Analyse les erreurs d'orthographe et de grammaire dans un texte"""
    if pd.isna(text) or str(text).strip() == '':
        return {
            'spelling_errors': 0,
            'spelling_error_rate': 0,
            'corrected_text': '',
            'word_count': 0,
            'unique_words': 0,
            'avg_word_length': 0,
            'punctuation_count': 0,
            'uppercase_words': 0,
            'lowercase_ratio': 0
        }
    
    text = str(text)
    
    # Analyse avec TextBlob
    blob = TextBlob(text)
    corrected_text = str(blob.correct())
    
    # Compter les erreurs d'orthographe
    original_words = text.lower().split()
    corrected_words = corrected_text.lower().split()
    
    spelling_errors = sum(1 for orig, corr in zip(original_words, corrected_words) if orig != corr)
    
    # Statistiques textuelles
    words = text.split()
    word_count = len(words)
    unique_words = len(set(word.lower().strip(string.punctuation) for word in words))
    avg_word_length = np.mean([len(word.strip(string.punctuation)) for word in words]) if words else 0
    
    # Analyse de la ponctuation
    punctuation_count = sum(1 for char in text if char in string.punctuation)
    
    # Analyse de la casse
    uppercase_words = sum(1 for word in words if word.isupper() and len(word) > 1)
    lowercase_ratio = sum(1 for char in text if char.islower()) / len(text) if len(text) > 0 else 0
    
    return {
        'spelling_errors': spelling_errors,
        'spelling_error_rate': spelling_errors / word_count if word_count > 0 else 0,
        'corrected_text': corrected_text,
        'word_count': word_count,
        'unique_words': unique_words,
        'avg_word_length': avg_word_length,
        'punctuation_count': punctuation_count,
        'uppercase_words': uppercase_words,
        'lowercase_ratio': lowercase_ratio
    }

def detect_common_student_errors(text):
    """DÃ©tecte les erreurs communes des Ã©tudiants"""
    text = str(text).lower()
    common_errors = {
        'number_word_confusion': 0,  # "to" vs "two", "for" vs "four"
        'math_term_misspelling': 0,  # "fraction" mal orthographiÃ©
        'informal_language': 0,      # "gonna", "wanna", etc.
        'repetitive_words': 0,       # Mots rÃ©pÃ©tÃ©s
        'incomplete_sentences': 0    # Phrases incomplÃ¨tes
    }
    
    # DÃ©tection de confusions nombre/mot
    number_confusions = [('to', 'two'), ('for', 'four'), ('ate', 'eight'), ('won', 'one')]
    for wrong, correct in number_confusions:
        if wrong in text and correct not in text:
            common_errors['number_word_confusion'] += text.count(wrong)
    
    # DÃ©tection d'erreurs sur les termes mathÃ©matiques
    math_terms_errors = ['fracion', 'fractoin', 'divison', 'multipication', 'addtion']
    for error_term in math_terms_errors:
        common_errors['math_term_misspelling'] += text.count(error_term)
    
    # DÃ©tection de langage informel
    informal_words = ['gonna', 'wanna', 'gotta', 'dunno', 'yeah', 'nah', 'kinda', 'sorta']
    for informal in informal_words:
        common_errors['informal_language'] += text.count(informal)
    
    # DÃ©tection de mots rÃ©pÃ©tÃ©s consÃ©cutivement
    words = text.split()
    for i in range(len(words) - 1):
        if words[i] == words[i + 1] and len(words[i]) > 2:
            common_errors['repetitive_words'] += 1
    
    # DÃ©tection de phrases incomplÃ¨tes (heuristique simple)
    if len(text.strip()) > 0 and not text.strip().endswith(('.', '!', '?')):
        common_errors['incomplete_sentences'] = 1
    
    return common_errors

# 9.3 Application de l'analyse linguistique
print("Application de l'analyse linguistique sur les explications...")

# Analyse sur un Ã©chantillon pour Ã©viter les temps de traitement trop longs
sample_size = min(5000, len(train_df))
sample_indices = np.random.choice(train_df.index, sample_size, replace=False)
sample_df = train_df.loc[sample_indices].copy()

print(f"Analyse sur un Ã©chantillon de {sample_size} explications...")

# Application de l'analyse linguistique
linguistic_analysis = sample_df['StudentExplanation'].apply(analyze_spelling_grammar)
student_errors = sample_df['StudentExplanation'].apply(detect_common_student_errors)

# Extraction des rÃ©sultats
for key in linguistic_analysis.iloc[0].keys():
    sample_df[f'ling_{key}'] = [analysis[key] for analysis in linguistic_analysis]

for key in student_errors.iloc[0].keys():
    sample_df[f'error_{key}'] = [errors[key] for errors in student_errors]

# 9.4 Analyse descriptive des erreurs linguistiques
print("\n=== Statistiques des erreurs linguistiques ===")
print(f"Taux moyen d'erreurs d'orthographe: {sample_df['ling_spelling_error_rate'].mean():.3f}")
print(f"Nombre moyen de mots par explication: {sample_df['ling_word_count'].mean():.1f}")
print(f"Longueur moyenne des mots: {sample_df['ling_avg_word_length'].mean():.1f} caractÃ¨res")
print(f"Ratio moyen de minuscules: {sample_df['ling_lowercase_ratio'].mean():.3f}")

print(f"\n=== Erreurs communes des Ã©tudiants ===")
print(f"Confusions nombre/mot: {sample_df['error_number_word_confusion'].sum()} occurrences")
print(f"Erreurs termes mathÃ©matiques: {sample_df['error_math_term_misspelling'].sum()} occurrences")
print(f"Langage informel: {sample_df['error_informal_language'].sum()} occurrences")
print(f"Mots rÃ©pÃ©titifs: {sample_df['error_repetitive_words'].sum()} occurrences")
print(f"Phrases incomplÃ¨tes: {sample_df['error_incomplete_sentences'].sum()} explications")

# 9.5 Visualisations des erreurs linguistiques par catÃ©gorie
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Taux d'erreurs d'orthographe par catÃ©gorie
axes[0, 0].set_title('Taux d\'erreurs d\'orthographe par CatÃ©gorie')
sns.boxplot(data=sample_df, x='Category', y='ling_spelling_error_rate', ax=axes[0, 0], palette='Set2')
axes[0, 0].tick_params(axis='x', rotation=45)

# Longueur moyenne des mots par catÃ©gorie
axes[0, 1].set_title('Longueur moyenne des mots par CatÃ©gorie')
sns.boxplot(data=sample_df, x='Category', y='ling_avg_word_length', ax=axes[0, 1], palette='Set2')
axes[0, 1].tick_params(axis='x', rotation=45)

# DiversitÃ© lexicale (unique_words/word_count) par catÃ©gorie
sample_df['lexical_diversity'] = sample_df['ling_unique_words'] / sample_df['ling_word_count'].replace(0, 1)
axes[1, 0].set_title('DiversitÃ© lexicale par CatÃ©gorie')
sns.boxplot(data=sample_df, x='Category', y='lexical_diversity', ax=axes[1, 0], palette='Set2')
axes[1, 0].tick_params(axis='x', rotation=45)

# Utilisation de la ponctuation par catÃ©gorie
sample_df['punctuation_rate'] = sample_df['ling_punctuation_count'] / sample_df['ling_word_count'].replace(0, 1)
axes[1, 1].set_title('Utilisation de la ponctuation par CatÃ©gorie')
sns.boxplot(data=sample_df, x='Category', y='punctuation_rate', ax=axes[1, 1], palette='Set2')
axes[1, 1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('linguistic_analysis_by_category.png', dpi=300, bbox_inches='tight')
plt.show()

# 9.6 Analyse des erreurs par type de misconception
misconception_sample = sample_df[sample_df['Misconception'].notna()]
if len(misconception_sample) > 0:
    top_misconceptions_sample = misconception_sample['Misconception'].value_counts().head(8).index
    misconception_subset = misconception_sample[misconception_sample['Misconception'].isin(top_misconceptions_sample)]
    
    plt.figure(figsize=(12, 8))
    sns.boxplot(data=misconception_subset, x='Misconception', y='ling_spelling_error_rate', palette='viridis')
    plt.title('Taux d\'erreurs d\'orthographe par type de Misconception')
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Taux d\'erreurs d\'orthographe')
    plt.tight_layout()
    plt.savefig('spelling_errors_by_misconception.png', dpi=300, bbox_inches='tight')
    plt.show()

# 9.7 CorrÃ©lations entre qualitÃ© linguistique et performance mathÃ©matique
print("\n=== CorrÃ©lations linguistique vs mathÃ©matique ===")

# Matrice de corrÃ©lation
linguistic_features = ['ling_spelling_error_rate', 'ling_avg_word_length', 'lexical_diversity', 
                      'ling_lowercase_ratio', 'punctuation_rate']
math_features = ['numbers_mentioned', 'fractions_mentioned', 'operations_mentioned', 
                'numerical_consistency']

# Calculer les corrÃ©lations pour l'Ã©chantillon
correlation_data = sample_df[linguistic_features + math_features].corr()
linguistic_math_corr = correlation_data.loc[linguistic_features, math_features]

plt.figure(figsize=(10, 6))
sns.heatmap(linguistic_math_corr, annot=True, cmap='coolwarm', center=0, 
            fmt='.3f', cbar_kws={'label': 'CorrÃ©lation'})
plt.title('CorrÃ©lations entre QualitÃ© Linguistique et Ã‰lÃ©ments MathÃ©matiques')
plt.tight_layout()
plt.savefig('linguistic_math_correlations.png', dpi=300, bbox_inches='tight')
plt.show()

# 9.8 Identification des patterns linguistiques caractÃ©ristiques
print("\n=== Patterns linguistiques par catÃ©gorie ===")

category_linguistic_summary = sample_df.groupby('Category')[linguistic_features].mean()
print(category_linguistic_summary.round(3))

# 9.9 Exemples d'erreurs typiques
print("\n=== Exemples d'erreurs d'orthographe dÃ©tectÃ©es ===")
spelling_errors_examples = []

for idx, row in sample_df.head(50).iterrows():
    if row['ling_spelling_errors'] > 0:
        original = str(row['StudentExplanation'])[:100] + "..."
        corrected = row['ling_corrected_text'][:100] + "..."
        spelling_errors_examples.append({
            'category': row['Category'],
            'misconception': row['Misconception'],
            'original': original,
            'corrected': corrected,
            'error_count': row['ling_spelling_errors']
        })

if spelling_errors_examples:
    print("Premiers exemples d'erreurs dÃ©tectÃ©es:")
    for i, example in enumerate(spelling_errors_examples[:3]):
        print(f"\nExemple {i+1} ({example['category']}):")
        print(f"Original: {example['original']}")
        print(f"CorrigÃ©: {example['corrected']}")
        print(f"Erreurs: {example['error_count']}")

print(f"\n=== RÃ©sumÃ© de l'analyse linguistique ===")
print(f"â€¢ Ã‰chantillon analysÃ©: {sample_size} explications")
print(f"â€¢ Taux moyen d'erreurs d'orthographe: {sample_df['ling_spelling_error_rate'].mean():.1%}")
print(f"â€¢ {len(spelling_errors_examples)} exemples avec erreurs d'orthographe identifiÃ©s")
print(f"â€¢ DiversitÃ© lexicale moyenne: {sample_df['lexical_diversity'].mean():.3f}")


# 10.1 Fonctions d'analyse numÃ©rique avancÃ©e
def parse_mathematical_expression(expr):
    """Parse une expression mathÃ©matique et extrait ses composants"""
    expr = str(expr).strip()

    result = {
        'type': 'unknown',
        'numerator': None,
        'denominator': None,
        'whole_part': None,
        'decimal_value': None,
        'is_negative': False,
        'contains_latex': False,
        'operations': []
    }

    # DÃ©tecter LaTeX
    if '\\(' in expr and '\\)' in expr:
        result['contains_latex'] = True
        # Nettoyer le LaTeX
        expr = re.sub(r'\\[()]', '', expr)
        expr = re.sub(r'\\frac{([^}]+)}{([^}]+)}', r'\1/\2', expr)

    # DÃ©tecter le signe nÃ©gatif
    if expr.startswith('-'):
        result['is_negative'] = True
        expr = expr[1:]

    # DÃ©tecter les opÃ©rations
    operations = []
    if '+' in expr: operations.append('addition')
    if '-' in expr: operations.append('subtraction')
    if 'Ã—' in expr or '*' in expr or '\\times' in expr: operations.append('multiplication')
    if 'Ã·' in expr or '/' in expr or '\\div' in expr: operations.append('division')
    result['operations'] = operations

    # Parser selon le type
    try:
        # Fraction mixte (ex: 3 1/3)
        mixed_match = re.match(r'(\d+)\s+(\d+)/(\d+)', expr)
        if mixed_match:
            result['type'] = 'mixed_fraction'
            result['whole_part'] = int(mixed_match.group(1))
            result['numerator'] = int(mixed_match.group(2))
            result['denominator'] = int(mixed_match.group(3))
            result['decimal_value'] = result['whole_part'] + result['numerator'] / result['denominator']

        # Fraction simple (ex: 3/4)
        elif '/' in expr and len(operations) <= 1:
            parts = expr.split('/')
            if len(parts) == 2 and parts[0].strip().isdigit() and parts[1].strip().isdigit():
                result['type'] = 'fraction'
                result['numerator'] = int(parts[0].strip())
                result['denominator'] = int(parts[1].strip())
                if result['denominator'] != 0:
                    result['decimal_value'] = result['numerator'] / result['denominator']

        # Nombre dÃ©cimal
        elif '.' in expr and len(operations) == 0:
            result['type'] = 'decimal'
            result['decimal_value'] = float(expr)

        # Nombre entier
        elif expr.isdigit():
            result['type'] = 'integer'
            result['decimal_value'] = int(expr)

        # Expression complexe
        elif operations:
            result['type'] = 'expression'
            # Tenter d'Ã©valuer l'expression simple
            try:
                # Remplacer les symboles mathÃ©matiques
                safe_expr = expr.replace('Ã—', '*').replace('Ã·', '/')
                # Ã‰valuation sÃ©curisÃ©e pour expressions simples
                if re.match(r'^[\d\s+\-*/().]+$', safe_expr):
                    result['decimal_value'] = eval(safe_expr)
            except:
                pass

    except Exception:
        pass

    # Appliquer le signe nÃ©gatif
    if result['is_negative'] and result['decimal_value'] is not None:
        result['decimal_value'] *= -1

    return result

def analyze_numerical_relationship(correct_answer, student_explanation):
    """Analyse la relation numÃ©rique entre rÃ©ponse correcte et explication"""
    correct_parsed = parse_mathematical_expression(correct_answer)

    # Extraire tous les nombres de l'explication
    explanation = str(student_explanation).lower()
    numbers_in_explanation = re.findall(r'-?\d+(?:\.\d+)?', explanation)
    fractions_in_explanation = re.findall(r'\d+/\d+', explanation)

    analysis = {
        'correct_type': correct_parsed['type'],
        'correct_value': correct_parsed['decimal_value'],
        'numbers_in_explanation': [float(n) for n in numbers_in_explanation],
        'fractions_in_explanation': fractions_in_explanation,
        'mentions_correct_numerator': False,
        'mentions_correct_denominator': False,
        'mentions_correct_value': False,
        'numerical_proximity_score': 0,
        'common_error_patterns': []
    }

    if correct_parsed['decimal_value'] is not None:
        # VÃ©rifier si la valeur correcte est mentionnÃ©e
        correct_val = correct_parsed['decimal_value']
        analysis['mentions_correct_value'] = any(abs(float(n) - correct_val) < 0.001 for n in numbers_in_explanation)

        # Calculer un score de proximitÃ© numÃ©rique
        if numbers_in_explanation:
            distances = [abs(float(n) - correct_val) for n in numbers_in_explanation]
            analysis['numerical_proximity_score'] = 1 / (1 + min(distances))

    # VÃ©rifier les composants de fraction
    if correct_parsed['type'] in ['fraction', 'mixed_fraction'] and correct_parsed['numerator'] is not None:
        analysis['mentions_correct_numerator'] = str(correct_parsed['numerator']) in explanation
        analysis['mentions_correct_denominator'] = str(correct_parsed['denominator']) in explanation

    # Identifier des patterns d'erreurs communes
    error_patterns = []

    # Erreur d'inversion (ex: 3/4 devient 4/3)
    if correct_parsed['type'] == 'fraction':
        inverted_fraction = f"{correct_parsed['denominator']}/{correct_parsed['numerator']}"
        if inverted_fraction in explanation:
            error_patterns.append('fraction_inversion')

    # Erreur additive sur fractions (additionner numÃ©rateurs et dÃ©nominateurs)
    if correct_parsed['type'] == 'fraction' and len(fractions_in_explanation) > 0:
        for frac in fractions_in_explanation:
            parts = frac.split('/')
            if len(parts) == 2:
                num, den = int(parts[0]), int(parts[1])
                # VÃ©rifier si c'est une addition incorrecte de fractions
                if num == correct_parsed['numerator'] + correct_parsed['denominator']:
                    error_patterns.append('additive_fraction_error')

    # Erreur de simplification
    if correct_parsed['type'] == 'fraction' and numbers_in_explanation:
        correct_decimal = correct_parsed['decimal_value']
        for num_str in numbers_in_explanation:
            num_val = float(num_str)
            # VÃ©rifier les multiples courants (erreur de non-simplification)
            if abs(num_val - correct_decimal * 2) < 0.001 or abs(num_val - correct_decimal * 3) < 0.001:
                error_patterns.append('non_simplified_fraction')

    analysis['common_error_patterns'] = error_patterns
    return analysis

print("Analyse des patterns numÃ©riques dans les rÃ©ponses...")

# 10.2 Application de l'analyse numÃ©rique
print("Parsing des rÃ©ponses correctes...")
correct_answer_analysis = train_df['MC_Answer'].apply(parse_mathematical_expression)

# Extraction des informations
train_df['answer_type'] = [analysis['type'] for analysis in correct_answer_analysis]
train_df['answer_decimal_value'] = [analysis['decimal_value'] for analysis in correct_answer_analysis]
train_df['answer_contains_latex'] = [analysis['contains_latex'] for analysis in correct_answer_analysis]
train_df['answer_is_negative'] = [analysis['is_negative'] for analysis in correct_answer_analysis]

print("Analyse des relations numÃ©riques...")
# Analyse sur un Ã©chantillon pour la performance
sample_size = min(3000, len(train_df))
sample_indices = np.random.choice(train_df.index, sample_size, replace=False)
numerical_sample = train_df.loc[sample_indices].copy()

numerical_relationships = []
for idx, row in numerical_sample.iterrows():
    relationship = analyze_numerical_relationship(row['MC_Answer'], row['StudentExplanation'])
    numerical_relationships.append(relationship)

# Extraction des rÃ©sultats
numerical_sample['num_mentions_correct_value'] = [rel['mentions_correct_value'] for rel in numerical_relationships]
numerical_sample['num_proximity_score'] = [rel['numerical_proximity_score'] for rel in numerical_relationships]
numerical_sample['num_error_patterns'] = [rel['common_error_patterns'] for rel in numerical_relationships]
numerical_sample['num_error_count'] = [len(rel['common_error_patterns']) for rel in numerical_relationships]

# 10.3 Analyse descriptive des types de rÃ©ponses
print("\n=== Distribution des types de rÃ©ponses correctes ===")
answer_type_dist = train_df['answer_type'].value_counts()
print(answer_type_dist)

plt.figure(figsize=(10, 6))
sns.countplot(data=train_df, x='answer_type', palette='Set3')
plt.title('Distribution des Types de RÃ©ponses Correctes')
plt.xticks(rotation=45)
plt.ylabel('Nombre de questions')
plt.tight_layout()
plt.savefig('answer_types_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

# 10.4 Analyse des valeurs numÃ©riques
print(f"\n=== Statistiques des valeurs numÃ©riques ===")
numeric_answers = train_df[train_df['answer_decimal_value'].notna()]
print(f"RÃ©ponses avec valeur numÃ©rique: {len(numeric_answers)}")
print(f"Valeur minimale: {numeric_answers['answer_decimal_value'].min()}")
print(f"Valeur maximale: {numeric_answers['answer_decimal_value'].max()}")
print(f"Valeur moyenne: {numeric_answers['answer_decimal_value'].mean():.3f}")
print(f"MÃ©diane: {numeric_answers['answer_decimal_value'].median():.3f}")

# Distribution des valeurs numÃ©riques
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
# Histogramme des valeurs (limitÃ© pour la lisibilitÃ©)
values_subset = numeric_answers[numeric_answers['answer_decimal_value'].between(-10, 10)]
plt.hist(values_subset['answer_decimal_value'], bins=50, alpha=0.7, color='skyblue')
plt.title('Distribution des Valeurs NumÃ©riques (-10 Ã  10)')
plt.xlabel('Valeur')
plt.ylabel('FrÃ©quence')

plt.subplot(1, 2, 2)
# Analyse des fractions
fractions_data = train_df[train_df['answer_type'] == 'fraction']
if len(fractions_data) > 0:
    fraction_values = fractions_data['answer_decimal_value'].dropna()
    plt.hist(fraction_values, bins=30, alpha=0.7, color='lightcoral')
    plt.title('Distribution des Valeurs de Fractions')
    plt.xlabel('Valeur dÃ©cimale')
    plt.ylabel('FrÃ©quence')

plt.tight_layout()
plt.savefig('numerical_values_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

# 10.5 Analyse par catÃ©gorie et type de rÃ©ponse
print("\n=== Analyse des types de rÃ©ponses par catÃ©gorie ===")
answer_category_crosstab = pd.crosstab(train_df['Category'], train_df['answer_type'])
print(answer_category_crosstab)

plt.figure(figsize=(12, 8))
sns.heatmap(answer_category_crosstab, annot=True, fmt='d', cmap='Blues')
plt.title('Types de RÃ©ponses par CatÃ©gorie')
plt.xlabel('Type de rÃ©ponse')
plt.ylabel('CatÃ©gorie')
plt.tight_layout()
plt.savefig('answer_types_by_category.png', dpi=300, bbox_inches='tight')
plt.show()

# 10.6 Analyse des patterns d'erreurs numÃ©riques
print(f"\n=== Patterns d'erreurs numÃ©riques (Ã©chantillon de {sample_size}) ===")
print(f"Ã‰tudiants mentionnant la valeur correcte: {numerical_sample['num_mentions_correct_value'].mean():.1%}")
print(f"Score moyen de proximitÃ© numÃ©rique: {numerical_sample['num_proximity_score'].mean():.3f}")

# Compter les types d'erreurs
all_error_patterns = []
for patterns in numerical_sample['num_error_patterns']:
    all_error_patterns.extend(patterns)

if all_error_patterns:
    error_pattern_counts = pd.Series(all_error_patterns).value_counts()
    print(f"\nPatterns d'erreurs identifiÃ©s:")
    print(error_pattern_counts)

    plt.figure(figsize=(10, 6))
    error_pattern_counts.plot(kind='bar', color='salmon')
    plt.title('FrÃ©quence des Patterns d\'Erreurs NumÃ©riques')
    plt.xlabel('Type d\'erreur')
    plt.ylabel('Nombre d\'occurrences')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('numerical_error_patterns.png', dpi=300, bbox_inches='tight')
    plt.show()

# 10.7 Analyse de la proximitÃ© numÃ©rique par catÃ©gorie
plt.figure(figsize=(10, 6))
sns.boxplot(data=numerical_sample, x='Category', y='num_proximity_score', palette='viridis')
plt.title('Score de ProximitÃ© NumÃ©rique par CatÃ©gorie')
plt.xlabel('CatÃ©gorie')
plt.ylabel('Score de proximitÃ© (0-1)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('numerical_proximity_by_category.png', dpi=300, bbox_inches='tight')
plt.show()

# 10.8 Analyse des fractions spÃ©cifiquement
fraction_data = train_df[train_df['answer_type'] == 'fraction'].copy()
if len(fraction_data) > 0:
    print(f"\n=== Analyse spÃ©cifique des fractions ({len(fraction_data)} questions) ===")

    # Extraire numÃ©rateurs et dÃ©nominateurs
    fraction_components = fraction_data['MC_Answer'].apply(
        lambda x: parse_mathematical_expression(x)
    )

    numerators = [comp['numerator'] for comp in fraction_components if comp['numerator'] is not None]
    denominators = [comp['denominator'] for comp in fraction_components if comp['denominator'] is not None]

    print(f"NumÃ©rateurs les plus frÃ©quents: {pd.Series(numerators).value_counts().head()}")
    print(f"DÃ©nominateurs les plus frÃ©quents: {pd.Series(denominators).value_counts().head()}")

    # Visualisation des composants de fractions
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    pd.Series(numerators).value_counts().head(10).plot(kind='bar', ax=axes[0], color='lightblue')
    axes[0].set_title('Top 10 NumÃ©rateurs')
    axes[0].set_xlabel('NumÃ©rateur')
    axes[0].set_ylabel('FrÃ©quence')

    pd.Series(denominators).value_counts().head(10).plot(kind='bar', ax=axes[1], color='lightgreen')
    axes[1].set_title('Top 10 DÃ©nominateurs')
    axes[1].set_xlabel('DÃ©nominateur')
    axes[1].set_ylabel('FrÃ©quence')

    plt.tight_layout()
    plt.savefig('fraction_components_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

# 10.9 RÃ©sumÃ© de l'analyse numÃ©rique
print(f"\n=== RÃ©sumÃ© de l'analyse des patterns numÃ©riques ===")
print(f"â€¢ {len(train_df)} rÃ©ponses analysÃ©es")
print(f"â€¢ Types de rÃ©ponses: {len(answer_type_dist)} catÃ©gories diffÃ©rentes")
print(f"â€¢ {len(numeric_answers)} rÃ©ponses avec valeur numÃ©rique extractible")
print(f"â€¢ Fractions: {len(fraction_data)} questions ({len(fraction_data)/len(train_df)*100:.1f}%)")
print(f"â€¢ Patterns d'erreurs identifiÃ©s: {len(set(all_error_patterns)) if all_error_patterns else 0} types")
print(f"â€¢ Score moyen de proximitÃ© numÃ©rique: {numerical_sample['num_proximity_score'].mean():.3f}")


print("=== 11. ANALYSE DE LA COHÃ‰RENCE DES ANNOTATIONS ===")
print("Ã‰valuation de la qualitÃ© et cohÃ©rence des labels dans le dataset")

# 11.1 Analyse de la cohÃ©rence Category vs Misconception
def analyze_category_misconception_coherence():
    """Analyse la cohÃ©rence entre les colonnes Category et Misconception"""
    print("\n11.1 CohÃ©rence Category â†” Misconception")
    
    # CrÃ©er une matrice de contingence
    coherence_matrix = pd.crosstab(train_df['Category'], 
                                  train_df['Misconception'], 
                                  dropna=False)
    
    # Analyser les patterns attendus vs observÃ©s
    expected_patterns = {
        'True_Correct': ['NaN'],  # Devrait toujours Ãªtre NaN
        'False_Misconception': 'any_misconception',  # Devrait avoir une misconception
        'True_Misconception': 'any_misconception',   # Devrait avoir une misconception
        'False_Neither': ['NaN'],     # Devrait Ãªtre NaN
        'True_Neither': ['NaN'],      # Devrait Ãªtre NaN  
        'False_Correct': ['NaN']      # Devrait Ãªtre NaN
    }
    
    inconsistencies = []
    
    for category in train_df['Category'].unique():
        cat_data = train_df[train_df['Category'] == category]
        misconception_counts = cat_data['Misconception'].value_counts(dropna=False)
        
        print(f"\n--- {category} ---")
        print(f"Total Ã©chantillons: {len(cat_data)}")
        
        if category in ['True_Correct', 'False_Neither', 'True_Neither', 'False_Correct']:
            # Ces catÃ©gories ne devraient avoir QUE des NaN
            non_nan_count = cat_data['Misconception'].notna().sum()
            if non_nan_count > 0:
                print(f"âš ï¸�  INCOHÃ‰RENCE: {non_nan_count} Ã©chantillons avec misconception (attendu: 0)")
                inconsistencies.append({
                    'category': category,
                    'issue': 'unexpected_misconception',
                    'count': non_nan_count,
                    'percentage': (non_nan_count / len(cat_data)) * 100
                })
            else:
                print("âœ… CohÃ©rent: Aucune misconception (comme attendu)")
                
        elif category in ['False_Misconception', 'True_Misconception']:
            # Ces catÃ©gories DOIVENT avoir des misconceptions
            nan_count = cat_data['Misconception'].isna().sum()
            if nan_count > 0:
                print(f"âš ï¸�  INCOHÃ‰RENCE: {nan_count} Ã©chantillons sans misconception (attendu: 0)")
                inconsistencies.append({
                    'category': category,
                    'issue': 'missing_misconception',
                    'count': nan_count,
                    'percentage': (nan_count / len(cat_data)) * 100
                })
            else:
                print("âœ… CohÃ©rent: Toutes ont des misconceptions")
                
            print(f"Top 3 misconceptions: {misconception_counts.head(3).to_dict()}")
    
    return inconsistencies, coherence_matrix

# 11.2 Analyse des doublons et quasi-doublons
def analyze_duplicates_and_near_duplicates():
    """Identifie les doublons exacts et les quasi-doublons"""
    print("\n11.2 Analyse des Doublons et Quasi-doublons")
    
    # Doublons exacts
    exact_duplicates = train_df.duplicated(subset=['QuestionText', 'MC_Answer', 'StudentExplanation'])
    print(f"Doublons exacts: {exact_duplicates.sum()}")
    
    if exact_duplicates.sum() > 0:
        duplicate_examples = train_df[exact_duplicates].head(3)
        print("Exemples de doublons exacts:")
        for idx, row in duplicate_examples.iterrows():
            print(f"- Question {row['QuestionId']}: {row['StudentExplanation'][:100]}...")
    
    # Quasi-doublons (mÃªme question, mÃªme explication, mais labels diffÃ©rents)
    quasi_duplicates = []
    grouped = train_df.groupby(['QuestionId', 'StudentExplanation'])
    
    for (question_id, explanation), group in grouped:
        if len(group) > 1:
            # VÃ©rifier si les labels sont diffÃ©rents
            unique_categories = group['Category'].nunique()
            unique_misconceptions = group['Misconception'].nunique()
            
            if unique_categories > 1 or unique_misconceptions > 1:
                quasi_duplicates.append({
                    'question_id': question_id,
                    'explanation': explanation[:100] + "...",
                    'count': len(group),
                    'categories': group['Category'].unique().tolist(),
                    'misconceptions': group['Misconception'].unique().tolist()
                })
    
    print(f"Quasi-doublons (mÃªme contenu, labels diffÃ©rents): {len(quasi_duplicates)}")
    
    if quasi_duplicates:
        print("Exemples de quasi-doublons:")
        for i, example in enumerate(quasi_duplicates[:3]):
            print(f"Exemple {i+1}:")
            print(f"  Question: {example['question_id']}")
            print(f"  Explication: {example['explanation']}")
            print(f"  CatÃ©gories: {example['categories']}")
            print(f"  Misconceptions: {example['misconceptions']}")
    
    return exact_duplicates.sum(), quasi_duplicates

# 11.3 Analyse de la distribution des annotateurs (simulation)
def analyze_annotator_consistency_simulation():
    """Simule une analyse de cohÃ©rence entre annotateurs basÃ©e sur les patterns"""
    print("\n11.3 Analyse de la CohÃ©rence Inter-Annotateurs (Simulation)")
    
    # Simuler des "annotateurs" basÃ©s sur des patterns dans les donnÃ©es
    def get_annotation_confidence_score(row):
        """Calcule un score de confiance basÃ© sur diffÃ©rents indicateurs"""
        score = 0
        
        # Longueur de l'explication (les trÃ¨s courtes ou trÃ¨s longues peuvent Ãªtre ambiguÃ«s)
        length = len(str(row['StudentExplanation']).split())
        if 5 <= length <= 25:
            score += 1
        
        # CohÃ©rence numÃ©rique
        if 'numerical_consistency' in row:
            score += row.get('numerical_consistency', 0)
        
        # PrÃ©sence d'Ã©lÃ©ments mathÃ©matiques clairs
        if 'numbers_mentioned' in row:
            if row.get('numbers_mentioned', 0) > 0:
                score += 0.5
        
        # ClartÃ© de l'explication (moins de fautes d'orthographe)
        explanation = str(row['StudentExplanation']).lower()
        if not any(char in explanation for char in ['?', 'dunno', 'idk', 'not sure']):
            score += 0.5
            
        return min(score, 3) / 3  # Normaliser entre 0 et 1
    
    # Calculer les scores de confiance
    confidence_scores = train_df.apply(get_annotation_confidence_score, axis=1)
    train_df['annotation_confidence'] = confidence_scores
    
    print(f"Score moyen de confiance d'annotation: {confidence_scores.mean():.3f}")
    print(f"Annotations Ã  faible confiance (<0.3): {(confidence_scores < 0.3).sum()}")
    print(f"Annotations Ã  haute confiance (>0.7): {(confidence_scores > 0.7).sum()}")
    
    # Analyser la distribution par catÃ©gorie
    confidence_by_category = train_df.groupby('Category')['annotation_confidence'].agg(['mean', 'std', 'count'])
    print("\nConfiance moyenne par catÃ©gorie:")
    print(confidence_by_category.round(3))
    
    return confidence_scores

# 11.4 Analyse des cas ambigus
def identify_ambiguous_cases():
    """Identifie les cas potentiellement ambigus nÃ©cessitant une attention particuliÃ¨re"""
    print("\n11.4 Identification des Cas Ambigus")
    
    ambiguous_cases = []
    
    # Cas 1: Explications trÃ¨s courtes avec misconceptions complexes
    short_complex = train_df[
        (train_df['explanation_length'] <= 5) & 
        (train_df['Misconception'].notna()) &
        (~train_df['Misconception'].isin(['Incomplete', 'Irrelevant']))
    ]
    
    print(f"Explications courtes (<= 5 mots) avec misconceptions complexes: {len(short_complex)}")
    
    # Cas 2: Explications longues marquÃ©es comme "Incomplete"
    long_incomplete = train_df[
        (train_df['explanation_length'] >= 20) & 
        (train_df['Misconception'] == 'Incomplete')
    ]
    
    print(f"Explications longues (>= 20 mots) marquÃ©es 'Incomplete': {len(long_incomplete)}")
    
    # Cas 3: True_Correct avec faible cohÃ©rence numÃ©rique
    if 'numerical_consistency' in train_df.columns:
        correct_but_inconsistent = train_df[
            (train_df['Category'] == 'True_Correct') &
            (train_df['numerical_consistency'] < 0.2)
        ]
        print(f"True_Correct avec faible cohÃ©rence numÃ©rique: {len(correct_but_inconsistent)}")
    
    # Cas 4: Explications avec mots d'incertitude mais marquÃ©es comme correctes
    uncertainty_words = ['maybe', 'perhaps', 'i think', 'not sure', 'might be', 'probably']
    uncertain_but_correct = []
    
    for idx, row in train_df.iterrows():
        explanation = str(row['StudentExplanation']).lower()
        if (row['Category'] in ['True_Correct'] and 
            any(word in explanation for word in uncertainty_words)):
            uncertain_but_correct.append(idx)
    
    print(f"Explications incertaines marquÃ©es comme correctes: {len(uncertain_but_correct)}")
    
    # CrÃ©er un rÃ©sumÃ© des cas ambigus
    ambiguous_summary = {
        'short_complex': len(short_complex),
        'long_incomplete': len(long_incomplete),
        'correct_inconsistent': len(correct_but_inconsistent) if 'numerical_consistency' in train_df.columns else 0,
        'uncertain_correct': len(uncertain_but_correct)
    }
    
    return ambiguous_summary

# 11.5 Validation croisÃ©e des patterns d'annotation
def cross_validate_annotation_patterns():
    """Valide la cohÃ©rence des patterns d'annotation Ã  travers diffÃ©rentes dimensions"""
    print("\n11.5 Validation CroisÃ©e des Patterns d'Annotation")
    
    # Pattern 1: Longueur vs ComplÃ©tude
    incomplete_by_length = train_df[train_df['Misconception'] == 'Incomplete'].groupby(
        pd.cut(train_df['explanation_length'], bins=[0, 5, 10, 20, 50, np.inf])
    ).size()
    
    print("Distribution 'Incomplete' par longueur d'explication:")
    print(incomplete_by_length)
    
    # Pattern 2: CohÃ©rence mathÃ©matique vs CatÃ©gorie
    if 'numerical_consistency' in train_df.columns:
        consistency_by_category = train_df.groupby('Category')['numerical_consistency'].mean()
        print(f"\nCohÃ©rence numÃ©rique moyenne par catÃ©gorie:")
        print(consistency_by_category.round(3))
    
    # Pattern 3: PrÃ©sence d'Ã©lÃ©ments mathÃ©matiques vs Misconceptions
    if 'numbers_mentioned' in train_df.columns:
        math_elements_by_misconception = train_df[train_df['Misconception'].notna()].groupby(
            'Misconception'
        )[['numbers_mentioned', 'fractions_mentioned', 'operations_mentioned']].mean()
        
        print(f"\nTop 5 misconceptions par Ã©lÃ©ments mathÃ©matiques:")
        print(math_elements_by_misconception.head())
    
    # Pattern 4: Images vs Types de questions
    if 'has_image' in train_df.columns:
        image_by_category = pd.crosstab(train_df['has_image'], train_df['Category'], normalize='columns')
        print(f"\nDistribution des images par catÃ©gorie (en %):")
        print((image_by_category * 100).round(1))
    
    return True

# 11.6 MÃ©triques de qualitÃ© globale du dataset
def calculate_dataset_quality_metrics():
    """Calcule des mÃ©triques globales de qualitÃ© du dataset"""
    print("\n11.6 MÃ©triques de QualitÃ© Globale du Dataset")
    
    metrics = {}
    
    # ComplÃ©tude des donnÃ©es
    completeness = {
        'total_rows': len(train_df),
        'complete_rows': len(train_df.dropna()),
        'missing_misconceptions': train_df['Misconception'].isna().sum(),
        'completeness_ratio': len(train_df.dropna()) / len(train_df)
    }
    
    # Distribution des labels
    label_distribution = {
        'category_balance': train_df['Category'].value_counts(normalize=True).min(),
        'misconception_coverage': train_df['Misconception'].nunique(),
        'category_coverage': train_df['Category'].nunique()
    }
    
    # DiversitÃ© des contenus
    content_diversity = {
        'unique_questions': train_df['QuestionId'].nunique(),
        'unique_explanations': train_df['StudentExplanation'].nunique(),
        'avg_explanation_length': train_df['explanation_length'].mean(),
        'explanation_length_std': train_df['explanation_length'].std()
    }
    
    # Score de qualitÃ© composite
    quality_score = (
        completeness['completeness_ratio'] * 0.3 +
        min(label_distribution['category_balance'] * 6, 1.0) * 0.3 +  # Normaliser la balance
        (content_diversity['unique_explanations'] / len(train_df)) * 0.4
    )
    
    metrics.update(completeness)
    metrics.update(label_distribution)
    metrics.update(content_diversity)
    metrics['overall_quality_score'] = quality_score
    
    print("=== MÃ‰TRIQUES DE QUALITÃ‰ ===")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.3f}")
        else:
            print(f"{key}: {value}")
    
    return metrics

# ExÃ©cution de toutes les analyses
print("DÃ©marrage de l'analyse de cohÃ©rence des annotations...")

# 11.1 CohÃ©rence Category vs Misconception
inconsistencies, coherence_matrix = analyze_category_misconception_coherence()

# 11.2 Doublons
exact_dups, quasi_dups = analyze_duplicates_and_near_duplicates()

# 11.3 CohÃ©rence simulÃ©e des annotateurs
confidence_scores = analyze_annotator_consistency_simulation()

# 11.4 Cas ambigus
ambiguous_summary = identify_ambiguous_cases()

# 11.5 Validation croisÃ©e
cross_validate_annotation_patterns()

# 11.6 MÃ©triques de qualitÃ©
quality_metrics = calculate_dataset_quality_metrics()

# 11.7 Visualisations finales
print("\n11.7 Visualisations de CohÃ©rence")

# Graphique 1: Matrice de cohÃ©rence Category vs Misconception (top misconceptions)
plt.figure(figsize=(14, 8))
top_misconceptions = train_df['Misconception'].value_counts().head(15).index
coherence_subset = coherence_matrix[top_misconceptions]
sns.heatmap(coherence_subset, annot=True, fmt='d', cmap='YlOrRd')
plt.title('Matrice de CohÃ©rence: Category vs Top 15 Misconceptions')
plt.xlabel('Misconceptions')
plt.ylabel('Categories')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('category_misconception_coherence.png', dpi=300, bbox_inches='tight')
plt.show()

# Graphique 2: Distribution des scores de confiance
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.hist(confidence_scores, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
plt.title('Distribution des Scores de Confiance d\'Annotation')
plt.xlabel('Score de Confiance (0-1)')
plt.ylabel('FrÃ©quence')
plt.axvline(confidence_scores.mean(), color='red', linestyle='--', label=f'Moyenne: {confidence_scores.mean():.3f}')
plt.legend()

plt.subplot(1, 2, 2)
sns.boxplot(data=train_df, x='Category', y='annotation_confidence', palette='Set2')
plt.title('Confiance d\'Annotation par CatÃ©gorie')
plt.xlabel('CatÃ©gorie')
plt.ylabel('Score de Confiance')
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig('annotation_confidence_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# Graphique 3: Analyse des cas ambigus
plt.figure(figsize=(10, 6))
ambiguous_data = pd.DataFrame([ambiguous_summary]).T
ambiguous_data.columns = ['Count']
ambiguous_data.plot(kind='bar', color='coral')
plt.title('Distribution des Cas Ambigus IdentifiÃ©s')
plt.xlabel('Type de Cas Ambigu')
plt.ylabel('Nombre de Cas')
plt.xticks(rotation=45, ha='right')
plt.legend(['Nombre de cas'])
plt.tight_layout()
plt.savefig('ambiguous_cases_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# 11.8 Recommandations finales
print("\n" + "="*60)
print("11.8 RECOMMANDATIONS POUR LA MODÃ‰LISATION")
print("="*60)

print(f"""
ğŸ“Š RÃ‰SUMÃ‰ DE LA COHÃ‰RENCE DES ANNOTATIONS:

ğŸ”� COHÃ‰RENCE GÃ‰NÃ‰RALE:
â€¢ Score de qualitÃ© global: {quality_metrics['overall_quality_score']:.3f}/1.0
â€¢ IncohÃ©rences dÃ©tectÃ©es: {len(inconsistencies)} types
â€¢ Doublons exacts: {exact_dups}
â€¢ Quasi-doublons: {len(quasi_dups)}

ğŸ“ˆ CONFIANCE DES ANNOTATIONS:
â€¢ Score moyen: {confidence_scores.mean():.3f}
â€¢ Annotations fiables (>0.7): {(confidence_scores > 0.7).sum():,} ({(confidence_scores > 0.7).sum()/len(train_df)*100:.1f}%)
â€¢ Annotations douteuses (<0.3): {(confidence_scores < 0.3).sum():,} ({(confidence_scores < 0.3).sum()/len(train_df)*100:.1f}%)

âš ï¸�  CAS AMBIGUS IDENTIFIÃ‰S:
â€¢ Explications courtes avec misconceptions complexes: {ambiguous_summary['short_complex']}
â€¢ Explications longues marquÃ©es 'Incomplete': {ambiguous_summary['long_incomplete']}
â€¢ Explications incertaines marquÃ©es correctes: {ambiguous_summary['uncertain_correct']}

ğŸ�¯ RECOMMANDATIONS STRATÃ‰GIQUES:

1. STRATÃ‰GIE DE VALIDATION CROISÃ‰E:
   - Utiliser les annotations Ã  haute confiance (>0.7) pour l'entraÃ®nement principal
   - CrÃ©er un ensemble de validation spÃ©cial avec les cas ambigus
   - ImplÃ©menter une validation k-fold stratifiÃ©e par catÃ©gorie ET misconception

2. PRÃ‰TRAITEMENT CIBLÃ‰:
   - Filtrer ou re-labelliser les {len(inconsistencies)} incohÃ©rences identifiÃ©es
   - Traiter spÃ©cialement les {len(quasi_dups)} quasi-doublons
   - Augmenter les donnÃ©es pour les misconceptions sous-reprÃ©sentÃ©es

3. ARCHITECTURE MODÃˆLE:
   - ImplÃ©menter un systÃ¨me de confiance dans les prÃ©dictions
   - Utiliser une approche multi-tÃ¢ches: Category + Misconception simultanÃ©ment
   - ConsidÃ©rer des modÃ¨les ensemblistes pour gÃ©rer l'incertitude

4. Ã‰VALUATION ROBUSTE:
   - Utiliser des mÃ©triques pondÃ©rÃ©es par la confiance d'annotation
   - Ã‰valuer sÃ©parÃ©ment sur les cas ambigus vs. non-ambigus
   - ImplÃ©menter une mÃ©trique de cohÃ©rence Category-Misconception

5. GESTION DES CAS LIMITES:
   - CrÃ©er des rÃ¨gles de post-traitement pour assurer la cohÃ©rence
   - ImplÃ©menter une dÃ©tection d'anomalies pour les prÃ©dictions incohÃ©rentes
   - Utiliser l'incertitude du modÃ¨le pour identifier les cas difficiles
""")

print("\nğŸ�� ANALYSE EDA COMPLÃˆTE")
print("="*60)
print("L'analyse exploratoire est maintenant terminÃ©e avec:")
print("â€¢ 11 sections d'analyse approfondies")
print("â€¢ Identification des patterns clÃ©s et des problÃ¨mes de qualitÃ©")
print("â€¢ Recommandations stratÃ©giques pour la modÃ©lisation")
print("â€¢ Base solide pour le dÃ©veloppement du modÃ¨le de compÃ©tition")
print("\nâœ… PrÃªt pour la phase de PrÃ©traitement !")


# Phase 2 : PrÃ©traitement des DonnÃ©es

print("="*60)
print("PHASE 2 : PRÃ‰TRAITEMENT DES DONNÃ‰ES")
print("="*60)

import pandas as pd
import numpy as np
import re
import string
from sklearn.preprocessing import LabelEncoder
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# 1. NETTOYAGE DES DOUBLONS ET QUASI-DOUBLONS

def remove_duplicates_and_quasi_duplicates(df):
    """
    Supprime les doublons exacts et traite les quasi-doublons identifiÃ©s dans l'EDA.
    
    Justification :
    - L'EDA a identifiÃ© 797 doublons exacts et 101 quasi-doublons
    - Les doublons peuvent biaiser l'entraÃ®nement et l'Ã©valuation
    - Les quasi-doublons (mÃªme contenu, labels diffÃ©rents) crÃ©ent de l'incohÃ©rence
    """
    print("1. Nettoyage des doublons et quasi-doublons...")
    print(f"   Taille initiale: {len(df)} Ã©chantillons")
    
    initial_size = len(df)
    
    # 1.1 Suppression des doublons exacts
    df_clean = df.drop_duplicates(
        subset=['QuestionText', 'MC_Answer', 'StudentExplanation', 'Category', 'Misconception'],
        keep='first'
    ).copy()
    
    exact_duplicates_removed = initial_size - len(df_clean)
    print(f"   â†’ Doublons exacts supprimÃ©s: {exact_duplicates_removed}")
    
    # 1.2 Traitement des quasi-doublons (mÃªme contenu, labels diffÃ©rents)
    # StratÃ©gie : garder l'annotation avec le score de confiance le plus Ã©levÃ©
    quasi_duplicates_groups = df_clean.groupby(['QuestionId', 'StudentExplanation'])
    quasi_duplicates_to_remove = []
    
    for (question_id, explanation), group in quasi_duplicates_groups:
        if len(group) > 1:
            # Calculer un score de prioritÃ© basÃ© sur la cohÃ©rence des annotations
            priorities = []
            for idx, row in group.iterrows():
                priority_score = 0
                
                # PrioritÃ© 1: CohÃ©rence Category-Misconception
                if row['Category'] in ['True_Correct', 'False_Neither', 'True_Neither', 'False_Correct']:
                    if pd.isna(row['Misconception']):
                        priority_score += 3  # CohÃ©rent
                    else:
                        priority_score += 1  # IncohÃ©rent
                elif row['Category'] in ['False_Misconception', 'True_Misconception']:
                    if pd.notna(row['Misconception']):
                        priority_score += 3  # CohÃ©rent
                    else:
                        priority_score += 1  # IncohÃ©rent
                
                # PrioritÃ© 2: PrÃ©fÃ©rer les labels plus informatifs
                if pd.notna(row['Misconception']):
                    priority_score += 1
                
                priorities.append((idx, priority_score))
            
            # Garder seulement l'Ã©chantillon avec le score le plus Ã©levÃ©
            priorities.sort(key=lambda x: x[1], reverse=True)
            best_idx = priorities[0][0]
            
            # Marquer les autres pour suppression
            for idx, _ in priorities[1:]:
                quasi_duplicates_to_remove.append(idx)
    
    # Supprimer les quasi-doublons
    df_clean = df_clean.drop(index=quasi_duplicates_to_remove)
    quasi_duplicates_removed = len(quasi_duplicates_to_remove)
    print(f"   â†’ Quasi-doublons supprimÃ©s: {quasi_duplicates_removed}")
    
    final_size = len(df_clean)
    print(f"   Taille finale: {final_size} Ã©chantillons")
    print(f"   RÃ©duction totale: {initial_size - final_size} ({((initial_size - final_size) / initial_size * 100):.1f}%)")
    
    return df_clean.reset_index(drop=True)

# 2. NORMALISATION DES TEXTES

def normalize_text_fields(df):
    """
    Normalise les champs textuels en corrigeant les problÃ¨mes identifiÃ©s dans l'EDA.
    
    Justification :
    - L'EDA a rÃ©vÃ©lÃ© un taux d'erreurs d'orthographe de 4.1%
    - 3545 occurrences de confusions nombre/mot (to/two, for/four)
    - PrÃ©sence de langage informel et de mots rÃ©pÃ©titifs
    - NÃ©cessitÃ© de standardiser pour amÃ©liorer la cohÃ©rence
    """
    print("\n2. Normalisation des champs textuels...")
    
    def clean_student_explanation(text):
        """Nettoie et normalise l'explication de l'Ã©tudiant"""
        if pd.isna(text):
            return ""
        
        text = str(text).strip()
        
        # 2.1 Corrections orthographiques courantes identifiÃ©es dans l'EDA
        common_corrections = {
            # Confusions nombre/mot identifiÃ©es
            r'\bto\b(?=\s*(many|much|times|the))': 'two',  # "to many" -> "two many"
            r'\bfor\b(?=\s*(triangles|squares|parts))': 'four',  # "for triangles" -> "four triangles"
            
            # Termes mathÃ©matiques mal orthographiÃ©s identifiÃ©s
            r'\bfracion\b': 'fraction',
            r'\bfractoin\b': 'fraction',
            r'\bdivison\b': 'division',
            r'\bmultipication\b': 'multiplication',
            r'\bequasion\b': 'equation',
            
            # Mots courants mal orthographiÃ©s
            r'\bawnser\b': 'answer',
            r'\bbecaus\b': 'because',
            r'\btheres\b': 'there are',
            r'\barentt\b': 'are not',
            
            # Contractions informelles identifiÃ©es
            r'\bgonna\b': 'going to',
            r'\bwanna\b': 'want to',
            r'\bdunno\b': 'do not know',
            r'\bkinda\b': 'kind of',
        }
        
        for pattern, replacement in common_corrections.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        # 2.2 Suppression des mots rÃ©pÃ©titifs consÃ©cutifs (identifiÃ©s dans l'EDA)
        words = text.split()
        cleaned_words = []
        prev_word = None
        for word in words:
            if word.lower() != prev_word:
                cleaned_words.append(word)
                prev_word = word.lower()
        text = ' '.join(cleaned_words)
        
        # 2.3 Normalisation de la ponctuation
        # Ajouter un point si la phrase ne se termine pas par une ponctuation
        if text and not text[-1] in '.!?':
            text += '.'
        
        # 2.4 Normalisation des espaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def clean_question_text(text):
        """Nettoie le texte de la question"""
        if pd.isna(text):
            return ""
        
        text = str(text).strip()
        
        # Normaliser les balises d'image
        text = re.sub(r'\[Image:\s*[^\]]*\]', '[IMAGE]', text)
        
        # Normaliser les espaces autour de la ponctuation
        text = re.sub(r'\s*([.!?])\s*', r'\1 ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def clean_mc_answer(text):
        """Nettoie la rÃ©ponse Ã  choix multiple"""
        if pd.isna(text):
            return ""
        
        text = str(text).strip()
        
        # Normaliser les expressions LaTeX
        # Uniformiser les dÃ©limiteurs LaTeX
        text = re.sub(r'\\?\(\s*', r'\\( ', text)
        text = re.sub(r'\s*\\?\)', r' \\)', text)
        
        # Normaliser les fractions LaTeX
        text = re.sub(r'\\frac\s*{([^}]+)}\s*{([^}]+)}', r'\\frac{\1}{\2}', text)
        
        return text
    
    # Application du nettoyage
    df_clean = df.copy()
    
    print("   â†’ Nettoyage des explications Ã©tudiantes...")
    df_clean['StudentExplanation'] = df_clean['StudentExplanation'].apply(clean_student_explanation)
    
    print("   â†’ Nettoyage des textes de questions...")
    df_clean['QuestionText'] = df_clean['QuestionText'].apply(clean_question_text)
    
    print("   â†’ Nettoyage des rÃ©ponses MC...")
    df_clean['MC_Answer'] = df_clean['MC_Answer'].apply(clean_mc_answer)
    
    # Statistiques post-nettoyage
    empty_explanations = (df_clean['StudentExplanation'] == "").sum()
    print(f"   â†’ Explications vides aprÃ¨s nettoyage: {empty_explanations}")
    
    return df_clean

# 3. GESTION DES VALEURS MANQUANTES

def handle_missing_values(df):
    """
    GÃ¨re les valeurs manquantes selon les patterns identifiÃ©s dans l'EDA.
    
    Justification :
    - L'EDA montre 26,836 valeurs manquantes dans Misconception (73% des donnÃ©es)
    - Ces valeurs sont cohÃ©rentes avec la logique mÃ©tier (pas de misconception = NaN)
    - Quelques explications trÃ¨s courtes ou vides nÃ©cessitent un traitement
    """
    print("\n3. Gestion des valeurs manquantes...")
    
    df_clean = df.copy()
    initial_size = len(df_clean)
    
    # 3.1 Traitement des explications vides ou trÃ¨s courtes
    # L'EDA montre des explications d'1 mot minimum
    very_short_explanations = df_clean['StudentExplanation'].str.len() <= 2
    print(f"   â†’ Explications trÃ¨s courtes (<=2 chars): {very_short_explanations.sum()}")
    
    # Remplacer les explications trÃ¨s courtes par un marqueur
    df_clean.loc[very_short_explanations, 'StudentExplanation'] = "No explanation provided."
    
    # 3.2 Gestion des Misconceptions manquantes
    # Ces valeurs sont normales selon la logique mÃ©tier - les convertir en chaÃ®ne pour traitement
    misconceptions_na = df_clean['Misconception'].isna().sum()
    print(f"   â†’ Misconceptions manquantes (normal): {misconceptions_na}")
    
    # Remplacer NaN par 'NA' pour faciliter le traitement ultÃ©rieur
    df_clean['Misconception'] = df_clean['Misconception'].fillna('NA')
    
    # 3.3 VÃ©rification de l'intÃ©gritÃ© des donnÃ©es obligatoires
    required_fields = ['QuestionText', 'MC_Answer', 'StudentExplanation', 'Category']
    missing_required = {}
    
    for field in required_fields:
        missing_count = df_clean[field].isna().sum()
        missing_required[field] = missing_count
        if missing_count > 0:
            print(f"   âš ï¸� Champ obligatoire manquant - {field}: {missing_count}")
    
    # Supprimer les lignes avec des champs obligatoires manquants
    before_removal = len(df_clean)
    df_clean = df_clean.dropna(subset=required_fields)
    after_removal = len(df_clean)
    
    if before_removal != after_removal:
        print(f"   â†’ Lignes supprimÃ©es (champs obligatoires manquants): {before_removal - after_removal}")
    
    print(f"   Taille finale aprÃ¨s gestion des valeurs manquantes: {len(df_clean)}")
    
    return df_clean

# 4. VALIDATION ET CORRECTION DE COHÃ‰RENCE

def validate_and_fix_consistency(df):
    """
    Valide et corrige les incohÃ©rences identifiÃ©es dans l'EDA.
    
    Justification :
    - L'EDA n'a trouvÃ© aucune incohÃ©rence majeure Category-Misconception
    - Cependant, il faut s'assurer de la cohÃ©rence pour les nouvelles donnÃ©es nettoyÃ©es
    - Correction des cas ambigus identifiÃ©s (3139 explications incertaines marquÃ©es correctes)
    """
    print("\n4. Validation et correction de cohÃ©rence...")
    
    df_clean = df.copy()
    inconsistencies_fixed = 0
    
    # 4.1 VÃ©rification de la cohÃ©rence Category-Misconception
    print("   â†’ VÃ©rification cohÃ©rence Category-Misconception...")
    
    # RÃ¨gles de cohÃ©rence basÃ©es sur l'analyse EDA
    categories_without_misconception = ['True_Correct', 'False_Neither', 'True_Neither', 'False_Correct']
    categories_with_misconception = ['False_Misconception', 'True_Misconception']
    
    # Cas 1: CatÃ©gories qui ne devraient PAS avoir de misconception
    inconsistent_no_misc = df_clean[
        (df_clean['Category'].isin(categories_without_misconception)) & 
        (df_clean['Misconception'] != 'NA')
    ]
    
    if len(inconsistent_no_misc) > 0:
        print(f"   âš ï¸� IncohÃ©rence trouvÃ©e: {len(inconsistent_no_misc)} Ã©chantillons avec misconception inattendue")
        df_clean.loc[inconsistent_no_misc.index, 'Misconception'] = 'NA'
        inconsistencies_fixed += len(inconsistent_no_misc)
    
    # Cas 2: CatÃ©gories qui DOIVENT avoir une misconception
    inconsistent_with_misc = df_clean[
        (df_clean['Category'].isin(categories_with_misconception)) & 
        (df_clean['Misconception'] == 'NA')
    ]
    
    if len(inconsistent_with_misc) > 0:
        print(f"   âš ï¸� IncohÃ©rence trouvÃ©e: {len(inconsistent_with_misc)} Ã©chantillons sans misconception attendue")
        # Pour ces cas, assigner 'Incomplete' comme misconception par dÃ©faut
        df_clean.loc[inconsistent_with_misc.index, 'Misconception'] = 'Incomplete'
        inconsistencies_fixed += len(inconsistent_with_misc)
    
    # 4.2 Traitement des cas ambigus identifiÃ©s dans l'EDA
    print("   â†’ Traitement des cas ambigus...")
    
    # Identifier les explications avec marqueurs d'incertitude
    uncertainty_patterns = [
        r'\bi think\b', r'\bmaybe\b', r'\bperhaps\b', r'\bnot sure\b', 
        r'\bmight be\b', r'\bcould be\b', r'\bprobably\b'
    ]
    
    uncertainty_regex = '|'.join(uncertainty_patterns)
    uncertain_explanations = df_clean['StudentExplanation'].str.contains(
        uncertainty_regex, case=False, na=False
    )
    
    # Pour les explications incertaines marquÃ©es comme "True_Correct", 
    # les reclasser comme "True_Neither" (pas complÃ¨tement sÃ»r)
    uncertain_but_correct = (
        uncertain_explanations & 
        (df_clean['Category'] == 'True_Correct')
    )
    
    if uncertain_but_correct.sum() > 0:
        print(f"   â†’ Reclassification d'explications incertaines: {uncertain_but_correct.sum()}")
        df_clean.loc[uncertain_but_correct, 'Category'] = 'True_Neither'
        inconsistencies_fixed += uncertain_but_correct.sum()
    
    # 4.3 Validation des longueurs d'explication vs labels
    # L'EDA montre 290 explications longues marquÃ©es 'Incomplete'
    long_but_incomplete = (
        (df_clean['StudentExplanation'].str.len() > 100) & 
        (df_clean['Misconception'] == 'Incomplete')
    )
    
    if long_but_incomplete.sum() > 0:
        print(f"   â†’ RÃ©vision d'explications longues marquÃ©es 'Incomplete': {long_but_incomplete.sum()}")
        # Ces cas nÃ©cessitent une rÃ©vision - les marquer comme 'Irrelevant' plutÃ´t qu'Incomplete
        df_clean.loc[long_but_incomplete, 'Misconception'] = 'Irrelevant'
    
    print(f"   Total d'incohÃ©rences corrigÃ©es: {inconsistencies_fixed}")
    
    return df_clean

# 5. ENCODAGE ET STANDARDISATION DES LABELS

def encode_and_standardize_labels(df):
    """
    Encode et standardise les labels pour la modÃ©lisation.
    
    Justification :
    - NÃ©cessaire pour la compatibilitÃ© avec les modÃ¨les ML
    - L'EDA montre 6 catÃ©gories et 35 misconceptions + NA
    - Besoin de crÃ©er des mappings cohÃ©rents pour train/test
    """
    print("\n5. Encodage et standardisation des labels...")
    
    df_encoded = df.copy()
    
    # 5.1 CrÃ©er les mappings pour les catÃ©gories
    print("   â†’ Encodage des catÃ©gories...")
    categories = sorted(df_encoded['Category'].unique())
    category_to_id = {cat: idx for idx, cat in enumerate(categories)}
    id_to_category = {idx: cat for cat, idx in category_to_id.items()}
    
    df_encoded['Category_encoded'] = df_encoded['Category'].map(category_to_id)
    
    print(f"   CatÃ©gories encodÃ©es: {len(category_to_id)}")
    for cat, idx in category_to_id.items():
        count = (df_encoded['Category'] == cat).sum()
        print(f"     {idx}: {cat} ({count} Ã©chantillons)")
    
    # 5.2 CrÃ©er les mappings pour les misconceptions
    print("   â†’ Encodage des misconceptions...")
    misconceptions = sorted(df_encoded['Misconception'].unique())
    misconception_to_id = {misc: idx for idx, misc in enumerate(misconceptions)}
    id_to_misconception = {idx: misc for misc, idx in misconception_to_id.items()}
    
    df_encoded['Misconception_encoded'] = df_encoded['Misconception'].map(misconception_to_id)
    
    print(f"   Misconceptions encodÃ©es: {len(misconception_to_id)}")
    
    # Afficher les top 10 plus frÃ©quentes
    misconception_counts = df_encoded['Misconception'].value_counts()
    print("   Top 10 misconceptions:")
    for i, (misc, count) in enumerate(misconception_counts.head(10).items()):
        idx = misconception_to_id[misc]
        print(f"     {idx}: {misc} ({count} Ã©chantillons)")
    
    # 5.3 CrÃ©er le label combinÃ© Category:Misconception pour le format de soumission
    print("   â†’ CrÃ©ation du label combinÃ© Category:Misconception...")
    df_encoded['Category_Misconception'] = df_encoded['Category'] + ':' + df_encoded['Misconception']
    
    # 5.4 Sauvegarder les mappings pour utilisation ultÃ©rieure
    label_mappings = {
        'category_to_id': category_to_id,
        'id_to_category': id_to_category,
        'misconception_to_id': misconception_to_id,
        'id_to_misconception': id_to_misconception
    }
    
    return df_encoded, label_mappings

# 6. VALIDATION FINALE ET STATISTIQUES

def final_validation_and_stats(df_original, df_processed, label_mappings):
    """
    Validation finale et gÃ©nÃ©ration de statistiques de prÃ©traitement.
    """
    print("\n6. Validation finale et statistiques...")
    
    # 6.1 Statistiques de transformation
    print("   ğŸ“Š STATISTIQUES DE PRÃ‰TRAITEMENT:")
    print(f"   â€¢ Taille originale: {len(df_original):,} Ã©chantillons")
    print(f"   â€¢ Taille finale: {len(df_processed):,} Ã©chantillons")
    reduction = len(df_original) - len(df_processed)
    print(f"   â€¢ RÃ©duction: {reduction:,} Ã©chantillons ({reduction/len(df_original)*100:.1f}%)")
    
    # 6.2 Distribution finale des labels
    print(f"\n   ğŸ“ˆ DISTRIBUTION FINALE DES LABELS:")
    print("   CatÃ©gories:")
    category_dist = df_processed['Category'].value_counts()
    for cat, count in category_dist.items():
        pct = count / len(df_processed) * 100
        print(f"     â€¢ {cat}: {count:,} ({pct:.1f}%)")
    
    print(f"\n   Misconceptions (top 10):")
    misconception_dist = df_processed['Misconception'].value_counts().head(10)
    for misc, count in misconception_dist.items():
        pct = count / len(df_processed) * 100
        print(f"     â€¢ {misc}: {count:,} ({pct:.1f}%)")
    
    # 6.3 Validation de l'intÃ©gritÃ©
    print(f"\n   âœ… VALIDATION D'INTÃ‰GRITÃ‰:")
    
    # VÃ©rifier qu'il n'y a pas de valeurs manquantes dans les champs critiques
    critical_fields = ['QuestionText', 'MC_Answer', 'StudentExplanation', 'Category', 'Misconception']
    for field in critical_fields:
        missing = df_processed[field].isna().sum()
        if missing == 0:
            print(f"     â€¢ {field}: âœ… Aucune valeur manquante")
        else:
            print(f"     â€¢ {field}: âš ï¸� {missing} valeurs manquantes")
    
    # VÃ©rifier la cohÃ©rence des encodages
    category_encoding_ok = len(df_processed['Category_encoded'].unique()) == len(label_mappings['category_to_id'])
    misconception_encoding_ok = len(df_processed['Misconception_encoded'].unique()) == len(label_mappings['misconception_to_id'])
    
    print(f"     â€¢ Encodage catÃ©gories: {'âœ…' if category_encoding_ok else 'âš ï¸�'}")
    print(f"     â€¢ Encodage misconceptions: {'âœ…' if misconception_encoding_ok else 'âš ï¸�'}")
    
    # 6.4 QualitÃ© des textes
    print(f"\n   ğŸ“� QUALITÃ‰ DES TEXTES:")
    avg_explanation_length = df_processed['StudentExplanation'].str.len().mean()
    print(f"     â€¢ Longueur moyenne des explications: {avg_explanation_length:.1f} caractÃ¨res")
    
    empty_explanations = (df_processed['StudentExplanation'] == "").sum()
    print(f"     â€¢ Explications vides: {empty_explanations}")
    
    return True

# 7. PIPELINE COMPLÃˆTE

def preprocess_training_data(train_df):
    """
    PrÃ©traitement pour les donnÃ©es d'entraÃ®nement.
    """
    print("ğŸ”„ DÃ‰MARRAGE DU PRÃ‰TRAITEMENT")
    print("=" * 60)
    
    # Copie des donnÃ©es originales
    df_processed = train_df.copy()
    
    # Ã‰tape 1: Nettoyage des doublons
    df_processed = remove_duplicates_and_quasi_duplicates(df_processed)
    
    # Ã‰tape 2: Normalisation des textes
    df_processed = normalize_text_fields(df_processed)
    
    # Ã‰tape 3: Gestion des valeurs manquantes
    df_processed = handle_missing_values(df_processed)
    
    # Ã‰tape 4: Validation et correction de cohÃ©rence
    df_processed = validate_and_fix_consistency(df_processed)
    
    # Ã‰tape 5: Encodage des labels
    df_processed, label_mappings = encode_and_standardize_labels(df_processed)
    
    # Ã‰tape 6: Validation finale
    final_validation_and_stats(train_df, df_processed, label_mappings)
    
    print("\n" + "=" * 60)
    print("âœ… PRÃ‰TRAITEMENT TERMINÃ‰ AVEC SUCCÃˆS")
    print("=" * 60)
    
    return df_processed, label_mappings

def preprocess_test_data(test_df, label_mappings):
    """
    PrÃ©traitement pour les donnÃ©es de test (sans les labels).
    """
    print("\nğŸ”„ PRÃ‰TRAITEMENT DES DONNÃ‰ES DE TEST")
    print("=" * 40)
    
    df_test_processed = test_df.copy()
    
    # Ã‰tapes applicables aux donnÃ©es de test (sans labels)
    print("1. Normalisation des textes de test...")
    
    # RÃ©utiliser les fonctions de nettoyage (adaptÃ©es pour les donnÃ©es sans labels)
    def clean_student_explanation(text):
        if pd.isna(text):
            return ""
        
        text = str(text).strip()
        
        # MÃªme nettoyage que pour train
        common_corrections = {
            r'\bto\b(?=\s*(many|much|times|the))': 'two',
            r'\bfor\b(?=\s*(triangles|squares|parts))': 'four',
            r'\bfracion\b': 'fraction',
            r'\bfractoin\b': 'fraction',
            r'\bdivison\b': 'division',
            r'\bmultipication\b': 'multiplication',
            r'\bequasion\b': 'equation',
            r'\bawnser\b': 'answer',
            r'\bbecaus\b': 'because',
            r'\btheres\b': 'there are',
            r'\barentt\b': 'are not',
            r'\bgonna\b': 'going to',
            r'\bwanna\b': 'want to',
            r'\bdunno\b': 'do not know',
            r'\bkinda\b': 'kind of',
        }
        
        for pattern, replacement in common_corrections.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        # Suppression des mots rÃ©pÃ©titifs
        words = text.split()
        cleaned_words = []
        prev_word = None
        for word in words:
            if word.lower() != prev_word:
                cleaned_words.append(word)
                prev_word = word.lower()
        text = ' '.join(cleaned_words)
        
        # Normalisation de la ponctuation
        if text and not text[-1] in '.!?':
            text += '.'
        
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    # Application du nettoyage
    df_test_processed['StudentExplanation'] = df_test_processed['StudentExplanation'].apply(clean_student_explanation)
    df_test_processed['QuestionText'] = df_test_processed['QuestionText'].apply(
        lambda x: re.sub(r'\[Image:\s*[^\]]*\]', '[IMAGE]', str(x).strip()) if pd.notna(x) else ""
    )
    df_test_processed['MC_Answer'] = df_test_processed['MC_Answer'].apply(
        lambda x: str(x).strip() if pd.notna(x) else ""
    )
    
    # Traitement des explications vides
    very_short = df_test_processed['StudentExplanation'].str.len() <= 2
    df_test_processed.loc[very_short, 'StudentExplanation'] = "No explanation provided."
    
    print(f"   DonnÃ©es de test prÃ©traitÃ©es: {len(df_test_processed)} Ã©chantillons")
    
    return df_test_processed

# EXÃ‰CUTION DE LA PIPELINE DE PRETRAITEMENT

print("Chargement des donnÃ©es prÃ©traitÃ©es...")

# Application de la pipeline sur les donnÃ©es d'entraÃ®nement
train_df_processed, label_mappings = preprocess_training_data(train_df)

# Application sur les donnÃ©es de test
test_df_processed = preprocess_test_data(test_df, label_mappings)

print(f"\nğŸ“‹ RÃ‰SUMÃ‰ FINAL:")
print(f"â€¢ DonnÃ©es d'entraÃ®nement prÃ©traitÃ©es: {len(train_df_processed):,} Ã©chantillons")
print(f"â€¢ DonnÃ©es de test prÃ©traitÃ©es: {len(test_df_processed):,} Ã©chantillons")
print(f"â€¢ CatÃ©gories disponibles: {len(label_mappings['category_to_id'])}")
print(f"â€¢ Misconceptions disponibles: {len(label_mappings['misconception_to_id'])}")

print(f"\nğŸ�¯ PRÃŠT POUR LA PHASE 3: FEATURE ENGINEERING")

# SAUVEGARDE DES RÃ‰SULTATS POUR UTILISATION ULTÃ‰RIEURE

# Sauvegarder les DataFrames traitÃ©s (simulation - remplacez par vos chemins)
print(f"\nğŸ’¾ SAUVEGARDE DES DONNÃ‰ES PRÃ‰TRAITÃ‰ES:")
print(f"â€¢ train_df_processed: {train_df_processed.shape}")
print(f"â€¢ test_df_processed: {test_df_processed.shape}")
print(f"â€¢ label_mappings: {len(label_mappings)} mappings crÃ©Ã©s")

# Affichage des premiers Ã©chantillons pour vÃ©rification
print(f"\nğŸ”� APERÃ‡U DES DONNÃ‰ES PRÃ‰TRAITÃ‰ES:")
print("Ã‰chantillon des donnÃ©es d'entraÃ®nement:")
print(train_df_processed[['QuestionText', 'StudentExplanation', 'Category', 'Misconception']].head(2))

print(f"\nÃ‰chantillon des donnÃ©es de test:")
print(test_df_processed[['QuestionText', 'StudentExplanation']].head(2))


# 1. Features BasÃ©es sur l'Analyse Textuelle
import re
import string

def add_text_analysis_features(df):
    """
    Ajoute les features textuelles simples :
    - explanation_length : nombre de mots dans StudentExplanation
    - question_length : nombre de mots dans QuestionText
    - spelling_error_rate : proportion de mots avec caractÃ¨res non alphabÃ©tiques
    - lexical_diversity : ratio mots uniques / mots totaux
    - punctuation_count : nombre de signes de ponctuation
    - digits_per_word : ratio de chiffres par mot
    - has_image : boolÃ©en, 1 si QuestionText contient '[IMAGE]'
    - explanation_to_question_length_ratio : longueur explication / longueur question
    """
    
    # Longueurs
    df['explanation_length'] = df['StudentExplanation'].apply(lambda x: len(str(x).split()))
    df['question_length'] = df['QuestionText'].apply(lambda x: len(str(x).split()))
    
    # Taux d'erreurs d'orthographe (approximation simple)
    df['spelling_error_rate'] = df['StudentExplanation'].apply(
        lambda x: sum(1 for w in str(x).split() if not re.match(r"^[A-Za-z]+$", w)) /
                  (len(str(x).split()) + 1e-6)
    )
    
    # DiversitÃ© lexicale
    df['lexical_diversity'] = df['StudentExplanation'].apply(
        lambda x: len(set(str(x).split())) / (len(str(x).split()) + 1e-6)
    )
    
    # Ponctuation
    df['punctuation_count'] = df['StudentExplanation'].apply(
        lambda x: sum(ch in string.punctuation for ch in str(x))
    )
    
    # Ratio chiffres/mots
    df['digits_per_word'] = df['StudentExplanation'].apply(
        lambda x: sum(ch.isdigit() for ch in str(x)) / (len(str(x).split()) + 1e-6)
    )
    
    # PrÃ©sence d'image
    df['has_image'] = df['QuestionText'].apply(lambda x: 1 if "[IMAGE]" in str(x) else 0)
    
    # Ratio explication / question
    df['explanation_to_question_length_ratio'] = df.apply(
        lambda row: row['explanation_length'] / (row['question_length'] + 1e-6),
        axis=1
    )
    
    return df

# Application sur les datasets
df_train = train_df_processed.copy()
df_test = test_df_processed.copy()

df_train = add_text_analysis_features(df_train)
df_test = add_text_analysis_features(df_test)

# VÃ©rification
print("Colonnes ajoutÃ©es :", df_train.columns[-8:].tolist())
print("\nAperÃ§u des statistiques :")
print(df_train[['explanation_length','question_length','spelling_error_rate',
                'lexical_diversity','punctuation_count','digits_per_word',
                'has_image','explanation_to_question_length_ratio']].describe())



# 2. Features BasÃ©es sur lâ€™Analyse MathÃ©matique et SÃ©mantique - Version optimisÃ©e
import re

# --- PrÃ©compilation des regex ---
REGEX_NUMBER = re.compile(r"\b\d+(?:\.\d+)?\b")
REGEX_FRACTION = re.compile(r"\b\d+\s*/\s*\d+\b")

operations_terms = ['add', 'plus', 'sum', 'subtract', 'minus', 'less', 
                    'multiply', 'times', 'product', 'divide', 'quotient']
REGEX_OPERATIONS = re.compile(r'\b(' + '|'.join(map(re.escape, operations_terms)) + r')\b', flags=re.IGNORECASE)

math_keywords = ['fraction', 'decimal', 'simplify', 'equal', 'denominator', 'numerator']
REGEX_MATH_KEYWORDS = re.compile(r'\b(' + '|'.join(map(re.escape, math_keywords)) + r')\b', flags=re.IGNORECASE)

ADDITIVE_PATTERNS = [re.compile(pat, flags=re.IGNORECASE) for pat in [
    r'add.*numerator', r'add.*denominator', r'add across', r'add.*top.*bottom'
]]

def add_math_semantic_features(df):
    """
    Ajoute les features mathÃ©matiques et sÃ©mantiques optimisÃ©es
    """
    # Comptages rapides
    df['numbers_mentioned_count'] = df['StudentExplanation'].str.count(REGEX_NUMBER)
    df['fractions_mentioned_count'] = df['StudentExplanation'].str.count(REGEX_FRACTION)
    df['operations_mentioned_count'] = df['StudentExplanation'].str.count(REGEX_OPERATIONS)
    df['math_keywords_count'] = df['StudentExplanation'].str.count(REGEX_MATH_KEYWORDS)
    
    # CohÃ©rence numÃ©rique (proxy simple)
    def numerical_consistency(exp, mc):
        nums_exp = set(REGEX_NUMBER.findall(str(exp)))
        nums_mc = set(REGEX_NUMBER.findall(str(mc)))
        if not nums_mc:
            return 0
        return len(nums_exp.intersection(nums_mc)) / len(nums_mc)
    
    df['numerical_consistency_score'] = [
        numerical_consistency(e, m) for e, m in zip(df['StudentExplanation'], df['MC_Answer'])
    ]
    
    # Mention valeur exacte
    df['mentions_correct_value'] = (df.apply(
        lambda row: str(row['MC_Answer']).strip() in str(row['StudentExplanation']), axis=1
    ).astype(int))
    
    # Fraction inversÃ©e
    def is_inverted_fraction(exp, mc):
        frac_mc = REGEX_FRACTION.findall(str(mc))
        frac_exp = REGEX_FRACTION.findall(str(exp))
        if frac_mc and frac_exp:
            mc_num, mc_den = frac_mc[0]
            for num, den in frac_exp:
                if num == mc_den and den == mc_num:
                    return 1
        return 0
    
    df['mentions_inverted_fraction'] = [
        is_inverted_fraction(e, m) for e, m in zip(df['StudentExplanation'], df['MC_Answer'])
    ]
    
    # Erreur additive sur fractions
    df['additive_fraction_error'] = df['StudentExplanation'].apply(
        lambda x: any(pat.search(str(x)) for pat in ADDITIVE_PATTERNS)
    ).astype(int)
    
    # Ratio opÃ©rations / nombres
    df['operations_per_number'] = df['operations_mentioned_count'] / (df['numbers_mentioned_count'] + 1e-6)
    
    return df

# --- Application ---
df_train = add_math_semantic_features(df_train)
df_test = add_math_semantic_features(df_test)

# VÃ©rification
print("Colonnes ajoutÃ©es :", df_train.columns[-9:].tolist())
print("\nAperÃ§u des statistiques :")
print(df_train[['numbers_mentioned_count','fractions_mentioned_count',
                'operations_mentioned_count','math_keywords_count',
                'numerical_consistency_score','mentions_correct_value',
                'mentions_inverted_fraction','additive_fraction_error',
                'operations_per_number']].describe())



# Partie 3 â€“ Features dâ€™Interaction et de Contexte

import numpy as np
from sklearn.model_selection import KFold

def add_interaction_context_features(df_train, df_test, target_col="Category", n_splits=5, seed=42):
    """
    Ajoute les features d'interaction et de contexte :
    - misconception_per_question_rate : taux moyen de misconceptions par QuestionId (train uniquement)
    - QuestionId_target_encoding : encodage K-fold des probas de chaque classe pour QuestionId
    - has_image_x_fraction_terms : interaction entre image et mention de fractions
    """

    # 1. Taux de misconceptions par QuestionId (tout sauf "True_Correct")
    misconception_rate = (
        df_train.groupby('QuestionId')[target_col]
        .apply(lambda x: np.mean(x != "True_Correct"))
        .to_dict()
    )
    df_train['misconception_per_question_rate'] = df_train['QuestionId'].map(misconception_rate)
    df_test['misconception_per_question_rate'] = df_test['QuestionId'].map(misconception_rate).fillna(0.0)

    # 2. Target encoding KFold (probas par classe de Category)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    classes = df_train[target_col].unique()

    # Initialiser les colonnes
    for c in classes:
        col_name = f"qid_te_{c}"
        df_train[col_name] = 0.0
        df_test[col_name] = 0.0

    # Encodage out-of-fold
    for train_idx, val_idx in kf.split(df_train):
        fold_train, fold_val = df_train.iloc[train_idx], df_train.iloc[val_idx]
        te = (
            fold_train.groupby("QuestionId")[target_col]
            .value_counts(normalize=True)
            .unstack()
            .fillna(0)
        )
        for c in classes:
            col_name = f"qid_te_{c}"
            df_train.loc[val_idx, col_name] = df_train.loc[val_idx, "QuestionId"].map(te[c]).fillna(0)

    # Sur test : mapping depuis tout train
    te_full = (
        df_train.groupby("QuestionId")[target_col]
        .value_counts(normalize=True)
        .unstack()
        .fillna(0)
    )
    for c in classes:
        col_name = f"qid_te_{c}"
        df_test[col_name] = df_test["QuestionId"].map(te_full[c]).fillna(0)

    # 3. Interaction : has_image Ã— fractions
    df_train['has_image_x_fraction_terms'] = df_train['has_image'] * (df_train['fractions_mentioned_count'] > 0).astype(int)
    df_test['has_image_x_fraction_terms'] = df_test['has_image'] * (df_test['fractions_mentioned_count'] > 0).astype(int)

    return df_train, df_test


# --- Application ---
df_train, df_test = add_interaction_context_features(df_train, df_test, target_col="Category")

# VÃ©rification
cols_added = [c for c in df_train.columns if c.startswith("qid_te_")] + [
    "misconception_per_question_rate", "has_image_x_fraction_terms"
]
print("Colonnes ajoutÃ©es :", cols_added)
print("\nAperÃ§u des statistiques :")
print(df_train[cols_added].describe())



# Partie 4 â€“ Features BasÃ©es sur les Embeddings (TF-IDF + SimilaritÃ©)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def add_embedding_features(df_train, df_test, max_features=20000, ngram_range=(1,3)):
    """
    Ajoute des features basÃ©es sur TF-IDF :
    - TF-IDF explications
    - TF-IDF questions
    - SimilaritÃ© cosinus entre explication et question
    """

    # 1. On crÃ©e un corpus global (questions + explications)
    all_texts = pd.concat([
        df_train['StudentExplanation'], df_test['StudentExplanation'],
        df_train['QuestionText'], df_test['QuestionText']
    ]).astype(str)

    # 2. EntraÃ®nement d'un seul TF-IDF
    tfidf = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)
    tfidf.fit(all_texts)

    # 3. Transformations
    X_train_tfidf_exp = tfidf.transform(df_train['StudentExplanation'].astype(str))
    X_test_tfidf_exp  = tfidf.transform(df_test['StudentExplanation'].astype(str))
    X_train_tfidf_qst = tfidf.transform(df_train['QuestionText'].astype(str))
    X_test_tfidf_qst  = tfidf.transform(df_test['QuestionText'].astype(str))

    # 4. SimilaritÃ© cosinus explication vs question
    sim_train = cosine_similarity(X_train_tfidf_exp, X_train_tfidf_qst).diagonal()
    sim_test  = cosine_similarity(X_test_tfidf_exp, X_test_tfidf_qst).diagonal()

    df_train['cosine_similarity_exp_qst'] = sim_train
    df_test['cosine_similarity_exp_qst']  = sim_test

    return df_train, df_test, X_train_tfidf_exp, X_test_tfidf_exp, X_train_tfidf_qst, X_test_tfidf_qst


# --- Application ---
df_train, df_test, X_train_tfidf_exp, X_test_tfidf_exp, X_train_tfidf_qst, X_test_tfidf_qst = add_embedding_features(
    df_train, df_test, max_features=20000, ngram_range=(1,3)
)

# VÃ©rification
print("Colonne ajoutÃ©e : cosine_similarity_exp_qst")
print(df_train['cosine_similarity_exp_qst'].describe())
print("Shape TF-IDF explications :", X_train_tfidf_exp.shape)
print("Shape TF-IDF questions    :", X_train_tfidf_qst.shape)


# Partie 5 â€“ Features SÃ©mantiques avancÃ©es (Sentence-BERT)

from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def add_sbert_embeddings(df_train, df_test, model_path="/kaggle/input/all-minilm-l6-v2/all-MiniLM-L6-v2", batch_size=256):
    """
    Ajoute des embeddings sÃ©mantiques issus de Sentence-BERT :
    - Embeddings explications
    - Embeddings questions
    - SimilaritÃ© cosinus explication-question
    """

    # 1. Chargement du modÃ¨le depuis le dataset Kaggle (offline)
    model = SentenceTransformer(model_path)

    # 2. GÃ©nÃ©ration des embeddings
    emb_train_exp = model.encode(df_train['StudentExplanation'].astype(str).tolist(), 
                                 batch_size=batch_size, show_progress_bar=True)
    emb_test_exp  = model.encode(df_test['StudentExplanation'].astype(str).tolist(), 
                                 batch_size=batch_size, show_progress_bar=True)

    emb_train_qst = model.encode(df_train['QuestionText'].astype(str).tolist(), 
                                 batch_size=batch_size, show_progress_bar=True)
    emb_test_qst  = model.encode(df_test['QuestionText'].astype(str).tolist(), 
                                 batch_size=batch_size, show_progress_bar=True)

    # 3. SimilaritÃ© cosinus explication-question
    sim_train = cosine_similarity(emb_train_exp, emb_train_qst).diagonal()
    sim_test  = cosine_similarity(emb_test_exp, emb_test_qst).diagonal()

    # Ajout au dataframe
    df_train['sbert_cosine_similarity_exp_qst'] = sim_train
    df_test['sbert_cosine_similarity_exp_qst']  = sim_test

    # 4. Retourne aussi les matrices dâ€™embeddings (optionnel pour NN)
    return df_train, df_test, emb_train_exp, emb_test_exp, emb_train_qst, emb_test_qst


# --- Application ---
df_train, df_test, emb_train_exp, emb_test_exp, emb_train_qst, emb_test_qst = add_sbert_embeddings(
    df_train, df_test, model_path="/kaggle/input/all-minilm-l6-v2/all-MiniLM-L6-v2", batch_size=256
)

# VÃ©rification
print("Colonne ajoutÃ©e : sbert_cosine_similarity_exp_qst")
print(df_train['sbert_cosine_similarity_exp_qst'].describe())
print("Shape embeddings explications :", np.array(emb_train_exp).shape)
print("Shape embeddings questions    :", np.array(emb_train_qst).shape)



# Feature additionnelle : QuestionId Frequency
# Objectif : capturer combien de fois une question apparaÃ®t 
#            dans lâ€™ensemble dâ€™entraÃ®nement et reporter cette info

def add_question_id_frequency(df_train, df_test):
    """Ajoute une feature reprÃ©sentant la frÃ©quence d'apparition 
    de chaque QuestionId dans le dataset train."""
    
    # Calcul des frÃ©quences dans le train
    qid_freq = df_train['QuestionId'].value_counts().to_dict()
    
    # Ajout sur train
    df_train['question_id_frequency'] = df_train['QuestionId'].map(qid_freq)
    
    # Pour le test : si un QuestionId nâ€™existe pas dans le train, on met 0
    df_test['question_id_frequency'] = df_test['QuestionId'].map(qid_freq).fillna(0)
    
    return df_train, df_test


# --- Application ---
df_train, df_test = add_question_id_frequency(df_train, df_test)

# VÃ©rification
print("Colonne ajoutÃ©e : question_id_frequency")
print(df_train['question_id_frequency'].describe())
print(f"Exemple valeurs test : {df_test['question_id_frequency'].tolist()}")



import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. Analyse de la distribution des classes ---
def analyze_class_distribution(df_train):
    print("\n--- Distribution Category ---")
    print(df_train["Category"].value_counts(dropna=False))

    print("\n--- Distribution Misconception (y compris NaN) ---")
    print(df_train["Misconception"].value_counts(dropna=False).head(20))  # top 20

    # Heatmap Category vs Misconception
    plt.figure(figsize=(12, 6))
    cross_tab = pd.crosstab(df_train["Category"], df_train["Misconception"])
    sns.heatmap(cross_tab, cmap="Blues", cbar=True)
    plt.title("RÃ©partition Category x Misconception")
    plt.show()

# --- 2. Analyse de corrÃ©lation entre features numÃ©riques ---
def analyze_correlations(df_train):
    # SÃ©lectionner seulement les colonnes numÃ©riques
    numeric_cols = df_train.select_dtypes(include=["int64", "float64"]).columns
    
    corr = df_train[numeric_cols].corr(method="pearson")
    
    plt.figure(figsize=(14, 10))
    sns.heatmap(corr, cmap="coolwarm", center=0)
    plt.title("Matrice de corrÃ©lation (Pearson) des features numÃ©riques")
    plt.show()
    
    return corr

# --- Application ---
analyze_class_distribution(df_train)
corr_matrix = analyze_correlations(df_train)



# ===============================
# Nettoyage des features
# ===============================

# 1. Fusion des colonnes qid_te_* en une seule variable catÃ©gorielle
qid_te_cols = [
    'qid_te_True_Correct',
    'qid_te_True_Neither',
    'qid_te_True_Misconception',
    'qid_te_False_Neither',
    'qid_te_False_Misconception',
    'qid_te_False_Correct'
]

def fuse_qid_te(row):
    for col in qid_te_cols:
        if row[col] == 1:
            return col
    return "unknown"

df_train['qid_te_label'] = df_train[qid_te_cols].apply(fuse_qid_te, axis=1)
df_test['qid_te_label']  = df_test[qid_te_cols].apply(fuse_qid_te, axis=1)

# 2. Colonnes Ã  supprimer
cols_to_drop = [
    # Redondantes
    'numbers_mentioned', 'fractions_mentioned', 'operations_mentioned',
    'explanation_to_question_length_ratio', 'numerical_consistency',

    # Encodages labels
    'Category_encoded', 'Misconception_encoded',

    # Features trop spÃ©cifiques
    'mentions_correct_value', 'mentions_inverted_fraction', 'additive_fraction_error',
    
    # One-hot remplacÃ©s par la fusion
] + qid_te_cols  # ajoute la liste des qid_te_*

# Suppression
df_train = df_train.drop(columns=[c for c in cols_to_drop if c in df_train.columns], errors="ignore")
df_test = df_test.drop(columns=[c for c in cols_to_drop if c in df_test.columns], errors="ignore")

# VÃ©rif du rÃ©sultat
print("Features finales df_train :", df_train.shape, df_train.columns.tolist())
print("Features finales df_test  :", df_test.shape, df_test.columns.tolist())


