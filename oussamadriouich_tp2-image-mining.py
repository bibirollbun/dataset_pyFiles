import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.datasets import cifar10
from skimage.feature import hog
from skimage import color, exposure
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import warnings

# Ignorer les avertissements pour une sortie plus propre
warnings.filterwarnings('ignore')

# --- 1. Chargement et Filtrage des Données (CIFAR-10) ---

print("Étape 1 : Chargement et filtrage des données...")

# Définition des classes cibles
# Classe 2 = Oiseau (Bird)
# Classe 9 = Camion (Truck)
CLASSES_CIBLES = [2, 9]
labels_noms = {2: 'Oiseau', 9: 'Camion'}

# Chargement du dataset
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# Aplatir y pour faciliter le filtrage
y_train = y_train.flatten()
y_test = y_test.flatten()

# Filtrage pour ne garder que les Oiseaux (2) et les Camions (9)
masque_train = np.isin(y_train, CLASSES_CIBLES)
masque_test = np.isin(y_test, CLASSES_CIBLES)

X_train_filtre = X_train[masque_train]
y_train_filtre = y_train[masque_train]
X_test_filtre = X_test[masque_test]
y_test_filtre = y_test[masque_test]

print(f"Données d'entraînement : {len(X_train_filtre)} images")
print(f"Données de validation : {len(X_test_filtre)} images")


# --- 2. Extraction des Caractéristiques (HOG) ---

print("\nÉtape 2 : Extraction des caractéristiques HOG...")

def extraire_hog(images):
    features_list = []
    for image in images:
        # Convertir l'image en niveaux de gris
        image_gray = color.rgb2gray(image)
        
        # Calculer les caractéristiques HOG
        # Les paramètres (pixels_per_cell, cells_per_block) sont cruciaux
        features = hog(image_gray, pixels_per_cell=(8, 8),
                       cells_per_block=(2, 2), visualize=False,
                       block_norm='L2-Hys')
        features_list.append(features)
    return np.array(features_list)

X_train_hog = extraire_hog(X_train_filtre)
X_test_hog = extraire_hog(X_test_filtre)

print(f"Taille du vecteur de caractéristiques HOG : {X_train_hog.shape[1]}")


# --- 3. Prétraitement (Mise à l'échelle) ---

print("\nÉtape 3 : Mise à l'échelle des caractéristiques...")

# Le scaling est crucial pour SVM et KNN
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_hog)
X_test_scaled = scaler.transform(X_test_hog)


# --- 4. Partie 1 : Modèle SVM (Entraînement, Prédiction, Évaluation) ---

print("\n--- Partie 1 : Modèle SVM ---")

# 4.1. Entraînement
print("Entraînement du modèle SVM (Noyau RBF)...")
# Nous utilisons un noyau RBF (Gaussien) qui est souvent performant
svm_model = SVC(kernel='rbf', C=10, random_state=42)
svm_model.fit(X_train_scaled, y_train_filtre)

# 4.2. Prédiction sur 2 images de validation
print("\nPrédiction sur 2 images de validation :")
images_a_predire = X_test_scaled[:2]
labels_reels = y_test_filtre[:2]
predictions = svm_model.predict(images_a_predire)

for i in range(2):
    label_pred = labels_noms[predictions[i]]
    label_reel = labels_noms[labels_reels[i]]
    print(f"  Image {i+1} : Prédit='{label_pred}', Réel='{label_reel}'")

# 4.3. Évaluation complète et Matrice de Confusion
print("\nÉvaluation complète du modèle SVM :")
y_pred_svm = svm_model.predict(X_test_scaled)

# Définir l'ordre des labels : Camion (Négatif), Oiseau (Positif)
# pos_label=2 signifie que 'Oiseau' est notre classe positive
labels_ordre = [9, 2] 

cm = confusion_matrix(y_test_filtre, y_pred_svm, labels=labels_ordre)

print("Matrice de Confusion :")
print(f"         Prédit: Camion | Prédit: Oiseau")
print(f"Réel: Camion   {cm[0][0]:<13} | {cm[0][1]:<13}")
print(f"Réel: Oiseau   {cm[1][0]:<13} | {cm[1][1]:<13}")

# Extraction des TP, TN, FP, FN
# Classe Positive = Oiseau (2)
# Classe Négative = Camion (9)
VN, FP, FN, VP = cm.ravel()

print(f"\nValeurs extraites (Positif = Oiseau):")
print(f"  Vrais Positifs (VP) : {VP} (Oiseaux correctement identifiés)")
print(f"  Vrais Négatifs (VN) : {VN} (Camions correctement identifiés)")
print(f"  Faux Positifs (FP)  : {FP} (Camions pris pour des Oiseaux)")
print(f"  Faux Négatifs (FN)  : {FN} (Oiseaux pris pour des Camions)")

# 4.4. Calcul des Métriques
# Nous utilisons pos_label=2 pour nous assurer que Précision et Rappel
# sont calculés pour la classe "Oiseau".
accuracy = accuracy_score(y_test_filtre, y_pred_svm)
precision = precision_score(y_test_filtre, y_pred_svm, pos_label=2)
recall = recall_score(y_test_filtre, y_pred_svm, pos_label=2)
f1 = f1_score(y_test_filtre, y_pred_svm, pos_label=2)

print("\nMétriques de performance :")
print(f"  Accuracy (Exactitude) : {accuracy:.4f}")
print(f"  Précision (Oiseau)    : {precision:.4f}")
print(f"  Rappel (Oiseau)       : {recall:.4f}")
print(f"  F-mesure (Oiseau)     : {f1:.4f}")


# --- 5. Partie 2 : Étude Comparative (SVM, KNN, DT) ---

print("\n\n--- Partie 2 : Étude Comparative ---")

# Définition des modèles et hyperparamètres à tester
modeles = {
    "SVM (Linéaire, C=1)": SVC(kernel='linear', C=1, random_state=42),
    "SVM (RBF, C=1)": SVC(kernel='rbf', C=1, random_state=42),
    "SVM (RBF, C=10)": SVC(kernel='rbf', C=10, random_state=42),
    "KNN (k=3)": KNeighborsClassifier(n_neighbors=3),
    "KNN (k=7)": KNeighborsClassifier(n_neighbors=7),
    "KNN (k=11)": KNeighborsClassifier(n_neighbors=11),
    "Arbre (Prof=5)": DecisionTreeClassifier(max_depth=5, random_state=42),
    "Arbre (Prof=10)": DecisionTreeClassifier(max_depth=10, random_state=42),
    "Arbre (Non limité)": DecisionTreeClassifier(random_state=42)
}

resultats = []

print("Entraînement et évaluation des différents modèles...")

for nom, modele in modeles.items():
    # Entraînement
    modele.fit(X_train_scaled, y_train_filtre)
    
    # Prédiction
    y_pred = modele.predict(X_test_scaled)
    
    # Évaluation (toujours avec 'Oiseau' (2) comme classe positive)
    acc = accuracy_score(y_test_filtre, y_pred)
    prec = precision_score(y_test_filtre, y_pred, pos_label=2)
    rapp = recall_score(y_test_filtre, y_pred, pos_label=2)
    
    resultats.append({
        "Modèle": nom,
        "Accuracy": acc,
        "Précision (Oiseau)": prec,
        "Rappel (Oiseau)": rapp
    })

# Affichage du tableau des résultats
resultats_df = pd.DataFrame(resultats)
resultats_df = resultats_df.set_index("Modèle")

print("\nTableau comparatif des résultats :")
print(resultats_df.to_string(float_format="%.4f"))


"""
Classification Supervisée CIFAR-10 (Oiseau vs Camion)
Utilisation des caractéristiques CBIR avec SVM, KNN, et Decision Tree
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from skimage.feature import graycomatrix, graycoprops
import pandas as pd
import seaborn as sns
from tensorflow.keras.datasets import cifar10

# ============================================================================
# 1. CHARGEMENT ET PRÉPARATION DES DONNÉES CIFAR-10
# ============================================================================

def load_cifar10_bird_truck():
    """
    Charge CIFAR-10 et filtre les classes Oiseau (2) et Camion (9)
    """
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    
    # Filtrer les classes: Oiseau (2) et Camion (9)
    train_mask = np.isin(y_train, [2, 9]).flatten()
    test_mask = np.isin(y_test, [2, 9]).flatten()
    
    x_train_filtered = x_train[train_mask]
    y_train_filtered = y_train[train_mask].flatten()
    x_test_filtered = x_test[test_mask]
    y_test_filtered = y_test[test_mask].flatten()
    
    # Convertir les labels: Oiseau=0, Camion=1
    y_train_filtered = (y_train_filtered == 9).astype(int)
    y_test_filtered = (y_test_filtered == 9).astype(int)
    
    print(f"Train set: {x_train_filtered.shape[0]} images")
    print(f"Test set: {x_test_filtered.shape[0]} images")
    print(f"Classes: Oiseau (0), Camion (1)")
    
    return x_train_filtered, y_train_filtered, x_test_filtered, y_test_filtered

# ============================================================================
# 2. EXTRACTION DES CARACTÉRISTIQUES CBIR
# ============================================================================

def extract_color_moments(image):
    """
    A. Couleurs 1: Moments statistiques (Moyenne et Std) pour R, G, B
    Retourne 6 caractéristiques
    """
    features = []
    for i in range(3):  # R, G, B
        channel = image[:, :, i]
        mean = np.mean(channel)
        std = np.std(channel)
        features.extend([mean, std])
    return np.array(features)

def extract_hsv_histogram(image, h_bins=8, s_bins=2, v_bins=2):
    """
    B. Couleurs 2: Histogramme quantifié HSV
    H=8 bins, S=2 bins, V=2 bins => 32 caractéristiques
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    
    # Calculer l'histogramme 3D
    hist = cv2.calcHist([hsv], [0, 1, 2], None, 
                        [h_bins, s_bins, v_bins],
                        [0, 180, 0, 256, 0, 256])
    
    # Normaliser
    hist = hist.flatten()
    hist = hist / (hist.sum() + 1e-7)
    
    return hist

def extract_texture_features(image):
    """
    C. Texture: GLCM (Contrast, Correlation, Energy, Homogeneity)
    Retourne 4 caractéristiques
    """
    # Convertir en niveaux de gris
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Calculer GLCM
    glcm = graycomatrix(gray, distances=[1], angles=[0], levels=256,
                        symmetric=True, normed=True)
    
    # Extraire les propriétés
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    correlation = graycoprops(glcm, 'correlation')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    
    return np.array([contrast, correlation, energy, homogeneity])

def extract_hu_moments(image):
    """
    D. Forme: 7 moments invariants de Hu
    Retourne 7 caractéristiques
    """
    # Convertir en niveaux de gris
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Calculer les moments de Hu
    moments = cv2.HuMoments(cv2.moments(gray)).flatten()
    
    # Appliquer log pour normaliser
    moments = -np.sign(moments) * np.log10(np.abs(moments) + 1e-10)
    
    return moments

def extract_all_features(image):
    """
    Extraire et concaténer toutes les caractéristiques
    Total: 6 + 32 + 4 + 7 = 49 caractéristiques
    """
    feat1 = extract_color_moments(image)           # 6 features
    feat2 = extract_hsv_histogram(image)           # 32 features
    feat3 = extract_texture_features(image)        # 4 features
    feat4 = extract_hu_moments(image)              # 7 features
    
    return np.concatenate([feat1, feat2, feat3, feat4])

def extract_features_dataset(images):
    """
    Extraire les caractéristiques pour un ensemble d'images
    """
    features_list = []
    for i, img in enumerate(images):
        if i % 500 == 0:
            print(f"Extraction: {i}/{len(images)}")
        features = extract_all_features(img)
        features_list.append(features)
    return np.array(features_list)

# ============================================================================
# 3. ENTRAÎNEMENT ET ÉVALUATION
# ============================================================================

def evaluate_model(y_true, y_pred, model_name):
    """
    Évaluer un modèle et afficher les métriques
    """
    # Matrice de confusion
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    # Métriques
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    print(f"\n{'='*60}")
    print(f"Résultats pour {model_name}")
    print(f"{'='*60}")
    print(f"Matrice de confusion:")
    print(f"  TN: {tn:5d}  |  FP: {fp:5d}")
    print(f"  FN: {fn:5d}  |  TP: {tp:5d}")
    print(f"\nMétriques:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F-measure: {f1:.4f}")
    
    return {
        'Model': model_name,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'TN': tn, 'FP': fp, 'FN': fn, 'TP': tp
    }

def plot_confusion_matrix(y_true, y_pred, model_name):
    """
    Afficher la matrice de confusion
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Oiseau', 'Camion'],
                yticklabels=['Oiseau', 'Camion'])
    plt.title(f'Matrice de Confusion - {model_name}')
    plt.ylabel('Vraie Classe')
    plt.xlabel('Classe Prédite')
    plt.tight_layout()
    plt.show()

# ============================================================================
# 4. PROGRAMME PRINCIPAL
# ============================================================================

def main():
    print("Chargement des données CIFAR-10...")
    x_train, y_train, x_test, y_test = load_cifar10_bird_truck()
    
    # Créer un ensemble de validation
    x_train, x_val, y_train, y_val = train_test_split(
        x_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    print(f"\nEnsembles finaux:")
    print(f"  Train: {x_train.shape[0]} images")
    print(f"  Validation: {x_val.shape[0]} images")
    print(f"  Test: {x_test.shape[0]} images")
    
    # Extraction des caractéristiques
    print("\n" + "="*60)
    print("EXTRACTION DES CARACTÉRISTIQUES")
    print("="*60)
    
    print("\nExtraction pour l'ensemble d'entraînement...")
    X_train = extract_features_dataset(x_train)
    
    print("\nExtraction pour l'ensemble de validation...")
    X_val = extract_features_dataset(x_val)
    
    print("\nExtraction pour l'ensemble de test...")
    X_test = extract_features_dataset(x_test)
    
    print(f"\nDimension des features: {X_train.shape[1]}")
    
    # Normalisation
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # ========================================================================
    # PARTIE 1: ENTRAÎNEMENT SVM ET PRÉDICTION
    # ========================================================================
    
    print("\n" + "="*60)
    print("PARTIE 1: ENTRAÎNEMENT DU MODÈLE SVM")
    print("="*60)
    
    svm_model = SVC(kernel='rbf', C=10, gamma='scale', random_state=42)
    svm_model.fit(X_train_scaled, y_train)
    
    # Prédire sur 2 images de validation
    print("\nPrédiction sur 2 images de validation:")
    for i in range(2):
        pred = svm_model.predict(X_val_scaled[i:i+1])[0]
        true = y_val[i]
        pred_label = "Camion" if pred == 1 else "Oiseau"
        true_label = "Camion" if true == 1 else "Oiseau"
        print(f"  Image {i+1}: Prédiction={pred_label}, Vraie classe={true_label}")
        
        # Afficher l'image
        plt.figure(figsize=(3, 3))
        plt.imshow(x_val[i])
        plt.title(f'Pred: {pred_label} | True: {true_label}')
        plt.axis('off')
        plt.show()
    
    # Évaluation sur validation
    y_val_pred = svm_model.predict(X_val_scaled)
    svm_results = evaluate_model(y_val, y_val_pred, "SVM (RBF, C=10)")
    plot_confusion_matrix(y_val, y_val_pred, "SVM")
    
    # ========================================================================
    # PARTIE 2: ÉTUDE COMPARATIVE
    # ========================================================================
    
    print("\n" + "="*60)
    print("PARTIE 2: ÉTUDE COMPARATIVE DES ALGORITHMES")
    print("="*60)
    
    results_list = [svm_results]
    
    # Définir les modèles à tester
    models_to_test = [
        # SVM avec différents hyperparamètres
        ('SVM (Linear, C=1)', SVC(kernel='linear', C=1, random_state=42)),
        ('SVM (Linear, C=10)', SVC(kernel='linear', C=10, random_state=42)),
        ('SVM (RBF, C=1)', SVC(kernel='rbf', C=1, gamma='scale', random_state=42)),
        ('SVM (RBF, C=100)', SVC(kernel='rbf', C=100, gamma='scale', random_state=42)),
        ('SVM (Poly, C=10)', SVC(kernel='poly', degree=3, C=10, random_state=42)),
        
        # KNN avec différents k
        ('KNN (k=3)', KNeighborsClassifier(n_neighbors=3)),
        ('KNN (k=5)', KNeighborsClassifier(n_neighbors=5)),
        ('KNN (k=7)', KNeighborsClassifier(n_neighbors=7)),
        ('KNN (k=10, weighted)', KNeighborsClassifier(n_neighbors=10, weights='distance')),
        
        # Decision Tree avec différents paramètres
        ('DT (max_depth=5)', DecisionTreeClassifier(max_depth=5, random_state=42)),
        ('DT (max_depth=10)', DecisionTreeClassifier(max_depth=10, random_state=42)),
        ('DT (max_depth=20)', DecisionTreeClassifier(max_depth=20, random_state=42)),
        ('DT (min_samples=10)', DecisionTreeClassifier(min_samples_split=10, random_state=42)),
        ('DT (entropy)', DecisionTreeClassifier(criterion='entropy', max_depth=15, random_state=42)),
    ]
    
    # Entraîner et évaluer chaque modèle
    for model_name, model in models_to_test:
        print(f"\nEntraînement: {model_name}...")
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_val_scaled)
        results = evaluate_model(y_val, y_pred, model_name)
        results_list.append(results)
    
    # ========================================================================
    # PARTIE 3: TABLEAU COMPARATIF
    # ========================================================================
    
    print("\n" + "="*60)
    print("TABLEAU COMPARATIF DES RÉSULTATS")
    print("="*60)
    
    df_results = pd.DataFrame(results_list)
    df_results = df_results.sort_values('Accuracy', ascending=False)
    
    # Afficher le tableau
    print("\n" + df_results.to_string(index=False))
    
    # Visualisation des résultats
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    metrics = ['Accuracy', 'Precision', 'Recall']
    colors = ['#2ecc71', '#3498db', '#e74c3c']
    
    for idx, (metric, color) in enumerate(zip(metrics, colors)):
        ax = axes[idx]
        data = df_results.sort_values(metric, ascending=True)
        ax.barh(range(len(data)), data[metric], color=color, alpha=0.7)
        ax.set_yticks(range(len(data)))
        ax.set_yticklabels(data['Model'], fontsize=8)
        ax.set_xlabel(metric, fontsize=12, fontweight='bold')
        ax.set_xlim([0, 1])
        ax.grid(axis='x', alpha=0.3)
        
        # Ajouter les valeurs
        for i, v in enumerate(data[metric]):
            ax.text(v + 0.01, i, f'{v:.3f}', va='center', fontsize=8)
    
    plt.suptitle('Comparaison des Performances des Modèles', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    # ========================================================================
    # PARTIE 4: DISCUSSION DES RÉSULTATS
    # ========================================================================
    
    print("\n" + "="*60)
    print("DISCUSSION DES RÉSULTATS")
    print("="*60)
    
    best_model = df_results.iloc[0]
    worst_model = df_results.iloc[-1]
    
    print(f"\n1. MEILLEUR MODÈLE:")
    print(f"   {best_model['Model']} avec Accuracy = {best_model['Accuracy']:.4f}")
    
    print(f"\n2. ANALYSE PAR ALGORITHME:")
    
    # SVM
    svm_models = df_results[df_results['Model'].str.contains('SVM')]
    print(f"\n   a) SVM:")
    print(f"      - Meilleure performance: {svm_models.iloc[0]['Model']}")
    print(f"      - Accuracy moyenne: {svm_models['Accuracy'].mean():.4f}")
    print(f"      - Le noyau RBF performe généralement mieux que linéaire")
    
    # KNN
    knn_models = df_results[df_results['Model'].str.contains('KNN')]
    print(f"\n   b) KNN:")
    print(f"      - Meilleure performance: {knn_models.iloc[0]['Model']}")
    print(f"      - Accuracy moyenne: {knn_models['Accuracy'].mean():.4f}")
    print(f"      - Sensible au choix de k et à la pondération")
    
    # Decision Tree
    dt_models = df_results[df_results['Model'].str.contains('DT')]
    print(f"\n   c) Decision Tree:")
    print(f"      - Meilleure performance: {dt_models.iloc[0]['Model']}")
    print(f"      - Accuracy moyenne: {dt_models['Accuracy'].mean():.4f}")
    print(f"      - Profondeur optimale importante pour éviter sur-apprentissage")
    
    print(f"\n3. OBSERVATIONS GÉNÉRALES:")
    print(f"   - Les caractéristiques CBIR sont efficaces pour cette tâche")
    print(f"   - SVM et KNN montrent des performances similaires")
    print(f"   - La normalisation des features est cruciale")
    print(f"   - Les hyperparamètres ont un impact significatif")
    
    print("\n" + "="*60)
    print("FIN DE L'ANALYSE")
    print("="*60)

if __name__ == "__main__":
    main()




