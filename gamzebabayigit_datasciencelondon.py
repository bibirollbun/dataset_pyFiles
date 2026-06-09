import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
import warnings
warnings.filterwarnings('ignore')

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
 


train = pd.read_csv('/kaggle/input/data-science-london-scikit-learn/train.csv', header=None)
train_labels = pd.read_csv('/kaggle/input/data-science-london-scikit-learn/trainLabels.csv', header=None)
test = pd.read_csv('/kaggle/input/data-science-london-scikit-learn/test.csv', header=None)


print('train shape:', train.shape)
print('test shape:', test.shape)
print('trainLabel shape:', train_labels.shape)
train.head()


train.info()


train.describe().T


X = train.values
y = train_labels.values.ravel()
X_test = test.values


# Veri ön işleme - Standartlaştırma
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)


# Cross-validation ayarları
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


# Model 1: Random Forest
print("\n1. Random Forest eğitiliyor...")
rf = RandomForestClassifier(
    n_estimators=500,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features='sqrt',
    random_state=42,
    n_jobs=-1
)
rf_scores = cross_val_score(rf, X_scaled, y, cv=cv, scoring='accuracy')
print(f"RF CV Accuracy: {rf_scores.mean():.4f} (+/- {rf_scores.std():.4f})")


# Model 2: Gradient Boosting
print("\n2. Gradient Boosting eğitiliyor...")
gb = GradientBoostingClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=5,
    min_samples_split=5,
    min_samples_leaf=2,
    subsample=0.8,
    random_state=42
)
gb_scores = cross_val_score(gb, X_scaled, y, cv=cv, scoring='accuracy')
print(f"GB CV Accuracy: {gb_scores.mean():.4f} (+/- {gb_scores.std():.4f})")


# Model 3: Support Vector Machine
print("\n3. SVM eğitiliyor...")
svm = SVC(
    C=10,
    kernel='rbf',
    gamma='scale',
    probability=True,
    random_state=42
)
svm_scores = cross_val_score(svm, X_scaled, y, cv=cv, scoring='accuracy')
print(f"SVM CV Accuracy: {svm_scores.mean():.4f} (+/- {svm_scores.std():.4f})")


# Model 4: Logistic Regression
print("\n4. Logistic Regression eğitiliyor...")
lr = LogisticRegression(
    C=1.0,
    penalty='l2',
    max_iter=1000,
    random_state=42
)
lr_scores = cross_val_score(lr, X_scaled, y, cv=cv, scoring='accuracy')
print(f"LR CV Accuracy: {lr_scores.mean():.4f} (+/- {lr_scores.std():.4f})")


# Model 5: Neural Network
print("\n5. Neural Network eğitiliyor...")
mlp = MLPClassifier(
    hidden_layer_sizes=(100, 50),
    activation='relu',
    solver='adam',
    alpha=0.0001,
    learning_rate='adaptive',
    max_iter=1000,
    random_state=42
)
mlp_scores = cross_val_score(mlp, X_scaled, y, cv=cv, scoring='accuracy')
print(f"MLP CV Accuracy: {mlp_scores.mean():.4f} (+/- {mlp_scores.std():.4f})")


# Ensemble Model (Voting Classifier)
print("\n6. Ensemble Model (Voting) oluşturuluyor...")
ensemble = VotingClassifier(
    estimators=[
        ('rf', rf),
        ('gb', gb),
        ('svm', svm),
        ('lr', lr),
        ('mlp', mlp)
    ],
    voting='soft',
    n_jobs=-1
)

ensemble_scores = cross_val_score(ensemble, X_scaled, y, cv=cv, scoring='accuracy')
print(f"Ensemble CV Accuracy: {ensemble_scores.mean():.4f} (+/- {ensemble_scores.std():.4f})")


# En iyi modeli seç ve tüm veri ile eğit
print("\n7. En iyi model tam veri ile eğitiliyor...")
best_model = ensemble
best_model.fit(X_scaled, y)


# Test seti tahminleri
print("\n8. Test seti için tahminler yapılıyor...")
predictions = best_model.predict(X_test_scaled)


# Submission dosyası oluşturma
submission = pd.DataFrame({
    "Id" : list(range(1, 9001)),
    'Solution': predictions
})
  


submission.to_csv('submission.csv', index=False)


pd.Series(predictions).value_counts()


# Her model için eğitim ve tahmin
models = {
    'rf': rf,
    'gb': gb,
    'svm': svm,
    'lr': lr,
    'mlp': mlp
}


for name, model in models.items():
    model.fit(X_scaled, y)
    preds = model.predict(X_test_scaled)
    sub = pd.DataFrame({'Solution': preds})
    sub.to_csv(f'submission_{name}.csv', index=False)
    print(f"{name.upper()} submission saved: submission_{name}.csv")

