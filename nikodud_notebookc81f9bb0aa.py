import pandas as pd
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords
from nltk.data import load
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from gensim.models import Word2Vec
import os
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import warnings
from bs4 import MarkupResemblesLocatorWarning
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Игнорируем предупреждение о том, что текст похож на URL
warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)

# Загрузка данных
train = pd.read_csv("/kaggle/input/word2vec-nlp-tutorial/labeledTrainData.tsv.zip", header=0, delimiter="\t", quoting=3, compression="zip")
test = pd.read_csv("/kaggle/input/word2vec-nlp-tutorial/testData.tsv.zip", header=0, delimiter="\t", quoting=3, compression="zip")
unlabeled_train = pd.read_csv("/kaggle/input/word2vec-nlp-tutorial/unlabeledTrainData.tsv.zip", header=0, delimiter="\t", quoting=3, compression="zip")

# ================================
# Раздел 1: Анализ данных и принятие решений
# ================================
print("="*80)
print("ЭТАП АНАЛИЗА ДАННЫХ И ПРИНЯТИЯ РЕШЕНИЙ")
print("="*80)

# 1.1 Исследование структуры данных
print("\n1. ИССЛЕДОВАНИЕ СТРУКТУРЫ ДАННЫХ:")
print(f"Размер тренировочных данных: {train.shape}")
print(f"Размер тестовых данных: {test.shape}")
print(f"Размер неразмеченных данных: {unlabeled_train.shape}")
print(f"\nКолонки тренировочных данных: {train.columns.tolist()}")
print(f"Типы данных:\n{train.dtypes}")

# 1.2 Проверка баланса классов
print(f"\n2. БАЛАНС КЛАССОВ В ТРЕНИРОВОЧНЫХ ДАННЫХ:")
sentiment_counts = train['sentiment'].value_counts()
print(f"Позитивных отзывов (1): {sentiment_counts[1]} ({sentiment_counts[1]/len(train)*100:.1f}%)")
print(f"Негативных отзывов (0): {sentiment_counts[0]} ({sentiment_counts[0]/len(train)*100:.1f}%)")

# 1.3 Анализ длины отзывов
print(f"\n3. АНАЛИЗ ДЛИНЫ ОТЗЫВОВ:")
train['review_length'] = train['review'].apply(lambda x: len(str(x)))
test['review_length'] = test['review'].apply(lambda x: len(str(x)))

print(f"Средняя длина отзыва (тренировка): {train['review_length'].mean():.0f} символов")
print(f"Средняя длина отзыва (тест): {test['review_length'].mean():.0f} символов")
print(f"Максимальная длина отзыва: {train['review_length'].max():.0f} символов")
print(f"Минимальная длина отзыва: {train['review_length'].min():.0f} символов")

# 1.4 Поиск пропущенных значений
print(f"\n4. ПРОВЕРКА НА НАЛИЧИЕ ПРОПУЩЕННЫХ ЗНАЧЕНИЙ:")
print(f"Пропуски в тренировочных данных:\n{train.isnull().sum()}")
print(f"Пропуски в тестовых данных:\n{test.isnull().sum()}")

# 1.5 Примеры данных для понимания структуры
print(f"\n5. ПРИМЕРЫ ИСХОДНЫХ ДАННЫХ:")
sample_data = train.head(3).copy()
for idx, row in sample_data.iterrows():
    print(f"\nПример {idx+1} (Сентимент: {'Позитивный' if row['sentiment'] == 1 else 'Негативный'}):")
    preview = str(row['review'])[:200] + "..." if len(str(row['review'])) > 200 else str(row['review'])
    print(f"Отзыв (превью): {preview}")
    print(f"Длина отзыва: {len(str(row['review']))} символов")


# Улучшенные функции для очистки отзывов
def review_to_words(raw_review):
    """Конвертирует сырой отзыв в строку очищенных слов"""
    try:
        # Проверяем, не пустой ли отзыв
        if pd.isna(raw_review) or raw_review == "":
            return ""
        
        # Используем BeautifulSoup только если текст содержит HTML-теги
        if '<' in str(raw_review) and '>' in str(raw_review):
            review_text = BeautifulSoup(raw_review, "html.parser").get_text()
        else:
            review_text = str(raw_review)
            
        letters_only = re.sub("[^a-zA-Z]", " ", review_text)
        words = letters_only.lower().split()
        stops = set(stopwords.words("english"))
        meaningful_words = [w for w in words if not w in stops and len(w) > 1]
        return " ".join(meaningful_words)
    except Exception as e:
        print(f"Ошибка в review_to_words: {e}")
        return ""

def review_to_wordlist(review, remove_stopwords=False):
    """Конвертирует сырой отзыв в список слов"""
    try:
        if pd.isna(review) or review == "":
            return []
            
        if '<' in str(review) and '>' in str(review):
            review_text = BeautifulSoup(review, "html.parser").get_text()
        else:
            review_text = str(review)
            
        review_text = re.sub("[^a-zA-Z]", " ", review_text)
        words = review_text.lower().split()
        if remove_stopwords:
            stops = set(stopwords.words("english"))
            words = [w for w in words if not w in stops and len(w) > 1]
        return words
    except Exception as e:
        print(f"Ошибка в review_to_wordlist: {e}")
        return []

def review_to_sentences(review, tokenizer, remove_stopwords=False):
    """Разбивает отзыв на предложения"""
    try:
        if pd.isna(review) or review == "":
            return []
            
        raw_sentences = tokenizer.tokenize(str(review).strip())
        sentences = []
        for raw_sentence in raw_sentences:
            if len(raw_sentence) > 0:
                sentences.append(review_to_wordlist(raw_sentence, remove_stopwords))
        return sentences
    except Exception as e:
        print(f"Ошибка в review_to_sentences: {e}")
        return []


print(f"\n" + "="*80)
print("ПРИМЕРЫ ПРЕОБРАЗОВАНИЯ ДАННЫХ")
print("="*80)

# Создаем таблицу с примерами преобразования
sample_size = 10  # Увеличим размер выборки для лучшей наглядности
sample_indices = train.sample(sample_size, random_state=42).index

transformation_examples = []

for idx in sample_indices:
    original_review = train.loc[idx, 'review']
    cleaned_review = review_to_words(original_review)
    wordlist_review = review_to_wordlist(original_review, remove_stopwords=True)
    sentiment = train.loc[idx, 'sentiment']
    
    transformation_examples.append({
        'ID': idx,
        'Исходный отзыв (первые 200 символов)': str(original_review)[:200] + "..." if len(str(original_review)) > 200 else str(original_review),
        'Очищенный отзыв': cleaned_review[:300] + "..." if len(cleaned_review) > 300 else cleaned_review,
        'Количество слов в очищенном отзыве': len(cleaned_review.split()),
        'Пример очищенных слов (первые 15)': ', '.join(wordlist_review[:15]),
        'Ожидаемый сентимент': 'Позитивный' if sentiment == 1 else 'Негативный',
        'Длина исходного отзыва': len(str(original_review))
    })

# Создаем DataFrame для наглядного отображения
transformation_df = pd.DataFrame(transformation_examples)
print("\nТАБЛИЦА ПРИМЕРОВ ПРЕОБРАЗОВАНИЯ ДАННЫХ:")
print(transformation_df.to_string(index=False, max_colwidth=50))

# Сохраняем таблицу преобразования в CSV
transformation_df.to_csv('data_transformation_examples.csv', index=False, encoding='utf-8-sig')
print(f"\nТаблица преобразований сохранена в: data_transformation_examples.csv")

# ================================
# Раздел 4: Разделение данных
# ================================
print(f"\n" + "="*80)
print("РАЗДЕЛЕНИЕ ДАННЫХ ДЛЯ ОБУЧЕНИЯ И ВАЛИДАЦИИ")
print("="*80)


# Разделение тренировочных данных на 99% для обучения и 1% для валидации
split_ratio = 0.99
split_index = int(len(train) * split_ratio)

train_99 = train[:split_index]
validation_1 = train[split_index:]

print(f"Размер тренировочного набора (99%): {len(train_99)}")
print(f"Размер валидационного набора (1%): {len(validation_1)}")

# Проверка баланса классов в разделенных данных
print(f"\nБаланс классов в тренировочном наборе:")
train_sentiment_counts = train_99['sentiment'].value_counts()
print(f"Позитивных: {train_sentiment_counts[1]} ({train_sentiment_counts[1]/len(train_99)*100:.1f}%)")
print(f"Негативных: {train_sentiment_counts[0]} ({train_sentiment_counts[0]/len(train_99)*100:.1f}%)")

print(f"\nБаланс классов в валидационном наборе:")
val_sentiment_counts = validation_1['sentiment'].value_counts()
print(f"Позитивных: {val_sentiment_counts[1]} ({val_sentiment_counts[1]/len(validation_1)*100:.1f}%)")
print(f"Негативных: {val_sentiment_counts[0]} ({val_sentiment_counts[0]/len(validation_1)*100:.1f}%)")

# ================================
# Раздел 5: KNN на Bag of Words
# ================================
print("\n" + "="*80)
print("МОДЕЛЬ 1: KNN НА BAG OF WORDS")
print("="*80)



# Очистка тренировочных данных (99%)
num_reviews = train_99["review"].size
clean_train_reviews = []
for i in range(num_reviews):
    clean_train_reviews.append(review_to_words(train_99["review"][i]))

# Очистка валидационных данных (1%)
clean_validation_reviews = []
for i in range(len(validation_1)):

    clean_validation_reviews.append(review_to_words(validation_1["review"].iloc[i]))

# Создание Bag of Words
print("Creating Bag of Words features...")
vectorizer = CountVectorizer(analyzer="word", tokenizer=None, preprocessor=None, stop_words=None, max_features=5000)
train_data_features = vectorizer.fit_transform(clean_train_reviews)
train_data_features = train_data_features.toarray()

# Анализ созданных признаков
print(f"\nАНАЛИЗ ПРИЗНАКОВ BAG OF WORDS:")
print(f"Размер матрицы признаков: {train_data_features.shape}")
print(f"Количество уникальных слов (признаков): {len(vectorizer.get_feature_names_out())}")
print(f"Примеры самых частых слов: {vectorizer.get_feature_names_out()[:10]}")

# Обучение KNN (k=5)
print("\nTraining KNN classifier...")
knn_bow = KNeighborsClassifier(n_neighbors=5)
knn_bow.fit(train_data_features, train_99["sentiment"])

# Преобразование валидационных данных в Bag of Words
validation_data_features = vectorizer.transform(clean_validation_reviews)
validation_data_features = validation_data_features.toarray()

# Предсказание на валидационных данных
validation_predictions_bow = knn_bow.predict(validation_data_features)
validation_true_bow = validation_1["sentiment"].values

# ================================
# Раздел 5: KNN на Bag of Words - таблица предсказаний
# ================================
# Создание таблицы с результатами предсказаний для Bag of Words
results_bow = []
for i in range(min(20, len(validation_1))):  # Показываем первые 20 примеров
    original_text = validation_1["review"].iloc[i][:150] + "..." if len(validation_1["review"].iloc[i]) > 150 else validation_1["review"].iloc[i]
    cleaned_text = clean_validation_reviews[i][:150] + "..." if len(clean_validation_reviews[i]) > 150 else clean_validation_reviews[i]
    
    # Получаем вероятности для обоих классов
    probas = knn_bow.predict_proba(validation_data_features[i:i+1])[0]
    
    results_bow.append({
        'ID': validation_1["id"].iloc[i],
        'Исходный текст (первые 150 символов)': original_text,
        'Очищенный текст (первые 150 символов)': cleaned_text,
        'Ожидаемый сентимент': 'Позитивный' if validation_true_bow[i] == 1 else 'Негативный',
        'Предсказанный сентимент': 'Позитивный' if validation_predictions_bow[i] == 1 else 'Негативный',
        'Совпадение': '✓' if validation_true_bow[i] == validation_predictions_bow[i] else '✗',
        'Вероятность позитивного класса': f"{probas[1]:.4f}",
        'Вероятность негативного класса': f"{probas[0]:.4f}",
        'Уверенность предсказания': f"{max(probas):.4f}"
    })

results_bow_df = pd.DataFrame(results_bow)
print("\nТАБЛИЦА ПРЕДСКАЗАНИЙ BAG OF WORDS (первые 20 примеров):")
print(results_bow_df.to_string(index=False, max_colwidth=50))

# Сохраняем таблицу предсказаний Bag of Words
results_bow_df.to_csv('bow_predictions_examples.csv', index=False, encoding='utf-8-sig')
print(f"Таблица предсказаний Bag of Words сохранена в: bow_predictions_examples.csv")

# ================================
# Раздел 6: KNN на Word2Vec
# ================================
print("\n" + "="*80)
print("МОДЕЛЬ 2: KNN НА WORD2VEC")
print("="*80)


# Подготовка предложений для Word2Vec
print("Preparing sentences for Word2Vec training...")
tokenizer = load('tokenizers/punkt/english.pickle')
sentences = []
for review in train["review"]:  # Используем все данные для обучения Word2Vec
    sentences += review_to_sentences(review, tokenizer)
for review in unlabeled_train["review"]:
    sentences += review_to_sentences(review, tokenizer)

print(f"Total sentences for Word2Vec training: {len(sentences)}")

# Обучение или загрузка Word2Vec модели
model_name = "300features_40minwords_10context"
if os.path.exists(model_name):
    print("Loading existing Word2Vec model...")
    model = Word2Vec.load(model_name)
else:
    print("Training new Word2Vec model...")
    num_features = 300
    min_word_count = 40
    num_workers = 4
    context = 10
    downsampling = 1e-3
    model = Word2Vec(sentences, workers=num_workers, vector_size=num_features, 
                     min_count=min_word_count, window=context, sample=downsampling)
    model.save(model_name)
    print("Word2Vec model saved.")

# Анализ обученной модели Word2Vec
print(f"\nАНАЛИЗ WORD2VEC МОДЕЛИ:")
print(f"Размер словаря: {len(model.wv)}")
print(f"Размерность векторов: {model.vector_size}")
print(f"\nПримеры семантических связей:")

# Тестирование модели на примерах
test_words = ['good', 'bad', 'movie', 'film', 'excellent', 'terrible']
for word in test_words:
    if word in model.wv:
        similar_words = model.wv.most_similar(word, topn=3)
        print(f"Слова, похожие на '{word}': {similar_words}")

# Функции для усреднения векторов
def makeFeatureVec(words, model, num_features):
    featureVec = np.zeros((num_features,), dtype="float32")
    nwords = 0.
    index2word_set = set(model.wv.index_to_key)
    for word in words:
        if word in index2word_set:
            nwords += 1.
            featureVec = np.add(featureVec, model.wv[word])
    if nwords > 0:
        featureVec = np.divide(featureVec, nwords)
    return featureVec

def getAvgFeatureVecs(reviews, model, num_features):
    counter = 0
    reviewFeatureVecs = np.zeros((len(reviews), num_features), dtype="float32")
    for review in reviews:
        reviewFeatureVecs[counter] = makeFeatureVec(review, model, num_features)
        counter += 1
    return reviewFeatureVecs

# Получение векторов для тренировочных данных (99%)
print("\nCreating Word2Vec features for training data...")
clean_train_reviews_w2v = [review_to_wordlist(review, remove_stopwords=True) for review in train_99["review"]]
trainDataVecs = getAvgFeatureVecs(clean_train_reviews_w2v, model, 300)

# Обучение KNN
print("Training KNN classifier with Word2Vec features...")
knn_w2v = KNeighborsClassifier(n_neighbors=5)
knn_w2v.fit(trainDataVecs, train_99["sentiment"])

# Получение векторов для валидационных данных (1%)
print("Creating Word2Vec features for validation data...")
clean_validation_reviews_w2v = [review_to_wordlist(review, remove_stopwords=True) for review in validation_1["review"]]
validationDataVecs = getAvgFeatureVecs(clean_validation_reviews_w2v, model, 300)

# Предсказание на валидационных данных
validation_predictions_w2v = knn_w2v.predict(validationDataVecs)
validation_true_w2v = validation_1["sentiment"].values

# Создание таблицы с результатами предсказаний для Word2Vec
# Создание таблицы с результатами предсказаний для Word2Vec
results_w2v = []
for i in range(min(20, len(validation_1))):  # Показываем первые 20 примеров
    original_text = validation_1["review"].iloc[i][:150] + "..." if len(validation_1["review"].iloc[i]) > 150 else validation_1["review"].iloc[i]
    cleaned_words = clean_validation_reviews_w2v[i][:15]  # первые 15 слов
    
    # Получаем вероятности для обоих классов
    probas = knn_w2v.predict_proba(validationDataVecs[i:i+1])[0]
    
    results_w2v.append({
        'ID': validation_1["id"].iloc[i],
        'Исходный текст (первые 150 символов)': original_text,
        'Пример слов для Word2Vec (первые 15)': ', '.join(cleaned_words),
        'Ожидаемый сентимент': 'Позитивный' if validation_true_w2v[i] == 1 else 'Негативный',
        'Предсказанный сентимент': 'Позитивный' if validation_predictions_w2v[i] == 1 else 'Негативный',
        'Совпадение': '✓' if validation_true_w2v[i] == validation_predictions_w2v[i] else '✗',
        'Вероятность позитивного класса': f"{probas[1]:.4f}",
        'Вероятность негативного класса': f"{probas[0]:.4f}",
        'Уверенность предсказания': f"{max(probas):.4f}"
    })

results_w2v_df = pd.DataFrame(results_w2v)
print("\nТАБЛИЦА ПРЕДСКАЗАНИЙ WORD2VEC (первые 20 примеров):")
print(results_w2v_df.to_string(index=False, max_colwidth=50))

# Сохраняем таблицу предсказаний Word2Vec
results_w2v_df.to_csv('word2vec_predictions_examples.csv', index=False, encoding='utf-8-sig')
print(f"Таблица предсказаний Word2Vec сохранена в: word2vec_predictions_examples.csv")

# ================================
# Раздел 7: Визуализация решений моделей
# ================================
print("\n" + "="*80)
print("ВИЗУАЛИЗАЦИЯ ПРИНЯТИЯ РЕШЕНИЙ МОДЕЛЯМИ")
print("="*80)

# 7.1 Визуализация пространства признаков с помощью PCA
print("\n1. ВИЗУАЛИЗАЦИЯ ПРОСТРАНСТВА ПРИЗНАКОВ (PCA):")

# PCA для Bag of Words
pca_bow = PCA(n_components=2)
bow_2d = pca_bow.fit_transform(validation_data_features[:100])  # первые 100 примеров

# PCA для Word2Vec
pca_w2v = PCA(n_components=2)
w2v_2d = pca_w2v.fit_transform(validationDataVecs[:100])

# Создание графиков
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# График 1: Bag of Words PCA
scatter1 = axes[0].scatter(bow_2d[:, 0], bow_2d[:, 1], 
                          c=validation_true_bow[:100], 
                          cmap='coolwarm', alpha=0.6)
axes[0].set_title('Bag of Words: Пространство признаков (PCA)', fontsize=12)
axes[0].set_xlabel('Главная компонента 1')
axes[0].set_ylabel('Главная компонента 2')
axes[0].legend(handles=scatter1.legend_elements()[0], 
               labels=['Негативный', 'Позитивный'])

# График 2: Word2Vec PCA
scatter2 = axes[1].scatter(w2v_2d[:, 0], w2v_2d[:, 1], 
                          c=validation_true_w2v[:100], 
                          cmap='viridis', alpha=0.6)
axes[1].set_title('Word2Vec: Пространство признаков (PCA)', fontsize=12)
axes[1].set_xlabel('Главная компонента 1')
axes[1].set_ylabel('Главная компонента 2')
axes[1].legend(handles=scatter2.legend_elements()[0], 
               labels=['Негативный', 'Позитивный'])

# График 3: Сравнение распределений вероятностей
bow_proba = knn_bow.predict_proba(validation_data_features[:100])
w2v_proba = knn_w2v.predict_proba(validationDataVecs[:100])

axes[2].hist(bow_proba[:, 1], alpha=0.5, bins=20, label='Bag of Words', color='blue')
axes[2].hist(w2v_proba[:, 1], alpha=0.5, bins=20, label='Word2Vec', color='green')
axes[2].set_title('Распределение вероятностей позитивного класса', fontsize=12)
axes[2].set_xlabel('Вероятность')
axes[2].set_ylabel('Количество примеров')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('decision_visualization.png', dpi=300, bbox_inches='tight')
plt.show()

# 7.2 Визуализация ближайших соседей
print("\n2. АНАЛИЗ БЛИЖАЙШИХ СОСЕДЕЙ ДЛЯ ПРИМЕРОВ:")

def analyze_neighbors(model, features, true_labels, train_features, train_labels, indices_to_analyze=[0, 1, 2]):
    """Анализирует ближайших соседей для выбранных примеров"""
    results = []
    for idx in indices_to_analyze:
        distances, neighbor_indices = model.kneighbors(features[idx:idx+1])
        
        neighbor_info = []
        for i, (dist, n_idx) in enumerate(zip(distances[0], neighbor_indices[0])):
            # Используем метки из обучающей выборки, так как соседи находятся там
            neighbor_info.append(f"Сосед {i+1}: расстояние={dist:.3f}, метка={'Поз' if train_labels[n_idx]==1 else 'Нег'}")
        
        results.append({
            'Пример': idx,
            'Истинная метка': 'Позитивный' if true_labels[idx] == 1 else 'Негативный',
            'Предсказанная метка': 'Позитивный' if model.predict(features[idx:idx+1])[0] == 1 else 'Негативный',
            'Ближайшие соседи': "\n".join(neighbor_info[:3])  # Показываем 3 ближайших соседа
        })
    
    return pd.DataFrame(results)

print("\nАнализ ближайших соседей для Bag of Words:")
neighbors_bow_df = analyze_neighbors(knn_bow, 
                                    validation_data_features, 
                                    validation_true_bow,
                                    train_data_features,
                                    train_99["sentiment"].values,
                                    indices_to_analyze=[0, 1, 2, 3, 4])  # 5 примеров

print(neighbors_bow_df.to_string(index=False))
neighbors_bow_df.to_csv('bow_neighbors_analysis.csv', index=False, encoding='utf-8-sig')
print(f"Таблица анализа соседей Bag of Words сохранена в: bow_neighbors_analysis.csv")

print("\nАнализ ближайших соседей для Word2Vec:")
neighbors_w2v_df = analyze_neighbors(knn_w2v, 
                                    validationDataVecs, 
                                    validation_true_w2v,
                                    trainDataVecs,
                                    train_99["sentiment"].values,
                                    indices_to_analyze=[0, 1, 2, 3, 4])  # 5 примеров

print(neighbors_w2v_df.to_string(index=False))
neighbors_w2v_df.to_csv('word2vec_neighbors_analysis.csv', index=False, encoding='utf-8-sig')
print(f"Таблица анализа соседей Word2Vec сохранена в: word2vec_neighbors_analysis.csv")

# 7.3 Визуализация важных слов
print("\n3. ВАЖНЫЕ СЛОВА ДЛЯ КЛАССИФИКАЦИИ:")

# Для Bag of Words получаем наиболее информативные слова
feature_names = vectorizer.get_feature_names_out()
positive_coef = np.mean(train_data_features[train_99["sentiment"] == 1], axis=0)
negative_coef = np.mean(train_data_features[train_99["sentiment"] == 0], axis=0)

# Находим слова с наибольшей разницей
diff = positive_coef - negative_coef
top_positive_indices = np.argsort(diff)[-20:][::-1]  # Топ-20 положительных
top_negative_indices = np.argsort(diff)[:20]  # Топ-20 отрицательных

# Создаем таблицу важных слов для Bag of Words
important_words_bow = []

print("\nТоп-20 слов для позитивного класса (Bag of Words):")
for idx in top_positive_indices:
    word_info = {
        'Слово': feature_names[idx],
        'Разница частот': f"{diff[idx]:.4f}",
        'Средняя частота в позитивных': f"{positive_coef[idx]:.4f}",
        'Средняя частота в негативных': f"{negative_coef[idx]:.4f}",
        'Относительная частота': f"{positive_coef[idx]/(negative_coef[idx]+1e-10):.2f}x"
    }
    important_words_bow.append(word_info)
    print(f"  {feature_names[idx]}: разница={diff[idx]:.4f}, отношение={positive_coef[idx]/(negative_coef[idx]+1e-10):.2f}x")

print("\nТоп-20 слов для негативного класса (Bag of Words):")
for idx in top_negative_indices:
    word_info = {
        'Слово': feature_names[idx],
        'Разница частот': f"{diff[idx]:.4f}",
        'Средняя частота в позитивных': f"{positive_coef[idx]:.4f}",
        'Средняя частота в негативных': f"{negative_coef[idx]:.4f}",
        'Относительная частота': f"{negative_coef[idx]/(positive_coef[idx]+1e-10):.2f}x"
    }
    important_words_bow.append(word_info)
    print(f"  {feature_names[idx]}: разница={diff[idx]:.4f}, отношение={negative_coef[idx]/(positive_coef[idx]+1e-10):.2f}x")

# Сохраняем таблицу важных слов
important_words_df = pd.DataFrame(important_words_bow)
important_words_df.to_csv('important_words_analysis.csv', index=False, encoding='utf-8-sig')
print(f"\nТаблица важных слов сохранена в: important_words_analysis.csv")

# ================================
# Раздел 8: Оценка качества моделей
# ================================
print("\n" + "="*80)
print("ОЦЕНКА КАЧЕСТВА МОДЕЛЕЙ")
print("="*80)

# Оценка качества Bag of Words
accuracy_bow = accuracy_score(validation_true_bow, validation_predictions_bow)
print(f"\nBag of Words Accuracy на валидационных данных: {accuracy_bow:.4f}")

# Оценка качества Word2Vec
accuracy_w2v = accuracy_score(validation_true_w2v, validation_predictions_w2v)
print(f"Word2Vec Accuracy на валидационных данных: {accuracy_w2v:.4f}")

# Получаем отчеты классификации в виде словарей
report_bow = classification_report(validation_true_bow, validation_predictions_bow, 
                                  target_names=['Negative', 'Positive'], output_dict=True)
report_w2v = classification_report(validation_true_w2v, validation_predictions_w2v, 
                                   target_names=['Negative', 'Positive'], output_dict=True)

# Детальные отчеты классификации
print("\n=== Детальный отчет Bag of Words ===")
print(classification_report(validation_true_bow, validation_predictions_bow, 
                          target_names=['Negative', 'Positive']))

print("\n=== Детальный отчет Word2Vec ===")
print(classification_report(validation_true_w2v, validation_predictions_w2v, 
                          target_names=['Negative', 'Positive']))

# ================================
# Раздел 8.1: Детальные метрики моделей (сохранение в CSV)
# ================================
print("\n" + "="*80)
print("ДЕТАЛЬНЫЕ МЕТРИКИ МОДЕЛЕЙ")
print("="*80)

# Создаем подробную таблицу метрик
detailed_metrics = []

for model_name, report, accuracy in [('Bag of Words', report_bow, accuracy_bow), 
                                     ('Word2Vec', report_w2v, accuracy_w2v)]:
    
    for class_name in ['Negative', 'Positive']:
        detailed_metrics.append({
            'Модель': model_name,
            'Класс': class_name,
            'Accuracy модели': f"{accuracy:.4f}",
            'Precision': f"{report[class_name]['precision']:.4f}",
            'Recall': f"{report[class_name]['recall']:.4f}",
            'F1-Score': f"{report[class_name]['f1-score']:.4f}",
            'Поддержка (Support)': int(report[class_name]['support'])
        })
    
    # Добавляем общие метрики
    detailed_metrics.append({
        'Модель': model_name,
        'Класс': 'Общие метрики',
        'Accuracy модели': f"{accuracy:.4f}",
        'Precision': f"{report['weighted avg']['precision']:.4f}",
        'Recall': f"{report['weighted avg']['recall']:.4f}",
        'F1-Score': f"{report['weighted avg']['f1-score']:.4f}",
        'Поддержка (Support)': int(report['weighted avg']['support'])
    })

detailed_metrics_df = pd.DataFrame(detailed_metrics)
print("\nДЕТАЛЬНЫЕ МЕТРИКИ МОДЕЛЕЙ:")
print(detailed_metrics_df.to_string(index=False))

# Сохраняем детальные метрики
detailed_metrics_df.to_csv('detailed_metrics.csv', index=False, encoding='utf-8-sig')
print(f"\nДетальные метрики сохранены в: detailed_metrics.csv")

# ================================
# Раздел 8.2: Финальная сравнительная таблица
# ================================
print("\n" + "="*80)
print("ФИНАЛЬНАЯ СРАВНИТЕЛЬНАЯ ТАБЛИЦА")
print("="*80)

# Создаем расширенную финальную таблицу
final_comparison_df = pd.DataFrame({
    'Метрика': ['Точность (Accuracy)', 
                'Precision (Позитивный)',
                'Recall (Позитивный)',
                'F1-Score (Позитивный)',
                'Precision (Негативный)',
                'Recall (Негативный)',
                'F1-Score (Негативный)',
                'Взвешенная Precision',
                'Взвешенная Recall',
                'Взвешенный F1-Score',
                'Макро-средняя Precision',
                'Макро-средняя Recall',
                'Макро-средний F1-Score'],
    'Bag of Words': [
        accuracy_bow,
        report_bow['Positive']['precision'],
        report_bow['Positive']['recall'],
        report_bow['Positive']['f1-score'],
        report_bow['Negative']['precision'],
        report_bow['Negative']['recall'],
        report_bow['Negative']['f1-score'],
        report_bow['weighted avg']['precision'],
        report_bow['weighted avg']['recall'],
        report_bow['weighted avg']['f1-score'],
        report_bow['macro avg']['precision'],
        report_bow['macro avg']['recall'],
        report_bow['macro avg']['f1-score']
    ],
    'Word2Vec': [
        accuracy_w2v,
        report_w2v['Positive']['precision'],
        report_w2v['Positive']['recall'],
        report_w2v['Positive']['f1-score'],
        report_w2v['Negative']['precision'],
        report_w2v['Negative']['recall'],
        report_w2v['Negative']['f1-score'],
        report_w2v['weighted avg']['precision'],
        report_w2v['weighted avg']['recall'],
        report_w2v['weighted avg']['f1-score'],
        report_w2v['macro avg']['precision'],
        report_w2v['macro avg']['recall'],
        report_w2v['macro avg']['f1-score']
    ],
    'Разница (Word2Vec - BoW)': [
        accuracy_w2v - accuracy_bow,
        report_w2v['Positive']['precision'] - report_bow['Positive']['precision'],
        report_w2v['Positive']['recall'] - report_bow['Positive']['recall'],
        report_w2v['Positive']['f1-score'] - report_bow['Positive']['f1-score'],
        report_w2v['Negative']['precision'] - report_bow['Negative']['precision'],
        report_w2v['Negative']['recall'] - report_bow['Negative']['recall'],
        report_w2v['Negative']['f1-score'] - report_bow['Negative']['f1-score'],
        report_w2v['weighted avg']['precision'] - report_bow['weighted avg']['precision'],
        report_w2v['weighted avg']['recall'] - report_bow['weighted avg']['recall'],
        report_w2v['weighted avg']['f1-score'] - report_bow['weighted avg']['f1-score'],
        report_w2v['macro avg']['precision'] - report_bow['macro avg']['precision'],
        report_w2v['macro avg']['recall'] - report_bow['macro avg']['recall'],
        report_w2v['macro avg']['f1-score'] - report_bow['macro avg']['f1-score']
    ]
})

# Форматируем числа для лучшей читаемости
for col in ['Bag of Words', 'Word2Vec', 'Разница (Word2Vec - BoW)']:
    final_comparison_df[col] = final_comparison_df[col].apply(lambda x: f"{x:.4f}")

print("\nФИНАЛЬНАЯ СРАВНИТЕЛЬНАЯ ТАБЛИЦА МЕТРИК:")
print(final_comparison_df.to_string(index=False))

# Сохраняем финальную сравнительную таблицу
final_comparison_df.to_csv('final_model_comparison.csv', index=False, encoding='utf-8-sig')
print(f"\nФинальная сравнительная таблица сохранена в: final_model_comparison.csv")

# ================================
# Раздел 9: Выводы и рекомендации (обновленная версия)
# ================================

files_summary = pd.DataFrame({
    'Файл': [
        'data_transformation_examples.csv',
        'bow_predictions_examples.csv',
        'word2vec_predictions_examples.csv',
        'bow_neighbors_analysis.csv',
        'word2vec_neighbors_analysis.csv',
        'important_words_analysis.csv',
        'detailed_metrics.csv',
        'final_model_comparison.csv',
        'decision_visualization.png',
        'final_comparison.png',
        'KNN_BagOfWords_model.csv',
        'KNN_Word2Vec_AverageVectors.csv'
    ],
    'Описание': [
        'Примеры преобразования исходных данных в очищенные',
        'Предсказания модели Bag of Words на валидационной выборке',
        'Предсказания модели Word2Vec на валидационной выборке',
        'Анализ ближайших соседей для модели Bag of Words',
        'Анализ ближайших соседей для модели Word2Vec',
        'Важные слова для классификации в модели Bag of Words',
        'Детальные метрики обеих моделей по классам',
        'Финальное сравнение всех метрик моделей',
        'Визуализация принятия решений моделями (график)',
        'Итоговые графики сравнения моделей (график)',
        'Финальные предсказания Bag of Words для тестовых данных',
        'Финальные предсказания Word2Vec для тестовых данных'
    ],
    'Тип': [
        'CSV', 'CSV', 'CSV', 'CSV', 'CSV', 'CSV', 'CSV', 'CSV', 
        'Изображение', 'Изображение', 'CSV', 'CSV'
    ]
})

# ================================
# Раздел 10: Визуализация F-меры и других метрик
# ================================
print("\n" + "="*80)
print("ВИЗУАЛИЗАЦИЯ F-МЕРЫ И КОМПЛЕКСНЫХ МЕТРИК")
print("="*80)

# Создаем расширенную визуализацию метрик
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# 10.1 Сравнение F1-Score по классам
categories = ['Negative', 'Positive', 'Weighted Avg', 'Macro Avg']
bow_f1 = [
    report_bow['Negative']['f1-score'],
    report_bow['Positive']['f1-score'],
    report_bow['weighted avg']['f1-score'],
    report_bow['macro avg']['f1-score']
]
w2v_f1 = [
    report_w2v['Negative']['f1-score'],
    report_w2v['Positive']['f1-score'],
    report_w2v['weighted avg']['f1-score'],
    report_w2v['macro avg']['f1-score']
]

x = np.arange(len(categories))
width = 0.35

axes[0, 0].bar(x - width/2, bow_f1, width, label='Bag of Words', color='skyblue', alpha=0.8)
axes[0, 0].bar(x + width/2, w2v_f1, width, label='Word2Vec', color='lightgreen', alpha=0.8)
axes[0, 0].set_xlabel('Тип F1-Score')
axes[0, 0].set_ylabel('Значение F1-Score')
axes[0, 0].set_title('Сравнение F1-Score по разным агрегациям', fontsize=12)
axes[0, 0].set_xticks(x)
axes[0, 0].set_xticklabels(categories, rotation=45)
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Добавляем значения на столбцы
for i, (b_val, w_val) in enumerate(zip(bow_f1, w2v_f1)):
    axes[0, 0].text(i - width/2, b_val + 0.01, f'{b_val:.3f}', 
                   ha='center', va='bottom', fontsize=9)
    axes[0, 0].text(i + width/2, w_val + 0.01, f'{w_val:.3f}', 
                   ha='center', va='bottom', fontsize=9)

# 10.2 Trade-off между Precision и Recall (кривая PR для обоих классов)
for class_idx, class_name in enumerate(['Negative', 'Positive']):
    metrics_data = {
        'Model': ['Bag of Words', 'Word2Vec', 'Bag of Words', 'Word2Vec'],
        'Metric': ['Precision', 'Precision', 'Recall', 'Recall'],
        'Value': [
            report_bow[class_name]['precision'],
            report_w2v[class_name]['precision'],
            report_bow[class_name]['recall'],
            report_w2v[class_name]['recall']
        ]
    }
    
    df_metrics = pd.DataFrame(metrics_data)
    pivot_df = df_metrics.pivot(index='Model', columns='Metric', values='Value')
    
    ax = axes[0, 1] if class_idx == 0 else axes[0, 2]
    pivot_df.plot(kind='bar', ax=ax, color=['lightcoral', 'lightblue'], alpha=0.8)
    ax.set_title(f'Precision-Recall для класса: {class_name}', fontsize=12)
    ax.set_ylabel('Значение')
    ax.set_ylim([0, 1.1])
    ax.legend(title='Метрика')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Добавляем значения на столбцы
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.3f}', 
                   (p.get_x() + p.get_width() / 2., p.get_height()), 
                   ha='center', va='bottom', fontsize=9)

# 10.3 Матрица ошибок для обеих моделей
for idx, (model_name, predictions, true_labels) in enumerate([
    ('Bag of Words', validation_predictions_bow, validation_true_bow),
    ('Word2Vec', validation_predictions_w2v, validation_true_w2v)
]):
    cm = confusion_matrix(true_labels, predictions)
    ax = axes[1, idx]
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'])
    ax.set_title(f'Матрица ошибок: {model_name}', fontsize=12)
    ax.set_xlabel('Предсказанный класс')
    ax.set_ylabel('Истинный класс')
    
    # Рассчитываем метрики из матрицы ошибок
    tn, fp, fn, tp = cm.ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # Добавляем метрики под графиком
    metrics_text = f'Accuracy: {accuracy:.3f}\nPrecision: {precision:.3f}\nRecall: {recall:.3f}\nF1-Score: {f1:.3f}'
    ax.text(0.5, -0.3, metrics_text, transform=ax.transAxes, 
            ha='center', va='top', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# 10.4 Radar chart для сравнения метрик
ax = axes[1, 2]
ax.axis('off')

# Создаем radar chart
metrics = ['Accuracy', 'Precision\n(Neg)', 'Recall\n(Neg)', 'F1\n(Neg)', 
           'Precision\n(Pos)', 'Recall\n(Pos)', 'F1\n(Pos)']

# Значения для обеих моделей
bow_values = [
    accuracy_bow,
    report_bow['Negative']['precision'],
    report_bow['Negative']['recall'],
    report_bow['Negative']['f1-score'],
    report_bow['Positive']['precision'],
    report_bow['Positive']['recall'],
    report_bow['Positive']['f1-score']
]

w2v_values = [
    accuracy_w2v,
    report_w2v['Negative']['precision'],
    report_w2v['Negative']['recall'],
    report_w2v['Negative']['f1-score'],
    report_w2v['Positive']['precision'],
    report_w2v['Positive']['recall'],
    report_w2v['Positive']['f1-score']
]

# Нормализуем значения для radar chart (приводим к шкале 0-1)
bow_norm = [v for v in bow_values]
w2v_norm = [v for v in w2v_values]

# Создаем углы для radar chart
angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
angles += angles[:1]  # Замыкаем круг

bow_norm += bow_norm[:1]
w2v_norm += w2v_norm[:1]
metrics_display = metrics + [metrics[0]]

# Создаем subplot для radar chart
ax_radar = fig.add_subplot(2, 3, 6, projection='polar')
ax_radar.plot(angles, bow_norm, 'o-', linewidth=2, label='Bag of Words', color='blue', alpha=0.7)
ax_radar.fill(angles, bow_norm, alpha=0.25, color='blue')
ax_radar.plot(angles, w2v_norm, 'o-', linewidth=2, label='Word2Vec', color='green', alpha=0.7)
ax_radar.fill(angles, w2v_norm, alpha=0.25, color='green')
ax_radar.set_xticks(angles[:-1])
ax_radar.set_xticklabels(metrics_display[:-1], fontsize=9)
ax_radar.set_ylim(0, 1)
ax_radar.set_title('Radar Chart: Сравнение всех метрик', fontsize=12, pad=20)
ax_radar.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
ax_radar.grid(True)

plt.tight_layout()
plt.savefig('f1_score_and_metrics_visualization.png', dpi=300, bbox_inches='tight')
plt.show()

# 10.5 Таблица с детальным расчетом F-меры
print("\n" + "="*80)
print("ДЕТАЛЬНЫЙ РАСЧЕТ F-МЕРЫ (F1-SCORE)")
print("="*80)

f1_calculation_table = []

for model_name, report in [('Bag of Words', report_bow), ('Word2Vec', report_w2v)]:
    for class_name in ['Negative', 'Positive']:
        precision = report[class_name]['precision']
        recall = report[class_name]['recall']
        f1_score = report[class_name]['f1-score']
        
        # Проверяем расчет F1
        calculated_f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        f1_calculation_table.append({
            'Модель': model_name,
            'Класс': class_name,
            'Precision (P)': f"{precision:.4f}",
            'Recall (R)': f"{recall:.4f}",
            'F1 = 2PR/(P+R)': f"{calculated_f1:.4f}",
            'Report F1': f"{f1_score:.4f}",
            'Совпадение': '✓' if abs(calculated_f1 - f1_score) < 0.001 else '✗'
        })

f1_calc_df = pd.DataFrame(f1_calculation_table)
print("\nРАСЧЕТ F1-SCORE ПО ФОРМУЛЕ: F1 = 2 * Precision * Recall / (Precision + Recall)")
print(f1_calc_df.to_string(index=False))

# Сохраняем таблицу расчета F1
f1_calc_df.to_csv('f1_score_calculation_details.csv', index=False, encoding='utf-8-sig')
print(f"\nДетали расчета F1-score сохранены в: f1_score_calculation_details.csv")

# Сохраняем сводную информацию
files_summary.to_csv('files_summary.csv', index=False, encoding='utf-8-sig')
print(f"\nСводная информация о файлах сохранена в: files_summary.csv")

# ================================
# Раздел 11: Создание финальных submission файлов для Kaggle
# ================================


# Очистка тестовых данных (оригинальный тестовый набор)
num_test_reviews = test["review"].size
clean_test_reviews = []
for i in range(num_test_reviews):
    clean_review = review_to_words(test["review"][i])
    clean_test_reviews.append(clean_review)

# 11.2 Bag of Words: финальные предсказания
print("\n2. BAG OF WORDS: ФИНАЛЬНЫЕ ПРЕДСКАЗАНИЯ...")

# Преобразование тестовых данных в Bag of Words
test_data_features = vectorizer.transform(clean_test_reviews)
test_data_features = test_data_features.toarray()

# Предсказание и сохранение
result_bow = knn_bow.predict(test_data_features)
output_bow = pd.DataFrame(data={"id": test["id"], "sentiment": result_bow})
output_bow.to_csv("submission_bag_of_words.csv", index=False, quoting=3)
print("✓ Bag of Words submission сохранен в: submission_bag_of_words.csv")

# Также сохраняем с вероятностями для анализа
probabilities_bow = knn_bow.predict_proba(test_data_features)[:, 1]
output_bow_proba = pd.DataFrame(data={
    "id": test["id"], 
    "sentiment": result_bow,
    "probability_positive": probabilities_bow,
    "probability_negative": 1 - probabilities_bow,
    "confidence": np.max(knn_bow.predict_proba(test_data_features), axis=1)
})
output_bow_proba.to_csv("submission_bag_of_words_with_probabilities.csv", index=False, quoting=3)
print("✓ Bag of Words с вероятностями сохранен в: submission_bag_of_words_with_probabilities.csv")

# 11.3 Word2Vec: финальные предсказания
print("\n3. WORD2VEC: ФИНАЛЬНЫЕ ПРЕДСКАЗАНИЯ...")

# Получение векторов для тестовых данных
print("Creating Word2Vec features for test data...")
clean_test_reviews_w2v = [review_to_wordlist(review, remove_stopwords=True) for review in test["review"]]
testDataVecs = getAvgFeatureVecs(clean_test_reviews_w2v, model, 300)

# Предсказание и сохранение
result_w2v = knn_w2v.predict(testDataVecs)
output_w2v = pd.DataFrame(data={"id": test["id"], "sentiment": result_w2v})
output_w2v.to_csv("submission_word2vec.csv", index=False, quoting=3)
print("✓ Word2Vec submission сохранен в: submission_word2vec.csv")

# Также сохраняем с вероятностями для анализа
probabilities_w2v = knn_w2v.predict_proba(testDataVecs)[:, 1]
output_w2v_proba = pd.DataFrame(data={
    "id": test["id"], 
    "sentiment": result_w2v,
    "probability_positive": probabilities_w2v,
    "probability_negative": 1 - probabilities_w2v,
    "confidence": np.max(knn_w2v.predict_proba(testDataVecs), axis=1)
})
output_w2v_proba.to_csv("submission_word2vec_with_probabilities.csv", index=False, quoting=3)
print("✓ Word2Vec с вероятностями сохранен в: submission_word2vec_with_probabilities.csv")


# Создаем ансамбль, усредняя вероятности обеих моделей
ensemble_probabilities = (probabilities_bow + probabilities_w2v) / 2
ensemble_predictions = (ensemble_probabilities >= 0.5).astype(int)

output_ensemble = pd.DataFrame(data={
    "id": test["id"], 
    "sentiment": ensemble_predictions,
    "probability_positive": ensemble_probabilities,
    "probability_negative": 1 - ensemble_probabilities,
    "confidence_bow": probabilities_bow,
    "confidence_w2v": probabilities_w2v,
    "confidence_ensemble": np.maximum(ensemble_probabilities, 1 - ensemble_probabilities)
})
output_ensemble.to_csv("submission_ensemble.csv", index=False, quoting=3)
print("✓ Ансамбль моделей сохранен в: submission_ensemble.csv")

# 11.5 Взвешенный ансамбль (по accuracy на валидации)
print("\n5. ВЗВЕШЕННЫЙ АНСАМБЛЬ (ПО ACCURACY)...")

# Взвешиваем по точности на валидации
weight_bow = accuracy_bow / (accuracy_bow + accuracy_w2v)
weight_w2v = accuracy_w2v / (accuracy_bow + accuracy_w2v)

print(f"Веса моделей:")
print(f"  Bag of Words: {weight_bow:.3f} (accuracy: {accuracy_bow:.4f})")
print(f"  Word2Vec: {weight_w2v:.3f} (accuracy: {accuracy_w2v:.4f})")

weighted_ensemble_probabilities = weight_bow * probabilities_bow + weight_w2v * probabilities_w2v
weighted_ensemble_predictions = (weighted_ensemble_probabilities >= 0.5).astype(int)

output_weighted = pd.DataFrame(data={
    "id": test["id"], 
    "sentiment": weighted_ensemble_predictions,
    "probability_positive": weighted_ensemble_probabilities,
    "probability_negative": 1 - weighted_ensemble_probabilities,
    "weight_bow": weight_bow,
    "weight_w2v": weight_w2v,
    "confidence_bow": probabilities_bow,
    "confidence_w2v": probabilities_w2v
})
output_weighted.to_csv("submission_weighted_ensemble.csv", index=False, quoting=3)

# 11.6 Статистика по предсказаниям
print("\n6. СТАТИСТИКА ПРЕДСКАЗАНИЙ:")

prediction_stats = pd.DataFrame({
    'Модель': ['Bag of Words', 'Word2Vec', 'Ансамбль (среднее)', 'Ансамбль (взвешенный)'],
    'Позитивных предсказаний': [
        np.sum(result_bow == 1),
        np.sum(result_w2v == 1),
        np.sum(ensemble_predictions == 1),
        np.sum(weighted_ensemble_predictions == 1)
    ],
    'Негативных предсказаний': [
        np.sum(result_bow == 0),
        np.sum(result_w2v == 0),
        np.sum(ensemble_predictions == 0),
        np.sum(weighted_ensemble_predictions == 0)
    ],
    'Доля позитивных': [
        f"{np.sum(result_bow == 1)/len(result_bow):.2%}",
        f"{np.sum(result_w2v == 1)/len(result_w2v):.2%}",
        f"{np.sum(ensemble_predictions == 1)/len(ensemble_predictions):.2%}",
        f"{np.sum(weighted_ensemble_predictions == 1)/len(weighted_ensemble_predictions):.2%}"
    ],
    'Средняя уверенность': [
        f"{np.mean(np.max(knn_bow.predict_proba(test_data_features), axis=1)):.4f}",
        f"{np.mean(np.max(knn_w2v.predict_proba(testDataVecs), axis=1)):.4f}",
        f"{np.mean(np.maximum(ensemble_probabilities, 1 - ensemble_probabilities)):.4f}",
        f"{np.mean(np.maximum(weighted_ensemble_probabilities, 1 - weighted_ensemble_probabilities)):.4f}"
    ]
})

print("\nСТАТИСТИКА ПРЕДСКАЗАНИЙ НА ТЕСТОВЫХ ДАННЫХ:")
print(prediction_stats.to_string(index=False))

# Сохраняем статистику
prediction_stats.to_csv('submission_statistics.csv', index=False, encoding='utf-8-sig')
print(f"\nСтатистика предсказаний сохранена в: submission_statistics.csv")

# 11.7 Анализ расхождений между моделями
print("\n7. АНАЛИЗ РАСХОЖДЕНИЙ МЕЖДУ МОДЕЛЯМИ:")

# Где модели расходятся во мнениях?
disagreements = result_bow != result_w2v
num_disagreements = np.sum(disagreements)
disagreement_rate = num_disagreements / len(result_bow)

print(f"Модели расходятся в {num_disagreements} из {len(result_bow)} предсказаний ({disagreement_rate:.2%})")

if num_disagreements > 0:
    # Создаем таблицу примеров расхождений
    disagreement_indices = np.where(disagreements)[0]
    disagreement_examples = []
    
    for idx in disagreement_indices[:10]:  # Первые 10 примеров
        disagreement_examples.append({
            'ID': test["id"].iloc[idx],
            'Bag of Words': 'Позитивный' if result_bow[idx] == 1 else 'Негативный',
            'Word2Vec': 'Позитивный' if result_w2v[idx] == 1 else 'Негативный',
            'Вероятность BoW': f"{probabilities_bow[idx]:.4f}",
            'Вероятность Word2Vec': f"{probabilities_w2v[idx]:.4f}",
            'Ансамбль': 'Позитивный' if ensemble_predictions[idx] == 1 else 'Негативный',
            'Текст (первые 100 символов)': test["review"].iloc[idx][:100] + "..."
        })
    
    disagreement_df = pd.DataFrame(disagreement_examples)
    print("\nПРИМЕРЫ РАСХОЖДЕНИЙ МЕЖДУ МОДЕЛЯМИ (первые 10):")
    print(disagreement_df.to_string(index=False, max_colwidth=50))
    
    # Сохраняем анализ расхождений
    disagreement_df.to_csv('model_disagreements_analysis.csv', index=False, encoding='utf-8-sig')
    print(f"Анализ расхождений сохранен в: model_disagreements_analysis.csv")


submission_files_summary = pd.DataFrame({
    'Файл для отправки': [
        'submission_bag_of_words.csv',
        'submission_word2vec.csv', 
        'submission_ensemble.csv',
        'submission_weighted_ensemble.csv'
    ],
    'Accuracy на валидации': [
        f"{accuracy_bow:.4f}",
        f"{accuracy_w2v:.4f}",
        f"{(accuracy_bow + accuracy_w2v)/2:.4f} (среднее)",
        f"{(weight_bow * accuracy_bow + weight_w2v * accuracy_w2v):.4f}"
    ]
})

print(submission_files_summary.to_string(index=False))

# Сохраняем сводную информацию по submission файлам
submission_files_summary.to_csv('submission_files_summary.csv', index=False, encoding='utf-8-sig')
print(f"\nСводная информация по submission файлам сохранена в: submission_files_summary.csv")


# Создаем таблицу с примерами предсказаний
prediction_samples = []
for i in range(min(5, len(test))):
    review_preview = test["review"].iloc[i][:100] + "..." if len(test["review"].iloc[i]) > 100 else test["review"].iloc[i]
    
    prediction_samples.append({
        'ID': test["id"].iloc[i],
        'Отзыв (фрагмент)': review_preview,
        'Bag of Words': 'Позитивный' if result_bow[i] == 1 else 'Негативный',
        'Word2Vec': 'Позитивный' if result_w2v[i] == 1 else 'Негативный',
        'Ансамбль': 'Позитивный' if ensemble_predictions[i] == 1 else 'Негативный',
        'Вероятность BoW': f"{probabilities_bow[i]:.3f}",
        'Вероятность W2V': f"{probabilities_w2v[i]:.3f}",
        'Рекомендовано': 'Word2Vec' if accuracy_w2v > accuracy_bow else 'Bag of Words'
    })

samples_df = pd.DataFrame(prediction_samples)
print("\nПРИМЕРЫ ПРЕДСКАЗАНИЙ ДЛЯ ПРОВЕРКИ:")
print(samples_df.to_string(index=False, max_colwidth=50))

# Сохраняем примеры предсказаний
samples_df.to_csv('prediction_samples_for_review.csv', index=False, encoding='utf-8-sig')
print(f"\nПримеры предсказаний сохранены в: prediction_samples_for_review.csv")

