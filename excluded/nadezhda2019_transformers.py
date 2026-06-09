!pip install transformers tensorflow


from transformers import DistilBertTokenizer, TFDistilBertForSequenceClassification
import tensorflow as tf

# Загрузка предобученных модели и токенизатора
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = DistilBertTokenizer.from_pretrained(model_name)
model = TFDistilBertForSequenceClassification.from_pretrained(model_name)


text = "I love using Transformers with Keras. It's seamless!"

# Токенизация текста (возвращает словарь input_ids и attention_mask)
inputs = tokenizer(text, return_tensors="tf", truncation=True, padding=True)


# Прямой проход через модель (инференс)
outputs = model(inputs)
logits = outputs.logits

# Преобразование logits в вероятности через softmax
probs = tf.nn.softmax(logits, axis=1).numpy()[0]


labels = ["0", "1"]
predicted_class = tf.argmax(probs).numpy()

print(f"Текст: '{text}'")
print(f"Тональность: {labels[predicted_class]} ({probs[predicted_class]:.2%})")


texts = [
    "The product is awful.",
    "Keras makes deep learning easy!",
    "I'm disappointed by the service."
]

# Токенизация батча
inputs = tokenizer(texts, return_tensors="tf", padding=True, truncation=True, max_length=512)

# Предсказание
outputs = model(inputs)
probs = tf.nn.softmax(outputs.logits, axis=1)

for i, text in enumerate(texts):
    print(f"{text[:30]}... -> {labels[tf.argmax(probs[i]).numpy()]} ({tf.reduce_max(probs[i]):.2%})")


'''
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

# Создание Keras-модели с DistilBERT
input_ids = Input(shape=(None,), dtype=tf.int32, name="input_ids")
attention_mask = Input(shape=(None,), dtype=tf.int32, name="attention_mask")

outputs = model.distilbert(input_ids=input_ids, attention_mask=attention_mask)[0]
outputs = model.pre_classifier(outputs)
logits = model.classifier(outputs)

keras_model = Model(inputs=[input_ids, attention_mask], outputs=logits)
keras_model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
'''


from transformers import pipeline

# Загрузка модели для суммаризации
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Исходный текст (например, рецензия с IMDB)
text = """
The movie was a fantastic blend of action and drama. The protagonist's journey from a troubled past to becoming a hero was compelling. 
However, the second act felt rushed, and some side characters were underdeveloped. 
Despite these flaws, the cinematography and soundtrack were outstanding.
"""

# Суммаризация
summary = summarizer(text, max_length=50, min_length=20, do_sample=False)
print(summary[0]['summary_text'])


from transformers import pipeline

generator = pipeline("text-generation", model="gpt2")

prompt = "In a distant future, humanity discovered interstellar travel"
generated_text = generator(prompt, max_length=100, num_return_sequences=1)
print(generated_text[0]['generated_text'])


from transformers import pipeline

# Русская модель для анализа тональности
model_name = "blanchefort/rubert-base-cased-sentiment"
classifier = pipeline("sentiment-analysis", model=model_name, tokenizer=model_name)

texts = [
    "Этот фильм просто потрясающий!",
    "Ужасное кино, никому не советую.",
    "Нормально, можно посмотреть.",
    "Очень понравилась игра актёров."
]

for text in texts:
    result = classifier(text)[0]
    print(f"Текст: {text}")
    print(f"Результат: {result}\n")


'''
from transformers import pipeline

# Готовые решения для NLP задач
classifier = pipeline("sentiment-analysis")
print(classifier("I love this library!"))

# Доступные пайплайны:
# - "sentiment-analysis" - анализ тональности
# - "text-generation" - генерация текста
# - "question-answering" - ответы на вопросы
# - "translation" - перевод
# - "summarization" - суммаризация
# - "ner" - извлечение сущностей
# - "fill-mask" - заполнение пропусков
'''


'''
from transformers import pipeline
# Указываем модель, обученную на русском
classifier = pipeline("sentiment-analysis", 
                     model="blanchefort/rubert-base-cased-sentiment")
texts = [
    "Этот фильм — просто шедевр, я в восторге!",
    "Ужасное обслуживание, больше никогда не приду.",
    "Нормальный продукт, соответствует описанию."
]
results = classifier(texts)
for text, result in zip(texts, results):
    print(f"Текст: {text}")
    print(f"Результат: {result['label']}, Уверенность: {result['score']:.3f}\n")
'''


'''
# minimal_kaggle_submission.py
import pandas as pd
from transformers import pipeline
import warnings
warnings.filterwarnings('ignore')

# 1. Загрузка данных (измените путь если нужно)
train = pd.read_csv('./data/labeledTrainData.tsv', sep='\t', quoting=3)
test = pd.read_csv('./data/testData.tsv', sep='\t', quoting=3)

print(f"Train shape: {train.shape}, Test shape: {test.shape}")
print(train.head())

# 2. Создаем пайплайн для анализа тональности
# Используем модель, которая уже обучена на отзывах (SST-2)
print("\nЗагрузка модели...")
classifier = pipeline("sentiment-analysis", 
                     model="distilbert-base-uncased-finetuned-sst-2-english")

# 3. Функция для очистки текста (минимальная)
def clean_text(text):
    """Базовая очистка текста"""
    if pd.isna(text):
        return ""
    # Убираем HTML-теги
    text = str(text).replace('<br />', ' ')
    text = text.replace('\\"', '"')
    # Ограничиваем длину (модель принимает до 512 токенов)
    words = text.split()
    if len(words) > 400:  # Оставляем запас
        text = ' '.join(words[:400])
    return text

# 4. Очищаем тестовые данные
print("\nОчистка текстов...")
test['clean_review'] = test['review'].apply(clean_text)

# 5. Предсказание (обрабатываем батчами для экономии памяти)
print("\nДелаем предсказания...")
predictions = []
batch_size = 32  # Можно увеличить если есть GPU

for i in range(0, len(test), batch_size):
    batch = test['clean_review'].iloc[i:i+batch_size].tolist()
    batch_results = classifier(batch)
    
    # Преобразуем результаты: POSITIVE -> 1, NEGATIVE -> 0
    for result in batch_results:
        sentiment = 1 if result['label'] == 'POSITIVE' else 0
        predictions.append(sentiment)
    
    # Прогресс
    if (i // batch_size) % 10 == 0:
        print(f"Обработано: {min(i+batch_size, len(test))}/{len(test)}")

# 6. Создаем файл для отправки
print("\nСоздание submission файла...")
submission = pd.DataFrame({
    'id': test['id'],
    'sentiment': predictions
})

# Проверяем распределение
print(f"\nРаспределение предсказаний:")
print(f"Положительных (1): {(submission['sentiment'] == 1).sum()}")
print(f"Отрицательных (0): {(submission['sentiment'] == 0).sum()}")

# 7. Сохраняем
submission_file = 'submission.csv'
submission.to_csv(submission_file, index=False)
print(f"\n✅ Файл сохранен: {submission_file}")
print("Можно загружать на Kaggle!")
'''


'''
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb

# 1. Загружаем данные
data = fetch_california_housing()
X = data.data  # Все признаки
y = data.target  # Целевая переменная (цена дома)

# 2. Делим на тренировочную и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Создаём и обучаем модель XGBoost
model = xgb.XGBRegressor(
    n_estimators=100,  # Количество деревьев
    max_depth=3,       # Глубина деревьев
    learning_rate=0.1, # Скорость обучения
    random_state=42    # Для воспроизводимости
)

model.fit(X_train, y_train)

# 4. Делаем предсказания
y_pred = model.predict(X_test)

# 5. Оцениваем качество
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Предсказания для первых 5 домов: {y_pred[:5]}")
print(f"Фактические значения: {y_test[:5]}")
print(f"\nСреднеквадратичная ошибка (MSE): {mse:.4f}")
print(f"Коэффициент детерминации (R²): {r2:.4f}")

# 6. Важность признаков (опционально)
feature_names = data.feature_names
importances = model.feature_importances_

print("\nВажность признаков:")
for name, importance in zip(feature_names, importances):
    print(f"{name}: {importance:.3f}")
'''


'''
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from catboost import CatBoostRegressor

# 1. Загружаем данные
data = fetch_california_housing()
X = data.data
y = data.target

# 2. Делим на тренировочную и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Создаём и обучаем модель CatBoost
model = CatBoostRegressor(
    iterations=100,   # Количество деревьев
    depth=6,         # Глубина деревьев
    learning_rate=0.1, # Скорость обучения
    random_seed=42,  # Для воспроизводимости
    verbose=0        # Отключаем логи обучения
)

model.fit(X_train, y_train)

# 4. Делаем предсказания
y_pred = model.predict(X_test)

# 5. Оцениваем качество
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Предсказания CatBoost (первые 5): {y_pred[:5]}")
print(f"Фактические значения: {y_test[:5]}")
print(f"\nСреднеквадратичная ошибка (MSE): {mse:.4f}")
print(f"Коэффициент детерминации (R²): {r2:.4f}")

# 6. Важность признаков (опционально)
feature_names = data.feature_names
importances = model.get_feature_importance()

print("\nВажность признаков (CatBoost):")
for name, importance in zip(feature_names, importances):
    print(f"{name}: {importance:.3f}")
'''




