# Установка библиотек
!pip install transformers datasets sentencepiece accelerate pandas scikit-learn matplotlib seaborn -q

# Импорт библиотек
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
from datasets import Dataset
from sklearn.model_selection import train_test_split
import torch
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Загрузка данных
def load_data():
    try:
        train = pd.read_csv('/kaggle/input/llm-prompt-recovery/train.csv')
        test = pd.read_csv('/kaggle/input/llm-prompt-recovery/test.csv')
        sample_sub = pd.read_csv('/kaggle/input/llm-prompt-recovery/sample_submission.csv')
    except:
        try:
            train = pd.read_csv('../input/llm-prompt-recovery/train.csv')
            test = pd.read_csv('../input/llm-prompt-recovery/test.csv')
            sample_sub = pd.read_csv('../input/llm-prompt-recovery/sample_submission.csv')
        except Exception as e:
            print(f"Ошибка загрузки данных: {e}")
            return None, None, None
    
    if train is None or test is None or sample_sub is None:
        print("Не удалось загрузить данные!")
        return None, None, None
    
    print("\nСтолбцы в train:", train.columns.tolist())
    print("Столбцы в test:", test.columns.tolist())
    print("Столбцы в sample_sub:", sample_sub.columns.tolist())
    
    return train, test, sample_sub

print("Загрузка данных...")
train_df, test_df, sample_sub = load_data()

if train_df is None or test_df is None or sample_sub is None:
    raise Exception("Не удалось загрузить данные! Проверьте пути к файлам.")

# EDA анализ
def perform_eda(df):
    print(f"\n=== EDA Analysis ===")
    print(f"Всего примеров: {len(df)}")
    
    if len(df) == 0:
        print("DataFrame пустой! Невозможно выполнить EDA.")
        return
    
    available_columns = df.columns.tolist()
    print("\nДоступные столбцы:", available_columns)
    
    print("\nПропущенные значения:")
    print(df.isnull().sum())
    
    text_columns = [col for col in df.columns if df[col].dtype == 'object' and col not in ['id', 'label']]
    
    for col in text_columns:
        try:
            df[f'{col}_length'] = df[col].astype(str).apply(len)
            plt.figure(figsize=(10, 4))
            sns.histplot(df[f'{col}_length'], bins=50)
            plt.title(f'Распределение длины {col}')
            plt.show()
            print(f"\nСтатистика длины {col}:")
            print(df[f'{col}_length'].describe())
        except Exception as e:
            print(f"Ошибка при анализе столбца {col}: {e}")
    
    print("\nПримеры данных (первые 3 строки):")
    for i in range(min(3, len(df))):
        print(f"\nПример {i+1}:")
        for col in df.columns:
            if col in ['id', 'label'] or '_length' in col: 
                continue
            try:
                content = str(df[col].iloc[i])
                print(f"{col}:", content[:100] + "..." if len(content) > 100 else content)
            except Exception as e:
                print(f"Ошибка при выводе столбца {col}: {e}")

print("\nEDA анализ тренировочных данных...")
perform_eda(train_df)

# Подготовка данных
print("\nОпределение структуры данных...")
input_col = 'text' if 'text' in train_df.columns else train_df.columns[0]
target_col = 'prompt' if 'prompt' in train_df.columns else train_df.columns[1] if len(train_df.columns) > 1 else input_col

print(f"Используем '{input_col}' как вход и '{target_col}' как цель")

# Очистка данных
train_df[input_col] = train_df[input_col].astype(str).fillna('')
train_df[target_col] = train_df[target_col].astype(str).fillna('')

# Проверка на минимальное количество данных
if len(train_df) < 2:
    print("\nСлишком мало данных - применяем аугментацию")
    original_input = train_df[input_col].iloc[0]
    original_target = train_df[target_col].iloc[0]
    
    augmented_data = {
        input_col:[original_input] + [original_input + " "] + [" " + original_input] + [original_input.lower()] + [original_input.capitalize()],
        target_col: [original_target] + [original_target + " "] + [" " + original_target] + [original_target.lower()] + [original_target.capitalize()]
    }
    train_df = pd.DataFrame(augmented_data)
    print(f"Создано {len(train_df)} аугментированных примеров")

# Инициализация токенизатора
model_checkpoint = "t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

def preprocess_function(examples):
    inputs = [str(x) for x in examples[input_col]]
    targets = [str(x) for x in examples[target_col]]
    
    model_inputs = tokenizer(
        inputs, 
        max_length=512, 
        truncation=True,
        padding="max_length"
    )
    
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            targets,
            max_length=128,
            truncation=True,
            padding="max_length"
        )
    
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

print("\nПодготовка Dataset...")
train_dataset = Dataset.from_pandas(train_df)
tokenized_datasets = train_dataset.map(preprocess_function, batched=True)

# Разделение данных
num_samples = len(tokenized_datasets)
print(f"\nКоличество примеров: {num_samples}")

if num_samples < 5:
    print("Используем все данные для обучения (без валидации)")
    train_val = {"train": tokenized_datasets, "test": tokenized_datasets}
else:
    test_size = max(0.1, min(0.2, 5/num_samples))
    print(f"Разделяем данные: test_size={test_size:.1%}")
    train_val = tokenized_datasets.train_test_split(test_size=test_size, seed=42)

# Определение модели
print("\nИнициализация модели...")
model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)

# Настройка обучения
training_args = Seq2SeqTrainingArguments(
    output_dir="./results",
    evaluation_strategy="no" if num_samples < 5 else "epoch",
    save_strategy="no" if num_samples < 5 else "epoch",
    learning_rate=3e-4,
    per_device_train_batch_size=4 if num_samples < 10 else 8,
    per_device_eval_batch_size=4 if num_samples < 10 else 8,
    num_train_epochs=15 if num_samples < 5 else (10 if num_samples < 10 else 3),
    weight_decay=0.01,
    predict_with_generate=True,
    fp16=True,
    report_to="none",
    logging_steps=1 if num_samples < 10 else 10,
    save_total_limit=1
)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# Обучение модели
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_val["train"],
    eval_dataset=train_val["test"] if num_samples >= 5 else None,
    data_collator=data_collator,
    tokenizer=tokenizer,
)

print("\nОбучение модели...")
trainer.train()

# Предсказание на тестовых данных
def generate_predictions(texts, model, tokenizer, max_length=128):
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    ).to(model.device)
    
    outputs = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=max_length
    )
    
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)

print("\nГенерация предсказаний...")
test_texts = test_df[input_col].astype(str).tolist()
predictions = []
batch_size = 8  # Уменьшенный размер батча для стабильности

for i in tqdm(range(0, len(test_texts), batch_size)):
    batch = test_texts[i:i+batch_size]
    preds = generate_predictions(batch, model, tokenizer)
    predictions.extend(preds)

# Создание сабмита
submission_col = sample_sub.columns[1] if len(sample_sub.columns) > 1 else 'prediction'

submission = pd.DataFrame({
    sample_sub.columns[0]: test_df[sample_sub.columns[0]],
    submission_col: predictions
})

submission.to_csv("submission.csv", index=False)
print("\nСабмит сохранен как submission.csv")
print("\nПример предсказаний:")
print(submission.head())

