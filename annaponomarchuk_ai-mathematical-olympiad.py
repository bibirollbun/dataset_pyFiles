# для начала посмотрим, с какими данными предстоит работать и отметим, что их ОЧЕНЬ мало
import pandas as pd

train_data = pd.read_csv('/kaggle/input/ai-mathematical-olympiad-prize/train.csv')
train_data


# теперь загрузим токенизатор и модель

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments

model_name = 'distilgpt2'
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)


# из обучающей выборки создадим .txt файл, в который внесём информацию в фиксированном формате
# Задача: *описание задачи* Ответ: *ответ*

with open("train.txt", "w", encoding="utf-8") as file:
    for index, row in train_data.iterrows():
        problem = row["problem"]
        answer = row["answer"]
        text = f"Задача: {problem} Ответ: {answer}\n"
        file.write(text)


# из полученного .txt файла соберём датасет для обучения

train_dataset = TextDataset(
    tokenizer=tokenizer,
    file_path='train.txt',
    block_size=32,
    overwrite_cache=True
)


data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)


# теперь запустим обучение на имеющихся данных

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device);

training_args = TrainingArguments(
    output_dir='./results',
    overwrite_output_dir=True,
    num_train_epochs=7,
    save_total_limit=1,
    prediction_loss_only=True,
    evaluation_strategy="no",
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
)


trainer.train()


test_data = pd.read_csv('/kaggle/input/ai-mathematical-olympiad-prize/test.csv')
test_data


# научимся генерировать ответ для задач из тестовой выборки

def generate_answer(problem_text, max_new_tokens=20):
    input_text = f"Задача: {problem_text} Ответ:"
    input_ids = tokenizer.encode(input_text, return_tensors="pt")
    input_ids = input_ids.to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids, 
            max_new_tokens=max_new_tokens, 
            num_return_sequences=1,
            temperature=0.8,
            top_p=0.5,
            do_sample=True,
        )

    answer = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    # Очищаем текст (оставляем только часть после "Ответ:")
    if "Ответ:" in answer:
        answer = answer.split("Ответ:")[-1].strip()

    return answer

test_data["answer"] = test_data["problem"].apply(generate_answer)


test_data

