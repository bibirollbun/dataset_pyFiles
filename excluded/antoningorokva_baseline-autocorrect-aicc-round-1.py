import datasets
import pandas as pd
from tqdm.auto import tqdm

ds = datasets.load_dataset("csv", data_files="/kaggle/input/autocorrect-aicc-round-1-2/train.csv")
ds = ds["train"].train_test_split(seed=42)
ds


train_ds = ds["train"]
val_ds = ds["test"]

# for later evaluation
val_ds_input = val_ds.select_columns("misspell")
val_ds_solution = val_ds["text"]


pd.DataFrame(val_ds)["text"].str.len().describe()


pd.DataFrame(val_ds)["misspell"].str.len().describe()


from tokenizers import Tokenizer, models, pre_tokenizers, processors

tokenizer = Tokenizer(models.WordLevel(unk_token="<UNK>"))
tokenizer.pre_tokenizer = pre_tokenizers.Split("", "isolated")
tokenizer.enable_padding(pad_token="<PAD>")


trainer = tokenizer.model.get_trainer()
trainer.vocab_size = 1000
trainer.special_tokens = ["<PAD>", "<UNK>", "<SOS>", "<EOS>"]


def ds_iterator():
    for row in train_ds:
        yield row["text"]
        yield row["misspell"]

tokenizer.train_from_iterator(ds_iterator(), trainer=trainer)


tokenizer.post_processor = processors.TemplateProcessing(
    single="<SOS> $0 <EOS>",
    special_tokens=[("<SOS>", 2), ("<EOS>", 3)]
)


from transformers import PreTrainedTokenizerFast
tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer)
tokenizer.add_special_tokens({"pad_token": "<PAD>", "unk_token": "<UNK>", "cls_token": "<SOS>", "eos_token": "<EOS>"})
tokenizer


def tokenize_fn(examples):
    input_tokens = tokenizer(examples['misspell'], padding='max_length', truncation=True, max_length=768, return_tensors='pt')['input_ids']
    label_tokens = tokenizer(examples['text'], padding='max_length', truncation=True, max_length=768, return_tensors='pt')['input_ids']

    input_lengths = (input_tokens != tokenizer.pad_token_id).sum(dim=1)
    label_lengths = (label_tokens != tokenizer.pad_token_id).sum(dim=1)

    return {
        'input_ids': input_tokens,
        'labels': label_tokens,
        'input_lengths': input_lengths,
        'label_lengths': label_lengths,
    }


from torch.utils.data import DataLoader

train_ds = train_ds.map(tokenize_fn, batched=True, remove_columns=['text', 'misspell'])
train_ds.set_format("torch")
val_ds = val_ds.map(tokenize_fn, batched=True, remove_columns=['text', 'misspell'])
val_ds.set_format("torch")
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)


import torch
import torch.nn as nn
import torch.optim as optim

class LSTMAutocorrect(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, lens):
        x = self.embedding(x)
        packed = nn.utils.rnn.pack_padded_sequence(x, lens.cpu(), batch_first=True, enforce_sorted=False)
        packed_outputs, (_, _) = self.lstm(packed)
        x, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs, batch_first=True)
        x = self.fc(x)
        return x


vocab_size = tokenizer.vocab_size
embed_size = 64
hidden_size = 128
num_epochs = 10
learning_rate = 0.001
device = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize model, loss function, and optimizer
model = LSTMAutocorrect(vocab_size, embed_size, hidden_size).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


for epoch in range(num_epochs):
    model.train()
    for batch in train_loader:
        optimizer.zero_grad()

        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)
        lengths = batch['input_lengths'].to(device)

        outputs = model(input_ids, lengths)
        loss = criterion(outputs.view(-1, vocab_size), labels.view(-1))
        loss.backward()
        optimizer.step()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")


def predict_sentence(model, tokenizer, input_ids, device="cpu"):
    """
    input_ids: tensor of shape (1, seq_len) with tokenized input
    """
    model.eval()
    with torch.no_grad():
        input_ids = input_ids.to(device)
        outputs = model(input_ids, torch.tensor([input_ids.size(1)], dtype=torch.int64))  # (1, seq_len, vocab_size)
        predictions = outputs.argmax(dim=-1)  # (1, seq_len)

    # Convert ids back to tokens / string
    predicted_tokens = tokenizer.convert_ids_to_tokens(predictions[0][1:-1].tolist())
    return "".join(predicted_tokens)

# Suppose you have a test sentence
test_text = " Good morning rveryone. How are you. "  # misspelled input

# Tokenize and convert to tensor
input_ids = tokenizer.encode(test_text, return_tensors="pt")

# Get prediction
corrected = predict_sentence(model, tokenizer, input_ids, device=device)
print("Input:", test_text)
print("Corrected:", corrected)


test_ds = datasets.load_dataset("csv", data_files="/kaggle/input/autocorrect-aicc-round-1-2/test.csv")
# test_ds = val_ds_input # for validation
test_ds = test_ds["train"] # for test
test_ds


def tokenize_test_fn(examples):
    input_tokens = tokenizer(examples['misspell'], padding='max_length', truncation=True, max_length=4096, return_tensors='pt')['input_ids']
    input_lengths = (input_tokens != tokenizer.pad_token_id).sum(dim=1)

    return {
        'input_ids': input_tokens,
        'input_lengths': input_lengths
    }

test_ds = test_ds.map(tokenize_test_fn, batched=True, remove_columns=['misspell'])
test_ds.set_format("torch")

test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)


def wrap_model_prediction(model, batch, device):
    input_ids = batch['input_ids'].to(device)
    lengths = batch['input_lengths'].to(device)

    return model(input_ids, lengths)


%%time
device = "cpu" # change to cuda for validation
preds_all = []

model.eval()
model.to(device)
with torch.no_grad():
    for batch in tqdm(test_loader):
        outputs = wrap_model_prediction(model, batch, device)
        predictions = outputs.argmax(dim=-1)
        preds_all.append(predictions.cpu())


results = [] # convert tokens back to string, excluded from timed evaluation as this takes quite a while
for pred in tqdm(preds_all):
    results += [ "".join(tokenizer.convert_ids_to_tokens(x, skip_special_tokens=True)) for x in pred ]


import pandas as pd

df = pd.DataFrame(results)
df = df.reset_index()
df.columns = ["id", "corrected"]
df.to_csv("/kaggle/working/test_submission.csv", index=False)

print("test_submission.csv generated!")


#! pip install jiwer evaluate --quiet
#import evaluate

#cer = evaluate.load("cer")


#import pandas as pd

#submission = pd.read_csv("test_submission.csv")
# solution = pd.DataFrame(pd.Series(val_ds_solution)) # for validation
#solution = pd.read_csv("test_sol.csv") # for testing


#cer.compute(
#    predictions=submission["corrected"],
#    references=solution.iloc[:, 0]
#)

