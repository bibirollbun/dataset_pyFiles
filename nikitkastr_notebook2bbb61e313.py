import pandas as pd
from nltk.corpus import stopwords
import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed import init_process_group, destroy_process_group
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import transformers
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import os
import numpy as np
import random
import time

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


full_dataset = pd.read_csv("/kaggle/input/jigsaw-agile-community-rules/train.csv")
submission = pd.read_csv("/kaggle/input/jigsaw-agile-community-rules/test.csv")


for rule in submission['rule'].unique():
    fraction = submission[submission['rule'] == rule]
    positive_examples = list(set((fraction['positive_example_1'].to_list() + 
                         fraction['positive_example_2'].to_list())))
    negative_examples = list(set((fraction['negative_example_1'].to_list() + 
                         fraction['negative_example_2'].to_list())))
    subreddit = fraction['subreddit']
    for i in range(len(positive_examples)):
        new_body = positive_examples[i]
        
        try:
            new_positive_examples = list(set(positive_examples) - set([new_body]))
            new_positive_example_1 = random.choice(new_positive_examples)
            new_positive_example_2 = random.choice(list(set(new_positive_examples) - set([new_positive_example_1])))
        except IndexError: 
            new_positive_example_1 = random.choice(positive_examples)
            new_positive_example_2 = random.choice(positive_examples)    

        try:
            new_negative_example_1 = random.choice(negative_examples)
            new_negative_example_2 = random.choice(list(set(negative_examples) - set([new_negative_example_1])))
        except IndexError:
            new_negative_example_1 = random.choice(negative_examples)
            new_negative_example_2 = random.choice(negative_examples)
        
        subreddit = fraction[(fraction['positive_example_1'] == new_body) | (fraction['positive_example_2'] == new_body)]['subreddit'].iloc[0]

        new_line = {'body': new_body, 'rule': rule, 'subreddit': subreddit,
                   'positive_example_1': new_positive_example_1, 'positive_example_2': new_positive_example_2,
                   'negative_example_1': new_negative_example_1, 'negative_example_2': new_negative_example_2,
                   'rule_violation': 1}
        full_dataset = pd.concat([full_dataset, pd.DataFrame([new_line])], ignore_index=True)

    for i in range(len(negative_examples)):
        new_body = negative_examples[i]

        try:
            new_positive_example_1 = random.choice(positive_examples)
            new_positive_example_2 = random.choice(list(set(new_positive_examples) - set([new_positive_example_1])))
        except IndexError:
            new_positive_example_1 = random.choice(positive_examples)
            new_positive_example_2 = random.choice(positive_examples)

        try:
            new_negative_examples = list(set(negative_examples) - set([new_body]))
            new_negative_example_1 = random.choice(new_negative_examples)
            new_negative_example_2 = random.choice(list(set(new_negative_examples) - set([new_negative_example_1])))
        except IndexError:
            new_negative_example_1 = random.choice(negative_examples)
            new_negative_example_2 = random.choice(negative_examples)
        
        subreddit = fraction[(fraction['negative_example_1'] == new_body) | (fraction['negative_example_2'] == new_body)]['subreddit'].iloc[0]

        new_line = {'body': new_body, 'rule': rule, 'subreddit': subreddit,
                   'positive_example_1': new_positive_example_1, 'positive_example_2': new_positive_example_2,
                   'negative_example_1': new_negative_example_1, 'negative_example_2': new_negative_example_2,
                   'rule_violation': 0}
        full_dataset = pd.concat([full_dataset, pd.DataFrame([new_line])], ignore_index=True)


train, test = train_test_split(full_dataset, test_size=0.2)


stop_words = set(stopwords.words('english'))
def preprocess_text(text):
    preprocessed_text = []
    for word in text.split(" "):
        if not(word in stop_words):
            preprocessed_text.append(word)

    return " ".join(preprocessed_text)

text_columns = ['body', 'rule', 'positive_example_1', 'positive_example_2', 'negative_example_1', 'negative_example_2']
for df in [train, test, submission]:
    for text_column in text_columns:   
        df[f'preprocessed_{text_column}'] = df[f'{text_column}'].map(preprocess_text)


subreddits = full_dataset['subreddit'].unique()
for sub in submission['subreddit'].unique():
    if not(sub in subreddits):
        subreddits.append(sub)
one_hot_subreddits = {} 
for i, subreddit in enumerate(subreddits):
    one_hot_subreddits[subreddit] = i

def one_hot(idx, length):
    one_hot = []
    for i in range(length):
        if i == idx:
            one_hot.append(1)
        else:
            one_hot.append(0)
    return one_hot

for df in [train, test, submission]:
    df['subreddit_one-hot'] = df['subreddit'].map(lambda x: one_hot_subreddits[x])


#model google-bert/bert-base-cased
tokenizer = transformers.AutoTokenizer.from_pretrained("/kaggle/input/google-bertbert-base-cased/pytorch/default/1/tokenizer")
model_body = transformers.AutoModel.from_pretrained("/kaggle/input/google-bertbert-base-cased/pytorch/default/1/model_body").to(device)


def is_link(text):
    for word in text.split(" "):
        if "http://" in word:
            return [0, 1, 0]
        elif "https://" in word:
            return [0, 0, 1]
    return [1, 0, 0]


def do_links(df):
    df['is_link_body'] = df['body'].map(is_link)
    df['is_link_positive_example_1'] = df['positive_example_1'].map(is_link)
    df['is_link_positive_example_2'] = df['positive_example_2'].map(is_link)
    df['is_link_negative_example_1'] = df['negative_example_1'].map(is_link)
    df['is_link_negative_example_2'] = df['negative_example_2'].map(is_link) 


do_links(train)
do_links(test)
do_links(submission)


def is_body_same_with_X(data, X):
    k = (data['is_link_body'] == data[f'{X}_1']) & (data['is_link_body'] == data[f'{X}_2'])
    result = []
    for s in k:
        if s:
            result.append(1)
        else:
            result.append(0)
    return result


for df in [train, test, submission]:
    df['is_body_same_positive_examples'] = is_body_same_with_X(df, "is_link_positive_example")
    df['is_body_same_negative_examples'] = is_body_same_with_X(df, "is_link_negative_example")


for df in [train, test, submission]:
    df['input'] = df['preprocessed_body'] + df['preprocessed_rule']
    df['positive_examples'] = df['preprocessed_positive_example_1'] + df['preprocessed_positive_example_2']
    df['negative_examples'] = df['preprocessed_negative_example_1'] + df['preprocessed_negative_example_2']


for df in [train, test, submission]:
    df['word_count'] = df['body'].map(lambda x: len(x.split(" ")))


def amount_exclamation_per_word(df, column):
    result = np.double(df[column].map(lambda x: x.count("!"))) / df[column].map(lambda x: len(x.split(" ")))
    return result


for df in [train, test, submission]:
    df['body_amount_exclamation_per_word'] = amount_exclamation_per_word(df, 'body')
    df['positive_examples_amount_exclamation_per_word'] = amount_exclamation_per_word(df, 
                                                                                         'positive_examples')
    df['negative_examples_amount_exclamation_per_word'] = amount_exclamation_per_word(df, 
                                                                                         'negative_examples')
    df['dif_body_positive_exclamation_per_word'] = df['body_amount_exclamation_per_word'] - df['positive_examples_amount_exclamation_per_word']
    df['dif_body_negative_exclamation_per_word'] = df['body_amount_exclamation_per_word'] - df['negative_examples_amount_exclamation_per_word']
    df['dif_of_difs_excalamtion'] = df['dif_body_positive_exclamation_per_word'] - df['dif_body_negative_exclamation_per_word']



columns_to_tensor = ['body_amount_exclamation_per_word', 'dif_body_positive_exclamation_per_word', 
                     'dif_body_negative_exclamation_per_word', 'dif_of_difs_excalamtion']
for df in [train, test, submission]:   
    df[columns_to_tensor] = df[columns_to_tensor].map(lambda x: torch.tensor(x))


def caps_ratio(text):
    caps = 0
    for character in text:
        if character.isupper():
            caps += 1
    return torch.tensor(caps / len(text))


for df in [train, test, submission]:
    df['caps_ratio_body'] = df['body'].map(caps_ratio)
    df['caps_ratio_positive'] = df['positive_examples'].map(caps_ratio)
    df['caps_ratio_negative'] = df['negative_examples'].map(caps_ratio)
    df['dif_caps_ratio_body_positive'] = df['caps_ratio_body'] - df['caps_ratio_positive']
    df['dif_caps_ratio_body_negative'] = df['caps_ratio_body'] - df['caps_ratio_negative']
    df['dif_of_difs_caps'] = df['dif_caps_ratio_body_positive'] - df['dif_caps_ratio_body_negative']


def count_punctuation(text):
    punctuation = 0
    for character in text:
        if character in [',', ':', '-', ';']:
            punctuation += 1
    return torch.tensor(punctuation / (len(text.split(" "))))

for df in [train, test, submission]:
    df['punctuation_ratio'] = df['body'].map(count_punctuation)


train['rule_violation'] = train['rule_violation'].map(lambda x: 0.95 if x > 0.5 else 0.05)


#model facebook/nllb-200-distilled-600M
translate_model = transformers.AutoModelForSeq2SeqLM.from_pretrained('/kaggle/input/facebooknllb-200-distilled-600m/transformers/default/1/translate/translate_model').to('cuda:1')
translate_tokenizer = transformers.AutoTokenizer.from_pretrained('/kaggle/input/facebooknllb-200-distilled-600m/transformers/default/1/translate/translate_tokenizer')

en_de = transformers.pipeline('translation', 
                             model=translate_model, tokenizer=translate_tokenizer, 
                             src_lang='eng_Latn', tgt_lang='deu_Latn', max_length = 1024,
                             device=1)
de_en = transformers.pipeline('translation', 
                             model=translate_model, tokenizer=translate_tokenizer, 
                             src_lang='deu_Latn', tgt_lang='eng_Latn', max_length = 1024,
                             device=1)

en_nn = transformers.pipeline('translation',
                             model=translate_model, tokenizer=translate_tokenizer, 
                             src_lang='eng_Latn', tgt_lang='nno_Latn', max_length = 1024,
                             device=1)
nn_en = transformers.pipeline('translation',
                             model=translate_model, tokenizer=translate_tokenizer, 
                             src_lang='nno_Latn', tgt_lang='eng_Latn', max_length = 1024,
                             device=1)

en_af = transformers.pipeline('translation',
                             model=translate_model, tokenizer=translate_tokenizer, 
                             src_lang='eng_Latn', tgt_lang='afr_Latn', max_length = 1024,
                             device=1)
af_en = transformers.pipeline('translation',
                             model=translate_model, tokenizer=translate_tokenizer, 
                             src_lang='afr_Latn', tgt_lang='eng_Latn', max_length = 1024,
                             device=1)

en_sp = transformers.pipeline('translation',
                             model=translate_model, tokenizer=translate_tokenizer, 
                             src_lang='eng_Latn', tgt_lang='spa_Latn', max_length = 1024,
                             device=1)
sp_en = transformers.pipeline('translation',
                             model=translate_model, tokenizer=translate_tokenizer, 
                             src_lang='spa_Latn', tgt_lang='eng_Latn', max_length = 1024,
                             device=1)


languages = ['nn', 'de', 'af', 'sp']
lang_to_model = {'nn': en_nn, 'de': en_de, 'af': en_af, 'sp': en_sp,
                 'back_nn': nn_en, 'back_de': de_en, 'back_af': af_en, 'back_sp': sp_en}

def translate(text, language):
    with torch.no_grad():
        translator = lang_to_model[language]
        translated_text = [x['translation_text'] for x in translator(text, max_length=1024)]
    
    return translated_text


def back_translate(text, n_iterations=1, languages=languages):
    language = languages[random.randint(0, len(languages)-1)]
    translated_text = translate(text, language)
    translated_text = translate(translated_text, f'back_{language}')
    
    if n_iterations > 1:
        for _ in range(n_iterations - 1):
            new_language = languages[random.randint(0, len(languages)-1)]
            while new_language == language:
                new_language = languages[random.randint(0, len(languages)-1)]

            translated_text = translate(translated_text, new_language)      
            translated_text = translate(translated_text, f'back_{new_language}')

            language = new_language
            
    return translated_text


#do vocabulary body - back_translated 
bodies = []
columns_to_translate = ['body', 
                        'positive_example_1', 'positive_example_2', 
                        'negative_example_1', 'negative_example_2'] 
for idx in range(len(train)):
    fraction = train.iloc[idx][columns_to_translate].to_list()
    for text in fraction:
        if not(text in bodies):
            bodies.append(text)
            
translated_bodies = {}
start = time.time()
for idx in range(len(bodies) // 100 + 1):
    fraction = bodies[idx*100:(idx+1)*100]
    translated_bodies_ = back_translate(fraction)
    for body, translated_body in zip(fraction, translated_bodies_):
        translated_bodies[body] = translated_body
    if (time.time() - start) >= 5 * 3600:
        print('break')
        break


rules = train['rule'].unique().tolist()
translated_rules = {}
for rule in rules:
    translated_rules_ = []
    for _ in range(4):
        translated_rule = back_translate(rule, random.randint(2, 4))
        translated_rules_.append(translated_rule)
    translated_rules[rule] = translated_rules_

print((time.time() - start) / 3600)
print(len(translated_bodies) / len(bodies))
print(len(translated_bodies))


feature_columns = ['is_link_body',
                   'is_body_same_positive_examples', 'is_body_same_negative_examples', 
                   'body_amount_exclamation_per_word', 'dif_body_positive_exclamation_per_word',
                   'dif_body_negative_exclamation_per_word', 'dif_of_difs_excalamtion',
                   'caps_ratio_body', 'dif_caps_ratio_body_positive',
                   'dif_caps_ratio_body_negative', 'dif_of_difs_caps',
                   'punctuation_ratio']


same_columns = list(train.columns)
for c in ['preprocessed_body', 'preprocessed_rule', 'positive_examples', 'negative_examples', 'row_id']:
    same_columns.remove(c)


def back_translate_line(line):
    new_body = preprocess_text(translated_bodies[line['body']])
    new_rule = translated_rules[line['rule']][random.randint(0, len(translated_rules[line['rule']]) - 1)][0]
    try:
        pos_example_1 = translated_bodies[line['positive_example_1']]
        pos_example_2 = translated_bodies[line['positive_example_2']]
        neg_example_1 = translated_bodies[line['negative_example_1']]
        neg_example_2 = translated_bodies[line['negative_example_2']]
    except KeyError:
        return 'Failed'

    positive_examples = preprocess_text(pos_example_1 + ' ' + pos_example_2)
    negative_examples = preprocess_text(neg_example_1 + ' ' + neg_example_2)

    new_line = {'preprocessed_body': new_body, 
                'preprocessed_rule': new_rule,
                'positive_examples': positive_examples,
                'negative_examples': negative_examples}

    for column in same_columns:
        new_line[column] = line[column]

    return new_line


initial_len = len(train)

for key in translated_bodies.keys():
    fraction = train[train['body'] == key]
    if isinstance(fraction, pd.Series):
        n_l = back_translate_line(fraction)
        if not (n_l == 'Failed'):
            train = pd.concat([train, pd.DataFrame([n_l])], ignore_index=True)
    elif isinstance(fraction, pd.DataFrame):
        for idx in range(len(fraction)):
            fraction_ = fraction.iloc[idx]
            n_l = back_translate_line(fraction_)
            if not (n_l == 'Failed'):
                train = pd.concat([train, pd.DataFrame([n_l])], ignore_index=True)

print(f'Succes rate: {(len(train) - initial_len) / len(translated_bodies.keys())}')


def print_gpu_memory():
    print(f"Allocated memory: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    print(f"Cached memory: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")


print_gpu_memory()
del en_de
del de_en
del en_sp
del sp_en
del en_af
del af_en
del en_nn
del nn_en
del translate_model
del translate_tokenizer
torch.cuda.empty_cache()
print_gpu_memory()


class RedditDataset(Dataset):
    def __init__(self, data, tokenizer, device, feature_columns, is_submission):
        self.data = data
        self.tokenizer = tokenizer
        self.device = device
        self.feature_columns = feature_columns
        self.is_submission = is_submission
    

    def __len__(self):
        return len(self.data)


    def __getitem__(self, idx):
        body = self.data['preprocessed_body'].iloc[idx]
        rule = self.data['preprocessed_rule'].iloc[idx]
        input_ = self.data['input'].iloc[idx]
        positive_examples = self.data['positive_examples'].iloc[idx]
        negative_examples = self.data['negative_examples'].iloc[idx]
        subreddit = self.data['subreddit_one-hot'].iloc[idx]
        if not(self.is_submission):
            rule_violation = self.data['rule_violation'].iloc[idx]
        
        if not(isinstance(body, str)):
            body = body.to_list()
            rule = rule.to_list()
            input_ = input_.to_list()
            positive_examples, negative_examples = positive_examples.to_list(), negative_examples.to_list()
            subreddit = subreddit.to_list(), 
            if not(self.is_submission):
                rule_violation = rule_violation.to_list()
                rule_violation = torch.tensor(rule_violation).to(self.device)

        body = tokenizer(body, padding = 'longest', truncation=True, return_tensors="pt").to(self.device)
        rule = tokenizer(rule, padding = 'longest', truncation=True, return_tensors="pt").to(self.device)
        input_ = tokenizer(input_, padding = 'longest', truncation=True, return_tensors="pt").to(self.device)
        positive_examples = tokenizer(positive_examples, padding = 'longest', truncation=True, return_tensors="pt").to(self.device)
        negative_examples = tokenizer(negative_examples, padding = 'longest', truncation=True, return_tensors="pt").to(self.device)

    
        features = self.data[self.feature_columns].iloc[idx]        
        if isinstance(features, pd.DataFrame):
            feature_of_idxs = []
            for idx_ in range(len(features)):
                feature_of_idx = []
                feature_line = features.iloc[idx_]
                for i in range(len(feature_line)):
                    if isinstance(feature_line.iloc[i], list):
                        for element in feature_line.iloc[i]:
                            feature_of_idx.append(element)
                    else:
                        feature_of_idx.append(feature_line.iloc[i])
                feature_of_idxs.append(feature_of_idx)
        else:
            feature_of_idxs = []
            for featr in features:
                if isinstance(featr, list):
                    for f in featr:
                        feature_of_idxs.append(f)
                else:
                    feature_of_idxs.append(featr)

        row_ids = self.data.iloc[idx]['row_id']
        if isinstance(row_ids, pd.Series):
            row_ids = row_ids.to_list()
        
        return_ = {'row_ids': row_ids,
                   'body': body, 'rule': rule, 'input': input_,
                   'positive_examples': positive_examples, 'negative_examples': negative_examples, 
                   'subreddit': subreddit, 'feature': feature_of_idxs}
        if not(self.is_submission):
            return_['rule_violation'] = rule_violation
        
        return return_


train_dataset = RedditDataset(train.sample(frac=1), tokenizer, device, feature_columns, False)
test_dataset = RedditDataset(test, tokenizer, device, feature_columns, False)
submission_dataset = RedditDataset(submission, tokenizer, device, feature_columns, True)


def metric(y_true, y_predicted):
    return roc_auc_score(y_true, y_predicted)


class RedditModel(nn.Module):
    def __init__(self, body_transformer, n_features, n_subreddits, device):
        super().__init__()
        self.body_transformer = body_transformer
        self.n_features = n_features
        self.n_subreddits = n_subreddits
        self.n_features = n_features + 3
        self.device = device
        
        self.features = nn.Sequential(
            nn.BatchNorm1d(self.n_features),
            nn.Dropout(0.5),
            nn.Linear(self.n_features, self.n_features // 2),
            nn.Tanh(),
            nn.Dropout(0.5),
            nn.Linear(self.n_features // 2, self.n_features),
            nn.Tanh()
        )

        input_size = 768*3+self.n_features
        self.Classifier = nn.Sequential(
            nn.BatchNorm1d(input_size),
            nn.Dropout(0.5),
            nn.Linear(input_size, input_size // 4),
            nn.Tanh(),
            nn.Dropout(0.5),
            nn.Linear(input_size // 4, 2),
            nn.Softmax(dim=1)
        )

    
    def forward(self, x):
        input_ = x['input']
        body = x['body']
        positive_examples, negative_examples = x['positive_examples'], x['negative_examples']
        feature = x['feature']

        body = {k:v.to(self.device) for k,v in body.items()}
        input_ = {k:v.to(self.device) for k,v in input_.items()}
        positive_examples = {k:v.to(self.device) for k,v in positive_examples.items()}
        negative_examples = {k:v.to(self.device) for k,v in negative_examples.items()}
        body = self.body_transformer(**body).last_hidden_state[:, 0]
        input_ = self.body_transformer(**input_).last_hidden_state[:, 0]
        positive_examples = self.body_transformer(**positive_examples).last_hidden_state[:, 0]
        negative_examples = self.body_transformer(**negative_examples).last_hidden_state[:, 0]

        dist_pos = (body - positive_examples).pow(2).sum(1).sqrt()
        dist_neg = (body - negative_examples).pow(2).sum(1).sqrt()

        for i in range(len(feature)):
            feature[i].append(dist_pos[i])
            feature[i].append(dist_neg[i])
            feature[i].append(dist_pos[i] - dist_neg[i])

            
        feature = torch.tensor(feature).to(self.device)

        feature = self.features(feature)

        ultimate = torch.cat((input_, positive_examples, negative_examples, feature), 
                             dim=1).to(self.device)
        predict = self.Classifier(ultimate)
        return predict[:, 0]
        


class Trainer():
    def __init__(self, 
                 model, criterion, optimizer,
                 train_dataset, test_dataset,
                 metric, batch_size):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.metric = metric
        self.batch_size = batch_size


    def predict_batch(self, batch):
        true_preds = batch['rule_violation'].float()
        preds = self.model(batch)
        loss = self.criterion(preds, true_preds)

        return loss, preds, true_preds
        
    
    def train_batch(self, batch):
        model.train()
        self.optimizer.zero_grad()
        loss, preds, true_preds = self.predict_batch(batch)
        loss.backward()
        self.optimizer.step()

        preds = [pos for pos in preds.detach().cpu().numpy()]
        true_preds = [1.0 if pos > 0.9 else 0.0 for pos in true_preds.float().detach().cpu().numpy()]
        
        return loss, preds, true_preds


    def predict(self, batch):
        self.model.eval()
        with torch.no_grad():
            preds = self.model(batch)
        return preds.detach().cpu().numpy()
         
    
    def train_epoch(self):
        all_train_preds = []
        all_train_true_preds = []
        all_train_losses = []
            
        for i in range((len(self.train_dataset) // self.batch_size) + 1):
            batch = self.train_dataset[i*self.batch_size:(i+1)*self.batch_size]
            train_loss, train_preds, train_true_preds = self.train_batch(batch)

            all_train_losses.append(train_loss.item())
            all_train_preds += train_preds
            all_train_true_preds += train_true_preds

        all_test_preds = []
        all_test_true_preds = []
        for k in range((len(self.test_dataset) // self.batch_size) + 1):
            test_batch = self.test_dataset[k*self.batch_size:(k+1)*self.batch_size]
            test_true_preds = test_batch['rule_violation'].float().detach().cpu().numpy()
            test_preds = self.predict(test_batch)
            test_preds = [pos for pos in test_preds]
            all_test_preds += [pos for pos in test_preds]
            all_test_true_preds += [1.0 if pos > 0.9 else 0.0 for pos in test_true_preds]

        loss = round(sum(all_train_losses) / len(all_train_losses), 3)
        train_metric = round(self.metric(all_train_true_preds, all_train_preds), 3)
        test_metric = round(self.metric(all_test_true_preds, all_test_preds), 3)

        return loss, train_metric, test_metric


    def train(self, num_epochs):
        for epoch in range(num_epochs):
            loss, train_metric, test_metric = self.train_epoch()
            print(f"Epoch: end of {epoch} epoch | loss: {loss}| train_metric: {train_metric}| test_metric: {test_metric}")


n_features = len(train_dataset[1]['feature'])
model = RedditModel(model_body, n_features, len(subreddits), device).to(device)


criterion = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=3e-6, weight_decay=0.1)
num_epochs = 25

trainer = Trainer(model, 
                  criterion, optimizer,
                  train_dataset, test_dataset, metric, 
                  8)


trainer.train(num_epochs)


model.eval()

ids = []
predictions = []
batch_size = 4
for j in range(len(submission_dataset) // batch_size + 1):
    submission_batch = submission_dataset[j*batch_size:(j+1)*batch_size]
    preds = trainer.predict(submission_batch)
    
    for l in range(len(preds)):
        predictions.append(round(preds[l], 3))
        ids.append(submission_batch['row_ids'][l])
submission = {'row_id': ids, 'rule_violation': predictions}
submission = pd.DataFrame(submission)
submission.to_csv('submission.csv')


submission

