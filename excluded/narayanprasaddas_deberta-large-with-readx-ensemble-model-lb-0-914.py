import pandas as pd
import numpy as np
import os 
path_train = '/kaggle/input/fake-or-real-the-impostor-hunt/data/train'
path_test = '/kaggle/input/fake-or-real-the-impostor-hunt/data/test'
train_labels_path = '/kaggle/input/fake-or-real-the-impostor-hunt/data/train.csv'
train_labels = pd.read_csv(train_labels_path)
def make_data_csv(path):
    df = []
    for articles in os.listdir(path):
        data_point = []
        data_point.append(articles)
        for text in os.listdir(path+'/'+articles):
            with open(path+'/'+articles+'/'+text,'r') as file:
                contents = file.read()
                data_point.append(contents)
                file.close()
        df.append(data_point)
    df = pd.DataFrame(df)
    df.columns = ['article_number','text_2','text_1']
    df['id'] = df['article_number'].str.split('_').str[1].astype(int)
    df = df.sort_values(by='id').reset_index(drop=True)
    return df
train_df = make_data_csv(path_train)
test_df = make_data_csv(path_test)
train_df['labels'] = train_labels['real_text_id']
train_df.to_csv('train_dataset.csv')
test_df.to_csv('test_data.csv')


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
TEST_DATA_PATH = '/kaggle/working/test_data.csv'
TRUE_TRAIN_DATA_PATH = '/kaggle/working/train_dataset.csv'
test_df = pd.read_csv(TEST_DATA_PATH)
train_df = pd.read_csv(TRUE_TRAIN_DATA_PATH)
aug_data = []
for _,row in train_df.iterrows():
    if row['labels']==1:
        aug_data.append({'text':row['text_1'],'label':1})
        aug_data.append({'text':row['text_2'],'label':0})
    else:
        aug_data.append({'text':row['text_1'],'label':0})
        aug_data.append({'text':row['text_2'],'label':1})
        
aug_data = pd.DataFrame(aug_data)
aug_data['text'] = aug_data['text'].fillna('')
train_df, val_df = train_test_split(aug_data, test_size=0.15, random_state=42, stratify=aug_data['label'])


import tensorflow as tf
from transformers import AutoTokenizer,TFAutoModelForSequenceClassification,create_optimizer
from transformers import logging
import gc
from numba import cuda
tf.keras.mixed_precision.set_global_policy('mixed_float16')
logging.set_verbosity_error()
LR = 3e-5
MAX_TOKEN_LEN = 512
NUM_EPOCHS = 3
PATIENCE = 2  
SAVE_PATH = './best_models_READX'
BATCH_SIZE = 2
STRATEGY = tf.distribute.MirroredStrategy()
original_weight = 1.0
pseudo_weight = 0.8  
WT_DECAY = 0.01
PRED_SAVE_PATH = './pred_proba'
MODEL_NAMES =[
    'google/electra-base-discriminator',
    'albert-base-v2',
    'microsoft/deberta-v3-base',
    'roberta-base',
    'distilbert-base-uncased'
]
SAVED_PATHS = []


def pack_data(df,tokenizer,mode='eval'):
    print(f'Tokenizing Data...')
    encodings = tokenizer(
        df['text'].fillna('').tolist(),
        truncation=True,
        padding=True,
        max_length=MAX_TOKEN_LEN,
        return_tensors="tf"
    )
    if mode == 'eval':
        labels = df['label'].values
        dataset = tf.data.Dataset.from_tensor_slices((
            dict(encodings),
            labels
        ))
    else:
        dataset = tf.data.Dataset.from_tensor_slices((dict(encodings)))
    dataset = dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    
    return dataset

def train_model(model_name,path):
    patience_counter = 0
    best_val_loss = float('inf')
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if 'deberta-v3' in model_name:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        
    train_dataset = pack_data(train_df,tokenizer)
    val_dataset = pack_data(val_df,tokenizer)
    
    with STRATEGY.scope():
        num_train_steps = len(train_df) * NUM_EPOCHS
        model = TFAutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
        optimizer, _ = create_optimizer(init_lr=LR, 
                                            num_warmup_steps=int(0.1 * num_train_steps), 
                                            num_train_steps=num_train_steps,
                                            weight_decay_rate=WT_DECAY
                                        )
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True) 
        model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
        
    safe_model_name = model_name.replace('/', '_')
    final_save_path = f'{SAVE_PATH}/{safe_model_name}'
    SAVED_PATHS.append(final_save_path)
    
    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")
        history = model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=1 
        )
        current_val_loss = history.history['val_loss'][0]
        if current_val_loss < best_val_loss:
            print(f"Validation loss improved from {best_val_loss:.4f} to {current_val_loss:.4f}. Saving model.")
            best_val_loss = current_val_loss
            
            print(f"Saving model to {final_save_path}")
            model.save_pretrained(final_save_path)
            tokenizer.save_pretrained(final_save_path)
            
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"Validation loss did not improve. Patience: {patience_counter}/{PATIENCE}")
    
        if patience_counter >= PATIENCE:
            print("Early stopping triggered. Training stopped.")
            break
    print("\n--- Training Finished ---")


for model_name in MODEL_NAMES:
    print(f"Starting Training of {model_name} model...")
    try:
        train_model(model_name=model_name,path=SAVE_PATH)
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
           print(f"Skipping {model_name} due to OutOfMemoryError.")
        else:
            raise e
    finally:
        print("Clearing session...")
        tf.keras.backend.clear_session()
        gc.collect()


model_predictions = {}
test_df_text1 = test_df[['text_1']].rename(columns={'text_1': 'text'})
test_df_text2 = test_df[['text_2']].rename(columns={'text_2': 'text'})

for model_path in SAVED_PATHS:
    model_name = os.path.basename(model_path)
    print(f"Predicting with {model_name}...")
    model = TFAutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    test_dataset1 = pack_data(test_df_text1, tokenizer, mode='pred')
    logits1 = model.predict(test_dataset1).logits
    probs1 = tf.nn.softmax(logits1)[:, 1].numpy() 

    test_dataset2 = pack_data(test_df_text2, tokenizer, mode='pred')
    logits2 = model.predict(test_dataset2).logits
    probs2 = tf.nn.softmax(logits2)[:, 1].numpy() 

    model_predictions[f'{model_name}_prob1'] = probs1
    model_predictions[f'{model_name}_prob2'] = probs2

    tf.keras.backend.clear_session()
    gc.collect()
del model
del tokenizer
predictions_df = pd.DataFrame(model_predictions)
predictions_df['id'] = test_df['id'] 
PRED_SAVE_PATH = 'Readx_ensemble_test_predictions.csv'
predictions_df.to_csv(PRED_SAVE_PATH, index=False)
print(f"\nIndividual model predictions saved to {PRED_SAVE_PATH}")
print(predictions_df.head())
# prob1_cols = [col for col in predictions_df.columns if '_prob1' in col]
# prob2_cols = [col for col in predictions_df.columns if '_prob2' in col]

# final_avg_probs_text1 = predictions_df[prob1_cols].mean(axis=1)
# final_avg_probs_text2 = predictions_df[prob2_cols].mean(axis=1)

# final_labels = np.where(final_avg_probs_text1 > final_avg_probs_text2, 1, 2)
# def make_submission_csv(results):
#     df_results = pd.DataFrame(results)
#     output_df = df_results.copy()
#     output_df.columns = ['real_text_id']
#     output_df.reset_index(inplace=True)
#     output_df.rename(columns={'index': 'id'}, inplace=True)
#     output_df.to_csv('Deberta_Roberta_Electra_DisitilBERT_xlnet.csv', index=False)
#     return output_df
# pred_df = make_submission_csv(final_labels) # This Yielded a score of 0.885


import pandas as pd
import numpy as np
PSUEDO_PROB_DATA_PATH = '/kaggle/working/Readx_ensemble_test_predictions.csv'
pseudo_train_data = pd.read_csv(PSUEDO_PROB_DATA_PATH)

prob_1_cols = [col for col in pseudo_train_data.columns if '_prob1' in col]
prob_2_cols = [col for col in pseudo_train_data.columns if '_prob2' in col]

pseudo_train_data['avg_prob1'] = pseudo_train_data[prob_1_cols].mean(axis=1)
pseudo_train_data['std_prob1'] = pseudo_train_data[prob_1_cols].std(axis=1)
pseudo_train_data['avg_prob2'] = pseudo_train_data[prob_2_cols].mean(axis=1)
pseudo_train_data['std_prob2'] = pseudo_train_data[prob_2_cols].std(axis=1)


CONFIDENCE_THRESH = 0.80
AGREEMENT_THRESH = 0.20
NUM_PSEUDO_PAIRS = 200
mask1  = (pseudo_train_data['avg_prob1'] > CONFIDENCE_THRESH) & (pseudo_train_data['avg_prob2'] < 1 - CONFIDENCE_THRESH)
mask2 = (pseudo_train_data['avg_prob2'] > CONFIDENCE_THRESH) & (pseudo_train_data['avg_prob1'] < 1 - CONFIDENCE_THRESH)
agreement_mask = (pseudo_train_data['std_prob1'] < AGREEMENT_THRESH) & (pseudo_train_data['std_prob2'] < AGREEMENT_THRESH)
high_confidence_pairs = pseudo_train_data[(mask1 | mask2) & agreement_mask]
pseudo_train_data['confidence_gap'] = (pseudo_train_data['avg_prob1'] - pseudo_train_data['avg_prob2']).abs()

pseudo_train_data_sorted = pseudo_train_data.sort_values(by='confidence_gap', ascending=False)
pseudo_train_data_sorted
top_confidence_pairs = pseudo_train_data_sorted.head(NUM_PSEUDO_PAIRS)

top_confidence_pairs['id'] = top_confidence_pairs['id'].astype('int64') 

print(top_confidence_pairs)
psuedo_label_dataset = pd.merge(test_df,
                                top_confidence_pairs,
                               left_on='id',
                               right_on='id',
                               how='inner')

psuedo_label_dataset
psuedo_label_dataset['label'] = np.where(psuedo_label_dataset['avg_prob1'] > psuedo_label_dataset['avg_prob2'], 1, 2)
psuedo_label_dataset = psuedo_label_dataset[['id', 'text_1', 'text_2', 'label']]
psuedo_label_dataset.to_csv('READX_top_200.csv')
print("Final Pseudo-Labeled DataFrame and made READX_top_200.csv :")
psuedo_label_dataset


# Hyper Parameters
import tensorflow as tf
from transformers import AutoTokenizer,TFAutoModelForSequenceClassification,create_optimizer,AutoConfig
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from numba import cuda

LR = 1.5e-5
MAX_TOKEN_LEN = 256
NUM_EPOCHS = 2
PATIENCE = 2  
BATCH_SIZE = 1
ORIGINAL_WEIGHT = 1.0
PSEUDO_WEIGHT = 0.6
WT_DECAY = 0.1
CONFIDENCE_THRESH = 0.85 
AGREEMENT_THRESH = 0.10
NUM_PSEUDO_PAIRS = 200
STRATEGY = tf.distribute.MirroredStrategy()
TEST_DATA_PATH = '/kaggle/working/test_data.csv'
TRUE_TRAIN_DATA_PATH = '/kaggle/working/train_dataset.csv'
PSEUDO_TRAIN_DATA_PATH = '/kaggle/working/READX_top_200.csv'
# PSEUDO_TRAIN_DATA_PATH = '/kaggle/input/assests-for-real-or-fake-the-imposter-hunt-in-text/READX_200_psuedo_label_IDs.csv'

MODEL_NAME = 'microsoft/deberta-v3-large'
SAVE_PATH = f'./best_model_deBERTa_large'
PRED_SAVE_PATH = './pred_proba'
SAVED_PATHS = []


def augment_data(df):
    if 'label' in df.columns:
        df['labels'] = df['label']
    real_texts = pd.concat([
        df.loc[df['labels'] == 1, 'text_1'],
        df.loc[df['labels'] == 2, 'text_2']
    ])
    fake_texts = pd.concat([
        df.loc[df['labels'] == 1, 'text_2'],
        df.loc[df['labels'] == 2, 'text_1']
    ])
    df_real = pd.DataFrame({'text': real_texts, 'label': 1})
    df_fake = pd.DataFrame({'text': fake_texts, 'label': 0})
    aug_data = pd.concat([df_real, df_fake], ignore_index=True)
    aug_data = aug_data.sample(frac=1).reset_index(drop=True)
    return aug_data
test_df = pd.read_csv(TEST_DATA_PATH)
train_df = pd.read_csv(TRUE_TRAIN_DATA_PATH)
pseudo_train_data = pd.read_csv(PSEUDO_TRAIN_DATA_PATH)

if 'input' in PSEUDO_TRAIN_DATA_PATH:
    new_df = test_df.loc[pseudo_train_data['id']] 
    new_df['labels'] = pseudo_train_data['labels'].values
    pseudo_train_data = new_df.copy()
    if 'Unnamed: 0' in pseudo_train_data.columns:
        pseudo_train_data.drop(['Unnamed: 0'],axis=1,inplace=True)

train_df_, val_df = train_test_split(train_df, 
                                     test_size=0.30, 
                                     random_state=42, 
                                     stratify=train_df['labels'])
aug_train_df = augment_data(train_df_)
val_df = augment_data(val_df)
aug_psuedo_train_df = augment_data(pseudo_train_data)
aug_train_df['source'] = 'original'
aug_psuedo_train_df['source'] = 'pseudo'
joint_train_df = pd.concat([aug_train_df,aug_psuedo_train_df],ignore_index=True)
joint_train_df['sample_weight'] = np.where(
    joint_train_df['source'] == 'original', 
    ORIGINAL_WEIGHT,      
    PSEUDO_WEIGHT     
)
joint_train_df = joint_train_df.sample(frac=1).reset_index(drop=True)


def pack_data(df,tokenizer,mode='train'):
    print(f'Tokenizing Data...')
    cols = df.columns
    encodings = tokenizer(
        df['text'].fillna('').tolist(),
        truncation=True,
        padding=True,
        max_length=MAX_TOKEN_LEN,
        return_tensors="tf"
    )
    if 'sample_weight' in cols:
        labels = df['label'].values
        weights = df['sample_weight'].values
        dataset = tf.data.Dataset.from_tensor_slices((
            dict(encodings),
            labels,
            weights
        ))
    elif 'labels' or 'label' in cols:
        labels = df['label'].values
        dataset = tf.data.Dataset.from_tensor_slices((
            dict(encodings),
            labels,
        ))
    else:
        dataset = tf.data.Dataset.from_tensor_slices((dict(encodings)))
    dataset = dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return dataset
from transformers import logging
import gc
tf.keras.mixed_precision.set_global_policy('mixed_float16')
logging.set_verbosity_error()
def train_model(model_name,
                train_df,
                val_df,
                path,
                config):
    patience_counter = 0
    best_val_loss = float('inf')
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if 'deberta-v3' in model_name:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        
    train_dataset = pack_data(train_df,tokenizer,mode='train')
    val_dataset = pack_data(val_df,tokenizer,mode='val')
    
    with STRATEGY.scope():
        num_train_steps = len(train_df) * NUM_EPOCHS
        model = TFAutoModelForSequenceClassification.from_pretrained(model_name,
                                                                     config=config)
        optimizer, _ = create_optimizer(init_lr=LR, 
                                            num_warmup_steps=int(0.1 * num_train_steps), 
                                            num_train_steps=num_train_steps,
                                            weight_decay_rate=WT_DECAY
                                        )
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True) 
        model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
        
    safe_model_name = model_name.replace('/', '_')
    final_save_path = f'{SAVE_PATH}/{safe_model_name}'
    SAVED_PATHS.append(final_save_path)
    
    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")
        history = model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=1 
        )
        current_val_loss = history.history['val_loss'][0]
        if current_val_loss < best_val_loss:
            print(f"Validation loss improved from {best_val_loss:.4f} to {current_val_loss:.4f}. Saving model.")
            best_val_loss = current_val_loss
            
            print(f"Saving model to {final_save_path}")
            model.save_pretrained(final_save_path)
            tokenizer.save_pretrained(final_save_path)
            
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"Validation loss did not improve. Patience: {patience_counter}/{PATIENCE}")
    
        if patience_counter >= PATIENCE:
            print("Early stopping triggered. Training stopped.")
            break
    print("\n--- Training Finished ---")


# Pre-Training 
config = AutoConfig.from_pretrained(MODEL_NAME)
config.classifier_dropout = 0.1 
config.num_labels = 2
train_model(model_name=MODEL_NAME,
            train_df=aug_train_df,
            val_df=val_df,
            path=SAVE_PATH,
            config=config
           )
tf.keras.backend.clear_session()
gc.collect()


# FIne-Tuning
SAVED_PATHS = ['/kaggle/working/best_model_deBERTa_large/microsoft_deberta-v3-large']
SAVE_PATH = SAVE_PATH+'_fine_tuned'
LR = 5e-6
config = AutoConfig.from_pretrained(SAVED_PATHS[0])
config.classifier_dropout = 0.1 
config.num_labels = 2
train_model(
            model_name=SAVED_PATHS[0],
            train_df=joint_train_df,
            val_df=val_df,
            path=SAVE_PATH,
            config=config
            )
tf.keras.backend.clear_session()
gc.collect()


tf.keras.backend.clear_session()
gc.collect()

CURR_SAVE_PATHS = ['']
test_df_text1 = test_df[['text_1']].rename(columns={'text_1': 'text'})
test_df_text2 = test_df[['text_2']].rename(columns={'text_2': 'text'})
model_predictions = {}
tf.keras.mixed_precision.set_global_policy('mixed_float16')
for model_path in CURR_SAVE_PATHS:
    print(f'---------------predicting with saved model{model_path}---------------')
    print('---------------Loading Model and Tokenizer---------------')
    with STRATEGY.scope():
        model = TFAutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    print('---------------Tokenizing and Predicting on Text1---------------')
    
    test_dataset1 = pack_data(test_df_text1, tokenizer, mode='pred')
    logits1 = model.predict(test_dataset1).logits
    probs1 = tf.nn.softmax(logits1)[:, 1].numpy() 
    
    print('---------------Tokenizing and Predicting on Text2---------------')
    test_dataset2 = pack_data(test_df_text2, tokenizer, mode='pred')
    logits2 = model.predict(test_dataset2).logits
    probs2 = tf.nn.softmax(logits2)[:, 1].numpy() 

    model_predictions[f'{model_path}_prob1'] = probs1
    model_predictions[f'{model_path}_prob2'] = probs2
    print('---------------Finished Model Predictions---------------')
    tf.keras.backend.clear_session()
    gc.collect()
predictions_df = pd.DataFrame(model_predictions)
predictions_df['id'] = test_df['id'] 
# PRED_SAVE_PATH = 'deberta(0.91493)_large_test_predictions.csv'
predictions_df.to_csv(PRED_SAVE_PATH, index=False)
print(f"\nIndividual model predictions saved to {PRED_SAVE_PATH}")
print(predictions_df.head())

prob1_cols = [col for col in predictions_df.columns if '_prob1' in col]
prob2_cols = [col for col in predictions_df.columns if '_prob2' in col]

final_avg_probs_text1 = predictions_df[prob1_cols].mean(axis=1)
final_avg_probs_text2 = predictions_df[prob2_cols].mean(axis=1)

final_labels = np.where(final_avg_probs_text1 > final_avg_probs_text2, 1, 2)
def make_submission_csv(results,name=None):
    df_results = pd.DataFrame(results)
    output_df = df_results.copy()
    output_df.columns = ['real_text_id']
    output_df.reset_index(inplace=True)
    output_df.rename(columns={'index': 'id'}, inplace=True)
    if name!=None:
        output_df.to_csv(name, index=False)
    return output_df
pred_df = make_submission_csv(final_labels,name='sample_submission.csv')

