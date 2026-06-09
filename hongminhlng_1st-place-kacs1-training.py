!mkdir /kaggle/tmp
!gdown https://drive.google.com/file/d/1RrwWM4o_pZI865sPT4pdluQMwHd-9QWC/view?usp=sharing --fuzzy -O /kaggle/tmp/VulGen.zip


%%capture
#!unzip "/kaggle/tmp/VulGen.zip" -d /kaggle/working/VulGen
!mkdir /kaggle/tmp/VulGen
!unzip "/kaggle/tmp/VulGen.zip" -d /kaggle/tmp/VulGen


!ls /kaggle/tmp/VulGen/VulGen


!pip install -q -U bitsandbytes
!pip install -q -U git+https://github.com/huggingface/transformers.git
!pip install -q -U git+https://github.com/huggingface/peft.git
# !pip install -q -U git+https://github.com/huggingface/accelerate.git
!pip install -q accelerate==1.6.0 


%%writefile vulrepair_vulgen_beam1_4bit.py
# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import, division, print_function
import argparse
import logging
import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler
# from transformers import (AdamW, get_linear_schedule_with_warmup, 
#                           T5ForConditionalGeneration, RobertaTokenizer)
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from tokenizers.pre_tokenizers import Whitespace
from tqdm import tqdm
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
import datasets
from sklearn.model_selection import train_test_split
from difflib import SequenceMatcher as SM

# Use all 4 GPUs
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

cpu_cont = 16
logger = logging.getLogger(__name__)

class InputFeatures(object):
    """A single training/test features for a example."""
    def __init__(self,
                 name,
                 input_ids,
                 label,
                 decoder_input_ids):
        self.name = name
        self.input_ids = input_ids
        self.label = label
        self.decoder_input_ids = decoder_input_ids
        

class TextDataset(Dataset):
    def __init__(self, tokenizer, args, train_data=None, val_data=None, file_type="train"):
        if file_type == "train":
            names = train_data["name"].tolist()
            sources = train_data["source"].tolist()
            labels = train_data["target"].tolist()
        elif file_type == "eval":
            names = val_data["name"].tolist()
            sources = val_data["source"].tolist()
            labels = val_data["target"].tolist()
        elif file_type == "test":
            data = datasets.load_dataset("MickyMike/cve_fixes", split="test")
            sources = data["source"]
            labels = data["target"]
        self.examples = []
        for i in tqdm(range(len(sources))):
            self.examples.append(convert_examples_to_features(names[i], sources[i], labels[i], tokenizer, args))
        if file_type == "train":
            for example in self.examples[:3]:
                    logger.info("*** Example ***")
                    logger.info("label: {}".format(example.label))
                    logger.info("input_ids: {}".format(' '.join(map(str, example.input_ids))))
                    logger.info("decoder_input_ids: {}".format(' '.join(map(str, example.decoder_input_ids))))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):       
        return self.examples[i].name, self.examples[i].input_ids, self.examples[i].input_ids.ne(0), self.examples[i].label, self.examples[i].decoder_input_ids


def convert_examples_to_features(name, source, label, tokenizer, args):
    # encode - subword tokenize
    source_ids = tokenizer.encode(source, truncation=True, max_length=args.encoder_block_size, padding='max_length', return_tensors='pt')
    decoder_input_ids = tokenizer.encode(label, truncation=True, max_length=args.decoder_block_size, padding='max_length', return_tensors='pt')
    label = tokenizer.encode(label, truncation=True, max_length=args.decoder_block_size, padding='max_length', return_tensors='pt')
    return InputFeatures(name, source_ids, label, decoder_input_ids)

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def train(args, train_dataset, model, tokenizer, eval_dataset):
    """ Train the model """
    # build dataloader
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, num_workers=0)
    
    args.max_steps = args.epochs * len(train_dataloader)

    # evaluate model per epoch
    args.save_steps = len(train_dataloader) * 1
   
    args.warmup_steps = args.max_steps // 5
    # model.to(args.device)

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                num_training_steps=args.max_steps)
    
    # multi-gpu training
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.train_batch_size//max(args.n_gpu, 1))
    logger.info("  Total train batch size = %d",args.train_batch_size*args.gradient_accumulation_steps)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", args.max_steps)
    
    global_step = 0
    tr_loss, logging_loss, avg_loss, tr_nb, tr_num, train_loss = 0.0, 0.0, 0.0, 0, 0, 0
    best_loss = 100

    writer_path = "tb/codet5_training_loss"
    writer = SummaryWriter(writer_path)

    model.zero_grad()

    for idx in range(args.epochs): 
        bar = tqdm(train_dataloader, total=len(train_dataloader))
        tr_num = 0
        train_loss = 0
        for step, batch in enumerate(bar):
            #(input_ids, attention_mask, labels, decoder_input_ids) = [x.squeeze(1).to(args.device) for x in batch[1:]]
            (input_ids, attention_mask, labels, decoder_input_ids) = [x.squeeze(1) for x in batch[1:]]
            model.train()
            # the forward function automatically creates the correct decoder_input_ids
            loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels).loss
            if args.n_gpu > 1:
                loss = loss.mean()
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            tr_loss += loss.item()
            tr_num += 1
            train_loss += loss.item()
            if avg_loss == 0:
                avg_loss = tr_loss
            avg_loss = round(train_loss/tr_num,5)
            bar.set_description("epoch {} loss {}".format(idx,avg_loss))
            
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()  
                global_step += 1
                output_flag = True
                avg_loss = round(np.exp((tr_loss - logging_loss) /(global_step- tr_nb)),4)
                if global_step % args.save_steps == 0:
                    # placeholder of evaluation
                    #test(args, model, tokenizer, eval_dataset)
                    eval_loss = evaluate(args, model, tokenizer, eval_dataset, eval_when_training=True)    
                    # Save model checkpoint
                    if eval_loss < best_loss:
                        best_loss = eval_loss
                        logger.info("  "+"*"*20)  
                        logger.info("  Best Loss:%s",round(best_loss,4))
                        logger.info("  "+"*"*20)                          
                        checkpoint_prefix = 'checkpoint-best-loss'
                        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))                        
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)                        
                        model_to_save = model.module if hasattr(model,'module') else model
                        output_dir = os.path.join(output_dir, '{}'.format('model.bin')) 
                        torch.save(model_to_save.state_dict(), output_dir)
                        logger.info("Saving model checkpoint to %s", output_dir)

def clean_tokens(tokens):
    tokens = tokens.replace("<pad>", "")
    tokens = tokens.replace("<s>", "")
    tokens = tokens.replace("</s>", "")
    tokens = tokens.strip("\n")
    tokens = tokens.strip()
    return tokens

def evaluate(args, model, tokenizer, eval_dataset, eval_when_training=False):
    #build dataloader
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, num_workers=0)
    # multi-gpu evaluate
    if args.n_gpu > 1 and eval_when_training is False:
        model = torch.nn.DataParallel(model)
    # Eval!
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    model.eval()
    
    eval_loss, num = 0, 0
    bar = tqdm(eval_dataloader, total=len(eval_dataloader))
    for batch in bar:
        (input_ids, attention_mask, labels, decoder_input_ids) = [x.squeeze(1).to(args.device) for x in batch[1:]]
        loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels).loss
        if args.n_gpu > 1:
            loss = loss.mean()
        eval_loss += loss.item()
        num += 1
    eval_loss = round(eval_loss/num,5)
    model.train()
    logger.info("***** Eval results *****")
    logger.info(f"Evaluation Loss: {str(eval_loss)}")
    return eval_loss


def test(args, model, tokenizer: AutoTokenizer, test_dataset, best_threshold=0.5, i=0):
    # Build dataloader
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.eval_batch_size, num_workers=0)
    
    # Test!
    logger.info("***** Running Test *****")
    logger.info("  Num examples = %d", len(test_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    nb_eval_steps = 0
    model.eval()
    accuracy = []
    names = []
    raw_predictions = []
    ground_truths = []
    bar = tqdm(test_dataloader, total=len(test_dataloader))

    # Access the underlying model for DataParallel
    model_to_use = model.module if isinstance(model, torch.nn.DataParallel) else model
    
    for batch in bar:
        (input_ids, attention_mask, labels, decoder_input_ids) = [x.squeeze(1) for x in batch[1:]]
        namesx = batch[0]

        with torch.no_grad():
            beam_outputs = model_to_use.generate(input_ids=input_ids,
                                                attention_mask=attention_mask,
                                                do_sample=False,
                                                num_beams=args.num_beams,
                                                num_return_sequences=args.num_beams,
                                                max_length=args.decoder_block_size)
        
        # Move beam_outputs and decoder_input_ids to CPU
        beam_outputs = beam_outputs.detach().cpu().tolist()
        decoder_input_ids = decoder_input_ids.detach().cpu().tolist()
        
        # Use batch_decode to reduce tokenization calls
        predictions = tokenizer.batch_decode(beam_outputs, skip_special_tokens=False)
        ground_truths_batch = tokenizer.batch_decode([decoder_input_ids[0]] * len(beam_outputs), skip_special_tokens=False)
        
        # Clean tokens
        predictions_cleaned = [clean_tokens(pred) for pred in predictions]
        ground_truths_cleaned = [clean_tokens(gt) for gt in ground_truths_batch]
        
        # Compute batch accuracy
        batch_accuracy = [1 if pred == gt else 0 for pred, gt in zip(predictions_cleaned, ground_truths_cleaned)]
        
        # Update lists
        names.extend([namesx[0]] * len(beam_outputs))
        raw_predictions.extend(predictions_cleaned)
        ground_truths.extend(ground_truths_cleaned)
        accuracy.extend(batch_accuracy)
        
        nb_eval_steps += 1
    
    # Calculate accuracy
    test_result = round(sum(accuracy) / len(accuracy), 4)
    logger.info("***** Test results *****")
    logger.info(f"Test Accuracy: {str(test_result)}")

    # Save results to file
    df = pd.DataFrame({
        "names": names,
        "raw_predictions": raw_predictions,
        "ground_truth": ground_truths,
        "is_correct": accuracy
    })
    df.to_csv(f"../data/raw_predictions/CodeT5/VulRepair_raw_preds_final_beam10_part{i}.csv")

def main():
    parser = argparse.ArgumentParser()
    # Params
    parser.add_argument("--output_dir", default=None, type=str, required=False,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--model_type", default="t5", type=str,
                        help="The model architecture to be fine-tuned.")
    parser.add_argument("--encoder_block_size", default=-1, type=int,
                        help="Optional input sequence length after tokenization.")
    parser.add_argument("--decoder_block_size", default=-1, type=int,
                        help="Optional input sequence length after tokenization.")
    parser.add_argument("--num_beams", default=50, type=int,
                        help="Beam size to use when decoding.")                          
    parser.add_argument("--model_name_or_path", default=None, type=str,
                        help="The model checkpoint for weights initialization.")
    parser.add_argument("--config_name", default="", type=str,
                        help="Optional pretrained config name or path if not the same as model_name_or_path")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path")

    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Run evaluation during training at each logging step.")

    parser.add_argument("--train_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--epochs', type=int, default=1,
                        help="training epochs")

    args = parser.parse_args()

    # Setup CUDA, GPU - Use all 4 GPUs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()  # Automatically detect number of GPUs
    args.device = device

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',datefmt='%m/%d/%Y %H:%M:%S',level=logging.INFO)
    logger.warning("device: %s, n_gpu: %s",device, args.n_gpu,)
    # Set seed
    set_seed(args)

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,  # or torch.bfloat16 if GPU supports it
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"  # or "fp4"
    )
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, trust_remote_code=True,encoding="utf-8")
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.add_tokens(["<S2SV_StartBug>", "<S2SV_EndBug>", "<S2SV_blank>", "<S2SV_ModStart>", "<S2SV_ModEnd>"])
    
    # Use auto device mapping for multi-GPU setup
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path, 
        quantization_config=quantization_config, 
        device_map="auto"  # Automatically distribute across all GPUs
    )
    model.resize_token_embeddings(len(tokenizer))

    logger.info("Training/evaluation parameters %s", args)
    # Training
    if args.do_train:
        pairs = os.listdir('T5_vulgen_train_translate_final_tokenized')
        dataset = {'name': [], 'source': [], 'target': []}
        for i, pair in enumerate(pairs):
            f = open('./T5_vulgen_train_translate_final_tokenized/' + pair + '/' + pair + '_in.c')
            source = f.read()
            source = source.replace(' ', '<S2SV_blank>')
            source = source.replace('!@#$', ' ')
            f.close()
            f = open('./T5_vulgen_train_translate_final_tokenized/' + pair + '/' + pair + '_out.c')
            target = f.read()
            target = target.replace(' ', '<S2SV_blank>')
            target = target.replace('!@#$', ' ')
            f.close()
            dataset['name'].append(pair)
            dataset['source'].append(source)
            dataset['target'].append(target)
        df = pd.DataFrame(dataset)
        train_data, val_data = train_test_split(df, test_size=0.1238)
        train_dataset = TextDataset(tokenizer, args, train_data, val_data, file_type='train')
        eval_dataset = TextDataset(tokenizer, args, train_data, val_data, file_type='eval')
        train(args, train_dataset, model, tokenizer, eval_dataset)
    # Evaluation
    results = {}  
    if args.do_test:
        pairs = os.listdir('T5Test_tokenized_final')
        dataset = {'name': [], 'source': [], 'target': []}
        for i, pair in enumerate(pairs):
            try:
                f = open('./T5Test_tokenized_final/' + pair + '/' + pair + '_in.c')
                source = f.read()
                source = source.replace(' ', '<S2SV_blank>')
                source = source.replace('!@#$', ' ')
                f.close()
                f = open('./T5Test_tokenized_final/' + pair + '/' + pair + '_out.c')
                target = f.read()
                target = target.replace(' ', '<S2SV_blank>')
                target = target.replace('!@#$', ' ')
                f.close()
                dataset['name'].append(pair)
                dataset['source'].append(source)
                dataset['target'].append(target)
            except:
                print(pair)
        df = pd.DataFrame(dataset)

        total_len = len(df)
        part_size = total_len // 4
        
        indices = [
            (0, part_size),  # part 0
            (part_size, 2 * part_size),  # part 1
            (2 * part_size, 3 * part_size),  # part 2
            (3 * part_size, total_len)  # part 3 (lấy đến hết để không bỏ sót)
        ]
        
        # Ví dụ: chọn part 2
        i = 3
        start, end = indices[i]
        df = df.iloc[start:end]
        print(f"start index: {start}, end index: {end}")
    
        test_dataset = TextDataset(tokenizer, args, df, df, file_type='train')
        test(args, model, tokenizer, test_dataset, best_threshold=0.5, i=i)
    return results

if __name__ == "__main__":
    main()


%cd /kaggle/tmp/VulGen/VulGen/T5/M1_VulRepair_PL-NL

# !python vulrepair_vulgen_beam1.py --output_dir=./saved_models \
# --tokenizer_name=ltgbao/DeepCoder-1.5b-merged-16bit-VulGen \
# --model_name_or_path=ltgbao/DeepCoder-1.5b-merged-16bit-VulGen \
# --do_test \
# --encoder_block_size 256 \
# --decoder_block_size 512 \
# --num_beams=1 \
# --eval_batch_size 32

# !accelerate launch --multi_gpu --num_processes=4 /kaggle/working/vulrepair_vulgen_beam1_4bit.py --output_dir=./saved_models \
# --tokenizer_name=ltgbao/Qwen3-32b-r256-4bit-VulGen \
# --model_name_or_path=ltgbao/Qwen3-32b-r256-4bit-VulGen \
# --do_test \
# --encoder_block_size 256 \
# --decoder_block_size 512 \
# --num_beams=1 \
# --eval_batch_size 2

!python /kaggle/working/vulrepair_vulgen_beam1_4bit.py --output_dir=./saved_models \
--tokenizer_name=ltgbao/Qwen3-32b-r256-4bit-VulGen \
--model_name_or_path=ltgbao/Qwen3-32b-r256-4bit-VulGen \
--do_test \
--encoder_block_size 256 \
--decoder_block_size 512 \
--num_beams=10  \
--eval_batch_size 18


!cp /kaggle/tmp/VulGen/VulGen/T5/data/raw_predictions/CodeT5/VulRepair_raw_preds_final_beam10_part*.csv /kaggle/working/

