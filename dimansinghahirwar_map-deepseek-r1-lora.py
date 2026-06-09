from dataclasses import dataclass

@dataclass
class CFG:
    # Auto fast path during Kaggle Batch submission
    FAST_SUBMIT: bool = (__import__('os').environ.get('KAGGLE_KERNEL_RUN_TYPE','Interactive')=='Batch')
    MODEL_DIR: str = "/kaggle/input/deepseek-r1/transformers/deepseek-r1-distill-qwen-1.5b/2"
    SEED: int = 42

    # Training
    MAX_TRAIN_SAMPLES: int = 2500 if FAST_SUBMIT else 6000
    MAX_STEPS: int = 700 if FAST_SUBMIT else 1500
    NUM_EPOCHS: int = 1
    LR: float = 2e-4 if FAST_SUBMIT else 1.6e-4
    BATCH_SIZE: int = 1
    GRAD_ACCUM: int = 24 if FAST_SUBMIT else 32

    # LoRA
    LORA_R: int = 4 if FAST_SUBMIT else 8
    LORA_ALPHA: int = 8 if FAST_SUBMIT else 16
    LORA_DROPOUT: float = 0.1
    LORA_TARGETS: tuple = ('q_proj','k_proj','v_proj','o_proj')

    # Sequence
    MAX_LEN_TOK: int = 320 if FAST_SUBMIT else 384

    # Ranking
    SHORTLIST_K: int = 8 if FAST_SUBMIT else 20
    PRED_TOPK: int = 1 if FAST_SUBMIT else 3

    # Prompt trimming
    Q_TRIM: int = 140 if FAST_SUBMIT else 300
    EX_TRIM: int = 700 if FAST_SUBMIT else 900

CFG = CFG()
print(CFG)



import os, sys, math, time, random
import numpy as np, pandas as pd, torch
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

print("FAST_SUBMIT:", CFG.FAST_SUBMIT)

if torch.cuda.is_available():
    torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

random.seed(CFG.SEED); np.random.seed(CFG.SEED); torch.manual_seed(CFG.SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(CFG.SEED)

def find_csv(root="/kaggle/input"):
    tr, te = None, None
    for p in Path(root).rglob("*.csv"):
        n = p.name.lower()
        if n=="train.csv" and tr is None: tr = p
        elif n=="test.csv" and te is None: te = p
    return tr, te

train_csv, test_csv = find_csv()
print("Train:", train_csv); print("Test:", test_csv)
assert train_csv is not None and test_csv is not None, "Attach competition dataset."

train_df = pd.read_csv(train_csv)
test_df  = pd.read_csv(test_csv)
display(train_df.head(2)); display(test_df.head(2))



VALID_CATS = [
    'True_Correct','True_Misconception','True_Neither',
    'False_Correct','False_Misconception','False_Neither'
]

def normalize_category(c):
    s = str(c).strip().replace('-', '_').replace(' ', '_')
    fixes = {'True__Correct':'True_Correct','False__Correct':'False_Correct',
             'True_Miscon':'True_Misconception','False_Miscon':'False_Misconception',
             'TrueNeither':'True_Neither','FalseNeither':'False_Neither'}
    s = fixes.get(s, s)
    if s in VALID_CATS: return s
    if s.startswith('True_'):  return 'True_Correct'
    if s.startswith('False_'): return 'False_Correct'
    return 'True_Correct'

def make_target(category, miscon):
    if pd.isna(miscon) or str(miscon).strip() in ['', 'NaN']:
        miscon = 'NA'
    return f"{normalize_category(category)}:{str(miscon).strip()}"

def chatml_prompt(qtext, mc_answer, explanation):
    sys_msg = ('You analyze student math explanations and output exactly one line: Category:Misconception. '
               'Category is one of {True_Correct, True_Misconception, True_Neither, '
               'False_Correct, False_Misconception, False_Neither}. Misconception is a canonical string from training or NA.')
    user_msg = (f'Question: {qtext}\n'
                f'SelectedOption: {mc_answer}\n'
                f'Explanation: {explanation}\n'
                'Return exactly: Category:Misconception')
    return ('<|im_start|>system\n' + sys_msg + '\n<|im_end|>\n'
            '<|im_start|>user\n' + user_msg + '\n<|im_end|>\n'
            '<|im_start|>assistant\n')

def pack_for_length(qtext, mc_answer, explanation, max_chars=850):
    qtext = qtext or ''; mc_answer = mc_answer or ''; explanation = explanation or ''
    ex = explanation[: int(max_chars*0.67)]
    q  = qtext[: int(max_chars*0.33)]
    return q, mc_answer, ex



from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType

tokenizer = AutoTokenizer.from_pretrained(CFG.MODEL_DIR, use_fast=True)
if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'right'

model = AutoModelForCausalLM.from_pretrained(
    CFG.MODEL_DIR,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map={'':0} if torch.cuda.is_available() else None,
    low_cpu_mem_usage=True
)
model.config.use_cache = False
try: model.gradient_checkpointing_enable()
except Exception as e: print('GC enable failed:', e)

lora_cfg = LoraConfig(task_type=TaskType.CAUSAL_LM,
                      r=CFG.LORA_R, lora_alpha=CFG.LORA_ALPHA, lora_dropout=CFG.LORA_DROPOUT,
                      target_modules=list(CFG.LORA_TARGETS))
model = get_peft_model(model, lora_cfg)
model.print_trainable_parameters()



from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

MAX_TRAIN_SAMPLES = CFG.MAX_TRAIN_SAMPLES

cats_norm = train_df['Category'].apply(normalize_category)
tmp_df = train_df.assign(_cat=cats_norm)

per_class = max(1, MAX_TRAIN_SAMPLES // len(VALID_CATS))
blocks = []
for c in VALID_CATS:
    blk = tmp_df[tmp_df['_cat']==c]
    if len(blk)>per_class: blk = blk.sample(n=per_class, random_state=CFG.SEED)
    blocks.append(blk)
mini_df = pd.concat(blocks).sample(frac=1.0, random_state=CFG.SEED).reset_index(drop=True)

train_fold, val_fold = train_test_split(mini_df, test_size=0.08 if not CFG.FAST_SUBMIT else 0.05,
                                        random_state=CFG.SEED, shuffle=True)

def build_pairs_from_df(df, speed_fast=False):
    pairs = []
    for _, r in df.iterrows():
        q,a,ex = pack_for_length(str(r['QuestionText']), str(r['MC_Answer']), str(r['StudentExplanation']),
                                 900 if not speed_fast else 700)
        prompt = chatml_prompt(q,a,ex)
        target = make_target(r['Category'], r['Misconception'])
        pairs.append((prompt, target))
    return pairs

train_pairs = build_pairs_from_df(train_fold, speed_fast=CFG.FAST_SUBMIT)
val_pairs   = build_pairs_from_df(val_fold, speed_fast=CFG.FAST_SUBMIT)

class CausalSFTDataset(Dataset):
    def __init__(self, pairs, tokenizer, max_length=384):
        self.pairs = pairs; self.tok = tokenizer; self.max_length=max_length
    def __len__(self): return len(self.pairs)
    def __getitem__(self, idx):
        prompt, target = self.pairs[idx]
        full = prompt.strip() + '\n' + target.strip() + tokenizer.eos_token
        enc_full   = self.tok(full, max_length=self.max_length, truncation=True, padding=False, return_tensors='pt')
        enc_prompt = self.tok(prompt.strip()+'\n', max_length=self.max_length, truncation=True, padding=False, return_tensors='pt')
        input_ids = enc_full['input_ids'][0]; attention_mask = enc_full['attention_mask'][0]
        labels = input_ids.clone(); labels[: enc_prompt['input_ids'].shape[1]] = -100
        return {'input_ids':input_ids, 'attention_mask':attention_mask, 'labels':labels}

train_ds = CausalSFTDataset(train_pairs, tokenizer, CFG.MAX_LEN_TOK)
val_ds   = CausalSFTDataset(val_pairs, tokenizer, CFG.MAX_LEN_TOK)
print('Train/Val sizes:', len(train_ds), len(val_ds))



from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
import time

args = TrainingArguments(
    output_dir='outputs',
    overwrite_output_dir=True,
    bf16=False,
    fp16=True if torch.cuda.is_available() else False,
    per_device_train_batch_size=CFG.BATCH_SIZE,
    per_device_eval_batch_size=CFG.BATCH_SIZE,
    gradient_accumulation_steps=CFG.GRAD_ACCUM,
    learning_rate=CFG.LR,
    num_train_epochs=CFG.NUM_EPOCHS,
    lr_scheduler_type='cosine',
    warmup_ratio=0.05,
    logging_steps=200000,
    max_steps=CFG.MAX_STEPS,
    save_total_limit=1,
    report_to=[],
)

data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
trainer = Trainer(model=model, args=args, train_dataset=train_ds, data_collator=data_collator)

t0=time.time(); trainer.train(); print('Train seconds:', time.time()-t0)
trainer.save_model('outputs/final_model'); tokenizer.save_pretrained('outputs/final_model')



from sklearn.feature_extraction.text import TfidfVectorizer

MISC_BANK = sorted(set(m for m in train_df['Misconception'].dropna().astype(str) if m.strip() and m.strip()!='NA'))
print('MISC_BANK size:', len(MISC_BANK))

tfidf = TfidfVectorizer(lowercase=True, ngram_range=(1,2), min_df=1)
X_labels = tfidf.fit_transform(MISC_BANK)

def shortlist_miscon(expl, k=20):
    q = tfidf.transform([expl or ''])
    sims = (X_labels @ q.T).toarray().ravel()
    if k >= len(MISC_BANK):
        idx = np.argsort(-sims)
    else:
        idx = np.argpartition(-sims, k-1)[:k]
        idx = idx[np.argsort(-sims[idx])]
    return [MISC_BANK[i] for i in idx]

def normalize_miscon(m):
    m = str(m or '').strip()
    if not m: return 'NA'
    return m[:128] if len(m)>128 else m

@torch.no_grad()
def score_combo(prompt, target, max_len):
    full = prompt + '\n' + target + tokenizer.eos_token
    enc_full   = tokenizer(full, return_tensors='pt', truncation=True, max_length=max_len)
    enc_prompt = tokenizer(prompt+'\n', return_tensors='pt', truncation=True, max_length=max_len)
    if torch.cuda.is_available():
        enc_full = {k:v.to('cuda') for k,v in enc_full.items()}
        enc_prompt = {k:v.to('cuda') for k,v in enc_prompt.items()}
    labels = enc_full['input_ids'].clone()
    labels[:, :enc_prompt['input_ids'].shape[1]] = -100
    out = model(input_ids=enc_full['input_ids'], attention_mask=enc_full['attention_mask'], labels=labels)
    return -float(out.loss)  # higher is better

def rank_topk(prompt, explanation_text, k_out=3, max_len=320, shortlist_k=20):
    # rank categories (using NA placeholder)
    scores = []
    for cat in VALID_CATS:
        s = score_combo(prompt, f'{cat}:NA', max_len)
        scores.append((s, cat))
    scores.sort(reverse=True)
    expand = [c for _,c in scores[: (2 if CFG.FAST_SUBMIT else 3)]]
    cand = []
    for cat in expand:
        if cat.endswith('_Misconception'):
            for m in shortlist_miscon(explanation_text, k=shortlist_k):
                tgt = f'{cat}:{normalize_miscon(m)}'
                cand.append((score_combo(prompt, tgt, max_len), tgt))
        else:
            tgt = f'{cat}:NA'
            cand.append((score_combo(prompt, tgt, max_len), tgt))
    cand.sort(reverse=True)
    out, seen = [], set()
    for _,t in cand:
        if t not in seen:
            out.append(t); seen.add(t)
        if len(out)==k_out: break
    return out or ['True_Correct:NA']



def mapk_3(y_true, y_pred_lists):
    # y_true: list of correct strings 'Category:Misconception'
    # y_pred_lists: list of list[str] (top-3)
    assert len(y_true)==len(y_pred_lists)
    total=0.0
    for gt, pred in zip(y_true, y_pred_lists):
        score=0.0
        for j,p in enumerate(pred[:3]):
            if p==gt:
                score = 1.0/(j+1); break
        total += score
    return total/len(y_true)

# Build prompts for val fold and score
val_targets = []
val_preds = []
for _, r in val_fold.iterrows():
    q = str(r['QuestionText']); a=str(r['MC_Answer']); ex=str(r['StudentExplanation'])
    q_trim = q[:CFG.Q_TRIM]; ex_trim = ex[:CFG.EX_TRIM]
    prompt = chatml_prompt(q_trim, a, ex_trim)
    preds = rank_topk(prompt, ex_trim, k_out=min(3, CFG.PRED_TOPK if CFG.FAST_SUBMIT else 3),
                      max_len=CFG.MAX_LEN_TOK, shortlist_k=CFG.SHORTLIST_K)
    val_preds.append(preds)
    val_targets.append(make_target(r['Category'], r['Misconception']))

val_map3 = mapk_3(val_targets, val_preds)
print('Validation MAP@3:', round(val_map3, 5))



rows = []
for _, r in test_df.iterrows():
    q = str(r['QuestionText']); a = str(r['MC_Answer']); ex = str(r['StudentExplanation'])
    q_trim = q[:CFG.Q_TRIM]; ex_trim = ex[:CFG.EX_TRIM]
    prompt = chatml_prompt(q_trim, a, ex_trim)
    preds = rank_topk(prompt, ex_trim, k_out=CFG.PRED_TOPK, max_len=CFG.MAX_LEN_TOK, shortlist_k=CFG.SHORTLIST_K)
    rows.append((r['row_id'], ' '.join(preds)))

sub = pd.DataFrame(rows, columns=['row_id','Category:Misconception'])
out_path = '/kaggle/working/submission.csv'
sub.to_csv(out_path, index=False)
display(sub.head()); print('Saved:', out_path)


