# %%
import os
os.environ["WANDB_DISABLED"] = "true"
os.environ["WANDB_MODE"] = "disabled"
os.environ["CUDA_VISIBLE_DEVICES"] = "0" 
import pandas as pd
import torch
import numpy as np
from argparse import Namespace
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import gc

# %% [markdown]
# ## 数据预处理

class DataProcessor:
	def __init__(self, args):
		self.args = args
		self.le = None
		self.isPreprocess = False
		self.correct_lookup = None

	def load_data(self):
		self.train_df = pd.read_csv(self.args.train_path)
		self.test_df = pd.read_csv(self.args.test_path)
		if self.args.use_extra_data:
			self.extra_df = pd.read_csv(self.args.extra_data_path)
			self.train_df = pd.concat([self.train_df, self.extra_df], ignore_index=True)

	def get_num_classes(self):
		if self.isPreprocess == False:
			return "please preprocess first"
		return self.train_df['label'].nunique()

	def get_label_encoder(self):
		if self.le is None:
			raise ValueError("LabelEncoder not initialized. Please run preprocess first.")
		return self.le

	@staticmethod
	def format_input(row):
		correct_text = "Yes" if row['IsCorrect'] else "No"
		return (
			f"Question: {row['QuestionText']}\n"
			f"Answer: {row['MC_Answer']}\n"
			f"Correct? {correct_text}\n"
			f"Student Explanation: {row['StudentExplanation']}\n"
		)

	def preprocess(self):
		self.load_data()
		self.train_df['target'] = self.train_df['Category'] + ':' + self.train_df['Misconception'].fillna('NA')
		correct_samples = self.train_df[self.train_df['Category'].str.startswith('True', na=False)].copy()
		correct_samples['count'] = correct_samples.groupby(['QuestionId', 'MC_Answer'])['MC_Answer'].transform('count')
		most_popular_correct = correct_samples.sort_values('count', ascending=False).drop_duplicates(['QuestionId'])
		self.correct_lookup = most_popular_correct[['QuestionId', 'MC_Answer']].copy()
		self.correct_lookup['IsCorrect_flag'] = True
		self.train_df = self.train_df.merge(self.correct_lookup, on=['QuestionId', 'MC_Answer'], how='left')
		self.train_df['IsCorrect'] = self.train_df['IsCorrect_flag'].notna()
		self.train_df = self.train_df.drop(columns=['IsCorrect_flag'])
		self.le = LabelEncoder()
		self.train_df['label'] = self.le.fit_transform(self.train_df['target'])
		self.train_df['text'] = self.train_df.apply(self.format_input, axis=1)
		self.isPreprocess = True
		return self.train_df

	def inference_processor(self):
		if self.isPreprocess == False:
			return "Have you do the train? please preprocess first"
		self.test_df = self.test_df.merge(self.correct_lookup, on=['QuestionId', 'MC_Answer'], how='left')
		self.test_df['IsCorrect'] = self.test_df['IsCorrect_flag'].notna()
		self.test_df = self.test_df.drop(columns=['IsCorrect_flag'])
		self.test_df['text'] = self.test_df.apply(self.format_input, axis=1)
		return self.test_df

# %%
args = Namespace(
	train_path='/kaggle/input/map-charting-student-math-misunderstandings/train.csv',
	test_path='/kaggle/input/map-charting-student-math-misunderstandings/test.csv',
	use_extra_data=False,
	extra_data_path='data/train_corrected.csv',
	test_size=0.2,
	random_state=42,
	model_dir='ettin-encoder-400m-cv',
	inference_model_dir='/kaggle/input/ettin-encoder-400m-10-folds/result-ettin-encoder-400m-cv',
	label_names=["labels"],
	mode='inference',  # 'train' or 'inference'
	model_name = "/kaggle/input/ettin-encoder-400m-10-folds/result-ettin-encoder-400m-cv/fold0",
	n_splits=10
)

DP = DataProcessor(args)
train_df = DP.preprocess()
print("Num classes:", DP.get_num_classes())

# %% [markdown]
# ## Tokenizer
tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=False)
if tokenizer.pad_token is None:
	tokenizer.pad_token = tokenizer.eos_token
MAX_LEN = 256

def tokenize_function(examples):
	tokenized_inputs = tokenizer(
		examples["text"], padding="max_length", truncation=True, max_length=MAX_LEN
	)
	if "label" in examples:
		tokenized_inputs["labels"] = examples["label"]
	return tokenized_inputs

# %% [markdown]
# ## 评估指标
def compute_map3(eval_pred):
	logits, labels = eval_pred
	probs = torch.nn.functional.softmax(torch.tensor(logits), dim=-1).numpy()
	top3 = np.argsort(-probs, axis=1)[:, :3]
	map3 = 0.0
	for i in range(len(labels)):
		true_label = int(labels[i])
		if true_label == top3[i, 0]:
			map3 += 1.0
		elif true_label == top3[i, 1]:
			map3 += 1.0 / 2
		elif true_label == top3[i, 2]:
			map3 += 1.0 / 3
	return {"map@3": map3 / len(labels)}

# %% [markdown]
# ## CV 训练
skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.random_state)

if args.mode == 'train':
	for fold, (train_idx, val_idx) in enumerate(skf.split(train_df, train_df['label'])):
		print(f"\n===== Fold {fold+1} / {args.n_splits} =====")

		train_split = train_df.iloc[train_idx]
		val_split = train_df.iloc[val_idx]

		train_ds = Dataset.from_pandas(train_split[['text', 'label']])
		val_ds = Dataset.from_pandas(val_split[['text', 'label']])

		train_ds = train_ds.map(tokenize_function, batched=True, remove_columns=["text"])
		val_ds = val_ds.map(tokenize_function, batched=True, remove_columns=["text"])
		train_ds.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
		val_ds.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

		model = AutoModelForSequenceClassification.from_pretrained(
			args.model_name,
			num_labels=DP.get_num_classes(),
			device_map="auto"
		)
		model.config.pad_token_id = tokenizer.pad_token_id

		training_args = TrainingArguments(
			output_dir=f"./results_fold{fold}",
			do_train=True,
			do_eval=True,
			eval_strategy="steps",
			save_strategy="steps",
			num_train_epochs=3,
			per_device_train_batch_size=8*2,
			per_device_eval_batch_size=16*2,
			learning_rate=5e-5,
			logging_dir=f"./logs_fold{fold}",
			logging_steps=50,
			save_steps=200,
			eval_steps=200,
			save_total_limit=1,
			metric_for_best_model="map@3",
			greater_is_better=True,
			load_best_model_at_end=True,
			report_to="none",
			bf16=True,
			fp16=False,
		)

		trainer = Trainer(
			model=model,
			args=training_args,
			train_dataset=train_ds,
			eval_dataset=val_ds,
			tokenizer=tokenizer,
			compute_metrics=compute_map3,
		)

		trainer.train()
		fold_dir = os.path.join(args.model_dir, f"fold{fold}")
		os.makedirs(fold_dir, exist_ok=True)
		trainer.save_model(fold_dir)
		tokenizer.save_pretrained(fold_dir)
		del trainer
		del model
		torch.cuda.empty_cache()
		gc.collect()

# %% [markdown]
# ## 推理 + Ensemble
if args.mode == 'inference':
	test_df = DP.inference_processor()
	ds_test = Dataset.from_pandas(test_df[['text']])
	ds_test = ds_test.map(tokenize_function, batched=True)

	all_probs = []

	for fold in range(args.n_splits):
		print(f"Loading fold {fold} model...")
		fold_dir = os.path.join(args.inference_model_dir, f"fold{fold}")
		model = AutoModelForSequenceClassification.from_pretrained(
            fold_dir,
            device_map="auto",
            torch_dtype=torch.float16
        )
		model.config.pad_token_id = tokenizer.pad_token_id

		trainer = Trainer(model=model, tokenizer=tokenizer)
		predictions = trainer.predict(ds_test)
		probs = torch.nn.functional.softmax(torch.tensor(predictions.predictions), dim=1).numpy()
		all_probs.append(probs)
		del trainer
		del model
		torch.cuda.empty_cache()
		gc.collect()

	# softmax 融合
	avg_probs = np.mean(all_probs, axis=0)

	top3 = np.argsort(-avg_probs, axis=1)[:, :3]
	flat_top3 = top3.flatten()
	le = DP.get_label_encoder()
	decoded_labels = le.inverse_transform(flat_top3)
	top3_labels = decoded_labels.reshape(top3.shape)
	joined_preds = [" ".join(row) for row in top3_labels]

	sub = pd.DataFrame({
		"row_id": test_df.row_id.values,
		"Category:Misconception": joined_preds
	})
	sub.to_csv("submission.csv", index=False)
	print(sub.head())

