import os
import gc
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding
from datasets import Dataset
from peft import PeftModel, PeftConfig
from collections import defaultdict

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # 修复tokenizer并行问题
os.environ["PYDEVD_DISABLE_FILE_VALIDATION"] = "1"  # 禁用调试器验证


# ==================== DATA PREPROCESSING ====================
def load_and_preprocess_data():
    """Load and preprocess training and test data"""
    le = LabelEncoder()
    train = pd.read_csv('/kaggle/input/map-charting-student-math-misunderstandings/train.csv')
    train.Misconception = train.Misconception.fillna('NA')
    train['target'] = train.Category + ":" + train.Misconception
    train['label'] = le.fit_transform(train['target'])
    
    # Identify correct answers
    idx = train.apply(lambda row: row.Category.split('_')[0], axis=1) == 'True'
    correct = train.loc[idx].copy()
    correct['c'] = correct.groupby(['QuestionId', 'MC_Answer']).MC_Answer.transform('count')
    correct = correct.sort_values('c', ascending=False)
    correct = correct.drop_duplicates(['QuestionId'])
    correct = correct[['QuestionId', 'MC_Answer']]
    correct['is_correct'] = 1

    train = train.merge(correct, on=['QuestionId', 'MC_Answer'], how='left')
    train.is_correct = train.is_correct.fillna(0)
    
    return train, correct, le

train, correct, le = load_and_preprocess_data()
n_classes = len(le.classes_)
print(f"Train shape: {train.shape} with {n_classes} target classes")

# Clean up
del correct
gc.collect()


train.head(10)


# ==================== MODEL 1: GEMMA-2-9B-IT ====================
def run_gemma_model():
    """Run Gemma-2-9b-it model training and inference"""
    print("=== Running Gemma-2-9b-it Model ===")
    
    # Text formatting function
    def format_input_gemma(row):
        x = "Yes" if row['is_correct'] else "No"
        return (
            f"Question: {row['QuestionText']}\n"
            f"Answer: {row['MC_Answer']}\n"
            f"Correct? {x}\n"
            f"Student Explanation: {row['StudentExplanation']}"
        )

    train['text'] = train.apply(format_input_gemma, axis=1)
    
    # Split data
    train_df, val_df = train_test_split(train, test_size=0.2, random_state=42)
    train_ds = Dataset.from_pandas(train_df[['text', 'label']])
    val_ds = Dataset.from_pandas(val_df[['text', 'label']])

    # Tokenization
    tokenizer = AutoTokenizer.from_pretrained("/kaggle/input/gemma2-9b-it-cv945")
    
    def tokenize(batch):
        return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=256)

    train_ds = train_ds.map(tokenize, batched=True)
    val_ds = val_ds.map(tokenize, batched=True)
    train_ds.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    val_ds.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

    # Model setup
    model = AutoModelForSequenceClassification.from_pretrained(
        "/kaggle/input/gemma2-9b-it-bf16",
        num_labels=n_classes,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(model, "/kaggle/input/gemma2-9b-it-cv945")

    # Training arguments
    training_args = TrainingArguments(
        output_dir="./ver_1",
        do_train=True,
        do_eval=True,
        eval_strategy="steps",
        save_strategy="steps",
        num_train_epochs=2,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        learning_rate=2e-5,
        logging_steps=50,
        save_steps=200,
        eval_steps=200,
        save_total_limit=1,
        metric_for_best_model="map@3",
        greater_is_better=True,
        load_best_model_at_end=True,
        report_to="none",
        fp16=True,
    )

    # MAP@3 metric
    def compute_map3(eval_pred):
        logits, labels = eval_pred
        probs = torch.nn.functional.softmax(torch.tensor(logits), dim=-1).numpy()
        top3 = np.argsort(-probs, axis=1)[:, :3]
        match = (top3 == labels[:, None])
        
        map3 = 0
        for i in range(len(labels)):
            if match[i, 0]:
                map3 += 1.0
            elif match[i, 1]:
                map3 += 1.0 / 2
            elif match[i, 2]:
                map3 += 1.0 / 3
        return {"map@3": map3 / len(labels)}

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        compute_metrics=compute_map3,
    )

    # Test inference
    test = pd.read_csv('/kaggle/input/map-charting-student-math-misunderstandings/test.csv')
    test = test.merge(train[['QuestionId', 'MC_Answer', 'is_correct']].drop_duplicates(), 
                     on=['QuestionId', 'MC_Answer'], how='left')
    test.is_correct = test.is_correct.fillna(0)
    test['text'] = test.apply(format_input_gemma, axis=1)
    
    ds_test = Dataset.from_pandas(test[['text']])
    ds_test = ds_test.map(tokenize, batched=True)
    
    predictions = trainer.predict(ds_test)
    probs = torch.nn.functional.softmax(torch.tensor(predictions.predictions), dim=1).numpy()
    
    # Get top predictions
    top3 = np.argsort(-probs, axis=1)[:, :]
    flat_top3 = top3.flatten()
    decoded_labels = le.inverse_transform(flat_top3)
    top3_labels = decoded_labels.reshape(top3.shape)
    joined_preds = ["|".join(row) for row in top3_labels]

    # Save submission
    sub = pd.DataFrame({
        "row_id": test.row_id.values,
        "Category:Misconception": joined_preds
    })
    sub.to_csv("submission_gemma.csv", index=False)
    
    # Clean up memory
    del model, trainer, training_args, train_ds, val_ds, tokenizer
    del predictions, probs, top3, flat_top3, decoded_labels, top3_labels, ds_test
    torch.cuda.empty_cache()
    gc.collect()
    
    return sub

gemma_predictions = run_gemma_model()


# ==================== MODEL 2: DEEPSEEKMATH-7B ====================
def run_deepseek_model():
    """Run DeepSeekMath-7b model inference"""
    print("=== Running DeepSeekMath-7b Model ===")
    
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    torch.cuda.empty_cache()
    gc.collect()
    
    # Text formatting function
    def format_input_deepseek(row):
        x = "This answer is correct." if row['is_correct'] else "This answer is incorrect."
        return (
            f"Question: {row['QuestionText']}\n"
            f"Answer: {row['MC_Answer']}\n"
            f"{x}\n"
            f"Student Explanation: {row['StudentExplanation']}"
        )

    test = pd.read_csv('/kaggle/input/map-charting-student-math-misunderstandings/test.csv')
    test = test.merge(train[['QuestionId', 'MC_Answer', 'is_correct']].drop_duplicates(), 
                     on=['QuestionId', 'MC_Answer'], how='left')
    test.is_correct = test.is_correct.fillna(0)
    test['text'] = test.apply(format_input_deepseek, axis=1)
    
    # Load model and tokenizer
    model = AutoModelForSequenceClassification.from_pretrained(
        "/kaggle/input/deekseepmath-7b-map-competition/MAP_EXP_09_FULL", 
        device_map="auto",  # 使用auto而不是固定设备
        torch_dtype=torch.bfloat16
    )
    tokenizer = AutoTokenizer.from_pretrained("/kaggle/input/deekseepmath-7b-map-competition/MAP_EXP_09_FULL")
    model.config.pad_token_id = tokenizer.pad_token_id

    # Tokenization
    def tokenize(batch):
        return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=256)

    ds_test = Dataset.from_pandas(test)
    ds_test = ds_test.map(tokenize, batched=True)

    # Inference
    test_args = TrainingArguments(
        output_dir="./",
        do_train=False,
        do_predict=True,
        per_device_eval_batch_size=16,  # 增加批次大小,default:16
        fp16=True,
        report_to='none',
        dataloader_pin_memory=True,
        dataloader_num_workers=2
    )

    # Trainer with updated parameter
    trainer = Trainer(
        model=model,
        args=test_args,
        processing_class=tokenizer,  # 修复弃用警告
        data_collator=DataCollatorWithPadding(tokenizer)
    )

    predictions = trainer.predict(ds_test)
    
    # Process predictions
    top3 = np.argsort(-predictions.predictions, axis=1)[:, :3]  # 只取top3
    flat_top3 = top3.flatten()
    decoded_labels = le.inverse_transform(flat_top3)
    top3_labels = decoded_labels.reshape(top3.shape)
    joined_preds = ["|".join(row) for row in top3_labels]

    # Save submission
    sub = pd.DataFrame({
        "row_id": test.row_id.values,
        "Category:Misconception": joined_preds
    })
    sub.to_csv("submission_deepseek.csv", index=False)
    
    # Clean up memory
    cleanup_objects = [model, trainer, tokenizer, predictions, top3, flat_top3, decoded_labels, top3_labels, ds_test, test]
    for obj in cleanup_objects:
        if 'obj' in locals() and obj is not None:
            del obj
    torch.cuda.empty_cache()
    gc.collect()
    
    return sub

deepseek_predictions = run_deepseek_model()


# ==================== MODEL 3: QWEN3-8B ====================
def run_qwen_model():
    """Run Qwen3-8b model inference with optimized GPU utilization"""
    print("=== Running Qwen3-8b Model ===")
    
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    torch.cuda.empty_cache()
    gc.collect()
    
    # Reuse the same formatting as DeepSeek
    def format_input_qwen(row):
        x = "This answer is correct." if row['is_correct'] else "This answer is incorrect."
        return (
            f"Question: {row['QuestionText']}\n"
            f"Answer: {row['MC_Answer']}\n"
            f"{x}\n"
            f"Student Explanation: {row['StudentExplanation']}"
        )

    test = pd.read_csv('/kaggle/input/map-charting-student-math-misunderstandings/test.csv')
    test = test.merge(train[['QuestionId', 'MC_Answer', 'is_correct']].drop_duplicates(), 
                     on=['QuestionId', 'MC_Answer'], how='left')
    test.is_correct = test.is_correct.fillna(0)
    test['text'] = test.apply(format_input_qwen, axis=1)
    
    # Load model with optimized settings
    model = AutoModelForSequenceClassification.from_pretrained(
        "/kaggle/input/qwen3-8b-map-competition/MAP_EXP_16_FULL", 
        device_map="auto",  # 使用auto设备映射
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    tokenizer = AutoTokenizer.from_pretrained("/kaggle/input/qwen3-8b-map-competition/MAP_EXP_16_FULL")
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id

    # Tokenization with optimized padding
    def tokenize(batch):
        return tokenizer(
            batch["text"], 
            padding=True,  # 动态padding
            truncation=True, 
            max_length=256,
            return_tensors="pt"
        )

    ds_test = Dataset.from_pandas(test)
    ds_test = ds_test.map(tokenize, batched=True)

    # Optimized inference with larger batch size
    test_args = TrainingArguments(
        output_dir="./",
        do_train=False,
        do_predict=True,
        per_device_eval_batch_size=4,  # 增加批次大小，default: 4
        fp16=True,
        report_to='none',
        dataloader_pin_memory=True,
        dataloader_num_workers=2
    )

    # Trainer with updated parameter
    trainer = Trainer(
        model=model,
        args=test_args,
        processing_class=tokenizer,  # 修复弃用警告
        data_collator=DataCollatorWithPadding(tokenizer, padding=True)
    )

    predictions = trainer.predict(ds_test)
    
    # Process predictions
    top3 = np.argsort(-predictions.predictions, axis=1)[:, :3]  # 只取top3
    flat_top3 = top3.flatten()
    decoded_labels = le.inverse_transform(flat_top3)
    top3_labels = decoded_labels.reshape(top3.shape)
    joined_preds = ["|".join(row) for row in top3_labels]

    # Save submission
    sub = pd.DataFrame({
        "row_id": test.row_id.values,
        "Category:Misconception": joined_preds
    })
    sub.to_csv("submission_qwen.csv", index=False)
    
    # Clean up memory efficiently
    cleanup_objects = [model, trainer, tokenizer, predictions, top3, flat_top3, decoded_labels, top3_labels, ds_test, test]
    for obj in cleanup_objects:
        if 'obj' in locals() and obj is not None:
            del obj
    torch.cuda.empty_cache()
    gc.collect()
    
    return sub

qwen_predictions = run_qwen_model()


# ==================== MODEL ENSEMBLE WITH VOTING ====================
def ensemble_predictions_voting():
    """Combine predictions from all three models using voting"""
    print("=== Ensemble Predictions with Voting ===")
    
    # Load individual model predictions
    df1 = pd.read_csv('submission_gemma.csv').rename(columns={'Category:Misconception': 'gemma_pred'})
    df2 = pd.read_csv('submission_deepseek.csv').rename(columns={'Category:Misconception': 'deepseek_pred'})
    df3 = pd.read_csv('submission_qwen.csv').rename(columns={'Category:Misconception': 'qwen_pred'})

    # Merge predictions
    df = pd.merge(df1, df2, on='row_id')
    df = pd.merge(df, df3, on='row_id')

    # Voting ensemble function
    def voting_ensemble(l1, l2, l3, top_k=3):
        """Majority voting ensemble with weighted preferences"""
        list1, list2, list3 = l1.split('|'), l2.split('|'), l3.split('|')
        
        # 收集所有预测的标签
        all_predictions = list1 + list2 + list3
        
        # 计算每个标签的出现次数（硬投票）
        vote_count = {}
        for label in all_predictions:
            vote_count[label] = vote_count.get(label, 0) + 1
        
        # 考虑排名位置的加权投票
        weighted_votes = {}
        for i, pred_list in enumerate([list1, list2, list3]):
            weight = 3 - i  # 第一个模型权重最高（3），第三个最低（1）
            for rank, label in enumerate(pred_list):
                weighted_votes[label] = weighted_votes.get(label, 0) + weight * (len(pred_list) - rank)
        
        # 结合硬投票和加权投票
        final_scores = {}
        for label in set(all_predictions):
            # 硬投票计数 + 加权投票分数（归一化）
            final_scores[label] = vote_count[label] + (weighted_votes.get(label, 0) / 10)
        
        # 按最终得分排序
        sorted_labels = sorted(final_scores.items(), key=lambda x: -x[1])
        
        # 返回top-k个标签
        return '|'.join([label for label, score in sorted_labels[:top_k]])

    # Alternative: Simple majority voting
    def simple_majority_voting(l1, l2, l3, top_k=3):
        """Simple majority voting"""
        lists = [l1.split('|'), l2.split('|'), l3.split('|')]
        
        # 收集所有预测
        all_labels = set()
        for lst in lists:
            all_labels.update(lst)
        
        # 计算每个标签的投票数
        votes = {}
        for label in all_labels:
            vote_count = 0
            for lst in lists:
                if label in lst:
                    vote_count += 1
            votes[label] = vote_count
        
        # 按投票数排序，如果平票则按平均排名排序
        def get_avg_rank(label):
            ranks = []
            for lst in lists:
                if label in lst:
                    ranks.append(lst.index(label))
                else:
                    ranks.append(len(lst))  # 如果不在列表中，给一个较差的排名
            return sum(ranks) / len(ranks)
        
        sorted_labels = sorted(votes.keys(), 
                              key=lambda x: (-votes[x], get_avg_rank(x)))
        
        return '|'.join(sorted_labels[:top_k])

    # Apply voting ensemble (choose one method)
    df['Category:Misconception'] = df.apply(
        lambda x: voting_ensemble(x['gemma_pred'], x['deepseek_pred'], x['qwen_pred']), axis=1
    )
    
    # 或者使用简单多数投票
    # df['Category:Misconception'] = df.apply(
    #     lambda x: simple_majority_voting(x['gemma_pred'], x['deepseek_pred'], x['qwen_pred']), axis=1
    # )

    # Save final submission
    final_sub = df[['row_id', 'Category:Misconception']]
    # final_sub.to_csv('submission_voting.csv', index=False)
    
    # 输出投票统计信息
    print("Voting ensemble completed.")
    print("Sample predictions:")
    print(final_sub.head(10))
    
    return final_sub

# 更高级的加权投票版本
def weighted_voting_ensemble():
    """Weighted voting based on model performance"""
    print("=== Weighted Voting Ensemble ===")
    
    # Load predictions
    df1 = pd.read_csv('submission_gemma.csv').rename(columns={'Category:Misconception': 'gemma_pred'})
    df2 = pd.read_csv('submission_deepseek.csv').rename(columns={'Category:Misconception': 'deepseek_pred'})
    df3 = pd.read_csv('submission_qwen.csv').rename(columns={'Category:Misconception': 'qwen_pred'})

    df = pd.merge(df1, df2, on='row_id')
    df = pd.merge(df, df3, on='row_id')

    # 假设的模型权重（可以根据验证集性能调整）
    model_weights = {
        'gemma': 1.0,
        'deepseek': 1.2,  # 假设deepseek性能更好
        'qwen': 0.8       # 假设qwen性能稍差
    }

    def weighted_voting(l1, l2, l3, top_k=3):
        lists = [l1.split('|'), l2.split('|'), l3.split('|')]
        weights = [model_weights['gemma'], model_weights['deepseek'], model_weights['qwen']]
        
        # 收集所有唯一标签
        all_labels = set()
        for lst in lists:
            all_labels.update(lst)
        
        # 计算加权得分
        scores = {}
        for label in all_labels:
            total_score = 0
            for i, lst in enumerate(lists):
                if label in lst:
                    rank = lst.index(label)
                    # 排名越高（数值越小），得分越高
                    score = (len(lst) - rank) * weights[i]
                    total_score += score
            scores[label] = total_score
        
        # 按得分排序
        sorted_labels = sorted(scores.items(), key=lambda x: -x[1])
        return '|'.join([label for label, score in sorted_labels[:top_k]])

    df['Category:Misconception'] = df.apply(
        lambda x: weighted_voting(x['gemma_pred'], x['deepseek_pred'], x['qwen_pred']), axis=1
    )

    final_sub = df[['row_id', 'Category:Misconception']]
    final_sub.to_csv('submission_weighted_voting.csv', index=False)
    
    return final_sub

# 使用投票集成
final_predictions = ensemble_predictions_voting()
# 或者使用加权投票
# final_predictions = weighted_voting_ensemble()
final_predictions.to_csv('submission_voting.csv', index=False)

print("Voting ensemble completed successfully!")
print(final_predictions.head())

# Final memory cleanup
cleanup_objects = [train, le, gemma_predictions, deepseek_predictions, qwen_predictions, final_predictions]
for obj in cleanup_objects:
    if 'obj' in locals() and obj is not None:
        del obj
torch.cuda.empty_cache()
gc.collect()

print("Memory allocated:", torch.cuda.memory_allocated())
print("Memory reserved:", torch.cuda.memory_reserved())


final_predictions







