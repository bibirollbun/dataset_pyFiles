import torch
import gc

# Optional: Delete model and other large objects from memory
# This is a good practice if you plan to load another model or start a new task
# del model
# del trainer
# del test_ds
# del predictions_output
# del predictions
# ... and any other large tensors or data structures on the GPU

# Release GPU memory cache
torch.cuda.empty_cache()

# Run Python's garbage collector to free up any unreferenced objects
gc.collect()

print("GPU resources released.")


!pip install --upgrade --no-index --find-links=/kaggle/input/transformers-4-56-1-and-deps transformers -qq


# import torch._dynamo
# torch._dynamo.config.suppress_errors = True
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer
import torch
from datasets import Dataset
import joblib
import pickle
import re
import os
import sys
from peft import PeftModel

# Define the directory where the model and components are saved
DIR = "/kaggle/input/qwen3-8b-v7/qwen_8b_lora_v7"
MAX_LEN = 640
TEST_FILE_PATH = "/kaggle/input/map-charting-student-math-misunderstandings/test.csv"

# Define the path to the base model and the fused adapter
BASE_MODEL_PATH = '/kaggle/input/qwen-3/transformers/8b/1'
FUSED_LORA_PATH = f"{DIR}/lora_adapter"

# --- NEW: Define a name for this specific model run for saving probabilities ---
MODEL_NAME = "qwen3-8b-v7"

# --- START: Feature Engineering Functions - MUST BE DEFINED HERE FOR PICKLE TO WORK ---
def format_options(options):
    """将选项列表格式化为字符串"""
    options = sorted(list(options))
    formatted_str = ""
    for i, opt in enumerate(options):
        formatted_str += f"{chr(65 + i)}) {opt}\n"
    return formatted_str.strip()

def extract_math_concepts(text):
    """Extract mathematical concepts from text"""
    concepts = []
    if re.search(r'\d+/\d+|fraction|numerator|denominator', text, re.I):
        concepts.append('fraction')
    if re.search(r'\d+\.\d+|decimal|point', text, re.I):
        concepts.append('decimal')
    if re.search(r'triangle|square|circle|shape|area|perimeter|angle|shaded', text, re.I):
        concepts.append('geometry')
    if re.search(r'greater|less|equal|compare|larger|smaller|highest|lowest', text, re.I):
        concepts.append('comparison')
    return ','.join(concepts) if concepts else 'other'

def format_input_with_new_template(row):
    """
    更新后的提示词格式化函数，包含提取出的选项。
    """
    judge_result = "CORRECT" if row['is_correct'] else "INCORRECT"

    # Mathematical context
    math_context = f"Concept: {row['question_concept']}"

    prompt = f"""You are an experienced math educator and a specialized AI assistant with a deep understanding of common student errors and pedagogical approaches. Your task is to analyze a student's response to a math problem and accurately classify it. You will focus on identifying the underlying mathematical misconception from a predefined list of categories.

Question: {row['QuestionText']}
Answer Choices:
{row['formatted_options']}
Student's Answer: {row['MC_Answer']}
Student's Explanation: {row['StudentExplanation']}
Judge: {judge_result}
Mathematical Context: {row['question_concept']}

--- IMPORTANT GUIDANCE ---
Based on the "Judge" field, the student's final answer has already been determined as either CORRECT or INCORRECT. This is a critical piece of information.
- If the Judge is "CORRECT", your classification MUST begin with "True_".
- If the Judge is "INCORRECT", your classification MUST begin with "False_".

CLASSIFICATION GUIDELINES:
• True_Correct:NA = Student demonstrates correct understanding
• False_Correct:NA = Student gives correct answer but for wrong reasons
• True_Neither:NA = Correct answer but unclear/incomplete reasoning
• False_Neither:NA = Incorrect answer but no specific misconception identified
• True_Misconception:[Type] = Correct answer but demonstrates specific misconception
• False_Misconception:[Type] = Incorrect answer with identifiable misconception

TASK: Based on the provided information and the guidelines above, classify this student's response into one of the categories. Pay close attention to both the answer and the explanation to determine the most fitting classification.

Classification:"""
    
    return prompt
# --- END: Feature Engineering Functions ---

# --- CRITICAL CHANGE: Load label encoder BEFORE the model to get n_classes ---
try:
    le = joblib.load(f"{DIR}/label_encoder.joblib")
    n_classes = len(le.classes_)
    print(f"Label encoder loaded successfully. Number of classes: {n_classes}")
except FileNotFoundError:
    print(f"Error: label_encoder.joblib not found at {DIR}/label_encoder.joblib")
    print("Please ensure the path is correct and the file exists.")
    sys.exit(1)



# --- Now load the base model with the correct number of classes ---
try:
    print("Loading base model with correct number of labels...")
    # 1. Load the base model with num_labels parameter
    base_model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL_PATH,
        num_labels=n_classes,
        device_map="auto",          # 多 GPU 自动切分
        torch_dtype=torch.float16   # 减少显存
    )
    
    print("Loading LoRA adapter...")
    # 2. Load the fused LoRA adapter on top of the base model
    model = PeftModel.from_pretrained(
        base_model,
        FUSED_LORA_PATH,
        device_map="auto",
        torch_dtype=torch.float16
    )
    
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    
    print("Model and tokenizer loaded successfully!")
except Exception as e:
    print(f"Error loading model or tokenizer: {e}")
    print("Please check if the model paths are correct:")
    print(f"  Base model path: {BASE_MODEL_PATH}")
    print(f"  LoRA adapter path: {FUSED_LORA_PATH}")
    sys.exit(1)

if model is None or tokenizer is None:
    print("Model or tokenizer failed to load. Exiting...")
    sys.exit(1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model.eval()

# Load feature engineering components
try:
    with open(f"{DIR}/feature_components.pkl", 'rb') as f:
        feature_components = pickle.load(f)
    print("Feature components loaded successfully!")
except FileNotFoundError:
    print(f"Error: feature_components.pkl not found at {DIR}/feature_components.pkl")
    print("Please ensure the path is correct.")
    sys.exit(1)

correct_train_data = feature_components['correct']
question_options = feature_components['question_options']
print("All components loaded successfully.")

try:
    test_df = pd.read_csv(TEST_FILE_PATH)
    print(f"Test data loaded successfully. Shape: {test_df.shape}")
except FileNotFoundError:
    print(f"Error: test.csv not found at {TEST_FILE_PATH}")
    print("Please ensure the path is correct.")
    sys.exit(1)

print("Applying feature engineering to test data...")
test_df = test_df.merge(question_options[['QuestionId', 'formatted_options']], on='QuestionId', how='left')
test_df = test_df.merge(correct_train_data, on=['QuestionId', 'MC_Answer'], how='left')
test_df['is_correct'] = test_df['is_correct'].fillna(0)
test_df['question_concept'] = test_df['QuestionText'].apply(extract_math_concepts)
test_df['text'] = test_df.apply(lambda row: format_input_with_new_template(row), axis=1)
print("Feature engineering applied to test data.")

def tokenize(batch):
    return tokenizer(
        batch["text"],
        padding="max_length",
        truncation=True,
        max_length=MAX_LEN
    )

print("Tokenizing test data...")
test_ds = Dataset.from_pandas(test_df[['text']])
test_ds = test_ds.map(tokenize, batched=True, batch_size=32)
test_ds.set_format(type='torch', columns=['input_ids', 'attention_mask'])
print("Test data tokenized.")

print("Making predictions...")
predictions = []
batch_size_inference = 4
total_batches = (len(test_ds) + batch_size_inference - 1) // batch_size_inference

for i in range(0, len(test_ds), batch_size_inference):
    batch_num = i // batch_size_inference + 1
    if batch_num % 10 == 0:
        print(f"Processing batch {batch_num}/{total_batches}")
    
    batch = test_ds[i:i+batch_size_inference]
    input_ids = batch['input_ids'].to(model.device)
    attention_mask = batch['attention_mask'].to(model.device)
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    # 修正之处：将logits移动回CPU以进行后续处理
    logits = outputs.logits.cpu().numpy()
    predictions.append(logits)

predictions = np.vstack(predictions)
probs = torch.nn.functional.softmax(torch.tensor(predictions), dim=-1).numpy()
print("Predictions completed.")

# --- Save probabilities for ensembling ---
output_filename = f"{MODEL_NAME}_predictions.npy"
np.save(output_filename, probs)
print(f"Probabilities saved successfully to {output_filename}")
print(f"Probabilities shape: {probs.shape}")

# Optional: Print a small sample to confirm the output
print("\nSample of the saved probability matrix:")
print(probs[:3, :5])


# 在代码末尾添加以下完整的GPU资源释放代码

print("\n" + "="*50)
print("开始释放GPU资源...")
print("="*50)

# 1. 删除模型和tokenizer对象
if 'model' in locals():
    del model
    print("✓ Model deleted")

if 'base_model' in locals():
    del base_model
    print("✓ Base model deleted")

if 'tokenizer' in locals():
    del tokenizer
    print("✓ Tokenizer deleted")

# 2. 删除数据相关对象
if 'test_ds' in locals():
    del test_ds
    print("✓ Test dataset deleted")

if 'test_df' in locals():
    del test_df
    print("✓ Test dataframe deleted")

# 3. 删除预测结果（如果不再需要）
if 'predictions' in locals():
    del predictions
    print("✓ Predictions deleted")

if 'probs' in locals():
    del probs
    print("✓ Probabilities deleted")

if 'logits' in locals():
    del logits
    print("✓ Logits deleted")

# 4. 删除batch数据
if 'batch' in locals():
    del batch
    print("✓ Batch deleted")

if 'input_ids' in locals():
    del input_ids
    print("✓ Input IDs deleted")

if 'attention_mask' in locals():
    del attention_mask
    print("✓ Attention mask deleted")

if 'outputs' in locals():
    del outputs
    print("✓ Outputs deleted")

# 5. 清空Python垃圾回收
import gc
gc.collect()
print("✓ Python garbage collected")

# 6. 清空PyTorch缓存
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print("✓ CUDA cache cleared")
    
    # 7. 同步CUDA设备
    torch.cuda.synchronize()
    print("✓ CUDA synchronized")
    
    # 8. 重置CUDA峰值内存统计
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.reset_accumulated_memory_stats()
    print("✓ CUDA memory stats reset")
    
    # 9. 显示当前GPU内存使用情况
    for i in range(torch.cuda.device_count()):
        allocated = torch.cuda.memory_allocated(i) / 1024**3
        reserved = torch.cuda.memory_reserved(i) / 1024**3
        print(f"  GPU {i}: Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")
else:
    print("⚠ CUDA not available, skipping CUDA-specific cleanup")

print("="*50)
print("GPU资源释放完成!")
print("="*50)


import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from datasets import Dataset
import joblib
import pickle
import re
import os
import sys
from peft import PeftModel

# Define the directory where the model and components are saved
DIR = "/kaggle/input/qwen3-4b-adapter-v3/whole_sub_adapter"
MAX_LEN = 640
TEST_FILE_PATH = "/kaggle/input/map-charting-student-math-misunderstandings/test.csv"

# Define the path to the base model and the fused adapter
BASE_MODEL_PATH = '/kaggle/input/qwen-3/transformers/4b/1'
FUSED_LORA_PATH = '/kaggle/input/qwen3-4b-adapter-v3/whole_sub_adapter/lora_adapter'

# --- NEW: Define a name for this specific model run for saving probabilities ---
MODEL_NAME = "qwen3_4b_v3_adapter"

# --- START: Feature Engineering Functions - MUST BE DEFINED HERE FOR PICKLE TO WORK ---
def format_options(options):
    options = sorted(list(options))
    formatted_str = ""
    for i, opt in enumerate(options):
        formatted_str += f"{chr(65 + i)}) {opt}\n"
    return formatted_str.strip()

def extract_math_concepts(text):
    concepts = []
    if re.search(r'\d+/\d+|fraction|numerator|denominator', text, re.I):
        concepts.append('fraction')
    if re.search(r'\d+\.\d+|decimal|point', text, re.I):
        concepts.append('decimal')
    if re.search(r'triangle|square|circle|shape|area|perimeter|angle|shaded', text, re.I):
        concepts.append('geometry')
    if re.search(r'greater|less|equal|compare|larger|smaller|highest|lowest', text, re.I):
        concepts.append('comparison')
    return ','.join(concepts) if concepts else 'other'

def format_input_with_new_template(row):
    """
    更新后的提示词格式化函数，包含提取出的选项。
    """
    judge_result = "CORRECT" if row['is_correct'] else "INCORRECT"

    # Mathematical context
    math_context = f"Concept: {row['question_concept']}"

    prompt = f"""You are an experienced math educator and a specialized AI assistant with a deep understanding of common student errors and pedagogical approaches. Your task is to analyze a student's response to a math problem and accurately classify it. You will focus on identifying the underlying mathematical misconception from a predefined list of categories.

Question: {row['QuestionText']}
Answer Choices:
{row['formatted_options']}
Student's Answer: {row['MC_Answer']}
Student's Explanation: {row['StudentExplanation']}
Judge: {judge_result}
Mathematical Context: {row['question_concept']}

--- IMPORTANT GUIDANCE ---
Based on the "Judge" field, the student's final answer has already been determined as either CORRECT or INCORRECT. This is a critical piece of information.
- If the Judge is "CORRECT", your classification MUST begin with "True_".
- If the Judge is "INCORRECT", your classification MUST begin with "False_".

CLASSIFICATION GUIDELINES:
• True_Correct:NA = Student demonstrates correct understanding
• False_Correct:NA = Student gives correct answer but for wrong reasons
• True_Neither:NA = Correct answer but unclear/incomplete reasoning
• False_Neither:NA = Incorrect answer but no specific misconception identified
• True_Misconception:[Type] = Correct answer but demonstrates specific misconception
• False_Misconception:[Type] = Incorrect answer with identifiable misconception

TASK: Based on the provided information and the guidelines above, classify this student's response into one of the categories. Pay close attention to both the answer and the explanation to determine the most fitting classification.

Classification:"""
    
    return prompt
# --- END: Feature Engineering Functions ---

# --- Load label encoder BEFORE the model to get n_classes ---
try:
    le = joblib.load(f"{DIR}/label_encoder.joblib")
    n_classes = len(le.classes_)
    print(f"Label encoder loaded successfully. Number of classes: {n_classes}")
except FileNotFoundError:
    print(f"Error: label_encoder.joblib not found at {DIR}/label_encoder.joblib")
    sys.exit(1)

# --- Load base model and LoRA adapter with multi-GPU ---
try:
    print("Loading base model with correct number of labels...")
    base_model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL_PATH,
        num_labels=n_classes,
        device_map="auto",
        torch_dtype=torch.float16
    )

    print("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(
        base_model,
        FUSED_LORA_PATH,
        device_map="auto",
        torch_dtype=torch.float16
    )

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    print("Model and tokenizer loaded successfully!")
except Exception as e:
    print(f"Error loading model or tokenizer: {e}")
    sys.exit(1)

model.eval()

# --- Load feature engineering components ---
try:
    with open(f"{DIR}/feature_components.pkl", 'rb') as f:
        feature_components = pickle.load(f)
    print("Feature components loaded successfully!")
except FileNotFoundError:
    print(f"Error: feature_components.pkl not found at {DIR}/feature_components.pkl")
    sys.exit(1)

correct_train_data = feature_components['correct']
question_options = feature_components['question_options']
print("All components loaded successfully.")

# --- Load test data ---
try:
    test_df = pd.read_csv(TEST_FILE_PATH)
    print(f"Test data loaded successfully. Shape: {test_df.shape}")
except FileNotFoundError:
    print(f"Error: test.csv not found at {TEST_FILE_PATH}")
    sys.exit(1)

print("Applying feature engineering to test data...")
test_df = test_df.merge(question_options[['QuestionId', 'formatted_options']], on='QuestionId', how='left')
test_df = test_df.merge(correct_train_data, on=['QuestionId', 'MC_Answer'], how='left')
test_df['is_correct'] = test_df['is_correct'].fillna(0)
test_df['question_concept'] = test_df['QuestionText'].apply(extract_math_concepts)
test_df['text'] = test_df.apply(lambda row: format_input_with_new_template(row), axis=1)
print("Feature engineering applied to test data.")

def tokenize(batch):
    return tokenizer(
        batch["text"],
        padding="max_length",
        truncation=True,
        max_length=MAX_LEN
    )

print("Tokenizing test data...")
test_ds = Dataset.from_pandas(test_df[['text']])
test_ds = test_ds.map(tokenize, batched=True, batch_size=32)
test_ds.set_format(type='torch', columns=['input_ids', 'attention_mask'])
print("Test data tokenized.")

# --- Inference ---
print("Making predictions...")
predictions = []
batch_size_inference = 8
total_batches = (len(test_ds) + batch_size_inference - 1) // batch_size_inference
for i in range(0, len(test_ds), batch_size_inference):
    batch_num = i // batch_size_inference + 1
    if batch_num % 10 == 0:
        print(f"Processing batch {batch_num}/{total_batches}")

    batch = test_ds[i:i + batch_size_inference]
    input_ids = batch['input_ids'].to(model.device)
    attention_mask = batch['attention_mask'].to(model.device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    logits = outputs.logits.cpu().numpy()
    predictions.append(logits)

predictions = np.vstack(predictions)
probs = torch.nn.functional.softmax(torch.tensor(predictions), dim=-1).numpy()
print("Predictions completed.")

# --- Save probabilities for ensembling ---
output_filename = f"{MODEL_NAME}_predictions.npy"
np.save(output_filename, probs)
print(f"Probabilities saved successfully to {output_filename}")
print(f"Probabilities shape: {probs.shape}")

# Optional: Print a small sample to confirm the output
print("\nSample of the saved probability matrix:")
print(probs[:3, :5])


# 在代码末尾添加以下完整的GPU资源释放代码

print("\n" + "="*50)
print("开始释放GPU资源...")
print("="*50)

# 1. 删除模型和tokenizer对象
if 'model' in locals():
    del model
    print("✓ Model deleted")

if 'base_model' in locals():
    del base_model
    print("✓ Base model deleted")

if 'tokenizer' in locals():
    del tokenizer
    print("✓ Tokenizer deleted")

# 2. 删除数据相关对象
if 'test_ds' in locals():
    del test_ds
    print("✓ Test dataset deleted")

if 'test_df' in locals():
    del test_df
    print("✓ Test dataframe deleted")

# 3. 删除预测结果（如果不再需要）
if 'predictions' in locals():
    del predictions
    print("✓ Predictions deleted")

if 'probs' in locals():
    del probs
    print("✓ Probabilities deleted")

if 'logits' in locals():
    del logits
    print("✓ Logits deleted")

# 4. 删除batch数据
if 'batch' in locals():
    del batch
    print("✓ Batch deleted")

if 'input_ids' in locals():
    del input_ids
    print("✓ Input IDs deleted")

if 'attention_mask' in locals():
    del attention_mask
    print("✓ Attention mask deleted")

if 'outputs' in locals():
    del outputs
    print("✓ Outputs deleted")

# 5. 清空Python垃圾回收
import gc
gc.collect()
print("✓ Python garbage collected")

# 6. 清空PyTorch缓存
import torch
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print("✓ CUDA cache cleared")
    
    # 7. 同步CUDA设备
    torch.cuda.synchronize()
    print("✓ CUDA synchronized")
    
    # 8. 重置CUDA峰值内存统计
    # torch.cuda.reset_peak_memory_stats() # 这些函数在新版本中可能不可用，暂且注释
    # torch.cuda.reset_accumulated_memory_stats() 
    # print("✓ CUDA memory stats reset")
    
    # 9. 显示当前GPU内存使用情况
    for i in range(torch.cuda.device_count()):
        allocated = torch.cuda.memory_allocated(i) / 1024**3
        reserved = torch.cuda.memory_reserved(i) / 1024**3
        print(f"  GPU {i}: Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")
else:
    print("⚠ CUDA not available, skipping CUDA-specific cleanup")

print("="*50)
print("GPU资源释放完成!")
print("="*50)


import pandas as pd
import numpy as np
import torch
import os
import sys
from sklearn.preprocessing import LabelEncoder
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding
from torch.utils.data import DataLoader
from scipy.special import softmax
from tqdm.auto import tqdm
import gc

# --- 0. 配置和路径 ---
# =================================================================
print("Starting Deepseek-Math-7B inference...")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 请根据您的环境修改以下路径
DEEPSEEK_MODEL_PATH = "/kaggle/input/deekseepmath-7b-map-competition/MAP_EXP_09_FULL"
TEST_FILE_PATH = "/kaggle/input/map-charting-student-math-misunderstandings/test.csv"
TRAIN_FILE_PATH = "/kaggle/input/map-charting-student-math-misunderstandings/train.csv"

# 推理参数
MAX_LEN = 256
BATCH_SIZE_INFERENCE = 4
MODEL_NAME = "deepseek_math_7b"

# --- 1. 数据准备和共享组件模拟 (DataProcessor) ---

class DataProcessor:
    """
    模拟多模型集成中用于共享 LabelEncoder 和正确答案查找表的类。
    """
    def __init__(self, train_path, test_path):
        self.train_df = pd.read_csv(train_path)
        self.test_df = pd.read_csv(test_path)
        self._le = None
        self._correct_lookup = None
        self._prepare_components()

    def _prepare_components(self):
        # 1. 准备 LabelEncoder (来自训练集)
        self.train_df['target'] = (
            self.train_df['Category'].fillna('NA') + ':' + self.train_df['Misconception'].fillna('NA')
        )
        self._le = LabelEncoder()
        self._le.fit(self.train_df['target'])
        
        # 2. 准备正确答案查找表
        correct_answers = self.train_df[self.train_df['Category'].str.startswith('True', na=False)].copy()
        # 统计出现次数，用于在重复项中选出最常见的作为“正确答案”
        correct_answers['count'] = correct_answers.groupby(['QuestionId', 'MC_Answer'])['MC_Answer'].transform('count')
        # 去重，保留出现次数最多的
        self._correct_lookup = correct_answers.sort_values('count', ascending=False).drop_duplicates(['QuestionId'])
        self._correct_lookup = self._correct_lookup[['QuestionId', 'MC_Answer']]
        self._correct_lookup['IsCorrect_flag'] = True # 使用您脚本中的列名

    def get_label_encoder(self):
        return self._le

    def get_num_classes(self):
        return len(self._le.classes_)

    @property
    def correct_lookup(self):
        return self._correct_lookup[['QuestionId', 'MC_Answer', 'IsCorrect_flag']]


# 初始化数据处理器
try:
    data_processor = DataProcessor(TRAIN_FILE_PATH, TEST_FILE_PATH)
    le_shared = data_processor.get_label_encoder()
    n_classes = data_processor.get_num_classes()
    correct_lookup_shared = data_processor.correct_lookup
    print(f"Data Processor initialized. Number of classes: {n_classes}")
except FileNotFoundError as e:
    print(f"Error loading data files: {e}")
    sys.exit(1)


# --- 2. Deepseek 输入格式化函数 ---

def format_deepseek_input(row):
    """根据模型的微调模板格式化输入文本。"""
    x = "This answer is correct."
    if not row['is_correct']:
        x = "This is answer is incorrect."
    return (
        f"Question: {row['QuestionText']}\n"
        f"Answer: {row['MC_Answer']}\n"
        f"{x}\n"
        f"Student Explanation: {row['StudentExplanation']}"
    )


# --- 3. 准备测试数据和输入 ---

print("Preparing test data for Deepseek-Math-7B...")
test_ds = data_processor.test_df.copy()

# 1. 合并正确答案查找表
test_ds = test_ds.merge(correct_lookup_shared, on=['QuestionId', 'MC_Answer'], how='left')
test_ds['is_correct'] = test_ds['IsCorrect_flag'].notna()
test_ds = test_ds.drop(columns=['IsCorrect_flag'])

# 2. 生成模型输入文本
test_ds['text'] = test_ds.apply(format_deepseek_input, axis=1)


# --- 4. 模型和 Tokenizer 加载 ---

try:
    print("Loading Deepseek-Math-7B model and tokenizer...")
    tokenizer_ds = AutoTokenizer.from_pretrained(DEEPSEEK_MODEL_PATH)
    model_ds = AutoModelForSequenceClassification.from_pretrained(
        DEEPSEEK_MODEL_PATH,
        num_labels=n_classes,  # 确保标签数量正确
        device_map="auto",
        torch_dtype=torch.float16
    )
    # 设置 Pad Token ID
    if tokenizer_ds.pad_token_id is None:
        if tokenizer_ds.eos_token_id is not None:
            tokenizer_ds.pad_token_id = tokenizer_ds.eos_token_id
        else:
            # 兼容极端情况
            tokenizer_ds.add_special_tokens({'pad_token': '[PAD]'})
            model_ds.config.pad_token_id = tokenizer_ds.pad_token_id
            
    model_ds.config.pad_token_id = tokenizer_ds.pad_token_id
    model_ds.eval()
    device = next(model_ds.parameters()).device
    print(f"Model loaded successfully. Using device: {device}")
except Exception as e:
    print(f"Error loading Deepseek model or tokenizer: {e}")
    sys.exit(1)


# --- 5. Tokenization 和 DataLoader ---

def tokenize_ds(batch):
    return tokenizer_ds(batch["text"], truncation=True, max_length=MAX_LEN)

print("Tokenizing test data...")
ds_test_ds = Dataset.from_pandas(test_ds[['text']])
# 移除 'text' 列以避免内存问题，并进行批量 Tokenization
ds_test_ds = ds_test_ds.map(tokenize_ds, batched=True, remove_columns=['text'])

# DataCollator 用于动态填充
data_collator = DataCollatorWithPadding(
    tokenizer=tokenizer_ds,
    padding=True,
    return_tensors="pt")

dataloader = DataLoader(
    ds_test_ds,
    batch_size=BATCH_SIZE_INFERENCE,
    shuffle=False,
    collate_fn=data_collator,
    pin_memory=True,
    num_workers=0)


# --- 6. 批量推理和结果保存 ---

print("Starting batch inference...")
all_logits = []
with torch.no_grad():
    for batch in tqdm(dataloader, desc="DeepSeek Inference"):
        # 将批次数据移动到 GPU
        batch = {k: v.to(device) for k, v in batch.items()}
        
        # 前向传播
        outputs = model_ds(**batch)
        logits = outputs.logits
        all_logits.append(logits.float().cpu().numpy())

# 合并所有 Logits
predictions_ds = np.concatenate(all_logits, axis=0)
# 应用 Softmax 转换为概率
deepseek_probs = softmax(predictions_ds, axis=1)

print(f"DeepSeek inference complete. Probabilities shape: {deepseek_probs.shape}")

# 保存概率数组
output_filename = f"{MODEL_NAME}_predictions.npy"
np.save(output_filename, deepseek_probs)
print(f"Probabilities saved successfully to {output_filename}")

# --- 7. 清理资源 ---
print("Cleaning up GPU memory...")
del model_ds, tokenizer_ds, dataloader, ds_test_ds, predictions_ds
gc.collect()
torch.cuda.empty_cache()
torch.cuda.synchronize()

print("\nDeepseek-Math-7B inference finished.")


import pandas as pd
import numpy as np
from collections import defaultdict
from scipy.special import softmax

# 你的函数保持不变
def extract_class_probabilities(row, model_suffix='', top_k=25):
    classes_col = f'top_classes{model_suffix}'
    if classes_col in row:
        classes = row[classes_col].split(' ')[:top_k]
    else:
        return {}
    class_probs = {}
    for i in range(min(top_k, len(classes))):
        prob_col = f'prob_{i}{model_suffix}'
        if prob_col in row:
            class_probs[classes[i]] = row[prob_col]
    return class_probs

def ensemble_with_disagreement_handling(prob_files, model_weights=None, top_k=3):
    n_models = len(prob_files)
    prob_dfs = []
    final_predictions = []

    for df in prob_files:
        prob_dfs.append(df)

    merged_df = prob_dfs[0]
    for i, df in enumerate(prob_dfs[1:], 1):
        suffix = f'_model{i+1}'
        df_renamed = df.rename(columns={
            col: f"{col}{suffix}" if col != 'row_id' else col for col in df.columns
        })
        merged_df = pd.merge(merged_df, df_renamed, on='row_id')

    for idx, row in merged_df.iterrows():
        all_class_probs = []
        for i in range(n_models):
            suffix = f'_model{i+1}' if i > 0 else ''
            class_probs = extract_class_probabilities(row, suffix, top_k=25)
            all_class_probs.append(class_probs)

        all_classes = set()
        for class_probs in all_class_probs:
            all_classes.update(class_probs.keys())

        class_votes = defaultdict(int)
        class_total_prob = defaultdict(float)
        class_max_prob = defaultdict(float)

        for i, class_probs in enumerate(all_class_probs):
            weight = model_weights[i]
            for class_name, prob in class_probs.items():
                class_votes[class_name] += 1
                class_total_prob[class_name] += prob * weight
                class_max_prob[class_name] = max(class_max_prob[class_name], prob * weight)

        final_scores = {}
        for class_name in all_classes:
            base_score = class_total_prob[class_name]
            agreement_bonus = class_votes[class_name] / n_models
            confidence_bonus = class_max_prob[class_name]
            final_scores[class_name] = (
                base_score * 0.7 +
                agreement_bonus * 0.2 +
                confidence_bonus * 0.1
            )

        sorted_classes = sorted(final_scores.items(), key=lambda x: -x[1])
        top_classes = [class_name for class_name, _ in sorted_classes[:top_k]]
        final_predictions.append(' '.join(top_classes))

    return final_predictions

# ✅ 加载 label encoder
le = joblib.load("/kaggle/input/map-competition-v17/whole_sub_adapter/label_encoder.joblib")

# ✅ 加载 .npy 概率文件
probs_1 = np.load("qwen3_4b_v3_adapter_predictions.npy")
probs_2 = np.load("qwen3-8b-v7_predictions.npy")
probs_3 = np.load("deepseek_math_7b_predictions.npy")

# ✅ 加载 test.csv 获取 row_id
test_df = pd.read_csv("/kaggle/input/map-charting-student-math-misunderstandings/test.csv")

# ✅ 将每个模型的概率转换为 CSV-like DataFrame
def probs_to_df(probs, model_name):
    topk = 15
    top_indices = np.argsort(-probs, axis=1)[:, :topk]
    top_probs = np.sort(probs, axis=1)[:, ::-1][:, :topk]
    top_classes = le.inverse_transform(top_indices.reshape(-1)).reshape(top_indices.shape)

    df = pd.DataFrame({'row_id': test_df['row_id']})
    df['top_classes'] = [' '.join(cls) for cls in top_classes]
    for i in range(topk):
        df[f'prob_{i}'] = top_probs[:, i]
    return df

# ✅ 转换为 DataFrame
df1 = probs_to_df(probs_1, "model1")
df2 = probs_to_df(probs_2, "model2")
df3 = probs_to_df(probs_3, "model3")

# ✅ 模型权重（可按性能调整）
w1, w2, w3 = 0.946, 0.945, 0.944

# ✅ 调用集成函数
predictions = ensemble_with_disagreement_handling(
    [df1, df2, df3],
    model_weights=[w1, w2, w3],
    top_k=3
)

# ✅ 生成提交文件
submission = pd.DataFrame({
    'row_id': test_df['row_id'],
    'Category:Misconception': predictions
})

submission.to_csv('submission.csv', index=False)
print("✅ 集成完成，submission.csv 已生成")
print(submission.head())




