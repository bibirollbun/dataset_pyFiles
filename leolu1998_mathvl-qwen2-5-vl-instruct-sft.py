!git clone https://github.com/modelscope/swift.git
!pip install git+https://github.com/huggingface/transformers.git



!git clone https://github.com/ECNU-ICALK/EduChat-Math.git


import json

file_path = '/kaggle/working/EduChat-Math/data/all_data.jsonl'
data = []

try:
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            # 移除行尾可能存在的换行符等空白字符，然后解析
            data.append(json.loads(line.strip()))
except FileNotFoundError:
    print(f"错误: 文件未找到，请确认路径 '{file_path}' 是否正确。")
except json.JSONDecodeError as e:
    print(f"错误: 文件内容不是有效的JSON Lines格式。错误发生在: {e}")

# 现在，`data` 是一个列表，其中每个元素都是一个从文件中读取的字典
if data:
    print(f"成功读取 {len(data)} 条数据。")
    print("第一条数据的预览:")
    print(data[1028])


import os
import json
import shutil
import re

# --- 常量与路径定义 ---
# 注意：请根据您的实际环境调整这些路径
SOURCE_JSONL_PATH = '/kaggle/working/EduChat-Math/data/all_data.jsonl'
SOURCE_IMAGES_BASE_PATH = '/kaggle/working/EduChat-Math/Images/All_Images'
OUTPUT_DATASET_DIR = '/kaggle/working/swift_vl_math_sft_data'
OUTPUT_IMAGES_DIR = os.path.join(OUTPUT_DATASET_DIR, 'images')
OUTPUT_JSONL_PATH = os.path.join(OUTPUT_DATASET_DIR, 'data.jsonl')


# --- 系统提示定义 ---

# 1. 用于处理带图片问题的系统提示 (原提示)
VL_MATH_SYSTEM_PROMPT = """你是一位精准高效的数学解题专家。你的任务是从所提供的图片中提取出数学题目，并提供简洁的解题思路，最终给出格式规范的答案。

要求：
    1. 请按以下结构组织你的回答：
    •   重述问题：清晰简洁地描述图像中的数学问题。
    •   解题思路：概述你的推理过程及解题步骤。
        * 简要总结解决问题所需的关键步骤与逻辑。
        * 要简明扼要，不要包含细小计算或冗长解释。
    2. 最终答案部分只包含最终结果，不能有其他内容。
    •  若为选择题：
        * 只输出对应的大写选项字母。
    •  若为非选择题：
        * 只输出数值结果，且格式为浮点数。

请严格遵循以下两段式格式进行回答：

###【分析】###
<在这里简要概述你的解题步骤和核心逻辑>

###【答案】###
<在这里给出清理过的、格式绝对正确的最终答案>""".strip()

# 2. 【新增】用于处理纯文本问题的系统提示
TEXT_MATH_SYSTEM_PROMPT = """你是一位精准高效的数学解题专家。你的任务是分析所提供的数学题目，提供简洁的解题思路，并最终给出格式规范的答案。

要求：
    1. 请按以下结构组织你的回答：
    •   重述问题：清晰简洁地描述问题。
    •   解题思路：概述你的推理过程及解题步骤。
        * 简要总结解决问题所需的关键步骤与逻辑。
        * 要简明扼要，不要包含细小计算或冗长解释。
    2. 最终答案部分只包含最终结果，不能有其他内容。
    •  若为选择题：
        * 只输出对应的大写选项字母。
    •  若为非选择题：
        * 只输出数值结果，且格式为浮点数。

请严格遵循以下两段式格式进行回答：

###【分析】###
<在这里简要概述你的解题步骤和核心逻辑>

###【答案】###
<在这里给出清理过的、格式绝对正确的最终答案>""".strip()


def clean_answer(answer_text: str, is_mcq: bool):
    """
    根据问题类型（选择题/填空题）清理答案字符串。

    Args:
        answer_text: 原始的答案字符串。
        is_mcq: 布尔值，如果为 True，则为选择题。

    Returns:
        清理和格式化后的答案字符串，如果无法提取则返回 None。
    """
    if not isinstance(answer_text, str):
        return None

    if is_mcq:
        # 对于选择题，查找答案中独立的单个大写字母。
        # 使用 \b 确保是独立的字母，而不是单词的一部分。
        match = re.search(r'\b([A-Z])\b', answer_text)
        if match:
            return match.group(1)
    else:
        # 对于填空题，查找所有数字（包括整数、浮点数和负数）。
        numbers = re.findall(r'-?\d+\.?\d*', answer_text)
        if numbers:
            try:
                # 返回找到的最后一个数字，并格式化为浮点数字符串。
                return str(float(numbers[-1]))
            except (ValueError, IndexError):
                # 如果转换失败，则返回 None。
                return None
    
    # 如果没有找到符合条件的答案，返回 None。
    return None


def convert_data_for_swift_vl(
    source_jsonl: str,
    source_images_base: str,
    target_jsonl: str,
    target_images_dir: str
):
    """
    将原始数据转换为 Swift-VL 格式，并根据新 prompt 进行调整。
    此版本已修正，会包含所有有效的样本，包括无图片的非选择题。
    """
    os.makedirs(target_images_dir, exist_ok=True)
    
    valid_samples_count = 0
    skipped_samples_count = 0
    total_lines = 0

    with open(source_jsonl, 'r', encoding='utf-8') as f_source, \
         open(target_jsonl, 'w', encoding='utf-8') as f_target:
        
        for line in f_source:
            total_lines += 1
            try:
                example = json.loads(line.strip())
            except json.JSONDecodeError:
                skipped_samples_count += 1
                continue

            # --- 1. 提取核心数据 ---
            question_text = example.get('question')
            analysis_text = example.get('analysis')
            raw_answer_text = example.get('answer')
            options_val = example.get('options')
            image_list = example.get('image')

            # --- 2. 判断题目类型 (选择题 vs. 非选择题) ---
            # 如果 'options' 字段是非空字符串或非空列表，则为选择题。
            is_mcq = (isinstance(options_val, str) and bool(options_val.strip())) or \
                     (isinstance(options_val, list) and bool(options_val))
            
            # --- 3. 清理答案并验证基础字段 ---
            cleaned_answer = clean_answer(raw_answer_text, is_mcq)
            
            # 如果缺少必要文本字段或无法解析出有效答案，则跳过此样本。
            if not all([question_text, analysis_text, raw_answer_text, cleaned_answer is not None]):
                skipped_samples_count += 1
                continue

            # --- 4. 验证图片有效性 ---
            has_image_field = isinstance(image_list, list) and len(image_list) > 0 and image_list[0]
            valid_image_present = False
            image_filename = None
            if has_image_field:
                image_filename = image_list[0]
                source_image_path = os.path.join(source_images_base, image_filename)
                if os.path.exists(source_image_path):
                    valid_image_present = True

            # --- 5. 过滤逻辑 ---
            # 确保所有通过了基础字段验证的样本（包括无图片的非选择题）都能被处理。

            # --- 6. 构建用户输入 (user content) ---
            cleaned_question = question_text.replace('<image>', '').strip()
            # 为了清晰，可以在问题前缀中标识题目类型
            user_content_prefix = '选择题：' if is_mcq else '非选择题：'
            user_content = user_content_prefix + cleaned_question
            
            if is_mcq:
                options_str = ""
                if isinstance(options_val, str):
                    options_str = options_val
                elif isinstance(options_val, list):
                    options_str = '\n'.join(map(str, options_val))
                
                if options_str.strip():
                     user_content += f"\n\n{options_str.strip()}"

            # --- 7. 构建助手回答 (assistant content) ---
            assistant_content = (
                f"###【分析】###\n{analysis_text.strip()}\n\n"
                f"###【答案】###\n{cleaned_answer}"
            )
            
            # --- 8. 根据是否有图，选择合适的 System Prompt ---
            system_prompt_to_use = VL_MATH_SYSTEM_PROMPT if valid_image_present else TEXT_MATH_SYSTEM_PROMPT

            # --- 9. 组装最终的 JSON 对象 ---
            # 最终输出的 swift_example 结构包含了 messages 列表和可选的 image 字段
            swift_example = {
                "messages": [
                    {"role": "system", "content": system_prompt_to_use},
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": assistant_content}
                ]
            }
            
            # --- 10. 处理图片：复制并添加相对路径到 JSON 中 ---
            if valid_image_present:
                source_path = os.path.join(source_images_base, image_filename)
                dest_path = os.path.join(target_images_dir, image_filename)
                shutil.copyfile(source_path, dest_path)
                # 使用相对路径，方便数据集迁移
                swift_example['image'] = os.path.join('images', image_filename)

            # --- 11. 写入目标文件 ---
            f_target.write(json.dumps(swift_example, ensure_ascii=False) + '\n')
            valid_samples_count += 1

    print("\n" + "="*50)
    print("数据转换完成！")
    print(f"共处理 {total_lines} 行原始数据。")
    print(f"成功转换 {valid_samples_count} 条有效样本。")
    print(f"跳过 {skipped_samples_count} 条不符合条件的样本。")
    print(f"数据描述文件已生成: {target_jsonl}")
    print(f"图片文件已复制至: {target_images_dir}")
    print("="*50 + "\n")


# --- 主执行块 ---
if __name__ == '__main__':
    # 确保输出目录存在
    if not os.path.exists(OUTPUT_DATASET_DIR):
        os.makedirs(OUTPUT_DATASET_DIR)

    # 运行转换函数
    # 在实际运行前，请确保 SOURCE_JSONL_PATH 和 SOURCE_IMAGES_BASE_PATH 指向您的数据。
    convert_data_for_swift_vl(
        source_jsonl=SOURCE_JSONL_PATH,
        source_images_base=SOURCE_IMAGES_BASE_PATH,
        target_jsonl=OUTPUT_JSONL_PATH,
        target_images_dir=OUTPUT_IMAGES_DIR
    )


import json

file_path = '/kaggle/working/swift_vl_math_sft_data/data.jsonl'
data = []

try:
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            # 移除行尾可能存在的换行符等空白字符，然后解析
            data.append(json.loads(line.strip()))
except FileNotFoundError:
    print(f"错误: 文件未找到，请确认路径 '{file_path}' 是否正确。")
except json.JSONDecodeError as e:
    print(f"错误: 文件内容不是有效的JSON Lines格式。错误发生在: {e}")

# 现在，`data` 是一个列表，其中每个元素都是一个从文件中读取的字典
if data:
    print(f"成功读取 {len(data)} 条数据。")
    print("第一条数据的预览:")
    print(data[11110])


%cd /kaggle/working/swift


pip check


!pip install -e .[llm]


!pip install qwen_vl_utils


!SIZE_FACTOR=8 MAX_PIXELS=602112 CUDA_VISIBLE_DEVICES=0 swift sft \
    --model_type qwen2_5_vl \
    --model Qwen/Qwen2.5-VL-3B-Instruct \
    --train_type lora \
    --dataset /kaggle/working/swift_vl_math_sft_data/data.jsonl#100\
    --output_dir ./finetuned_qwen2_5_vl \
    --save_strategy "epoch"


from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch

# Path to your fine-tuned model
model_checkpoint = "finetuned_qwen2_5_vl/v1-20250623-113208/checkpoint-3-merged"
initial_prompt="""You are an expert in solving mathematical problems. Please begin by extracting the math problem from the provided image, and then solve it.

Requirements:
	1.	All mathematical formulas and symbols in your response must be written in LaTeX format.
	2.	Organize your response according to the following structure:
	•	Restate the Problem: Clearly and concisely describe the math problem shown in the image.
	•	Solution Approach: Outline your reasoning and the steps taken to solve the problem.
	•	Final Answer: Present the complete solution.
	3. The final answer can only contain the final result number or option.

Strictly follow the format below in your output:
### Think ###
<Restate the problem and outline the solution approach>

### Answer ###
<Final answer>"""
model =Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_checkpoint,
    torch_dtype=torch.float16,  # Use float16 for efficiency
    device_map="auto"  # Auto-assign to GPU if available
)

# Load processor
processor = AutoProcessor.from_pretrained(model_checkpoint)

# Image file path
file_name = "/kaggle/input/math-vl/images/0.JPEG"

# Prepare input message with image + prompt
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": file_name,
            },
            {
                "type": "text",
                "text": initial_prompt
            }
        ]
    }
]

# Apply chat template
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

# Process image
image_inputs, video_inputs = process_vision_info(messages)

# Prepare model inputs
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)

# Move inputs to GPU if available
device = "cuda"
model.to(device)
inputs = inputs.to(device)

# Generate response
with torch.no_grad():
    generated_ids = model.generate(**inputs, max_new_tokens=2048)

# Trim input tokens from output
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]

# Decode output
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=True
)

# Print final JSON output
for i in output_text[0].split("\n"):
    print (i)

