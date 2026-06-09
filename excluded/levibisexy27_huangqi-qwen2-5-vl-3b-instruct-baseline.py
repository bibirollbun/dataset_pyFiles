!pip install qwen_vl_utils


from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import os
from tqdm import tqdm
import json
import re
import torch
import sys

initial_prompt="""你是一位擅长数学推理的老师。你的任务是分析所提供图像中的数学题目，准确提取题干，并给出解答。

要求如下：

1.所有数学符号、公式必须使用 LaTeX 语法进行书写。

2.请按照以下结构组织你的回答：
    题目重述：清晰、简洁地描述图像中的数学题。如果是英文题则翻译至中文再解答。在OCR识别过程中仔细识别，分清楚O和0，l和1等容易混淆的符号。
    解题思路：简要列出你的解题逻辑、关键步骤或使用的公式。如果是几何题，重点描述图形特征解答，如果是代数题注意符合和公式的运用，如果是应用题，重点理解文字描述和逻辑关系。
    最终答案：只写出最终结果（如一个数字，或选项 A/B/C/D）。
    例：填空题答案为浮点数如0.5，不能为分数如1/2；选择题答案为大写字母选项如C，不能如C.9。
3.得出答案后，进行再次检查，再次确认无误则可以输出答案。
请严格按照如下格式输出：

### Think ###
<题目重述 + 解题思路（使用 LaTeX 格式）>

### Answer ###
<最终结果，仅限数字或字母>"""

def extract_steps_and_answer(response):
    """
    从模型响应中提取 <Restate the problem and outline the solution approach> 和 <Final answer>
    """
    # 定义正则表达式模式
    restate_pattern = r"### Think ###\n(.*?)\n### Answer ###"
    answer_pattern = r"### Answer ###\n(.*)"

    # 匹配 <Restate the problem and outline the solution approach>
    restate_match = re.search(restate_pattern, response, re.DOTALL)
    step = restate_match.group(1).strip() if restate_match else ""

    # 匹配 <Final answer>
    answer_match = re.search(answer_pattern, response, re.DOTALL)
    answer = answer_match.group(1).strip() if answer_match else ""

    return step, answer



def run(image_dir,model_path):
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.float16, device_map="auto")
    
    processor = AutoProcessor.from_pretrained(model_path)

    image_path = ['0.JPEG',
                 '1.png',
                 '2.JPEG',
                 '3.png',
                 '4.JPEG',
                 '5.png',
                 '6.JPEG',
                 '7.png',
                 '8.JPEG',
                 '9.png',
                 '10.JPEG',
                 '11.png']
    res = []
    paths = [os.path.join(image_dir, filename) for filename in image_path]
    for image in tqdm(paths, desc="Processing"):
        
        # Prepare the messages for the model
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": initial_prompt},
                ],
            }
        ]

        # Preparation for inference
        # apply_chat_template：将messages转换为模型要求的字符串格式
        # add_generation_prompt=True：在末尾添加模型生成的起始标记
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(model.device)

        # Inference: Generation of the output
        generated_ids = model.generate(**inputs, max_new_tokens=1024)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        step, answer = extract_steps_and_answer(output_text)
        print('图片：',image,'\n')
        print('解题步骤：',step,'\n')
        print('答案：',answer,'\n')
        print(18*"*")
        res.append(answer)
    return res





!cp /kaggle/input/qwen2.5-vl/transformers/3b-instruct/2 -r ./


import json

config_data = {
  "min_pixels": 3136,
  "max_pixels": 12845056,
  "patch_size": 14,
  "temporal_patch_size": 2,
  "merge_size": 2,
  "image_mean": [
    0.48145466,
    0.4578275,
    0.40821073
  ],
  "image_std": [
    0.26862954,
    0.26130258,
    0.27577711
  ],
  "image_processor_type": "Qwen2VLImageProcessor",
  "processor_class": "Qwen2_5_VLProcessor"
}

output_filename = "2/preprocessor_config.json"

with open(output_filename, 'w', encoding='utf-8') as f:
    json.dump(config_data, f, ensure_ascii=False, indent=4)


import glob
import os
image_dir = "/kaggle/input/math-vl/images/"
model_path = "/kaggle/working/2"
res = run(image_dir,model_path)


res


import pandas as pd
submission = pd.read_csv('/kaggle/input/math-vl/submission.csv')
submission['answer'] = res
submission.to_csv('submission.csv',index=False)


submission




