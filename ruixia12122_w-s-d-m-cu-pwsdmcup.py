


# !pip uninstall torch --y
import torch
torch.__version__


# !pip install vllm
# # !pip install triton
# !pip install /kaggle/input/sxcdscds/*.whl
# !pip install /kaggle/input/final/*.whl
# !pip install /kaggle/input/complementrary/*.whl
# !pip install /kaggle/input/xformers/xformers-0.0.28.post3-cp310-cp310-manylinux_2_28_x86_64.whl
# !pip install /kaggle/input/xjsaioj/prometheus_fastapi_instrumentator-7.0.2-py3-none-any.whl


#!/bin/bash

# !pip install /kaggle/input/sfca/transformers/default/1/triton-3.1.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
# # 使用 wget 下载文件到目标文件夹
# !pip install /kaggle/input/sfca/transformers/default/1/vllm-0.6.6.post1-cp38-abi3-manylinux1_x86_64.whl
# # !wget https://files.pythonhosted.org/packages/b0/14/9790c07959456a92e058867b61dc41dde27e1c51e91501b18207aef438c5/vllm-0.6.6.post1-cp38-abi3-manylinux1_x86_64.whl



# import torch
# from vllm import LLM, SamplingParams
# import pandas as pd
# import torch
# import torch.distributed as dist

# print(f"Number of GPUs available: {torch.cuda.device_count()}")
# # 初始化模型
# model_name = "/kaggle/input/qw/transformers/default/1/qwen_merged_lora_model"
# llm = LLM(
#     model=model_name,
#     dtype=torch.float16,
#     max_model_len=2048,
#     tensor_parallel_size=2,
#     gpu_memory_utilization=0.96
# )

# # 定义分块函数
# def chunk_list(data, chunk_size=5):
#     """
#     将列表按指定大小拆分成小组
#     :param data: 原始列表
#     :param chunk_size: 每组的元素个数，默认为5
#     :return: 拆分后的列表组
#     """
#     return [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]

# # 加载数据
# path = '/kaggle/input/wsdm-cup-multilingual-chatbot-arena/'
# train = pd.read_parquet(path + "train.parquet", engine='pyarrow')
# test = pd.read_parquet(path + "test.parquet", columns=['id', 'prompt', 'response_a', 'response_b', 'scored'])

# # 生成测试集和训练集的 prompt
# test_list = []
# for p, a, b in zip(test.prompt.values, test.response_a.values, test.response_b.values):
#     text = (
#         f"<Lang>\n{'zh'}\n</Lang>\n"
#         f"<Query>\n{p}\n</Query>\n"
#         f"-----------\n"
#         f"<Response_A>\n{a}\n</Response_A>\n"
#         f"-----------\n"
#         f"<Response_B>\n{b}\n</Response_B>\n"
#         f"Which is better?\n"
#         f"Choice:"
#     )
#     test_list.append(text)

# train_list = []
# for p, a, b in zip(train.prompt.values, train.response_a.values, train.response_b.values):
#     text = (
#         f"<Lang>\n{'zh'}\n</Lang>\n"
#         f"<Query>\n{p}\n</Query>\n"
#         f"-----------\n"
#         f"<Response_A>\n{a}\n</Response_A>\n"
#         f"-----------\n"
#         f"<Response_B>\n{b}\n</Response_B>\n"
#         f"Which is better?\n"
#         f"Choice:"
#     )
#     train_list.append(text)

# # 定义采样参数
# sampling_params = SamplingParams(
#     temperature=0,
#     top_p=1,
#     n=1,
#     top_k=1,
#     seed=777,
#     skip_special_tokens=False
# )

# # 分块处理测试集
# chunked = chunk_list(test_list, chunk_size=10)
# result = []

# # 逐块推理
# for i, v in enumerate(chunked):
#     conversation = [[
#         {
#             "role": "system",
#             "content": "You are a helpful assistant"
#         },
#         {
#             "role": "user",
#             "content": prompt
#         }
#     ] for prompt in v]

#     outputs = llm.chat(
#         conversation,
#         sampling_params=sampling_params,
#         use_tqdm=False
#     )

#     # 处理输出
#     for output in outputs:
#         generated_text = output.outputs[0].text
#         if generated_text != "":
#             result.append(generated_text)
#         else:
#             result.append("model_a")

# # 生成提交文件
# sub = pd.DataFrame({
#     'id': test.id.values,
#     'winner': result
# })
# sub.to_csv("submission.csv", index=False)



!pip install /kaggle/input/language3/langdetect_py-1.1.1-py3-none-any.whl


import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
from langdetect import detect  # 用于语言检测

# 检查可用的 GPU 数量
num_gpus = torch.cuda.device_count()
print(f"Number of GPUs available: {num_gpus}")

# 初始化模型和分词器
model_name = "/kaggle/input/qw/transformers/default/1/qwen_merged_lora_model"
device = "cuda" if torch.cuda.is_available() else "cpu"

# 加载模型和分词器
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)

# 如果有多块 GPU，提示用户
if num_gpus > 1:
    print(f"Multiple GPUs detected ({num_gpus}). Consider using DistributedDataParallel for better performance.")
    model = torch.nn.DataParallel(model)

# 将模型转移到 GPU（或 CPU）
model = model.to(device)

# 加载数据
path = '/kaggle/input/wsdm-cup-multilingual-chatbot-arena/'
train = pd.read_parquet(path + "train.parquet", engine='pyarrow')
test = pd.read_parquet(path + "test.parquet", columns=['id', 'prompt', 'response_a', 'response_b', 'scored'])

# 语言检测函数
def detect_language(text):
    """
    检测文本的语言
    :param text: 输入的文本
    :return: 语言代码（如 'zh' 表示中文）
    """
    try:
        return detect(text)
    except Exception as e:
        print(f"Language detection failed for text: {text}. Error: {e}")
        return "en"  # 如果检测失败，默认返回英文

# 生成测试集和训练集的 prompt
def generate_prompt(prompt, response_a, response_b):
    """
    生成模型输入的prompt
    """
    language = detect_language(prompt)  # 检测语言
    return (
        f"<Lang>\n{language}\n</Lang>\n"
        f"<Query>\n{prompt}\n</Query>\n"
        f"-----------\n"
        f"<Response_A>\n{response_a}\n</Response_A>\n"
        f"-----------\n"
        f"<Response_B>\n{response_b}\n</Response_B>\n"
        f"Which is better?\n"
        f"Choice:"
    )

# 为测试集生成 prompts
test_prompts = [
    generate_prompt(p, a, b)
    for p, a, b in zip(test.prompt.values, test.response_a.values, test.response_b.values)
]

# 定义推理函数
def generate_responses(prompts, model, tokenizer, device, max_length=2048):
    """
    使用模型生成响应
    """
    results = []
    for prompt in prompts:
        try:
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length).to(device)
            print(f"Input length: {inputs['input_ids'].shape[1]}")  # 调试输入长度

            # 检查是否使用了 DataParallel，如果是，访问内部模型
            if isinstance(model, torch.nn.DataParallel):
                outputs = model.module.generate(  # 访问内部模型
                    **inputs,
                    max_length=max_length,
                    do_sample=False,  # 确定性生成
                    top_p=1.0,
                    top_k=1,
                    num_return_sequences=1
                )
            else:
                outputs = model.generate(
                    **inputs,
                    max_length=max_length,
                    do_sample=False,  # 确定性生成
                    top_p=1.0,
                    top_k=1,
                    num_return_sequences=1
                )

            decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"Decoded output: {decoded_output}")  # 调试输出

            # 提取生成结果（选择 model_a 或 model_b）
            if "model_a" in decoded_output.lower():
                results.append("model_a")
            elif "model_b" in decoded_output.lower():
                results.append("model_b")
            else:
                results.append("model_a")  # 默认选择 A
        except Exception as e:
            print(f"Error during generation for prompt: {prompt}. Error: {e}")
            results.append("model_a")  # 如果发生错误，默认选择 A
    return results

# 对测试集逐条推理
print("Starting inference on the test set...")
test_results = generate_responses(test_prompts, model, tokenizer, device)

# 生成提交文件
submission = pd.DataFrame({
    'id': test.id.values,
    'winner': test_results
})
submission.to_csv("submission.csv", index=False)
print("Submission file created: submission.csv")


# import torch
# import pandas as pd
# from transformers import AutoModelForCausalLM, AutoTokenizer

# # 1. 加载合并后的模型和分词器
# MERGED_MODEL_PATH = "/kaggle/input/merge_lora/transformers/default/1/merged_lora_model"  # 替换为你的模型路径
# model = AutoModelForCausalLM.from_pretrained(MERGED_MODEL_PATH, torch_dtype=torch.float16)  # 使用 FP16 加载模型
# tokenizer = AutoTokenizer.from_pretrained(MERGED_MODEL_PATH)

# # 解决 tokenizer 缺失 pad_token 的问题
# if tokenizer.pad_token is None:
#     tokenizer.pad_token = tokenizer.eos_token  # 将 eos_token 用作 pad_token
# tokenizer.padding_side = 'left'

# # 2. 将模型转移到 GPU 上
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# model.to(device)

# # 3. 加载测试数据
# test_data_path = "/kaggle/input/wsdm-cup-multilingual-chatbot-arena/test.parquet"  # 替换为你的测试数据路径
# test_df = pd.read_parquet(test_data_path)

# # 4. 构造推理用的 prompt
# def construct_prompt(prompt, response_a, response_b):
#     return f"""
# <Query> {prompt} </Query>
# <Response_A> {response_a} </Response_A>
# <Response_B> {response_b} </Response_B>
# Which is better?
# Choice: model_"""

# # 5. 批量推理函数
# def batch_infer(prompts, batch_size=8):
#     results = []
#     for i in range(0, len(prompts), batch_size):
#         batch = prompts[i:i+batch_size]
#         inputs = tokenizer(batch, return_tensors="pt", max_length=512, truncation=True, padding=True)  # 动态调整长度
#         inputs = {k: v.to(device) for k, v in inputs.items()}
#         with torch.no_grad():
#             # 使用生成的限制参数加速推理
#             outputs = model.generate(
#                 **inputs,
#                 max_length=513,  # 限制生成长度
#                 top_k=20,        # 限制 top_k
#                 top_p=0.9,       # 限制 top_p
#                 num_beams=1      # 禁用多束搜索
#             )
#         decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
#         results.extend(decoded_outputs)
#     return results

# # 6. 构造所有 prompts
# prompts = [
#     construct_prompt(row["prompt"], row["response_a"], row["response_b"])
#     for _, row in test_df.iterrows()
# ]

# # 7. 批量推理
# batch_size = 16  # 根据显存大小调整
# model_outputs = batch_infer(prompts, batch_size=batch_size)

# # 8. 解析输出并保存结果
# results = []
# for i, output in enumerate(model_outputs):
#     if "model_a" in output:
#         winner = "model_a"
#     elif "model_b" in output:
#         winner = "model_b"
#     else:
#         winner = "model_a"  # 默认值
#     results.append({"id": test_df.iloc[i]["id"], "winner": winner})

# # 9. 将推理结果转换为 DataFrame
# results_df = pd.DataFrame(results)

# # 10. 保存结果到指定路径的 CSV 文件
# output_path = "submission.csv"
# results_df.to_csv(output_path, index=False)

# print(f"推理完成，结果已保存到 {output_path}")

