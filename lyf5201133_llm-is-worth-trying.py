# 【Task】
# You are an expert in bank marketing, Your goal is to predict whether a client will subscribe(0 or 1) to a bank term deposit based on client data.
# 【Category Definition】
# - 1:client will subscribe
# - 0:client will not subscribe
# 【Example 1】
# - Input:age:43;job:technician;marital:married;education:secondary;default:no;balance:-274;housing_loan:yes;personal_loan:no;contact:cellular;day:26;month:aug;duration:160;campaign:2;pdays:-1;previous:0;poutcome:unknown
# - Output:0
# 【Example 2】
# - Input:age:27;job:management;marital:single;education:tertiary;default:no;balance:376;housing_loan:no;personal_loan:no;contact:cellular;day:3;month:dec;duration:641;campaign:2;pdays:205;previous:1;poutcome:failure
# - Output:1
# 【Client Data to Predict】
# - age:32;job:technician;marital:married;education:tertiary;default:no;balance:3716;housing_loan:yes;personal_loan:no;contact:cellular;day:19;month:nov;duration:167;campaign:1;pdays:-1;previous:0;poutcome:unknown
# 【Output Requirements】
# - ONLY output "0" or "1" with NO additional text
# - DO NOT explain or analyze
# 【Result】
# - Output:


# {
#       "conversations": [
#          {
#             "value": "【Task】
# You are an expert in bank marketing, Your goal is to predict whether a client will subscribe(0 or 1) to a bank term deposit based on client data.
# 【Category Definition】
# - 1:client will subscribe
# - 0:client will not subscribe
# 【Example 1】
# - Input:age:43;job:technician;marital:married;education:secondary;default:no;balance:-274;housing_loan:yes;personal_loan:no;contact:cellular;day:26;month:aug;duration:160;campaign:2;pdays:-1;previous:0;poutcome:unknown
# - Output:0
# 【Example 2】
# - Input:age:27;job:management;marital:single;education:tertiary;default:no;balance:376;housing_loan:no;personal_loan:no;contact:cellular;day:3;month:dec;duration:641;campaign:2;pdays:205;previous:1;poutcome:failure
# - Output:1
# 【Client Data to Predict】
# - age:32;job:technician;marital:married;education:tertiary;default:no;balance:3716;housing_loan:yes;personal_loan:no;contact:cellular;day:19;month:nov;duration:167;campaign:1;pdays:-1;previous:0;poutcome:unknown
# 【Output Requirements】
# - ONLY output "0" or "1" with NO additional text
# - DO NOT explain or analyze
# 【Result】
# - Output:",
#             "from": "user"
#          },
#          {
#             "value": "0",
#             "from": "gpt"
#          }
#       ]
#    }


# model_name_or_path: /data/jupyter/dev/lyf/llms/LLM-Research/Meta-Llama-3.1-8B-Instruct
# trust_remote_code: true


# ###
# stage: sft
# do_train: true
# finetuning_type: lora
# lora_rank: 32
# lora_target: q_proj,v_proj


# deepspeed: examples/deepspeed/ds_z2_config.json
# ### dataset
# dataset: kaggle_clf_train_data
# template: llama3
# cutoff_len: 360
# overwrite_cache: true
# preprocessing_num_workers: 16
# dataloader_num_workers: 0

# ### output
# output_dir: /data/jupyter/dev/lyf/LLaMA-Factory/examples/save/adapter_llama31_kaggle_v5
# logging_steps: 100
# save_steps: 500
# plot_loss: true
# overwrite_output_dir: true
# save_only_model: false
# report_to: none  

# ### train
# per_device_train_batch_size: 16
# gradient_accumulation_steps: 4
# learning_rate: 3.0e-5
# num_train_epochs: 12.0
# lr_scheduler_type: cosine
# warmup_ratio: 0.1
# bf16: true
# ddp_timeout: 180000000
# resume_from_checkpoint: null

# ### eval
# eval_dataset: kaggle_clf_val_data
# per_device_eval_batch_size: 32
# eval_strategy: steps
# eval_steps: 500


# import torch
# from transformers import AutoTokenizer, AutoModelForCausalLM
# # 1. 加载模型与分词器
# model_name = "/data/jupyter/dev/lyf/LLaMA-Factory/examples/save/kaggle_llama31_sft_v5"  # 可选其他版本如 0.5B/1.5B/14B
# tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
# model = AutoModelForCausalLM.from_pretrained(
#     model_name,
#     device_map="cuda:1",
#     torch_dtype=torch.bfloat16,  # 优化显存，精度足够
#     trust_remote_code=True
# )


# res_data = []
# prompt = """【Task】
# You are an expert in bank marketing, Your goal is to predict whether a client will subscribe(0 or 1) to a bank term deposit based on client data.
# 【Category Definition】
# - 1:client will subscribe
# - 0:client will not subscribe
# 【Example 1】
# - Input:age:43;job:technician;marital:married;education:secondary;default:no;balance:-274;housing_loan:yes;personal_loan:no;contact:cellular;day:26;month:aug;duration:160;campaign:2;pdays:-1;previous:0;poutcome:unknown
# - Output:0
# 【Example 2】
# - Input:age:27;job:management;marital:single;education:tertiary;default:no;balance:376;housing_loan:no;personal_loan:no;contact:cellular;day:3;month:dec;duration:641;campaign:2;pdays:205;previous:1;poutcome:failure
# - Output:1
# 【Client Data to Predict】
# - {0}
# 【Output Requirements】
# - ONLY output "0" or "1" with NO additional text
# - DO NOT explain or analyze
# 【Result】
# - Output:"""
# for index, row in test_data.iterrows():
#     id_no = row['id']
#     temp_str = "age:" + str(row['age']) + ";" + "job:" + row['job'] + ";" + "marital:" + row['marital'] + ";" + "education:" + row['education'] +";" + "default:" + row['default'] + ";" + "balance:" + str(row['balance']) + ";" + "housing_loan:" + row['housing'] + ";" + "personal_loan:" + row['loan'] +";"+ "contact:" + row['contact']  + ";"+ "day:" + str(row['day']) + ";" + "month:" + row['month']+ ";" + "duration:" + str(row['duration']) + ";" + "campaign:" + str(row['campaign']) + ";" + "pdays:" + str(row['pdays']) + ";" + "previous:" + str(row['previous'])+ ";" + "poutcome:" + row['poutcome']
#     input_text = prompt.format(temp_str)
# #     print(input_text)
# #     input_text = "Your goal is to predict whether a client will subscribe to a bank term deposit.The following is the relevant information of the data.\nage:26,job:technician,marital:married,education:secondary,default:no,balance:889,housing:yes,loan:no,contact:cellular,day:3,month:feb,duration:902,campaign:1,pdays:-1,previous:0,poutcome:unknown\nOnly 0 or 1 needs to be output"
#     if index%100==0:
#         print(index)
    
#     inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
#     outputs = model.generate(inputs.input_ids,do_sample=False, max_new_tokens=1,num_beams=2,early_stopping=True)

#     #Output a single token result (debug)
#     final_output = tokenizer.decode(outputs[0][-1], skip_special_tokens=True)
#     token_id_1 = tokenizer.convert_tokens_to_ids("1")
#     token_id_0 = tokenizer.convert_tokens_to_ids("0")
#     with torch.no_grad():
#         outputs = model(**inputs)
#         logits = outputs.logits[:, -1, [token_id_1, token_id_0]]
#         print(logits)
#         probs = torch.softmax(logits, dim=-1)  
#         #temperature
#         probs_05 = torch.softmax(logits/0.5, dim=-1)[0].tolist()[0]
#         probs_08 = torch.softmax(logits/0.8, dim=-1)[0].tolist()[0]
#         probs_12 = torch.softmax(logits/1.2, dim=-1)[0].tolist()[0]
#     prob_1, prob_0 = probs[0].tolist()
#     res_data.append([id_no, prob_1,probs_05,probs_08,probs_12])
#     print(f"id_no {id_no},token result: {final_output} |probability: 1={prob_1:.6f}, 1_05={probs_05:.6f},1_08={probs_08:.6f},1_12={probs_12:.6f}")



import pandas as pd
sub = pd.read_csv('/kaggle/input/submission-97338/submission_097338.csv')
sub_llm = sub[['id','y']]
sub_llm[0:10]


import pandas as pd
sub_ml = pd.read_csv('/kaggle/input/submission-97772-ml/submission_97772.csv')
sub_ml[0:10]


merged_df = pd.merge(sub_ml,sub_llm ,on='id', suffixes=('_A', '_B')) # suffixes用于区分来源[7](@ref)

# # 2. 计算新列
merged_df['y'] = 0.97 * merged_df['y_A'] + 0.03 * merged_df['y_B']
merged_df= merged_df[['id','y']]
merged_df[0:10]


merged_df.to_csv('submission.csv',index=False)


# import pandas as pd
# sub_final = pd.read_csv('/kaggle/input/submission-97768/submission_97768.csv')
# sub_final[0:10]


# sub_final.to_csv('/kaggle/working/submission.csv',index=False)




