import pandas as pd

# 请将下面的文件路径替换为您本地的实际路径
file_path = r'/kaggle/input/map-charting-student-math-misunderstandings/train.csv'

try:
    # --- 1. 加载数据 ---
    df = pd.read_csv(file_path)

    print("--- 表格宏观统计分析 ---\n")

    # --- 2. 整体信息分析 ---
    print("--- 1. 数据整体概览 ---")
    print(f"total line: {len(df)}")
    print(f"total column: {len(df.columns)}")
    print(f"all features' names: {df.columns.tolist()}\n")


    # --- 3. 非重复问题分析 ---
    print("--- 2. 非重复问题分析 ---")
    # 创建一个只包含非重复QuestionId和对应文本的DataFrame
    unique_questions = df[['QuestionId', 'QuestionText']].drop_duplicates()
    print(f"there are {len(unique_questions)} questions in dataset。\n")

    print("the questions' text:")
    for index, row in unique_questions.head(15).iterrows():
        print(f"  - QuestionId: {row['QuestionId']}")
        print(f"    QuestionText: {row['QuestionText']}\n")


    # --- 4. 分类 (Category) 分析 ---
    print("--- 3. 'Category' 列统计分析 ---")
    category_counts = df['Category'].value_counts()
    print("all 'Category' and their number:")
    print(category_counts)
    print("\n")


    # --- 5. 错误概念 (Misconception) 分析 ---
    print("--- 4. 'Misconception' 列统计分析 ---")
    # 统计非NA的错误概念
    misconception_counts = df[df['Misconception'] != 'NA']['Misconception'].value_counts()
    print(f"there are {len(misconception_counts)} misunderstandings in dataset")
    print("最常见的几种错误概念及其出现次数 (Top 10):")
    print(misconception_counts.head(10))


except FileNotFoundError:
    print(f"错误：无法在路径 '{file_path}' 找到文件。")
    print("请确认文件路径是否正确，并重新运行脚本。")
except Exception as e:
    print(f"处理文件时发生了一个错误: {e}")


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import textwrap
import warnings

# --- 1. 全局设置 ---
# 忽略所有 UserWarning，特别是来自matplotlib的字体警告
warnings.filterwarnings('ignore', category=UserWarning)
# 使用更适合科学绘图的样式
plt.style.use('seaborn-v0_8-notebook')

# --- 2. 辅助函数 ---
def format_latex_answer(answer_str):
    """将CSV中的类LaTeX字符串转换为matplotlib可渲染的标准LaTeX格式。"""
    if not isinstance(answer_str, str):
        return str(answer_str)
    # 例如, 将 '\( \frac{1}{3} \)' 转换为 '$\frac{1}{3}$'
    return answer_str.replace('\\( ', '$').replace(' \\)', '$')

# --- 3. 加载和预处理数据 ---
# 请将文件路径替换为您的实际路径
file_path = r'/kaggle/input/map-charting-student-math-misunderstandings/train.csv'

try:
    train_df = pd.read_csv(file_path)
    train_df['Misconception'] = train_df['Misconception'].fillna('NA')

    # 识别每个问题的最常见正确答案
    true_answers = train_df[train_df['Category'].str.startswith('True')].copy()
    correct_answer_counts = true_answers.groupby(['QuestionId', 'MC_Answer']).size().reset_index(name='count')
    idx = correct_answer_counts.groupby('QuestionId')['count'].idxmax()
    top_correct_answers = correct_answer_counts.loc[idx][['QuestionId', 'MC_Answer']]
    top_correct_answers.rename(columns={'MC_Answer': 'CorrectAnswer'}, inplace=True)
    train_df = pd.merge(train_df, top_correct_answers, on='QuestionId', how='left')

    # --- 4. 循环为每个问题生成可视化报告 ---
    unique_question_ids = train_df['QuestionId'].unique()

    print("--- Generating Professional Visual Reports for each Question ---")
    
    # 为了演示，我们只分析前5个问题，您可以移除 [:5] 来分析所有问题
    for q_id in unique_question_ids:
        
        # --- 数据准备 ---
        q_data = train_df[train_df['QuestionId'] == q_id].copy()
        question_text = q_data['QuestionText'].iloc[0]
        correct_answer = q_data['CorrectAnswer'].iloc[0]

        # --- 创建一个 1x2 的网格布局 ---
        fig, axes = plt.subplots(1, 2, figsize=(20, 8), gridspec_kw={'width_ratios': [1, 1.5]})
        fig.suptitle(f'Analysis Report for Question ID: {q_id}', fontsize=20, y=1.03)
        
        # --- (左侧图表) 答案选项统计与可视化 ---
        ax1 = axes[0]
        answer_counts = q_data['MC_Answer'].value_counts()
        total_responses = len(q_data)
        
        # 确定颜色和干扰项
        distractors = answer_counts.drop(correct_answer, errors='ignore')
        top_distractor = distractors.idxmax() if not distractors.empty else None
        
        palette_dict = {}
        for ans in answer_counts.index:
            if ans == correct_answer:
                palette_dict[ans] = 'mediumseagreen'
            elif ans == top_distractor:
                palette_dict[ans] = 'salmon'
            else:
                palette_dict[ans] = 'darkgrey'

        # *** 准备LaTeX格式的标签用于绘图 ***
        plot_labels_latex = [format_latex_answer(ans) for ans in answer_counts.index]
        
        bars = sns.barplot(x=plot_labels_latex, y=answer_counts.values, ax=ax1, 
                           palette=[palette_dict[k] for k in answer_counts.index], 
                           order=plot_labels_latex)
        
        # 添加百分比标签
        for bar in bars.patches:
            height = bar.get_height()
            percentage = 100 * height / total_responses
            ax1.text(bar.get_x() + bar.get_width() / 2., height,
                     f'{int(height)}\n({percentage:.1f}%)', ha="center", va="bottom", fontsize=11)

        ax1.set_ylim(0, answer_counts.max() * 1.15) # 调整Y轴上限为标签留出空间
        ax1.set_title('Answer Choice Distribution', fontsize=16, pad=20)
        ax1.set_xlabel('Multiple Choice Answer', fontsize=12)
        ax1.set_ylabel('Number of Students', fontsize=12)
        ax1.tick_params(axis='x', labelsize=16) # 增大标签字号以适应LaTeX

        # --- (右侧图表) 关键信息文本展示 ---
        ax2 = axes[1]
        ax2.axis('off')
        
        y_pos, x_pos, step = 1.0, 0.0, 0.14
        
        # 显示问题原文
        wrapped_question = textwrap.fill(f"Question: {question_text}", width=90)
        ax2.text(x_pos, y_pos, wrapped_question, ha='left', va='top', fontsize=13, weight='bold', wrap=True)
        y_pos -= (wrapped_question.count('\n') + 2) * 0.05
        
        # 显示经典解释
        ax2.text(x_pos, y_pos, "Key Student Explanations:", ha='left', va='top', fontsize=13, weight='bold')
        y_pos -= 0.08
        
        for answer in answer_counts.index:
            # *** 移除Emoji, 使用纯文本标签 ***
            label = "(Other Incorrect)"
            if answer == correct_answer: label = "(Correct Answer)"
            elif answer == top_distractor: label = "(Top Distractor)"
            
            explanation = q_data[q_data['MC_Answer'] == answer]['StudentExplanation'].iloc[0]
            wrapped_expl = textwrap.fill(f'“{explanation}”', width=85, subsequent_indent='   ')
            
            # *** 使用LaTeX格式渲染答案 ***
            answer_latex = format_latex_answer(answer)
            ax2.text(x_pos, y_pos, f"For Answer {answer_latex} {label}:", ha='left', va='top', fontsize=12, style='italic')
            y_pos -= 0.05
            ax2.text(x_pos, y_pos, wrapped_expl, ha='left', va='top', fontsize=11)
            y_pos -= step
            if y_pos < 0.3: break
            
        # 显示常见错误概念
        wrong_answers_df = q_data[q_data['MC_Answer'] != correct_answer]
        if not wrong_answers_df.empty:
            misconception_counts = wrong_answers_df[wrong_answers_df['Misconception'] != 'NA']['Misconception'].value_counts()
            if not misconception_counts.empty:
                y_pos = 0.20
                ax2.text(x_pos, y_pos, "Top Misconceptions for this Question:", ha='left', va='top', fontsize=13, weight='bold')
                y_pos -= 0.07
                for miscon, count in misconception_counts.head(2).items():
                    ax2.text(x_pos, y_pos, f"  - {miscon} (Observed {count} times)", ha='left', va='top', fontsize=12)
                    y_pos -= 0.05

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

except FileNotFoundError:
    print(f"Error: The file was not found at '{file_path}'. Please check the path.")
except Exception as e:
    print(f"An error occurred while processing the file: {e}")


!pip install \
    /kaggle/input/spacy3/en_core_web_md-3.0.0-py3-none-any.whl \
    --no-index \
    --no-dependencies

!pip install \
    /kaggle/input/spacy3/en_core_web_trf-3.0.0-py3-none-any.whl \
    --no-index \
    --no-dependencies

!pip install \
    /kaggle/input/spacy3/en_core_web_lg-3.0.0-py3-none-any.whl \
    --no-index \
    --no-dependencies

!pip install \
    /kaggle/input/spacy3/en_core_web_sm-3.0.0-py3-none-any.whl \
    --no-index \
    --no-dependencies

!pip install \
    /kaggle/input/spacy3/spacy_transformers-1.0.2-py2.py3-none-any.whl \
    --no-index \
    --no-dependencies



import sys
print(sys.version)

!pip install \
    /kaggle/input/spacy-alignments-0-9-2-cp311/spacy_alignments-0.9.2-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl \
    --no-index \
    --no-dependencies


!pip install \
    /kaggle/input/d/xx98x98/bitsandbytes/bitsandbytes-0.42.0-py3-none-any.whl \
    --no-index \
    --no-dependencies


!pip install \
    /kaggle/input/accelerate-wheelwhl/accelerate-0.29.1-py3-none-any.whl \
    --no-index \
    --no-dependencies


# ==============================================================================
# 单元格 2: 阶段一 - spaCy 特征工程 -> 输出 structured_features.json
# ==============================================================================
import pandas as pd
import json
from tqdm import tqdm
import os
import gc
import torch

# 确保导入了必要的spacy库 (假设已在单元格1安装)
try:
    import spacy
    import spacy_transformers
except ImportError as e:
    print(f"Import Error: {e}. Please ensure spaCy and spacy-transformers are correctly installed and you have restarted the kernel.")
    # 如果库不存在，后续代码会失败，所以在这里停止是合理的
    raise

def stage1_spacy_feature_extraction(input_csv_path, output_json_path):
    """
    完整的阶段一流程：加载数据，加载spaCy，提取特征，保存JSON，清理内存。
    """
    print("--- Stage 1: Running spaCy Feature Engineering ---")
    
    # 1. 加载原始数据
    print(f"Loading data from {input_csv_path}...")
    try:
        df = pd.read_csv(input_csv_path)
        df['Misconception'] = df['Misconception'].fillna('NA')
        # 为了快速调试，使用.head()。在最终运行时，请移除.head()或增加数量。
        df = df
        print(f"Successfully loaded {len(df)} rows.")
    except FileNotFoundError:
        print(f"ERROR: Input file not found at {input_csv_path}")
        return

    # 2. 加载spaCy Transformer模型 (一次性)
    spacy_model_name = "en_core_web_trf"
    print(f"Loading spaCy model: '{spacy_model_name}'...")
    try:
        nlp = spacy.load(spacy_model_name)
    except OSError:
        print(f"ERROR: Could not load spaCy model '{spacy_model_name}'.")
        print("Please ensure the model .whl was installed correctly in the first cell and you have restarted the kernel.")
        return
    print("spaCy model loaded successfully.")

    # 3. 定义特征提取函数
    def analyze_question(text):
        if not isinstance(text, str): text = ""
        doc = nlp(text)
        known_conditions, requirement, question_goal = [], [], None
        numbers = [tok.text for tok in doc if tok.like_num or tok.ent_type_ == "CARDINAL"]
        for sent in doc.sents:
            s_text = sent.text.strip()
            if not s_text: continue
            if "?" in s_text: question_goal = s_text
            elif sent.root.pos_ == "VERB" and sent.root.tag_ == "VB": requirement.append(s_text)
            else: known_conditions.append(s_text)
        return {"inputs": {"known_conditions": known_conditions, "numbers": list(dict.fromkeys(numbers))}, "requirement": requirement, "question_goal": question_goal}

    def analyze_student_answer(row):
        text, mc_answer = row['StudentExplanation'], row['MC_Answer']
        if not isinstance(text, str): text = ""
        doc = nlp(text)
        numbers = [tok.text for tok in doc if tok.like_num or tok.ent_type_ == "CARDINAL"]
        reasoning_steps = [s.text.strip() for s in doc.sents if s.text.strip()]
        return {"inputs": {"numbers": list(dict.fromkeys(numbers))}, "reasoning": {"steps": reasoning_steps}, "output": {"goal": mc_answer}}

    # 4. 应用特征提取
    tqdm.pandas(desc="Analyzing QuestionText")
    df['QuestionFeature'] = df['QuestionText'].progress_apply(analyze_question)
    tqdm.pandas(desc="Analyzing StudentExplanation")
    df['StudentAnswerFeature'] = df.progress_apply(analyze_student_answer, axis=1)

    # 5. 生成并保存JSON
    print("Converting DataFrame to JSON format...")
    json_output_list = []
    for index, row in tqdm(df.iterrows(), total=len(df), desc="Creating JSON records"):
        try:
            cat_parts = row['Category'].split('_', 1)
            answer_correctness, inference_correctness = cat_parts[0], cat_parts[1]
        except IndexError:
            answer_correctness, inference_correctness = row['Category'], None

        json_record = {
            "row_id": row['row_id'], # 添加row_id用于后续合并
            "QuestionId": row['QuestionId'],
            "QuestionText": row['QuestionText'],
            "QuestionFeature": row['QuestionFeature'],
            "StudentAnswer": {
                "MC_Answer": row['MC_Answer'],
                "StudentExplanation": row['StudentExplanation'],
                "StudentAnswerFeature": row['StudentAnswerFeature'],
                "Category": {
                    "Category_answer_correctness": answer_correctness,
                    "Category_inference_correctness": inference_correctness,
                    "Category_combination": row['Category']
                },
                "Misconception": {
                    "Category_combination": row['Category'],
                    "Misconception": row['Misconception']
                }
            }
        }
        json_output_list.append(json_record)

    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(json_output_list, f, indent=4, ensure_ascii=False)
        
    print(f"\nStage 1 complete. Structured features saved to '{output_json_path}'")
    
    # 6. 彻底清理内存
    del nlp, df, json_output_list
    gc.collect()

# --- 执行阶段一 ---
stage1_spacy_feature_extraction(
    input_csv_path='/kaggle/input/map-charting-student-math-misunderstandings/train.csv',
    output_json_path='/kaggle/working/structured_features.json' # 保存到可写的/kaggle/working/目录
)

