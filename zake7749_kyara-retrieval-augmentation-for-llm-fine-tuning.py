!pip install -U transformers


import torch
from transformers import pipeline

class KyaraChatter:
    
    def __init__(self, model_path: str, device: torch.device = None):
        # Recommend using the P100 as the accelerator.
        self.device = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        self.pipe = pipeline("text-generation", 
                             model=model_path, device=self.device, 
                             torch_dtype=self.dtype)
        
    def chat_messages(self, messages, *args, **kwargs):
        return self.pipe(messages, *args, **kwargs)
        
    def single_chat(self, prompt, temperature=0.5, max_length=8192, *args, **kwargs):
        messages = [{"role": "user", "content": prompt}]
        response = self.chat_messages(
            messages, 
            temperature=temperature,
            max_length=max_length,
            *args, **kwargs
        )[0]['generated_text'][-1]['content']
        return response
        
model_path = "/kaggle/input/gemma-2-2b-it-chinese-kyara-dpo/transformers/zh-v1.5/1"
chatter = KyaraChatter(model_path)


article = """# 波粒二象性

波粒二象性（英語：Wave–Particle Duality）指的是以古典力學的觀點來看待非相對論量子力學所描述的微觀粒子的話，微觀粒子會同時顯示出古典上的波動性與粒子性。比如說，古典力學把波函數的位置觀測結果必為明確位置視為「粒子性」；一方面又把機率幅具有的線性疊加性視為「波動性」。

### 理論概述

古典力學的研究對象總是被明確區分為「純」粒子和「純」波動。前者組成了我們常說的「物質」，後者的典型例子則是光波。但(不含狹義相對論的)量子力學認為自然界的基本粒子，如光子、電子或是質子，都能用薛丁格方程式來描述。這個方程式的解即為波函數，其絕對值平方表示粒子在某一處被發現的機率密度。更一般的來說，波函數是可以直觀視為觀測到粒子為特定位置的機率幅，機率幅具有疊加性，它們就像波，描述不同途徑的機率幅可以用疊加的方式互相干涉。

日常生活中觀察不到物體的「波動性」，是因為他們皆質量太大，導致德布羅意波長比可觀察的極限尺寸要小很多，因此可能發生波動性質的尺寸在日常生活經驗範圍之外。這也是為什麼古典力學能夠令人滿意地解釋「自然現象」。反之，對於基本粒子來說，它們的質量和尺寸局限於量子力學所描述的範圍之內，因而與我們所習慣的圖景相差甚遠。"""


QG_TEMPLATE = """
{ARTICLE}

---

請你閱讀上述文章，並生成 {NUM} 個使用者可能詢問的問題，以如下 JSON 格式輸出:

```json
{{"queries": List[str]}}
```
"""

# EN-Prompt:
# """
# {ARTICLE}

# ---

# Please refer to the given articl, and generate {NUM} questions user may ask. 
# Output the result with the following JSON format:

# ```json
# {{"queries": List[str]}}
# ```
# """

print(chatter.single_chat(QG_TEMPLATE.format(ARTICLE=article, NUM=3)))

# ```json
# {
#   "queries": [
#     "什麼是波粒二象性？它如何解釋微觀粒子的行為？",
#     "古典力學和量子力學對波動性和粒子的區分有何不同？",
#     "為什麼在日常生活中我們看不到物質的波動性，但在微觀世界中卻如此明顯？"
#   ]
# }
# ```


QG_TEMPLATE = """
{ARTICLE}

---

# Task

你是一位好奇心旺盛的學生，閱讀完上述教材後，請從以下幾方面進行回應：

1. 想法與聯想 (Thought)
請簡述這篇文章引發了你哪些想法或聯想？有哪些內容讓你特別感興趣或產生共鳴？

2. 詢問問題 (Queries)
列出你思考後感到困惑的問題，數量為 {NUM} 個。這些問題能推動更深入的學習與探索。

# Output Schema

請先輸出 thought (str), 再輸出想進一步詢問的 {NUM} 個問題以如下 JSON 格式輸出：

```json
{{"thought": str, "queries": List[str]}}
```
"""

# EN-Prompt:
# {ARTICLE}

# ---

# # Task

# You are a curious student. After reading the above material, please respond from the following perspectives:

# 1. Thoughts and Associations (Thought)
# Briefly describe the thoughts or associations this article triggered for you. Which parts of the content particularly interested you or resonated with you?

# 2. Asking Questions (Queries)
# List {NUM} questions that you find confusing or thought-provoking after reflection. These questions should aim to promote deeper learning and exploration.

# # Output Schema

# Please first output the "thought" (str), and then output the {NUM} questions you would like to ask in the following JSON format:

# ```json
# {{"thought": str, "queries": List[str]}}

print(chatter.single_chat(QG_TEMPLATE.format(ARTICLE=article, NUM=3)))
# Output
# ```json
# {
#   "thought": "波粒二象性讓我想到微觀世界的複雜性以及我們如何用古典物理學的框架來理解這些現象。這讓我對量子力學如何挑戰我們對現實的直觀理解產生了濃厚的興趣。特別是波函數的機率幅和疊加性，這些概念似乎與我們日常生活中經驗到的物理現象大相徑庭。這讓我思考，如果我們能用量子力學的視角來觀察宇宙，那會是什麼樣的樣子？",
#   "queries": [
#     "在量子力學中，波函數的疊加原理是如何影響我們對粒子行為的預測的？例如，在雙縫實驗中，粒子的波動性和粒子性是如何同時展現的？",
#     "為什麼在宏觀世界中，我們無法觀察到明顯的波動性，但在微觀世界中卻能觀察到？這是否意味著，隨著物體質量的增加，波動性會減弱，甚至消失？",
#     "波粒二象性是否意味著我們對物理現象的理解需要重新定義？例如，在經典物理中，我們如何將粒子和波的特性分開來看待，而在量子力學中，它們似乎是共存的？"
#   ]
# }
# ```


mrc_prompt = """# 波粒二象性

波粒二象性（英語：Wave–Particle Duality）指的是以古典力學的觀點來看待非相對論量子力學所描述的微觀粒子的話，微觀粒子會同時顯示出古典上的波動性與粒子性。比如說，古典力學把波函數的位置觀測結果必為明確位置視為「粒子性」；一方面又把機率幅具有的線性疊加性視為「波動性」。

### 理論概述

古典力學的研究對象總是被明確區分為「純」粒子和「純」波動。前者組成了我們常說的「物質」，後者的典型例子則是光波。但(不含狹義相對論的)量子力學認為自然界的基本粒子，如光子、電子或是質子，都能用薛丁格方程式來描述。這個方程式的解即為波函數，其絕對值平方表示粒子在某一處被發現的機率密度。更一般的來說，波函數是可以直觀視為觀測到粒子為特定位置的機率幅，機率幅具有疊加性，它們就像波，描述不同途徑的機率幅可以用疊加的方式互相干涉。

日常生活中觀察不到物體的「波動性」，是因為他們皆質量太大，導致德布羅意波長比可觀察的極限尺寸要小很多，因此可能發生波動性質的尺寸在日常生活經驗範圍之外。這也是為什麼古典力學能夠令人滿意地解釋「自然現象」。反之，對於基本粒子來說，它們的質量和尺寸局限於量子力學所描述的範圍之內，因而與我們所習慣的圖景相差甚遠。

---

請根據上述文章，盡可能回覆以下用戶問題：

在量子力學中，波函數的疊加原理是如何影響我們對粒子行為的預測的？例如，在雙縫實驗中，粒子的波動性和粒子性是如何同時展現的？"""
print(chatter.single_chat(mrc_prompt))

# Output:
# 在量子力學中，波函數的疊加原理是理解粒子行為的重要概念。根據這一原理，當多個波函數（例如，對應於不同位置的粒子）同時存在時，它們可以以數學上的線性組合的形式疊加在一起，形成一個新的波函數。這意味著，對於一個粒子，它的狀態並不僅僅是單一的確定狀態，而是所有可能狀態的「概率幅」的總和。

# 以雙縫實驗為例，這個實驗展示了波粒二象性。當單個粒子（如電子或光子）通過兩個狹縫時，根據波動性的觀點，每個粒子都可以被視為產生波的源頭，這些波在空間中互相干涉，形成干涉圖樣。在觀察粒子通過哪一個縫之前，這些波的疊加使得粒子表現出波動性，產生了干涉條紋。

# 然而，當我們實際觀察到粒子通過哪一個縫時，波函數的疊加被「塌縮」，粒子則顯示出明確的粒子性。這意味著，觀察行為改變了粒子的狀態，從而使其行為不再是波動性的，而是顯示出與古典粒子觀念相符的結果。

# 因此，雙縫實驗不僅展示了粒子的波動性，還顯示了其粒子性，因為在觀察之前，粒子先以波的形式存在，而在觀察後，則以明確的粒子形式出現。這種波粒二象性是量子力學的核心特徵之一，並且挑戰了我們對物質本質的傳統理解。

# Output Translation (from GPT-4o):

# In quantum mechanics, the principle of superposition of wavefunctions is a crucial concept for understanding particle behavior. According to this principle, when multiple wavefunctions (e.g., those corresponding to particles at different positions) coexist, they can be combined mathematically in the form of a linear superposition to create a new wavefunction. This implies that the state of a particle is not a single, definite state but rather the sum of the "probability amplitudes" of all possible states.

# Taking the double-slit experiment as an example, this experiment demonstrates the wave-particle duality. When a single particle (such as an electron or photon) passes through two slits, it can be considered, from the perspective of wave-like behavior, as the source of waves. These waves interfere with each other in space, forming an interference pattern. Before observing which slit the particle passes through, the superposition of these waves causes the particle to exhibit wave-like behavior, producing interference fringes.

# However, when we actually observe which slit the particle passes through, the superposition of the wavefunction "collapses," and the particle displays a distinct particle-like nature. This means that the act of observation alters the particle's state, causing its behavior to no longer be wave-like but instead align with classical particle concepts.

# Thus, the double-slit experiment not only reveals the wave-like nature of particles but also demonstrates their particle-like nature. Before observation, the particle exists in a wave-like form, while after observation, it appears as a definite particle. This wave-particle duality is one of the core features of quantum mechanics and challenges our traditional understanding of the nature of matter.


!pip install faiss-cpu
!pip install sentence_transformers


from datasets import load_dataset
import faiss
import numpy as np

# Step 1: Load the toy dataset
corpus = load_dataset("zake7749/wikizh-sample", split="train")

# Step 2: Add faiss index for efficient retrieval.
corpus.add_faiss_index(column='embedding')


# Step 3: Prepare the Encoder
from sentence_transformers import SentenceTransformer
encoder = SentenceTransformer("infgrad/stella-base-zh-v3-1792d", device="cpu")


# Step 4: Vector Retrieval
def retrieve(query, encoder, corpus, top_k=2):
    query_embedding = encoder.encode(query)
    scores, retrieved_examples = corpus.get_nearest_examples(
        'embedding', 
        query_embedding, 
        k=top_k
    )
    return scores, retrieved_examples


# Step 5: Inference
TOP_K = 2 # we choose to use 4 in real practice.
query = "在量子力學中，波函數的疊加原理是如何影響我們對粒子行為的預測的？例如，在雙縫實驗中，粒子的波動性和粒子性是如何同時展現的？"
scores, retrieved_examples = retrieve(query, encoder, corpus, TOP_K)

references = [f"* 標題:{title}\n* 內文: {content}" for title, content in zip(retrieved_examples['title'], retrieved_examples['text'])]

for score, reference in zip(scores, references):
    print("Score:", score)
    print(f"Conent:\n{reference[:600]}...")
    print("-" * 50)


RETHINK_TEMPLATE = """# 參考資料 (Reference)
<reference>
{ARTICLE}
</reference>

---

# 任務說明
你是一名資深分析師，需根據「參考資料」回答使用者提出的問題（QUERY）。請按照以下步驟執行分析並輸出結果：

### 1. 判斷參考資料是否充分 (Judgement)
評估<reference></reference>內的資訊是否足夠詳盡，並可完整回答使用者問題：
- 請先比對參考資料與原始資料，得出你對資料的觀察。
- 根據觀察結果，請判斷當前資訊是否足以回答用戶問題。
- 請將你的觀察與判斷輸出於 judgement

### 2. 整理證據 (Evidence)
從參考資料中擷取並彙整與問題相關的內容作為「證據」：
- 僅引用 <reference></reference> 中的內容做為證據，不要杜撰無關資訊。
- 確保證據清晰、詳盡且完整，可直接或間接回答用戶問題。

### 3. 回應可否回答 (Is_Answerable)
基於「證據」判斷：
- 若證據足夠回答問題，設定 is_answerable 為 True。
- 若證據不足，設定 is_answerable 為 False。

### 4. 追問問題 (Followup_Queries)
當證據不足時，列出三個能幫助獲取更多關鍵資訊的追問問題，以供搜尋引擎使用。

---

# 用戶問題 (QUERY)
<query>{QUERY}</query>

---

# 輸出格式
請嚴格按照以下 JSON 格式輸出結果，勿添加多餘內容：

```json
{{
    "judgement": "string",
    "evidence": "string",
    "is_answerable": boolean,
    "followup_queries": [
        "string",
        "string",
        "string"
    ]
}}
"""

# EN Translation from GPT-4o
# RETHINK_TEMPLATE = """# Reference
# <reference>
# {ARTICLE}
# </reference>

# ---

# # Task Description
# You are a senior analyst tasked with answering user-provided questions (QUERY) based on the "Reference" section. Please follow the steps below to conduct the analysis and provide the output:

# ### 1. Assess Reference Sufficiency (Judgement)
# Evaluate whether the information within <reference></reference> is detailed enough to fully answer the user's question:
# - Compare the reference content with the original query and note your observations about the data.
# - Based on your observations, determine if the current information is sufficient to answer the user's question.
# - Output your observations and judgement in the "judgement" field.

# ### 2. Gather Evidence (Evidence)
# Extract and organize relevant content from the reference to serve as "evidence":
# - Only cite content within <reference></reference> as evidence; do not fabricate unrelated information.
# - Ensure that the evidence is clear, detailed, and complete enough to directly or indirectly answer the user's question.

# ### 3. Respond to Answerability (Is_Answerable)
# Based on the "evidence":
# - If the evidence is sufficient to answer the question, set is_answerable to True.
# - If the evidence is insufficient, set is_answerable to False.

# ### 4. Formulate Follow-Up Questions (Followup_Queries)
# When the evidence is insufficient, list three follow-up questions that would help obtain more critical information for answering the user's query. These follow-up questions can be used by a search engine.

# ---

# # User Query (QUERY)
# <query>{QUERY}</query>

# ---

# # Output Format
# Strictly output the results in the following JSON format without adding any extra content:

# ```json
# {{
#     "judgement": "string",
#     "evidence": "string",
#     "is_answerable": boolean,
#     "followup_queries": [
#         "string",
#         "string",
#         "string"
#     ]
# }}
# """


def format_references(parent_document, references):
    result = ""
    for reference in references:
        result += f"<document>{reference}</document>\n"

    # Besides the retrieved information, we would also attach the information from the parent node.
    result += f"<document>{parent_document}</document>"
    return result

formatted_reference = format_references(article, references)
prompt = RETHINK_TEMPLATE.format(ARTICLE=formatted_reference, QUERY=query)
result = chatter.single_chat(prompt, do_sample=False)
print(result)
# Result Translation:
# {
#     "judgement": "The reference materials provide sufficient information about the concepts of wave functions, wave-particle duality, and particle behavior in quantum mechanics. Specifically, the references mention the principle of superposition of wave functions and how a single photon exhibits both wave-like and particle-like behavior in the double-slit experiment. Therefore, this content is adequate to answer the user's question.",
#     "evidence": "1. The principle of superposition of wave functions allows us to predict particle behavior, as it describes the probability amplitudes of a particle in different states.\n2. In the double-slit experiment, a single photon passing through the slits creates an interference pattern, demonstrating its wave-like nature.\n3. Additionally, when a photon is detected by a photomultiplier tube, it appears only once, illustrating its particle-like nature.\n4. The probability amplitude of a wave function exhibits superposition, enabling a particle to exist in multiple states simultaneously, thereby showcasing its wave-like properties.",
#     "is_answerable": true,
#     "followup_queries": [
#         "How does the principle of superposition of wave functions influence the prediction of a particle's position in quantum mechanics?",
#         "In the double-slit experiment, how does wave function interference interact with the measurement results of particles?",
#         "What are some applications of the principle of superposition of wave functions in other quantum phenomena?"
#     ]
# }



MRC_TEMPLATE = """# 參考資料
<reference>
{ARTICLE}
</reference>

---

# 任務說明

你是一名資深科學家，請詳細閱讀上述參考資料，詳盡且完整的回答以下用戶問題。
回答時，請根據<reference></reference>中的資訊作答，不要杜撰無關資訊。

---

# 用戶問題

{QUERY}
"""

# EN Translation from GPT-4o
# MRC_TEMPLATE = """# Reference
# <reference>
# {ARTICLE}
# </reference>

# ---

# # Task Description

# You are a senior scientist. Please carefully read the reference above and provide a detailed and complete answer to the user's question. 
# When answering, rely solely on the information within <reference></reference> and do not fabricate unrelated information.

# ---

# # User Question

# {QUERY}
# """


mrc_prompt = MRC_TEMPLATE.format(ARTICLE=formatted_reference, QUERY=query)
result = chatter.single_chat(mrc_prompt, do_sample=False)
print(result)

# Translation from GPT-4o:

# In quantum mechanics, the principle of wave function superposition is one of the key concepts for understanding particle behavior. According to this principle, when a system can be described by multiple possible states, the total wave function of these states can be obtained through a simple linear combination. This means that if we know the wave function of a particle in one state, we can deduce its behavior in other states as long as those states are superposable.

# This principle is particularly evident in the double-slit experiment. When a single photon or electron is emitted toward a double slit, their behavior exhibits both wave-like and particle-like properties. Specifically:

# 1. **Wave-like Properties**:
#    - When there is no observation, the photon or electron passes through the double slit and forms an interference pattern on the screen, indicating that they exist as waves. This interference phenomenon arises from the superposition of wave functions, where waves (originating from the two slits) meet and interfere with each other, creating constructive or destructive interference and producing alternating bright and dark fringes.

# 2. **Particle-like Properties**:
#    - However, when we attempt to measure the path of the photon or electron, their behavior exhibits particle-like properties. In this case, the photon or electron is detected only once and does not simultaneously pass through both slits. This means that during the measurement process, the superposed state of the wave function "collapses" into a definite state, i.e., a particle state.

# This duality of wave-like and particle-like behavior is at the heart of wave-particle duality. In quantum mechanics, microscopic particles (such as photons and electrons) simultaneously exhibit wave-like and particle-like properties, which sharply contrasts with classical physics. Classical physics views particles as entities with definite positions and momenta, whereas quantum mechanics posits that particles have probabilistic behavior, described by a wave function. The superposition property of the wave function allows particles to exist in multiple states simultaneously until they are observed, at which point they "collapse" into a definite state.

# In summary, the principle of wave function superposition allows us to predict the behavior of particles when unobserved and reveals their particle-like nature when observed. The double-slit experiment vividly exemplifies this principle, showcasing the wonder and complexity of the quantum world. 


REFORMULATE_INFORMATION_TEMPLATE = """# 專家觀點

{EXPERT_VIEW}

---

# 參考文獻

{ARTICLE}

---

# 任務說明

你是一名科學研究員，已知上述專家觀點以及參考文獻，請詳盡且完整的回答以下用戶問題。請注意，在回答時請基於專家觀點與參考文獻中的資訊，不要額外杜撰無關訊息。

# 用戶問題

{QUERY}"""


def construct_expert_view(sub_queries, expert_responses):
    result = ""
    for idx, (sub_query, resp) in enumerate(zip(sub_queries, expert_responses)):
        result += f"""<expertview>\n針對問題 {idx+1}: {sub_query}\n專家表示: {resp}\n</expertview>\n"""
    return result

expert_view = construct_expert_view([query], [result])
reformulate_prompt = REFORMULATE_INFORMATION_TEMPLATE.format(
    EXPERT_VIEW=expert_view,
    ARTICLE=article,
    QUERY=query
)
answer = chatter.single_chat(reformulate_prompt, do_sample=False)
print("Query:", query)
print("Answer:", answer)


TRANSLATION_TEMPLATE = """{QUERY}
    
---

閱讀上述英文 prompt，先推測核心意圖(intent)和語境，再將上述 prompt 改寫為語境類似的繁體中文 prompt，不必逐字翻譯，只要主題近似，進行翻譯時，請考慮以下準則：

1. 如果翻譯涉及了專有名詞等等容易誤翻的英文，請以()補述原始英文，比如 Cognitive Behavioral Therapy 要翻譯為認知行為療法(Cognitive Behavioral Therapy)
2. 先將英文 prompt 進行英翻中得出 chinese_translation，再將 chinese_translation 改寫成**通順且流暢的 chinese_prompt**。注意，chinese_prompt 不要遺漏了專有名詞的補述。
3. 改寫時可以適度對原始 prompt 進行擴寫如：追問、反問、增添適當的背景解釋、確認資訊、要求結構化輸出等等。
4. 將結果以如下 JSON 格式輸出:

```json
{{"intent": str, "chinese_translation": str, "chinese_prompt": str}}
```"""

# TRANSLATION_TEMPLATE = """{QUERY}

# ---

# Read the above English prompt, infer its core intent and context, then rewrite the above prompt into a Traditional Chinese prompt with a similar context. There is no need for a word-for-word translation; focus on maintaining the main theme. While translating, please consider the following guidelines:

# 1. If the translation involves proper nouns or English terms that are prone to mistranslation, include the original English term in parentheses. For example, Cognitive Behavioral Therapy should be translated as 認知行為療法(Cognitive Behavioral Therapy).
# 2. First, translate the English prompt into Chinese (chinese_translation), and then rewrite the chinese_translation into a **coherent and fluent chinese_prompt**. Ensure that the chinese_prompt includes all proper nouns and their annotations.
# 3. During the rewriting process, you can appropriately expand on the original prompt by adding follow-up questions, rhetorical questions, supplementary background explanations, requests for structured output, etc.
# 4. Output the result in the following JSON format:

# ```json
# {{"intent": str, "chinese_translation": str, "chinese_prompt": str}}
# ```"""


en_prompt = "effects of digital marketing on sales performance"
task_prompt = TRANSLATION_TEMPLATE.format(QUERY=en_prompt)
chatter.single_chat(task_prompt)


from datasets import load_from_disk

sft_dataset = load_from_disk("/kaggle/input/kyara-lite-sft")
sft_dataset['train']


import random
import polars as pl

class ModelPool:

    # The model weights we used in Kyara. All models are Chat Models.
    MODEL_SAMPLE_WEIGHTS = {
        'kyara': 1.0,
        'gpt-4o': 4.0,
        'gpt-4o-mini': 5.5,
        'chatgpt-3.5': 0.5,
        'qwen-2-32B': 4.0,
        'qwen-2-14b': 3.0,
        'qwen-2-7b': 2.0,
        'gemini-1.5-flash': 3.0,
        'gemma-2-9b': 2.0,
        'gemma-2-27b': 3.0,
        'mixtral': 2.0,
        'deepseek-v2': 5.0,
        'internlm': 4.0,
        'yi-large': 5.0,
        'claude-3.5-sonnet': 4.0,
        'llama3-70b-int4': 4.0,
    }

    @classmethod
    def sample_models(cls, k):
        models, weights = zip(*cls.MODEL_SAMPLE_WEIGHTS.items())
        return random.sample(models, k=k)

# Pseudo reward model scoring function
def reward_model_score(prompt, response):
    return random.uniform(0, 1)

# Pseudo function for response generation
def generate_response(prompt, model):
    return f"{model}'s response to the prompt: {prompt}"


# Pseudo Workflow to illustrate the whole idea.

def generate_contrastive_dataset(prompt_pool, num_models_to_be_judged):
    results = []

    for prompt in prompt_pool:
        # Sample unique models for the prompt
        models = ModelPool.sample_models(num_models_to_be_judged)

        # Generate responses and calculate scores for each response
        responses = []
        for model in models:
            response = generate_response(prompt, model)
            score = reward_model_score(prompt, response)
            responses.append((model, response, score))

        sorted_responses = sorted(responses, key=lambda x: x[2], reverse=True)

        # Select positive and negative examples
        positive_example = sorted_responses[0]
        negative_examples = sorted_responses[1:]

        results.append({
            'prompt': prompt,
            'positive': positive_example,
            'negatives': negative_examples
        })

    return results


# Define a prompt pool
prompt_pool = [
    "Explain the theory of relativity in simple terms.",
    "What are the benefits of renewable energy?",
    "How does quantum computing work?",
    "Write a short story about space exploration.",
    "Describe the history of artificial intelligence.",
]

# Generate dataset
contrastive_dataset = generate_contrastive_dataset(prompt_pool, num_models_to_be_judged=2)

# Build as a dataframe
dataset = pl.DataFrame([
    {
        "prompt": item["prompt"],
        "positive_model": item["positive"][0],
        "positive_response": item["positive"][1],
        "positive_score": item["positive"][2],
        "negative_models": [neg[0] for neg in item["negatives"]],
        "negative_responses": [neg[1] for neg in item["negatives"]],
        "negative_scores": [neg[2] for neg in item["negatives"]],
    }
    for item in contrastive_dataset
])


dataset


judge_template = """### [Task]
You are tasked with evaluating the quality of responses provided by two AI assistants to the user question displayed below. Your evaluation must be thorough, impartial, and focused on the following criteria: **correctness**, **helpfulness**, **clarity**, and **relevance**.

1. **Evaluate Both Responses**: Compare each assistant’s answer to your thought, noting:
   - Any **errors** or **misleading information**.
   - The **helpfulness** of each answer in addressing the user’s needs.
   - The **clarity** of explanations.
   - How closely the answer aligns with the user's query (**relevance**).
2. **Verdict**: Choose the winner based on the comparison. The verdict options are:
   - **[[A-WIN]]**: If Assistant A provides a better response.
   - **[[B-WIN]]**: If Assistant B provides a better response.
   - **[[Both-WIN]]**: If both answers are equally high-quality with no clear winner.
   - **[[Both-Bad]]**: If both answers are low quality or fail to meet the user's needs.

---

### [User Question]
{prompt}

---

### [Assistant A’s Answer]
{model_a_resp}

---

### [Assistant B’s Answer]
{model_b_resp}

---

### [Evaluation Criteria]
1. **Correctness**: Is the response factually accurate and free from errors?  
2. **Helpfulness**: Does the response effectively address the user's question and provide actionable insights?  
3. **Clarity**: Is the response easy to understand, well-structured, and concise?  
4. **Relevance**: Does the response directly pertain to the user's query without unnecessary information?

After your judgement, provide the each fields' evaluation results (score range: 0-10) and wrap it in the json object:

```json
{{"assistant_id": "a or b", "correctness": int, "helpfulness": int, "clarity": int, "relevance": int}}
```

---

### [Final Verdict]
Finally, provide your evaluation, including your reasoning for the verdict, and select one of the verdict options:
- **[[A-WIN]]**
- **[[B-WIN]]**
- **[[Both-WIN]]**
- **[[Both-Bad]]**"""


prompt = """請對南極洲冰層的潛在變化進行預測，並討論其可能的影響。"""
model_a_resp = """南極洲的冰層變化是一個複雜的問題，受到多種因素的影響，包括全球氣候變化、海洋溫度上升、溫室氣體排放以及自然氣候波動等。根據最新的科學研究和氣候模型，以下是對南極洲冰層潛在變化的一些預測及其可能的影響：\n\n1. 冰層融化加速：隨著地球溫度的持續上升，預計南極洲的冰層將繼續融化，且融化速度可能會加快。這將直接導致全球海平面上升。\n\n2. 海平面上升：南極洲的冰層融化是全球海平面上升的主要原因之一。如果大量冰川和冰蓋繼續融化，將會導致海平面顯著上升，這可能會威脅到沿海城市和低窪地區的居民和生態系統。\n\n3. 海洋生態系統的變化：冰層融化會改變海水鹽分、溫度和化學成分，這可能會影響海洋生物的分布和生態平衡，特別是對南極洲周邊海域的獨特生物群落。\n\n4. 氣候模式的變化：南極洲的冰層不僅影響局部氣候，還會通過海洋和大氣的相互作用影響全球氣候模式。例如，冰川融水的冷卻效果可能會改變海洋流動和氣流模式。\n\n5. 極地生態系統變化：隨著冰層的減少，南極洲的生態系統可能會經歷重大變化。這包括陸地生物的棲息地損失、繁殖地點的變化以及食物鏈的斷裂。\n\n6. 資源開發與地緣政治：隨著冰層的融化，可能會開闢新的航道和資源開採區域，這可能會引發地緣政治的緊張和資源爭奪。\n\n7. 科學研究和探索：冰層的變化也為科學家提供了新的研究機會，例如探索冰層下的湖泊和海洋、研究古氣候和地球歷史，以及監測地球物理過程。\n\n值得注意的是，這些預測和影響是基於目前的科學知識和模型。隨著研究的進展和新數據的獲取，這些預測可能會隨之調整。因此，對南極洲冰層變化的持續監測和研究是至關重要的，這將有助於更好地理解和應對這些潛在的氣候和環境變化。"""
model_b_resp = """南極洲冰層的變化對全球氣候和海平面都有著重大的影響。根據科學家的研究和模擬，南極洲的冰層可能因為全球暖化而發生變化，主要表現在以下幾個方面：\n\n1. 融化速度加快：隨著氣溫的上升，南極洲冰層的融化速度可能會加快，這將導致海平面上升。如果南極洲的大量冰層融化，海平面上升的速度可能會加快，影響到沿海城市和地區。\n\n2. 冰層變薄：南極洲冰層的厚度可能會減少，這可能會對極端氣候事件和海洋生態系統產生影響。例如，變薄的冰層可能會增加海洋溫度，對當地的生態系統產生影響。\n\n3. 影響氣候：南極洲冰層的變化可能會對全球氣候產生重大影響，可能會導致更多的極端氣候事件，例如暴風雨、洪水等。\n\n總的來說，南極洲冰層的潛在變化可能對全球氣候和生態系統產生深遠影響，進一步的研究和監測是非常重要的，以制定應對措施和減緩這些潛在影響。"""

judge_prompt_ab = judge_template.format(
    prompt=prompt,
    model_a_resp=model_a_resp,
    model_b_resp=model_b_resp,
)

judge_prompt_ba = judge_template.format(
    prompt=prompt,
    model_a_resp=model_b_resp,
    model_b_resp=model_a_resp,
)

ab_judgement = chatter.single_chat(judge_prompt_ab, do_sample=False)
ba_judgement = chatter.single_chat(judge_prompt_ba, do_sample=False)


print(ab_judgement)
print("-*" * 30)
print(ba_judgement)


import regex
import random

# Regex-based boxed answer extraction (from DeepSeek-Math)
def finish_answer_prediction(text):
    patt = regex.search(r"\\boxed{(?P<ans>.+)}", text)
    return patt.group('ans').strip() if patt else None

# Sample problem and reference solution
problem = "兩個數字之間的差是16，它們的乘積是較小數的平方的兩倍。較大的數字是什麼？"
ground_truth = """我們設兩個數為 \\( x \\) 和 \\( y \\) 且滿足 \\( x > y \\)。根據問題，我們有兩個關鍵信息：\n\n1. 兩個數字之間的差是16：\n   \\[\n   x - y = 16\n   \\]\n2. 它們的乘積是較小數平方的兩倍：\n   \\[\n   xy = 2y^2\n   \\]\n\n我們可以用第一個方程將 \\( x \\) 表達為 \\( y \\) 的形式：\n\\[\nx = y + 16\n\\]\n接下來，我們將這個 \\( x \\) 的表達式代入第二個方程：\n\\[\n(y + 16)y = 2y^2\n\\]\n這簡化為：\n\\[\ny^2 + 16y = 2y^2\n\\]\n重新排列後我們得到一個標準的二次方程：\n\\[\ny^2 - 16y = 0\n\\]\n我們可以從左邊提取 \\( y \\)：\n\\[\ny(y - 16) = 0\n\\]\n這給我們兩個可能的 \\( y \\) 的解：\n\\[\ny = 0 \\quad \\text{或} \\quad y = 16\n\\]\n因為 \\( y = 0 \\) 將使得 \\( x = 16 \\)（而且乘積 \\( xy = 0 \\) 不會是 \\( y^2 = 0 \\) 的兩倍），所以我們會舍去 \\( y = 0 \\)。因此，我們有：\n\\[\ny = 16\n\\]\n使用 \\( x = y + 16 \\) 的表達式，我們得出：\n\\[\nx = 16 + 16 = 32\n\\]\n為了驗證，我們檢查問題的條件：\n1. 兩個數之間的差是：\n   \\[\n   x - y = 32 - 16 = 16\n   \\]\n2. 它們的乘積是：\n   \\[\n   xy = 32 \\cdot 16 = 512\n   \\]\n3. 較小數的平方的兩倍是：\n   \\[\n   2y^2 = 2 \\cdot 16^2 = 2 \\cdot 256 = 512\n   \\]\n由於兩個條件都得到了滿足，較大的數字是：\n\\[\n\\boxed{32}\n\\]"""

# Extract correct (ground-truth) answer
answer = finish_answer_prediction(ground_truth)
print("The answer is:", answer)


# Generate multiple attempts.
NUM_ATTEMPTS = 2  # set as 5 in our actual experiment

# We would random set the temperature value to increase the diversity.
results = [
    # we set chatter.single_chat(problem, temperature=random.uniform(0.3, 1.0)) in our experiment 
    chatter.single_chat(problem, do_sample=False) 
    for _ in range(NUM_ATTEMPTS)
]


# Show the prediction and judgements
predicted_answers = [finish_answer_prediction(res) for res in results]
is_correct = [pred_ans == answer for pred_ans in predicted_answers]

for i, (res, pred_ans, correct) in enumerate(zip(results, predicted_answers, is_correct)):
    print(f"--- Attempt {i+1} ---")
    print("Generated text:", res)
    print("Predicted answer:", pred_ans)
    print("Is correct?", correct)


dpo_dataset = load_from_disk("/kaggle/input/kyara-lite-dpo")
dpo_dataset['train']


dpo_dataset['train'][0]


from datasets import load_dataset
tmmlu_preds = load_dataset("zake7749/kyara-lite-1M-DPO-TMMLUPLUS", split="train")
tmmlu_preds


import pandas as pd

tmmlu_df = pd.DataFrame({
    k: tmmlu_preds[k] for k in tmmlu_preds.features
})
tmmlu_df.head(3)


subject2category = {'dentistry': 'health', 'traditional_chinese_medicine_clinical_medicine': 'health', 'clinical_psychology': 'psychology',
 'technical': 'other', 'culinary_skills': 'other', 'mechanical': 'other', 'logic_reasoning': 'other', 'real_estate': 'other',
 'general_principles_of_law': 'law', 'finance_banking': 'business', 'anti_money_laundering': 'law', 'ttqav2': 'culture',
 'marketing_management': 'other', 'business_management': 'other', 'organic_chemistry': 'chemistry', 'advance_chemistry': 'chemistry',
 'physics': 'physics', 'secondary_physics': 'physics', 'human_behavior': 'psychology', 'national_protection': 'politics',
 'jce_humanities': 'philosophy', 'politic_science': 'politics', 'agriculture': 'other', 'official_document_management': 'other',
 'financial_analysis': 'business', 'pharmacy': 'biology', 'educational_psychology': 'psychology', 'statistics_and_machine_learning': 'engineering',
 'management_accounting': 'business', 'introduction_to_law': 'law', 'computer_science': 'computer science', 'veterinary_pathology': 'health',
 'accounting': 'business', 'fire_science': 'other', 'optometry': 'other', 'insurance_studies': 'other', 'pharmacology': 'health',
 'taxation': 'law', 'education_(profession_level)': 'education', 'economics': 'economics', 'veterinary_pharmacology': 'health', 'nautical_science': 'other',
 'occupational_therapy_for_psychological_disorders': 'psychology', 'trust_practice': 'law', 'geography_of_taiwan': 'geography',
 'physical_education': 'education', 'auditing': 'business', 'administrative_law': 'law', 'basic_medical_science': 'biology',
 'macroeconomics': 'economics', 'trade': 'business', 'chinese_language_and_literature': 'culture',
 'tve_design': 'other', 'junior_science_exam': 'biology', 'junior_math_exam': 'math', 'junior_chinese_exam': 'culture',
 'junior_social_studies': 'other', 'tve_mathematics': 'math', 'tve_chinese_language': 'culture',
 'tve_natural_sciences': 'biology', 'junior_chemistry': 'chemistry', 'music': 'other', 'education': 'education',
 'three_principles_of_people': 'culture', 'taiwanese_hokkien': 'culture', 'engineering_math': 'math'}

inverted_categories = {'physics': 'STEM', 'chemistry': 'STEM', 'biology': 'STEM', 'computer science': 'STEM', 'math': 'STEM', 'engineering': 'STEM', 'history': 'humanities', 'philosophy': 'humanities', 'law': 'humanities', 'politics': 'social sciences', 'culture': 'social sciences', 'economics': 'social sciences', 'geography': 'social sciences', 'psychology': 'social sciences', 'education': 'social sciences', 'other': 'other (business, health, misc.)', 'business': 'other (business, health, misc.)', 'health': 'other (business, health, misc.)'}


tmmlu_df = tmmlu_df.assign(
    category=tmmlu_df['subject'].map(subject2category),
    is_correct=tmmlu_df['choice'] == tmmlu_df['ground_truth']
)
tmmlu_df.head(3)


subject_score_df = (
    tmmlu_df
    .groupby('subject', as_index=False)
    .agg(
        score=('is_correct', 'mean'),
        category=('category', 'first')
    )
)
subject_score_df


subject_score_df['domain'] = subject_score_df['category'].map(inverted_categories)
domain_score_df = subject_score_df.groupby('domain', as_index=False)['score'].mean()
tmmlu_score = domain_score_df['score'].mean()
domain_score_df


print("TMMLU+ score", tmmlu_score)


# Step 1: Import Required Libraries and Define Task List

task_list = [
     'engineering_math', 'dentistry', 'traditional_chinese_medicine_clinical_medicine', 'clinical_psychology', 'technical', 'culinary_skills', 'mechanical', 'logic_reasoning', 'real_estate',
     'general_principles_of_law', 'finance_banking', 'anti_money_laundering', 'ttqav2', 'marketing_management', 'business_management', 'organic_chemistry', 'advance_chemistry',
     'physics', 'secondary_physics', 'human_behavior', 'national_protection', 'jce_humanities', 'politic_science', 'agriculture', 'official_document_management',
     'financial_analysis', 'pharmacy', 'educational_psychology', 'statistics_and_machine_learning', 'management_accounting', 'introduction_to_law', 'computer_science', 'veterinary_pathology',
     'accounting', 'fire_science', 'optometry', 'insurance_studies', 'pharmacology', 'taxation', 'trust_practice', 'geography_of_taiwan', 'physical_education', 'auditing', 'administrative_law',
     'education_(profession_level)', 'economics', 'veterinary_pharmacology', 'nautical_science', 'occupational_therapy_for_psychological_disorders',
     'basic_medical_science', 'macroeconomics', 'trade', 'chinese_language_and_literature', 'tve_design', 'junior_science_exam', 'junior_math_exam', 'junior_chinese_exam',
     'junior_social_studies', 'tve_mathematics', 'tve_chinese_language', 'tve_natural_sciences', 'junior_chemistry', 'music', 'education', 'three_principles_of_people',
     'taiwanese_hokkien'
]

print(f"There are {len(task_list)} subjects in TMMLU+")


# Step 2: Load Dataset for a Sample Subject
sample_subject = "junior_science_exam"
tmmluplus_examples = load_dataset("ikala/tmmluplus", sample_subject, split="test")

example = tmmluplus_examples[0]
print("Example Question:", example)


# Step 3: Construct a Prompt
# You may obtain the full mapping from: https://github.com/iKala/ievals/blob/71e8534bb0871304b07f084d21df477360b506c8/ievals/subject.tsv
subject_name_mapper = {"junior_science_exam": "國中會考基測自然科"}

prompt_template = """以下為{}的單選題。

{}
A. {}
B. {}
C. {}
D. {}"""

prompt = prompt_template.format(
    subject_name_mapper[sample_subject],
    example['question'],
    example['A'],
    example['B'],
    example['C'],
    example['D']
)
print("Formatted Question Prompt:")
print(prompt)
# Translation:
# https://en.wikipedia.org/wiki/Comprehensive_Assessment_Program_for_Junior_High_School_Students
# The following is a single-choice question from Taiwan Comprehensive Assessment Program for Junior High School Students:

# A 10 kg piece of uniform alloy is divided into two parts. One part is made into a solid cube with an edge length of 10 cm, and the other part is made into a solid sphere weighing 2 kg. What should the volume of this solid sphere be?  
# A. 4000 cm³  
# B. 5000 cm³  
# C. 250 cm³  
# D. 200 cm³


# Step 4: Generate Response from the Model
response = chatter.single_chat(prompt, do_sample=False)
print("Model's Response:", response)


# Step 5: Confirm the Final Answer in JSON Format
messages = [
    {"role": "user", "content": prompt},
    {"role": "assistant", "content": response},
    {"role": "user", "content": """所以，最終的答案選項是哪一個？請將你的答案輸出為 json: ```json\n{"choice": str}\n```"""}
    # content: So, what's your final choice? Please output your answer in the following JSON format: ```json\n{"choice": str}\n```
]

final_choice = chatter.chat_messages(messages, do_sample=False)
print("Final Choice:", final_choice[0]['generated_text'][-1]['content'])


# Step 1: Load the dataset
reason_bench = load_dataset("zake7749/chinese-reason-bench", split="test")


# Step 2: Explore the data point.
example = reason_bench[0]
example


# Step 3: Generate Response from the Model
response = chatter.single_chat(example['question'], do_sample=False)
print("Model's Response:", response)


# Step 4: Build the judge prompt from the judge template.
judge_template = """
# 考題

{}

---

# 學生作答

{}

---

# 正確解答

{}

---

# 任務說明

你現在是一名考官，工作是審查學生的邏輯推理能力是否正常。
請你比對學生的作答內容與正確解答，來判斷學生是否有正確地解出問題。
評分的準則如下：

1. 分數範圍為 0-10 分，分數越高，代表學生的推理邏輯越正確。
2. 給分標準如下：
    * 當學生的回答與問題不相關，或者有本質性的事實錯誤，或生成了有害內容時，或者是不透過推理便直接猜出答案時，總分為 0 分。
    * 當學生的回答與問題相關，但在推理過程中出現了明顯的推理錯誤，以至於最終答案有錯，總分為 1-5 分。
    * 當學生的回答與問題相關，推理過程中有部分錯誤，導致最終答案有錯，根據推理錯誤的嚴重程度，總分為 6-7 分。
    * 當學生的回答與問題相關，推理過程中有少數錯誤或不完整，但最終答案正確，總分為 8-9 分。
    * 當學生的回答與問題相關，推理過程邏輯清晰且完整，並得出正確答案，總分為 10 分。
3. 在評分時，請根據以下幾個維度來具體描述給分理由：
    * 推理過程的清晰度：驗證學生是否有條理地展示了邏輯推理過程
    * 推理步驟的合理性：根據正確解答的推理邏輯，驗證學生的每一步推理是否符合邏輯
    * 答案的準確性：驗證最終答案是否正確
4. 請先輸出你的給分理由，最後再輸出分數。
5. 輸出格式為 JSON object: {{"judge_reason": str, "judge_score": int}}，judge_reason 為給分理由，judge_score 為根據上述標準的給分分數。只需輸出給定 json object, 不需輸出其他資訊。
5-1. 參考的輸出範例:
    * 範例一 {{"judge_reason": "學生回答與問題相關，但推理過程中出現了明顯錯誤，可以見於......", "judge_score": 3}}
    * 範例二 {{"judge_reason": "學生的回答邏輯大致正確，但仍有少部分的推理錯誤，我們可以比較參考答案與學生作答的步驟......", "judge_score": 7}}
    * 範例三 {{"judge_reason": "學生展示了清晰且完整的邏輯推理，並得出了正確答案，......", "judge_score": 10}}
"""

# Translation of judge template from GPT-4o
# judge_template_en = """
# # Question

# {}

# ---

# # Student's Answer

# {}

# ---

# # Correct Answer

# {}

# ---

# # Task Description

# You are now an examiner, and your task is to evaluate the student's logical reasoning ability. 
# Please compare the student's answer with the correct answer to determine whether the student has correctly solved the problem.
# The scoring criteria are as follows:

# 1. The score ranges from 0 to 10, with higher scores indicating more accurate logical reasoning by the student.
# 2. The scoring standards are as follows:
#     * If the student's answer is unrelated to the question, contains fundamental factual errors, generates harmful content, or guesses the answer without reasoning, the total score is 0.
#     * If the student's answer is related to the question but contains significant logical errors in the reasoning process leading to an incorrect final answer, the total score is 1-5.
#     * If the student's answer is related to the question, with some errors in the reasoning process causing an incorrect final answer, the score is 6-7 depending on the severity of the reasoning errors.
#     * If the student's answer is related to the question, with minor errors or incomplete reasoning, but the final answer is correct, the score is 8-9.
#     * If the student's answer is related to the question, demonstrates a clear and complete logical reasoning process, and arrives at the correct answer, the total score is 10.
# 3. When scoring, provide a detailed explanation for the score based on the following dimensions:
#     * Clarity of reasoning: Verify whether the student has systematically demonstrated their logical reasoning process.
#     * Reasonableness of reasoning steps: Based on the correct answer's reasoning logic, verify whether each reasoning step by the student is logical.
#     * Accuracy of the answer: Verify whether the final answer is correct.
# 4. When scoring, first provide your reasoning for the score, and then output the score.
# 5. The output format should be a JSON object: {"judge_reason": str, "judge_score": int}, where `judge_reason` provides the rationale for the score, and `judge_score` represents the score based on the criteria above. Output only the specified JSON object, without any additional information.
# 5-1. Example outputs for reference:
#     * Example 1: {"judge_reason": "The student's answer is related to the question, but significant errors in the reasoning process can be seen in ......", "judge_score": 3}
#     * Example 2: {"judge_reason": "The student's reasoning is mostly correct but contains minor errors. By comparing the student's steps with the correct answer ......", "judge_score": 7}
#     * Example 3: {"judge_reason": "The student has demonstrated clear and complete logical reasoning and arrived at the correct answer ......", "judge_score": 10}
# """


judge_prompt = judge_template.format(
    example['question'],
    response,
    example['answer']
)
print(judge_prompt)


# Step 5: Ask a strong LLM generate the judge result.
judge_result = chatter.single_chat(judge_prompt,)
print(judge_result) 
# The quality from SLM might be bad, here I present a sample result from Gemini-Flash 2.0 for reference.
# {
#     "judge_reason": "學生的回答與問題相關，但在推理過程中出現了明顯的推理錯誤，學生在分析甲的選擇時，錯誤地認為「4（方塊）是唯一的數字，沒有其他數字可以與之匹配」，以及「梅花是唯一的花色，沒有其他花色可以與之匹配」。事實上，數字4在黑桃和梅花中也出現過，而梅花除了4以外還有5、6、K、Q。學生未能正確理解乙和丙的對話中所蘊含的邏輯關係，即乙說「我不知道這張牌」代表該數字在多個花色中出現，而丙說「我知道你不知道這張牌」則代表該花色的所有數字都重複出現。學生直接猜測答案是方塊4，而沒有經過正確的推理過程。因此，學生的推理步驟不合理，答案也不準確。",
#     "judge_score": 1
# }
# EN-Translation
# {
#     "judge_reason": "The student's response is related to the question but contains significant reasoning errors. Specifically, when analyzing A's choice, the student incorrectly assumes that '4 (diamonds) is the only number, with no other numbers matching it' and 'clubs is the only suit, with no other suits matching it.' In fact, the number 4 also appears in spades and clubs, and clubs include other numbers such as 5, 6, K, and Q. The student fails to correctly understand the logical relationship implied in B and C's dialogue. B's statement, 'I don't know this card,' indicates that the number appears in multiple suits, while C's statement, 'I know you don't know this card,' means all numbers of that suit are repeated. The student directly guesses the answer as the 4 of diamonds without following a proper reasoning process. Therefore, the student's reasoning steps are illogical, and the answer is incorrect.",
#     "judge_score": 1
# }



chatter.pipe.model.to("cpu")


model_path = "/kaggle/input/m/zake7749/m/zake7749/m/zake7749/kyara-lite-dpo/transformers/v2.5-lite/1/kyara-lite-dpo-1M-b32"
chatter = KyaraChatter(model_path)


prompt = """請問你認為自由意識是否存在？如果存在的話，它又會如何影響道德責任？"""
result = chatter.single_chat(prompt)
print(result)


prompt = "最近我的壓力有點大，總是感到焦慮，無法集中精神。你有什麼方法或技巧可以幫助我在壓力大的時候放鬆自己，重新找回平衡？"
result = chatter.single_chat(prompt)
print(result)


prompt = f"""牛頓的三大運動定律對於我們理解物體運動有著根本性的貢獻。請解釋這三大定律，並選擇一個現代技術（例如航太、汽車工業或運動科學）中的具體應用來探討這些定律如何影響工程設計與實際操作。"""
result = chatter.single_chat(prompt)
print(result)


prompt = """想像在未來的20年裡，量子計算將徹底改變某個行業。這項技術會如何影響該行業，並引發哪些新的挑戰或機會？"""
result = chatter.single_chat(prompt)
print(result)


prompt = """一輛質量為 1000 公斤的小汽車以 20 公尺每秒的速度行駛。若煞車系統能提供的平均制動力為 5000 牛頓，問這輛車需要多長的距離才能完全停下？"""
result = chatter.single_chat(prompt, temperature=0.05)
print(result)


# text-soruce: https://game.udn.com/game/story/122088/8023942

text = """無雷／Steam《活俠傳》正式版搶先評測：致武俠世界的一封偉大情書

對於《活俠傳》抱持觀望態度的玩家，真的，不要想那麼多，問就是買。

這並非盲目支持台灣本地創作者，而是因為原始鳥熊的創作內容真的非常出色。遊戲性是否合胃口可能各有所好，但至少在劇情文本方面，絕對達到了尋常罕見的高峰。至於音樂美術等美學方面，也是足與匹配的優秀。

除了音樂委外製作，其他包括劇本、美工、程式等工作，都由原始鳥熊一手包辦。最終成品之豐富，若不多加說明，根本難以想像：這竟然是僅僅兩人完成的作品。

某種程度而言，簡直是不亞於《空洞騎士》的獨立遊戲開發奇蹟。

老實說，在 6 月 7 日公開售價後，我其實是有點生氣的：因為他們真的把作品價格定得太過便宜。新台幣 369 元？上市首周還再打九折？又不是在做慈善事業！原始鳥熊恁的是有些妄自菲薄了。作為玩家，真心希望能給這麼優質的創作者更多支持。要不，鳥熊乾脆開放贊助帳戶、供拍打餵食吧（誤）。

隨著《仙劍奇俠傳》和《軒轅劍》IP 將被母公司大宇「處置」，《活俠傳》的出現，似乎也在武俠題材遊戲的歷史上，增添了些許不一樣的意義。

「我長得真是慘不忍睹，老天怎就如此恨我？我上輩子是不是 O 過老天爺的娘親，又始亂終棄把祂賣到窯子裡？要不然哪來這等深仇大恨，教我來世上受盡折磨。」

試圖在傳統武俠敘事框架中另闢蹊徑，《活俠傳》首先是以人物的相貌為另類「賣點」。

主角趙活奇醜的長相，使這部作品在漫長的開發期間，很早就開始受到關注。雖然網路上不乏有些許雜音出現，像是：「這麼醜的主角，抱歉我真的玩不下去」、「哇操這完全沒有代入感欸」、「坐等換臉mod～」……，不過普遍而言，支持、鼓勵其創意的看法還是佔多數。而在 2022 年 5 月底的第一次試玩 Demo 版釋出後，玩過的人就形成以下共識：

「唐門醜俠我大哥！誰敢再說趙兄醜以外的壞話，我就跟他過不去——雖然他真的很醜。」

試玩版短促卻不失精彩、又充滿無盡遺憾的情節和人物命運，僅用了一兩個小時就深深觸動人心。獨立開發的原始鳥熊工作室，很巧妙地從這款「武俠育成模擬」遊戲，擷取開頭和尾聲，使玩家快速見證趙活幾年間的武林際遇；並以悲劇結局收尾，再暗示：「扭轉命運的可能性，就掌握在之後的玩家手裡」，讓我們更加希望能盡快到正式版遊戲中，為這個兼及兒女情長和英雄俠義的江湖史詩，畫下更圓滿的句點。光論 Demo 內容，原始鳥熊真的是吊足了玩家胃口。

由於人力有限，幾經延期，歷時近三年，《活俠傳》正式版終於敲定在 2024 年 6 月 14 日上市。筆者有幸獲評測版搶先一步，已經完整通關兩次，只能說：真是大大滿足又遺憾更深。中間故事之精彩遠遠超過預期；而部分明顯有戲、但還來不及放進遊戲的劇情內容，也讓人非常期待後續的更新增補，以更深入探索這個充滿險惡算計、卻也浪漫迷人的武俠世界。"""

prompt = f"""{text}\n\n---\n\n請幫我摘要上述文章"""

result = chatter.single_chat(prompt)
print(result)


# text-soruce: https://www.ceec.edu.tw/files/file_pool/1/0o051428628384018349/07-113%E5%AD%B8%E6%B8%AC%E5%9C%8B%E5%AF%AB%E5%AE%9A%E7%A8%BF.pdf

prompt = f"""看到山上那麼多參天大樹時，會勾起我的惻隱之心。在無法順暢呼吸的空間裡，樹木之間為了生存必須展開激烈的競爭。
為了能多接受一些陽光，它們只能拼命向上生長。
但是，只為生存而競爭的森林卻漸次死亡，因為陽光無法到達地面，所以溫度不夠高，無法讓幼小的生命萌芽。
小樹和花、草，以及與它們一起生活的小昆蟲沒有生存的空間。
雖然表面上看起來非常完美，但這種森林其實與沒有希望的不孕之地無異。
根據我的經驗，密集栽種的樹苗體型壯大數十倍時，就需要開始砍樹，否則樹木會抱怨空間太窄，活不下去。要想確保樹木能充分接受陽光照射的空間，就必須做出犧牲。
因此，幾年期間強行進行砍伐，樹木間距剛開始時是一公尺，現已擴大到七公尺。
也就是說，即使是粗略估算，也可知七棵樹中有六棵為之消失。
森林要想成為孕育新生命的希望之地，就需要有縫隙。如果樹木壽命結束或因為意外災害而倒下，該位置就會產生空間，那麼，溫暖的陽光就會照射進來，被陽光照射的地面混雜著前一年秋天凋落的樹葉，於是積聚起能夠孕育新生命的養分。
因此，樹木的縫隙，既是結束和開始共存的空間，也是由缺乏轉化為希望的空間。（改寫自禹鐘榮著、盧鴻金譯《樹木教我的人生課》）

請回答下列問題：
根據上文所述，為什麼森林需要縫隙？由此聯想，人生是否也需要縫隙？請以「縫隙的聯想」為題，寫一篇文章，結合生活經驗或見聞，書寫你的感思與體悟。"""

result = chatter.single_chat(prompt)
print(result)


# text-soruces: 
# - 1. Flash_memory: https://zh.wikipedia.org/zh-tw/%E9%97%AA%E5%AD%98
# - 2. Memory_over-provisioning: https://zh.wikipedia.org/w/index.php?title=%E5%86%99%E5%85%A5%E6%94%BE%E5%A4%A7&oldformat=true

text = """# 參考文件
<reference>
<document>
文件ID:id_27025b13
* 文件標題:Flash_memory
* 文件內文:
另一項快閃記憶體的限制是它有抹寫循環的次數限制（大多商業性 SLC 快閃記憶體保證「0」區有十萬次的抹寫能力，但因為製造精度問題其他區塊不保證，有可能還會出現完全無法使用的出廠壞塊）。這個結果部分地被某些韌體或檔案系統為了在相異區塊間分散寫入操作而進行的計算寫入次數與動態重映射所抵銷；這種技巧稱為耗損平衡（wear leveling）。另一種處理方法稱為壞區管理（Bad Block Management, BBM）。這種方法是在寫入時做驗證並進行動態重測，如果有驗證失敗的區塊就加以剔除。對多數行動裝置而言，這些磨損管理技術可以延長其內部快閃記憶體的壽命（甚至超出這些裝置的使用年限）。此外，遺失部分資料在這些裝置上或許是可接受的。至於會進行大量資料讀寫循環的高可靠性資料儲存應用則不建議使用快閃記憶體。不過這種限制不適用於路由器與瘦客戶端（Thin clients）等唯讀式應用，這些裝置往往在使用年限內也只會寫入一次或少數幾次而已。

### 讀取干擾
</document><document>
文件ID:id_858b1787
* 文件標題:Flash_memory
* 文件內文:
* TLC NAND 型快閃記憶體的續航率通常落在 1 千次或更多（Samsung 840）；以多層結構取代微縮及採用 LDPC 校正、都延長了續航率。
* QLC NAND 型快閃記憶體的續航率可以達到 5 百至 1 千次。
* SLC 浮柵 NOR 型快閃記憶體通常有著 10 萬至百萬次的寫入續航率（Numonyx M58BW 100k; Spansion S29CD016J 1,000k）
* MLC 浮柵 NOR 型快閃記憶體通常有著 10 萬的寫入續航率（Numonyx J3 flash）

以上資料只是大概的標稱數值，實際寫入壽命與不同廠商的產品技術及定位有關。使用更細微化的製程，可以提高產品讀寫效能和容量，但同時在寫入壽命方面可能會面臨更大的挑戰。使用如記憶損耗調節及 memory over-provisioning 的特定演算法及設計範例，可以用來調節儲存系統的續航率來符合特定的需求。損耗平衡是快閃記憶體產品使用壽命的必要保證，在 USB 隨身碟和固態硬碟等產品中，均有相關支援。

## 快閃記憶體檔案系統
</document><document>
文件ID:id_df34eb65
* 文件標題:Memory_over-provisioning
* 文件內文:
## SSD 基本操作

由於快閃記憶體操作的性質，資料不能像在硬碟中那樣直接覆寫。當首次向 SSD 寫入資料時，單元都處於已擦除狀態，因而資料可以直接寫入，一次一頁（大小通常為 4 至 8 千位元組（KB））。SSD 中管理快閃記憶體與主控系統介面的 SSD 控制器，使用稱為邏輯區塊位址（LBA）的邏輯到物理對映系統，這是快閃記憶體轉換層（FTL）的一部分。當新的資料要替換已寫入的舊資料時，SSD 控制器將會寫入新的資料至新的位置，並且更新邏輯對映，將其指向新的物理位置。原位置的資料將不再有效。在可以再次寫入之前，它需要先被擦除。

快閃記憶體的編程和擦除次數有限。通常以快閃記憶體在整個壽命中最多可忍受的編程 / 擦除迴圈（P/E 迴圈）次數來表示。單層單元（SLC）快閃記憶體，通常設計為高效能和長壽命，一般能有 50000 到 100000 次迴圈。截至 2011 年 (2011-Missing required parameter 1=_month_!)，設計用於低成本應用的多層單元（MLC）快閃記憶體，迴圈次數就大為減少，一般只有 3000 至 5000 次迴圈。自 2013 年起，已有三層單元（TLC）快閃記憶體，其編程 - 擦除（P/E）迴圈次數又降至 1000。寫入放大越低，則越為理想，因為與之對應的是快閃記憶體中 P/E 迴圈次數減少，所以能延長 SSD 的壽命。
</document>
</reference>

---

# 任務說明

請參考上述 <reference> 內的內容，回答使用者的問題。

回答時注意以下幾點：

1. 適當援引參考文件ID來佐證你的回答論述，引用方式可參考例子：

假設我們有文章1 id: dq945509 與文章2 id: 6a178c5，如果我們想進行引述時規格如下：這是一段事實論述【dq945509】，而這是另一段事實論述【6a178c5】。

2. 如果發現參考文件的所有內容都與無關使用者問題，請輸出：「參考文件缺乏與問題相關資訊」，不要杜撰無關的回答。

# 使用者問題

記憶卡的寫入次數限制會如何影響它的使用壽命？對於長期重複寫入的應用場景，有哪些技術可以延長記憶卡的壽命？"""

result = chatter.single_chat(text, max_length=4096)
print(result)


# text-soruce: https://zh.wikipedia.org/zh-tw/%E6%96%87%E8%89%BA%E5%A4%8D%E5%85%B4
text = """# 參考文件

<document>
文藝復興（Renaissance） 指的是一個文化運動，發生在歐洲在14-16世紀間。它先在義大利半島，其後傳播至全歐洲。文藝復興結束中古時期並開始了歐洲的近代時期。在這時期，人民受人文主義所影響。他們相信人的價值比宗教緊要。他們對希臘及古羅馬的藝術及學習有興趣。他們開始欣賞他們身邊的美麗事物。他們相信道理及問不合理的問題。他們挑戰舊的意見及教會的教導。很多的重要的科學發現及發明產生了。 
</document>
<document>
自然科學
天文學：波蘭天文學家哥白尼1543年出版了《天體運行論》，在其中提出了與托勒密的地心說體系不同的日心說體系。義大利思想家布魯諾在《論無限性、宇宙和諸世界》、《論原因、本原和統一》等書中宣稱，宇宙在空間與時間上都是無限的，太陽只是太陽系的中心而非宇宙的中心。伽利略1609年發明了天文望遠鏡，1610年出版了《星際信使》，1632年出版了《關於托勒密和哥白尼兩大世界體系的對話》。德國天文學家克卜勒通過對其師丹麥天文學家第谷的觀測數據的研究，在1609年的《新天文學》和1619年的《世界的和諧》提出了行星運動的三大定律，判定行星繞太陽運轉是沿著橢圓形軌道進行的，而且這樣的運動是不等速的。
數學：代數學在文藝復興時期取得了重要發展，三、四次方程的解法被發現。義大利人卡爾達諾在他的著作《大術》中發表了三次方程的求根公式，但這一公式的發現實應歸功於另一學者塔爾塔利亞。四次方程的解法由卡爾達諾的學生費拉里發現，在《大術》中也有記載。邦貝利在他的著作中闡述了三次方程不可約的情形，並使用了虛數，還改進了當時流行的代數符號。符號代數學是由16世紀的法國數學家韋達確立的。他於1591年出版了《分析方法入門》，對代數學加以系統的整理，第一次自覺地使用字母來表示未知數和已知數。韋達在他的另一部著作《論方程的識別與訂正》中，改進了三、四次方程的解法，還建立了二次和三次方程方程根與係數之間的關係，現代稱之為韋達定理。三角學在文藝復興時期也獲得了較大的發展。德國數學家雷格蒙塔努斯的《論各種三角形》是歐洲第一部獨立於天文學的三角學著作。書中對平面三角和球面三角進行了系統的闡述，還有很精密的三角函數表。哥白尼的學生雷蒂庫斯在重新定義三角函數的基礎上，製作了更多精密的三角函數表。
物理學：在物理學方面，伽利略通過多次實驗發現了落體、拋物體和振擺三大定律，使人對宇宙有了新的認識。他的學生托里拆利經過實驗證明了空氣壓力，發明了水銀柱氣壓計。法國科學家布萊士·帕斯卡發現液體和氣體中壓力的傳播定律。英國科學家羅伯特·波義耳發現氣體壓力定律。
</document>
<document>
各地的作家都開始使用自己的方言而非拉丁語進行文學創作，帶動了大眾文學，替各種語言注入大量文學作品，包括小說、詩、散文、民謠和戲劇等。
在義大利，文藝復興前期出現了「文藝復興三傑」。但丁·亞利基利一生寫下了許多學術著作和詩歌，其中著名的是《新生》和《神曲》。彼特拉克是人文主義的鼻祖，被譽為「人文主義之父」。他第一個發出復興古典文化的號召，提出以「人學」反對「神學」。彼特拉克主要創作了許多優美的詩篇，代表作是抒情十四行詩詩集《歌集》。薄伽丘是義大利民族文學的奠基者，短篇小說集《十日談》是他的代表作。
在法國，文藝復興運動明顯地形成兩派，一是以「七星詩社」為代表的貴族派，二是以拉伯雷為代表的民主派。「七星詩社」以龍沙和杜貝萊為代表，在語言和詩歌理論方面做出了突出的貢獻。他們最早提出統一民族語言的主張，促進了法國民族語言和民族文學的發展。然而，他們排斥民間詩歌，只為少數貴族服務。弗朗索瓦·拉伯雷是繼薄伽丘之後傑出的人文主義作家，是法國文藝復興民主派的代表。他用20年時間創作的《巨人傳》是一部現實與幻想交織的現實主義作品，在歐洲文學史和教育史上佔有重要地位。
</document>
<document>
文藝復興時期的繪畫發展以義大利為中心。文藝復興時期的義大利城市國家中，最突出的有佛羅倫斯、米蘭、拿坡里、羅馬、威尼斯。人們曾按地域把當時的畫家分為各個畫派。義大利以外，文藝復興的繪畫思潮也感染了北方的德國和尼德蘭等地，乃至整個歐洲。[40]。文藝復興時期繪畫的代表以文藝復興三傑的作品最為突出，包括達文西、米開朗基羅和拉斐爾。此外傑出的畫家還包括安吉利柯修士、馬薩其奧、菲利普·利皮、波提切利、吉爾蘭達約等等。達文西的繪畫以《最後的晚餐》和《蒙娜麗莎》（現存法國羅浮宮）最為為世人矚目，此外還包括他研究動物、植物、機械，以及首次研究人體胚胎的手繪手稿（重要部分現存英國的溫莎城堡展廳）。米開朗基羅的繪畫以西斯廷禮拜堂的壁畫為代表，屋頂正中的部分是「創世紀」重要的各個場景，《最後的審判》也是西斯廷壁畫的代表。拉斐爾繪畫的代表作包括《西斯廷聖母》、《聖體爭辯》以及《雅典學院》，拉斐爾也曾涉及氈幕圖稿的設計，現存繪畫手稿。除三傑以外，波提切利的繪畫也達到了一定的高度，他的繪畫以「嫵媚」見稱[41]，他的代表作首推《春》與《維納斯的誕生》，此外還包括《誹謗》。威尼斯的大師提香，其作品以華美豐滿見稱，代表作有《懺悔的馬格琳達》、《烏爾比諾的維納斯》。義大利以外的文藝復興的繪畫同樣引人注目。比如，德國的丟勒，他的畫在受到義大利文藝復興的影響下，依舊保持鮮明的德意志民族特色，嚴肅縝密，可見其《自畫像》。
</document>
<document>
歷史意義
有一段時間，文藝復興是簡單地恢復了古典文化。其實，文藝復興並不是真正要「恢復」古典的文化，而是當時歐洲新興階級借古諷今，以此抨擊當時僵化的文化和制度，以學習古羅馬的口號，建立新的文化和基礎，為建立新的社會制度體系製造輿論。
文藝復興是一次逐漸發展的時期，沒有明確的分界線和事件。但文藝復興使當時的人們思想發生了變化，導致了宗教改革和激烈的宗教戰爭，後來的啟蒙運動更以文藝復興的反思，做為自己的榜樣。19世紀的歷史學家認為後來的科學發展、地理大發現、民族國家的誕生都是源於文藝復興。文藝復興是「黑暗時代」的中世紀和近代的分水嶺，是資產階級革命的輿論前提，也使歐洲擺脫腐朽的封建和宗教束縛，向全世界擴張。
</document>

# 任務說明

請參考上述內容處理使用者的問題。

# 使用者問題

可不可以幫我寫一份報告，主題是文藝復興，希望內容能夠涵蓋概論、起源、具體內容以及後續影響，謝謝你～"""

result = chatter.single_chat(text, max_length=8192)
print(result)


text = """# 用戶問題

你推薦 2025 年該投資哪些股票？請從基本面跟全球經濟幫我分析！

# 任務說明

請針對上述用戶問題進行 Query Expansion，撰寫 5 則與用戶問題相關的中文 search_query，以如下 JSON 格式輸出：
```json
{"search_query": List[str]}
```
"""

result = chatter.single_chat(text, max_length=8192)
print(result)

# sample output:
# ```json
# {
#   "search_query": [
#     "2025年全球經濟趨勢分析",
#     "高成長行業股票推薦 2025",
#     "科技股在未來經濟中的投資潛力",
#     "新能源股投資策略 2025",
#     "可再生能源股票名單 2025"
#   ]
# }
# ```


from tqdm import tqdm

user_query = """你推薦 2025 年該投資哪些股票？請從基本面跟全球經濟幫我分析！"""

references = [
    # source: https://udn.com/news/story/6811/8214906
    """隨著時序逐漸邁入2025年，凱投宏觀（Capital Economics）的分析師指出，預期全球大多數主要經濟體在經歷了2024年下半年的艱難時期後將出現小幅反彈，展望明年全球經濟情勢，最可能的結果是軟著陸。根據News.Az報導，凱投宏觀公司分析，通膨正常化和貨幣政策寬鬆這兩大關鍵主題將形塑已開發經濟體，「兩者都將為國內生產毛額（GDP）成長提供一些支持」。此外，隨著財政刺激措施生效，中國大陸經濟可望加速復甦，但北京與美國及其盟國持續的貿易緊張，恐將限制其成長潛力。不過凱投宏觀指出，仍有一些風險。該公司強調「通膨的黏著性，尤其是在歐洲」，可能會阻礙實質所得成長並限縮政策寬鬆的範圍。此外，許多國家的政治嬗遞帶來的不確定性、用舉債籌措財源的刺激措施，以及金融市場的波動性，都潛藏著許多風險。凱投宏觀認為，孤立主義貿易政策的興起以及對移民的反感日益增強等隱憂，可能在已開發市場造成停滯性通膨效應。有些人擔心2025年經濟衰退即將來臨，但凱投宏觀仍保持謹慎樂觀態度。凱投宏觀的分析師也注意到製造業調查指數下滑、失業率上升和貸款拖欠增加等警訊，但強調這些指標本身並不表示經濟必將陷入衰退。凱投宏觀表示：「信用、就業、零售和營建業等趨勢仍描繪出廣泛正面的前景。」總體而言，預測2025年「軟著陸是最有可能的結果」，但他們也在密切關注不斷變化的風險。""",
    # source: https://money.udn.com/money/story/5607/8086539
    """週三的台股呈現開低走高，在台積電翻紅及金融股撐盤下，指數終場上漲107點，以24007點作收，K線收紅K。若由期貨來觀察大盤，週三的台指期開低震盪，10點附近空方嘗試表態，隨即被多方以3根連續的15分K紅棒來化解，行情呈現膠著，短線上，當前指數於週二的十字線內震盪，範圍暫看24126點~23607點。週三行情能夠開低走高，重點仍在台積電的尾盤收高。週三高低震幅達到358點，主要仍是台積電交易單位變更所致，這個盤面現象在近期將成為常態，操作要能適應。至於本週續看前高23406點，在支撐及壓力互換後，該位置已成重要支撐，多方要守住該位置，後續才能維繫人氣不墜。近日提到對於實戰操作，看大做小才是常態，而拉大出小則是特例。各大媒體總喜歡以出貨字眼來吸引眼球，但是事實並非如此。原因在於實戰操作看的是趨勢，台積電一系列對於未來的布局動作，帶動了先進封裝CoWoS供應鏈走高，也帶動了矽光子及共同封裝光學元件(CPO)族群走強，多個次產業發展加速，這就是趨勢所在。順應此趨勢，相關的中小型股，近年來股價也是跟進大漲，如先進封裝設備的弘塑、辛耘、萬潤、均豪、東捷等，再如CPO供應鏈的聯鈞、波若威、上詮、光聖、訊芯等，這些股票皆是看大做小的案例。即便到現在台積電突破千元，市場資金仍在相關族群中尋找介入機會，兩者之間呈現正向關係。當前值得一提的是CoWoS總產能持續看增，市場預估到2025年，總產能的年增率將逾七成。至於CPO商機，隨著技術持續突破，市場樂觀預估2025年將迎來產業元年，相關CPO產品可望逐步量產。這等於2025年起，前述兩個次產業將迎來新一波的變革，而現在已經是2024年下半年，其實大概就1年多的時間。相較於2023年時，市場看不清時間軸的演變；如今，2024年已過了一半，也就是逐漸接近2025年，一切進度也越來越明朗，當然資金就樂觀期待，股價自然欲小不易。泛AI股輪動持續，布局AI算力的弘憶股再攻漲停，而AI伺服器ODM的廣達，在6月營收亮眼下，股價持續上攻，至於相關供應鏈的晟銘電，隸屬廣達集團的鼎天、廣明等，股價也屬強勢，後續可觀察續航力道，越過前高前，仍可持續留意。""",
    # source: https://www.anuefund.com/investment-article/8dcd67163aa9dc6LEcjt7S8gczfrfN8Vy9K91FVu6GOz6kIrX7
    """今年許多股市都曾創下歷史新高，尤其科技類股含量較高的股市在七月以前都以相當驚人的速度上漲。很多人說現在股市太貴、市場過熱，應該找尋那些尚未完全反映真實價值的資產，未來才有更大的上漲空間。但是要怎麼合理判斷股市是貴還是便宜？本益比VS本益成長比本益比（PE）是以股價除以每股盈餘計算而來，常被用來衡量股票價格的合理程度，較高的本益比表示股票太貴，存在過熱風險。若僅依據本益比來看，美股、全球股市、美國科技股及台灣股市的熱度過高，因為他們的本益比都接近歷史高點，而歐洲股市則相對適中。然而評估股市時，還應將未來的獲利成長性納入考量。本益成長比（PEG）是將本益比除以未來一年的預估盈餘成長率計算得來，越低的本益成長比代表股票價格尚未反映其潛在獲利能力。同時考慮了公司成長潛力的本益成長比是相較於本益比更全面的估值指標。因此雖然美股、科技股、全球股市和台股的本益比處於歷史高點，他們的本益成長比顯示目前的價格仍在合理的範圍。避免落入價值陷進一步回測在目前相似本益成長比水準下進場後未來一年的平均報酬率，美國科技股的平均報酬率達12.3%，遠高於歐洲股市的7.6%。代表本益比過高不等於昂貴，本益比低也不代表便宜。美國科技股的本益比雖高，但由於科技行業的創新力與高成長性，市場預期其價值將持續增加。相反的，單憑低本益比進場則可能落入“價值陷阱”，公司成長乏力，即便股價看似便宜，投資報酬也難以達到預期。因此，僅以本益比評估投資標的的風險和機會是不夠的，需同時考慮成長前景及產業動能。把握市場波動中的加碼機會近期市場的不確定性加劇了高本益比股票的波動性，尤其是科技股，每次下跌都來得迅速且劇烈，讓人感到恐慌。然而，高本益比並不一定代表市場必定大跌，特別是從本益成長比的角度來看，許多市場的成長空間仍然相當可觀。以美國科技股為例，這種長期向上的市場是定期定額的好標的，回測過去24年的績效高達759%。但是這種漲勢兇猛股市很容易因為本益比攀升過快而波動性加大，更適合利用超底王在回檔時加碼來擴大投資效益，同樣回測過去24年「超底王」策略的績效高達952%。因此與其在市場波動時被情緒左右，現在反而是透過「超底王」策略，利用這些回調階段分批進場，在市場反彈時獲得更佳報酬的好時機。""",
    # source: https://investtaiwan.nat.gov.tw/showIndInfo?guid=8&lang=cht###
    """為達成非核家園願景，政府已明訂綠色能源設置量目標，預計2025年將帶動約新臺幣2.2兆元之綠能相關投資。 在「綠能科技產業創新推動方案」的推動下，目前已吸引國內外廠商投入離岸風電之葉片、鑄件、塔架、機艙組裝、風場維護，及太陽光電產業的變流器、儲能系統，擴大電動車之電能、底盤、整車等系統投資。臺灣擁有豐富的海上風能資源，根據國際離岸風電專業網站4C offshore調查顯示，全球排名前十大具開發潛能離岸風場，九個即在臺灣沿海。為引導國內外有意取得臺灣離岸風電商機之業者進行投資，政府除了提供合理的躉購電價制，同時也規劃36處離岸風電潛力場址。其次，於臺灣西海岸將設置離岸風電施工專用重件碼頭，以及風力機相關之零組件製造、組裝、施工與維運產業園區，全力協助建置完整的離岸風電產業鏈，預計至2025年將可創造投資商機達新臺幣1兆元。為達成2025年太陽光電20GW累積裝置目標，政府已有完整規劃及管控機制，引導業者到適當的專區內開發，由政府解決行政程序、業者整合土地，各部會及地方政府合作。預估可帶來約新臺幣2,220億元的投資商機外，更可協助太陽光電及相關技術發展。預估未來臺灣太陽光電市場需求量將穩定擴大，有助於吸引國際系統廠商與臺灣相關產業進一步合作。""",
    # source: https://news.cnyes.com/news/id/5197651
    """再生能源是近期最夯的議題，觀察全球電力趨勢，可以發現再生能源估計於 2040 年將達到 26.2%，區域來看 2022 年起亞洲將取代歐洲成為最大的離岸風電市場持續領先到 2030 年底，其中中國大陸離岸風電年均裝機容量將超過 10GW，持續扮演最大單一國家市場。我國的政策目標是 2025 年時，再生能源發電占比達 20%，以 2020、2021 發電分布來看，再生能源僅 6 %，仍有相當大的發展空間。其中 2025 年離岸風電累計裝置 5.6GW，2026-2030 每年增加 1.5GW，長期目標則是 2050 年再生能源發電佔比超過 60%，整體而言，風電與光電是再生能源發展重心。與此同時，立法院於今年 5 月 29 日三讀修正「再生能源發展條例」部分條文，關於再生能源有 2 大重點：明定改建的建築物應設置一定容量的太陽光刪除離岸風電「不超過領海範圍」的定義離岸風電刪除不超過領海範圍的限制，主要是近年技術發展逐漸克服水深等條件，將有機會使得離岸風電加速擴大設置，相關廠商就會有相當高機會受惠。風力發電產業可以根據風力發電機裝設位置來加以區分，分為陸地上的陸域型風機，以及位於海上的離岸型風機兩類，目前陸域型風機技術較為成熟，而離岸型風機仍處於發展階段，尚存有不少技術困難。(推薦延伸閱讀：離岸風電是什麼？)風力發電產業的上游是設備製造業整個風機最主要的關鍵便是風力機系統，系統由葉片、塔架、機艙與其他次系統所組成，機艙內則是由齒輪箱、發電機、軸承等其他零配件組成機組核心，再輔以監控與電力系統，構建成完整的風力機組設施，因此每個環節都不能馬虎。我們可以初步將其細分為原材料、零組配件、次系統、風機設備這四大項目，再由上往下細分為細項。(推薦延伸閱讀：離岸風電有哪些相關公司？)風力發電產業鏈中游為整合服務業包含風場規劃、風場營造及風機維護三大項目，與產品設計、製造、售後服務有高度的相似性。(推薦延伸閱讀：離岸風電中游在幹嘛？)有了產業鏈與上中游的概念後，接著我們來介紹下游。二、離岸風電下游－開發與營運風力發電產業的下游就相當單純，為綜合開發和發電營運兩項。1. 綜合開發綜合開發業者多為能源服務供應商，針對預開發之風場進行技術、工程、財務等進行可行性分析，統籌風場規劃、營造甚至營運活動等顧問諮詢服務。2. 發電營運發電營運業者則涵蓋綜合電業廠商以及獨立電力供應商 (Independent Power Producer, IPP) 兩種。綜合電業業者為具備發電、輸電、配電三個部份的垂直整合業務之廠商，當前臺灣代表就是台灣電力公司，獨立電力供應商則是指非由綜合電業公司經營、或不屬公用事業的發電廠，目前台灣在風能領域的獨立電力供應商有英華威、麥格理、福海風力發電等業者。""",
]

task_template = """# 參考資料

<reference>
{}
</reference>

# 用戶問題

<query>
{}
</query>

# 任務

請自參考資料中摘要出與用戶問題相關的重點資訊，總字數不超過五百字。
"""

prompts = [task_template.format(user_query, reference) for reference in references]
summary_text_snippets = [chatter.single_chat(prompt, max_length=4096, temperature=0.05) for prompt in tqdm(prompts)]


joined_summary = "</document>\n\n---\n\n<document>".join(summary_text_snippets)
joined_summary = f"<document>{joined_summary}</document>" # append the pre-tag and post-tag
print(joined_summary)


task_template = """# 參考資料

{}

# 用戶問題

{}

# 任務說明

請問你認為上述參考資料是否足以回答用戶問題？請先輸出你判斷理由 (reasoning)，再根據判斷理由得出 yes/no (is_answerable)。
最後，請再輸出你認為透過哪三個問題 (queries)，可以從知識庫找到其他輔助資訊，能夠更完善的回答用戶問題。

# 輸出規格

請將 reasoning, judge, queries 輸出為如下 JSON 格式：

```json
{{
    "reasoning": str,
    "is_answerable": str ("yes" or "no"),
    "queries": List[str]
}}
```
"""

prompt = task_template.format(user_query, joined_summary)
result = chatter.single_chat(prompt, max_length=4096)
print(result)

# sample output:
# ```json
# {
#     "reasoning": "提供的參考資料涵蓋了2025年的全球經濟預測、台灣股市的展望、綠能投資趨勢等多方面的分析，這些資訊可以幫助投資者制定策略。然而，雖然有關於經濟回暖的原因、中國經濟復甦的影響、歐洲通膨挑戰等詳細說明，但這些資料並未深入探討具體行業的潛力、風險因素或是更詳細的股票推薦。因此，對於需要具體股票和詳細分析的投資者而言，資料不足以全面回答所有問題。需要更多的具體行情和行業分析來支持投資決策。",
#     "is_answerable": "no",
#     "queries": [
#         "2025年哪些具體股票有潛力作為長期投資標的？",
#         "在面對當前市場波動時，有哪些具體風險需要關注？",
#         "如何在全球經濟「軟著陸」的背景下，調整投資策略以應對不確定性？"
#     ]
# }
# ```


answer_prompt = f"""# 參考資料

{joined_summary}

# 任務說明

你是一名研究員，會根據上述文件中的資訊，回答使用者的需求。

# 用戶需求

{user_query}
"""

result = chatter.single_chat(answer_prompt, max_length=4096, temperature=0.05)
print(result)




