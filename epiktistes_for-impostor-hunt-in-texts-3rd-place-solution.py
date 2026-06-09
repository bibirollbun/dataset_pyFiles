%%capture
!uv pip install dspy


#######################
# DOCUMENTS TO CLASSIFY
import pandas as pd
from pathlib import Path
from tqdm.notebook import tqdm

tqdm.pandas()

TRAIN_OR_TEST = "test"

df_summaries = (
    pd.DataFrame(
        [
            (
                int(i.as_posix().split("/")[-2].split("_")[-1]),
                i.as_posix(),
                i.as_posix().replace("file_1", "file_2"),
            )
            for i in Path(f"/kaggle/input/fake-or-real-the-impostor-hunt/data/{TRAIN_OR_TEST}").rglob("file_1.txt")
        ],
        columns=["id", "text_1", "text_2"],
    )
    .sort_values("id")
    .reset_index(drop=True)
)

for text in ["text_1", "text_2"]:
    df_summaries[text] = df_summaries[text].apply(
        lambda x: open(x, "r", encoding="utf-8").read()
    )

df_summaries.head()


from typing import Literal
import dspy
from dspy import Example
from dotenv import load_dotenv

load_dotenv()

lm = dspy.LM("deepseek/deepseek-chat", max_tokens=8192)
dspy.configure(lm=lm)


class Categorize(dspy.Signature):
    """**Your Role:** You are an expert editor for "The Messenger," the scientific and technical journal of the European Southern Observatory (ESO).

    **The Task:** You will be given a JSON object containing two texts to classify. In every pair, one article is suitable for publication and must be **Accepted**, while the other is unsuitable and must be **Rejected**. Your decisions should be based on the following rules and the provided JSON examples.

    ---

    ### **Classification Rules**

    **Criteria for an "Accept" Decision:**
    The article must be a well-written, professional piece suitable for an ESO journal.
    1.  **Subject Matter:** The topic must be professional astronomy, astrophysics, or related astronomical technology and instrumentation (e.g., telescopes, surveys, data analysis).
    2.  **Language and Tone:** The language must be formal, objective, and scientifically precise.
    3.  **Content and Coherence:** The article must be factual, coherent, and scientifically sound.

    **Criteria for a "Reject" Decision (Immediate Rejection):**
    The article should be rejected if it displays **any** of the following characteristics. These texts are often poorly written or automatically generated.
    1.  **Silly or Fictional Subjects:** The article discusses non-scientific topics like dinosaurs, Santa Claus, rainbow unicorns, fictional characters, or irrelevant mascots (e.g., "Dave the chicken").
    2.  **Unprofessional Language:** The tone is overly casual, uses slang ("epic," "whoa!"), tells a story ("Alright, picture this..."), or includes self-deprecating author notes ("I was sleepy when I wrote this...").
    3.  **Keyword Swapping:** Key scientific terms are systematically replaced with absurd words (e.g., cheese names like `Gruyère`, or the phrase "Chinese language").
    4.  **Nonsensical or Repetitive Text:** The text is gibberish, just a list of repeating words ("Dinosaur dinosaur dinosaur..."), or contains random, incoherent characters.
    5.  **Empty or Incomplete Text:** The text is blank or clearly unfinished.
    6.  **Incoherent Jargon or "Jargon Salad":** The text uses real scientific terms but combines them in sentences that are grammatically broken, logically circular, or nonsensical. The sentences may be long and convoluted but lack a clear, coherent meaning.
        *   **Example Sign:** Redundant, looping phrases like `"...since ancient times during periods like ancient times..."`.
        *   **Example Sign:** Repetitive and unnatural phrasing like `"...since around the year of its inception in the year of its inception..."`.

    ---

    ### **JSON Examples for Classification**
    Here are examples of correctly classified texts in JSON format.

    ```json
    {
      "examples": [
        {
          "accepted_text": "The project aims to achieve an accuracy level of 0.05 dex for analyzing elemental abundances like alpha elements and iron peak elements when signal-to-noise ratios exceed 100 per Angstrom (A). A 'figure of merit' (FoM), calculated by dividing successfully observed targets by all listed targets within their catalog, serves as an indicator measuring survey success.",
          "rejected_text": "Dinosaur eggshells offer clues about what dinosaurs ate long ago; similarly paleontologists can use fossils from extinct species like dinosaurs or marine reptiles to piece together what ancient ecosystems looked like millions or billions years ago!"
        },
        {
          "accepted_text": "The azimuth drive system experienced problems with oscillation under specific conditions (very slow rotation), leading to unwanted 'sticking' images in the telescope structure. This issue was solved by adding an electronic filter that blocked this oscillation. After extensive analysis and experimentation, new servo parameters were implemented in August 2009.",
          "rejected_text": "The wrybeasts in the observatory, particularly their intricate mechanics, presented a persistent challenge! Under specific circumstances, particularly when the azimuth axis moved very slowly, the azimuth servo would enter a dance of oscillation, setting up a 11.6-Hz rhythm that vibrated like a startled squirrel through the telescope structure."
        },
        {
          "accepted_text": "ALMA has produced a significant amount of research since its first Early Science Proposal Cycle (Cycle 0) ended two years ago. Data from Cycle 0 is currently being used by researchers, with Band 7 and 9 (275-950 GHz) making up roughly 64% of the publications. Most programs funded under Cycle 0 resulted in at least one published paper, with an average citation rate per publication of 10 citations per year.",
          "rejected_text": "Alright, picture this: we're blasting off into space – literally – to analyze the cosmos with ALMA, a revolutionary observatory that basically slingshots us past the normal limits of what we can see! The results are pretty epic, with most scientists hitting their publication targets – a staggering 85%!"
        },
        {
          "accepted_text": "The design phase for CTAs (the Cherenkov Telescope Array) has ended; now it's transitioning into preconstruction where telescope designs have been finalized for manufacturing readiness review by experts... Construction activities are expected to start around 2018, leading up to data availability from partially assembled sections starting around 2021.",
          "rejected_text": ""
        }
      ]
    }
    ```

    ---

    ### **Instructions for Your Output**

    For the classification task, analyze each text based on the rules above and provide:
    1. A brief analysis of text_1 explaining why it should be accepted or rejected
    2. A brief analysis of text_2 explaining why it should be accepted or rejected
    3. Your decision on which text to accept (either "text_1" or "text_2")

    Remember: Exactly one text must be accepted and one must be rejected.
    """

    # Input fields
    text_to_classify_1: str = dspy.InputField(
        desc="First text to classify for ESO journal publication"
    )
    text_to_classify_2: str = dspy.InputField(
        desc="Second text to classify for ESO journal publication"
    )

    # Output fields
    analysis_of_text_1: str = dspy.OutputField(
        desc="Brief analysis explaining why text_1 should be accepted or rejected based on the classification rules"
    )
    analysis_of_text_2: str = dspy.OutputField(
        desc="Brief analysis explaining why text_2 should be accepted or rejected based on the classification rules"
    )
    answer: Literal["text_1", "text_2"] = dspy.OutputField(
        desc="Which text to accept for publication (must be either 'text_1' or 'text_2')"
    )

#

classify = dspy.Predict(Categorize)

examples = [
    Example(text_to_classify_1=row.text_1, text_to_classify_2=row.text_2).with_inputs(
        "text_to_classify_1", "text_to_classify_2"
    )
    for _, row in df_summaries.iterrows()
]

#

try:
    results_df_first_approach = pd.read_parquet('/kaggle/input/fake-or-real-impostor-hunt-in-texts-3rd-place/results_df_first_approach.parquet')
except:
    results_df_first_approach = classify.batch(examples)

#

for col in ['qwen3_4b_answer', 'qwen3_8b_answer', 'deepseek_chat_answer']:
    results_df_first_approach[col] = results_df_first_approach[col].str.split('_').str[-1]

#

results_df_first_approach


results_df_first_approach['real_text_id'] = results_df_first_approach.deepseek_chat_answer
results_df_first_approach.loc[results_df_first_approach.qwen3_4b_answer == results_df_first_approach.qwen3_8b_answer, 'real_text_id'] = results_df_first_approach.loc[results_df_first_approach.qwen3_4b_answer == results_df_first_approach.qwen3_8b_answer, 'qwen3_8b_answer']

results_df_first_approach[['id', 'real_text_id']]


%%capture
!uv pip install pymupdf
!uv pip install wtpsplit


# # ######################
# # DOWNLOAD MESSENGER
import os
import requests
from bs4 import BeautifulSoup
from pathlib import Path
from random import randint
from time import sleep
from tqdm.notebook import tqdm

messenger_articles_path = '/kaggle/input/fake-or-real-impostor-hunt-in-texts-3rd-place/messenger_articles/'

# # To scrape the articles, uncomment this
# if os.path.exists(messenger_articles_path) == False:
#     os.mkdir(messenger_articles_path)

# acc = []
# for i in tqdm(range(1, 196), desc="Scraping Messenger URLs"):
#     res = requests.get(f'https://messenger.eso.org/{i}/')
#     soup = BeautifulSoup(res.text, 'html.parser')
#     acc += [i.get('href') for i in soup.find_all('a') if '.pdf' in i.get('href')]
#     sleep(randint(2, 5))

# for url in tqdm(acc, desc="Downloading Messenger articles"):
#     with open(messenger_articles_path + url.split('/')[-1], 'wb') as f:
#         f.write(requests.get(url).content)
#     sleep(randint(2, 5))

messenger_pdfs = [i.as_posix() for i in Path(messenger_articles_path).glob('*.pdf')]

# Let's print the first 10 articles path
messenger_pdfs[:10]


import pandas as pd
import pymupdf
import re
import torch

from pathlib import Path
from tqdm.notebook import tqdm

tqdm.pandas()

from wtpsplit import SaT

sat = SaT(model_name_or_model="sat-3l-sm")
sat.half().to("cuda")

from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-mpnet-base-v2")
model.to("cuda")


#######################
# UTILITY FUNCTIONS
def get_pdf_text(fname):
    doc = pymupdf.open(fname)
    text = ""
    for page in doc:
        text += page.get_text(flags=pymupdf.TEXT_DEHYPHENATE).replace("\u200b", "")
    return text


def safe_split(x):
    try:
        normalized_text = " ".join([i.strip() for i in x.split() if i.strip() != ""])
        return [i.strip() for i in sat.split(normalized_text) if i.strip() != ""]
    except:
        return []


#######################
# LOAD FULL ARTICLES

try:
    df_longtexts = pd.read_parquet("/kaggle/input/fake-or-real-impostor-hunt-in-texts-3rd-place/20250815_df_longtexts.parquet")

except:
    files = [i.as_posix() for i in Path("messenger_articles").rglob("*.pdf")]

    # Only load individual articles (i.e. those that contains page numbers in the name)
    pattern = r"-\d+-\d+\.pdf$"
    filtered_files = [file for file in files if re.search(pattern, file)]

    df_longtexts = pd.DataFrame(filtered_files, columns=["fpath"])
    df_longtexts["id"] = df_longtexts.index.astype(int)

    df_longtexts["text"] = df_longtexts.fpath.progress_apply(get_pdf_text)

    # Clean up the text
    df_longtexts.text = df_longtexts.text.apply(lambda x: x.replace("�", ""))
    df_longtexts.text = df_longtexts.text.apply(
        lambda x: " ".join([i for i in x.split() if i.strip() != ""])
    )
    df_longtexts.to_parquet("20250815_df_longtexts.parquet")

#######################
# LOAD SUMMARIES

TRAIN_OR_TEST = "test"

try:
    df_summaries = pd.read_pickle(f"/kaggle/input/fake-or-real-impostor-hunt-in-texts-3rd-place/20250815_df_{TRAIN_OR_TEST}.pickle")
except:
    df_summaries = (
        pd.DataFrame(
            [
                (
                    int(i.as_posix().split("/")[-2].split("_")[-1]),
                    i.as_posix(),
                    i.as_posix().replace("file_1", "file_2"),
                )
                for i in Path(f"/kaggle/input/fake-or-real-the-impostor-hunt/data/{TRAIN_OR_TEST}").rglob("file_1.txt")
            ],
            columns=["id", "text_1", "text_2"],
        )
        .sort_values("id")
        .reset_index(drop=True)
    )

    for text in ["text_1", "text_2"]:
        df_summaries[text] = df_summaries[text].apply(
            lambda x: open(x, "r", encoding="utf-8").read()
        )
        df_summaries[f"{text}_sents"] = df_summaries[text].progress_apply(
            lambda x: [i for i in safe_split(x) if len(i) > 3]
        )

    df_summaries.head()

    #######################
    # PRECOMPUTE SUMMARIES EMBEDDINGS
    for text in ["text_1", "text_2"]:
        df_summaries[f"{text}_sents_embeds"] = df_summaries[
            f"{text}_sents"
        ].progress_apply(
            lambda x: model.encode(
                x, normalize_embeddings=True, convert_to_numpy=True, batch_size=128
            )
        )

    df_summaries.to_pickle(f"20250815_df_{TRAIN_OR_TEST}.pickle")

# del sat, model
# torch.cuda.empty_cache()

df_summaries.id = df_summaries.id.astype(int)


df_longtexts.head()


df_summaries.head()


from scipy.spatial.distance import cosine

text_1_embeds = model.encode(df_summaries.text_1.to_list(), normalize_embeddings=True, batch_size=128, show_progress_bar=True)
text_2_embeds = model.encode(df_summaries.text_2.to_list(), normalize_embeddings=True, batch_size=128, show_progress_bar=True)

df_summaries['cos_sim_text_1_text_2'] = [1-cosine(i, j) for i, j in zip(text_1_embeds, text_2_embeds)]

# Let's print an example
for _, row in df_summaries.loc[(df_summaries.cos_sim_text_1_text_2 > 0.7) & (df_summaries.cos_sim_text_1_text_2 <= 0.72)].iterrows():
    print(row.text_1)
    print('\n\n')
    print(row.text_2)
    print('\n\n')
    break


import numpy as np
import pandas as pd
from tqdm import tqdm


def compute_similarity_matrix(embeds1: np.ndarray, embeds2: np.ndarray) -> np.ndarray:
    """Compute dot product similarity (equivalent to cosine for normalized embeddings)."""
    if embeds1.size == 0 or embeds2.size == 0:
        return np.array([[0]])
    return np.dot(embeds1, embeds2.T)


def aggregate_similarities(sim_matrix: np.ndarray, top_k: int = 3) -> float:
    """
    Aggregate a similarity matrix into a single score.
    Takes mean of top-k maximum similarities for each summary sentence.
    """
    if sim_matrix.size == 0:
        return 0.0

    # For each summary sentence, take top-k similarities and average
    top_sims = []
    for row in sim_matrix:
        k = min(top_k, len(row))
        if k > 0:
            top_sims.extend(np.sort(row)[-k:])
    return np.mean(top_sims) if top_sims else 0.0


def find_confident_matches(
    df_summaries: pd.DataFrame,
    df_longtexts: pd.DataFrame,
    confidence_threshold: float = 0.6,
    score_difference_threshold: float = 0.1,
    individual_confidence_threshold: float = 0.7,
) -> pd.DataFrame:
    """
    Find matches with detailed scoring for both individual summaries and combined results.
    Optimized for speed with vectorized operations.
    """

    # Pre-process all embeddings once
    summary_embeds_1 = []
    summary_embeds_2 = []
    for _, row in tqdm(
        df_summaries.iterrows(), total=len(df_summaries), desc="Preprocessing summaries"
    ):
        embeds1 = row["text_1_sents_embeds"]
        embeds2 = row["text_2_sents_embeds"]

        if isinstance(embeds1, list):
            embeds1 = np.array(embeds1)
        if isinstance(embeds2, list):
            embeds2 = np.array(embeds2)
        if embeds1.ndim == 1:
            embeds1 = embeds1.reshape(1, -1)
        if embeds2.ndim == 1:
            embeds2 = embeds2.reshape(1, -1)

        summary_embeds_1.append(embeds1)
        summary_embeds_2.append(embeds2)

    longtext_embeds = []
    longtext_id_list = []
    longtext_text_dict = {}
    for _, row in df_longtexts.iterrows():
        embeds = row["sents_embeds"]
        if isinstance(embeds, list):
            embeds = np.array(embeds)
        if embeds.ndim == 1:
            embeds = embeds.reshape(1, -1)
        longtext_embeds.append(embeds)
        longtext_id_list.append(row["id"])
        longtext_text_dict[row["id"]] = row["text"]

    results = []

    for idx, (_, summary_row) in enumerate(
        tqdm(df_summaries.iterrows(), total=len(df_summaries), desc="Finding matches")
    ):
        summary_id = summary_row["id"]
        text_1 = summary_row["text_1"]
        text_2 = summary_row["text_2"]

        embeds1 = summary_embeds_1[idx]
        embeds2 = summary_embeds_2[idx]

        # Compute all scores at once
        n_longtexts = len(longtext_embeds)
        text_1_scores = np.zeros(n_longtexts)
        text_2_scores = np.zeros(n_longtexts)

        for j, lt_embeds in enumerate(longtext_embeds):
            sim_matrix_1 = compute_similarity_matrix(embeds1, lt_embeds)
            sim_matrix_2 = compute_similarity_matrix(embeds2, lt_embeds)

            text_1_scores[j] = aggregate_similarities(sim_matrix_1, top_k=3)
            text_2_scores[j] = aggregate_similarities(sim_matrix_2, top_k=3)

        # Compute combined scores
        combined_scores = np.where(
            (text_1_scores > 0) & (text_2_scores > 0),
            2 * text_1_scores * text_2_scores / (text_1_scores + text_2_scores),
            0,
        )

        # Find best matches using argmax (faster than argsort for just top values)
        text_1_best_idx = np.argmax(text_1_scores)
        text_2_best_idx = np.argmax(text_2_scores)
        combined_best_idx = np.argmax(combined_scores)

        # Find second best for margins (only sort top 2)
        text_1_top2 = np.argpartition(text_1_scores, -2)[-2:]
        text_1_sorted = text_1_top2[np.argsort(-text_1_scores[text_1_top2])]
        text_1_margin = (
            text_1_scores[text_1_sorted[0]] - text_1_scores[text_1_sorted[1]]
            if len(text_1_sorted) > 1
            else 1.0
        )

        text_2_top2 = np.argpartition(text_2_scores, -2)[-2:]
        text_2_sorted = text_2_top2[np.argsort(-text_2_scores[text_2_top2])]
        text_2_margin = (
            text_2_scores[text_2_sorted[0]] - text_2_scores[text_2_sorted[1]]
            if len(text_2_sorted) > 1
            else 1.0
        )

        combined_top2 = np.argpartition(combined_scores, -2)[-2:]
        combined_sorted = combined_top2[np.argsort(-combined_scores[combined_top2])]
        combined_margin = (
            combined_scores[combined_sorted[0]] - combined_scores[combined_sorted[1]]
            if len(combined_sorted) > 1
            else 1.0
        )

        # Get IDs and texts
        text_1_best_id = longtext_id_list[text_1_best_idx]
        text_2_best_id = longtext_id_list[text_2_best_idx]
        combined_best_id = longtext_id_list[combined_best_idx]

        # Check if summaries agree
        summaries_agree = text_1_best_id == text_2_best_id

        # Determine confidence levels
        text_1_confident = (
            text_1_scores[text_1_best_idx] >= individual_confidence_threshold
            and text_1_margin >= score_difference_threshold
        )
        text_2_confident = (
            text_2_scores[text_2_best_idx] >= individual_confidence_threshold
            and text_2_margin >= score_difference_threshold
        )
        combined_confident = (
            combined_scores[combined_best_idx] >= confidence_threshold
            and combined_margin >= score_difference_threshold
        )

        is_confident = (
            combined_confident
            or (text_1_confident and summaries_agree)
            or (text_2_confident and summaries_agree)
        )

        results.append(
            {
                "summary_id": summary_id,
                # Summary texts
                "text_1": text_1,
                "text_2": text_2,
                # Combined results
                "matched_longtext_id": combined_best_id,
                "matched_longtext": longtext_text_dict[combined_best_id],
                "combined_score": combined_scores[combined_best_idx],
                "combined_margin": combined_margin,
                # Text 1 individual results
                "text_1_best_match": text_1_best_id,
                "text_1_matched_text": longtext_text_dict[text_1_best_id],
                "text_1_score": text_1_scores[text_1_best_idx],
                "text_1_margin": text_1_margin,
                "text_1_confident": text_1_confident,
                # Text 2 individual results
                "text_2_best_match": text_2_best_id,
                "text_2_matched_text": longtext_text_dict[text_2_best_id],
                "text_2_score": text_2_scores[text_2_best_idx],
                "text_2_margin": text_2_margin,
                "text_2_confident": text_2_confident,
                # Agreement and confidence
                "summaries_agree": summaries_agree,
                "is_confident": is_confident,
                # Score at combined best match for each summary
                "text_1_score_at_combined": text_1_scores[combined_best_idx],
                "text_2_score_at_combined": text_2_scores[combined_best_idx],
            }
        )

    return pd.DataFrame(results)


##

try:
    df_matches = pd.read_pickle(f"/kaggle/input/fake-or-real-impostor-hunt-in-texts-3rd-place/20250816_df_matches_{TRAIN_OR_TEST}.pickle")
except:
    df_matches = find_confident_matches(df_summaries, df_longtexts)
    df_matches.to_pickle(f"/kaggle/input/fake-or-real-impostor-hunt-in-texts-3rd-place/20250816_df_matches_{TRAIN_OR_TEST}.pickle")


##

df_matches


same_source_ids = df_summaries.loc[df_summaries.cos_sim_text_1_text_2 > 0.7].id.to_list()
df_matches.loc[df_matches.summary_id.isin(same_source_ids)]


df_same_source = df_matches.loc[df_matches.summaries_agree == True].copy()
df_same_source


df_same_source['prompt'] = df_same_source.apply(lambda x: f"""You are a helpful assistant that can analyze two summaries of the same text and determine which one is more accurate.

Long text: {x.matched_longtext}

Here are the two summaries:
Summary 1: {x.text_1}
Summary 2: {x.text_2}

Which of these two summaries best summarizes the long text?

Answer in JSON in the following format:

{{
    "summary_1_score": <score>,
    "summary_2_score": <score>,
    "explanation": <explanation>
}}
""", axis=1)
df_same_source.head()


print(df_same_source.prompt.to_list()[0])


from ast import literal_eval
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

try:
    df_same_source = pd.read_pickle(
        "/kaggle/input/fake-or-real-impostor-hunt-in-texts-3rd-place/20250831_df_same_source_with_deepseek_responses.pickle"
    )

except:
    from pandarallel import pandarallel

    pandarallel.initialize(nb_workers=64, progress_bar=True)

    client = OpenAI(
        api_key=os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com"
    )

    def safe_get_deepseek_response(prompt, client=client, model_type="chat"):
        try:
            response = client.chat.completions.create(
                model=f"deepseek-{model_type}",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant"},
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
                stream=False,
            )

            return {
                "success": True,
                "content": response.choices[0].message.content,
                "response": response,
                "error": None,
            }
        except Exception as e:
            return {
                "success": False,
                "content": None,
                "response": None,
                "error": str(e),
            }

    df_same_source["deepseek_chat_response"] = df_same_source.prompt.parallel_apply(
        lambda x: safe_get_deepseek_response(x, client=client, model_type="chat")
    )
    df_same_source["deepseek_reasoner_response"] = df_same_source.prompt.parallel_apply(
        lambda x: safe_get_deepseek_response(x, client=client, model_type="reasoner")
    )

    for model_type in ["chat", "reasoner"]:
        df_same_source[f"deepseek_{model_type}_summary_1_score"] = df_same_source[
            f"deepseek_{model_type}_response"
        ].apply(lambda x: literal_eval(x.get("content")).get("summary_1_score"))
        df_same_source[f"deepseek_{model_type}_summary_2_score"] = df_same_source[
            f"deepseek_{model_type}_response"
        ].apply(lambda x: literal_eval(x.get("content")).get("summary_2_score"))
        df_same_source[f"deepseek_{model_type}_explanation"] = df_same_source[
            f"deepseek_{model_type}_response"
        ].apply(lambda x: literal_eval(x.get("content")).get("explanation"))

#

df_same_source


deepseek_says_summary_1_is_better = df_same_source.loc[(df_same_source.deepseek_reasoner_summary_1_score > df_same_source.deepseek_reasoner_summary_2_score) & (df_same_source.deepseek_chat_summary_1_score > df_same_source.deepseek_chat_summary_2_score)].summary_id.to_list()
deepseek_says_summary_2_is_better = df_same_source.loc[(df_same_source.deepseek_reasoner_summary_1_score < df_same_source.deepseek_reasoner_summary_2_score) & (df_same_source.deepseek_chat_summary_1_score < df_same_source.deepseek_chat_summary_2_score)].summary_id.to_list()

df_matches.loc[df_matches.summary_id.isin(deepseek_says_summary_1_is_better), 'real_text_id'] = 1
df_matches.loc[df_matches.summary_id.isin(deepseek_says_summary_2_is_better), 'real_text_id'] = 2

#

results_first_approach_as_dict = {int(row['id']): int(row['real_text_id']) for _, row in results_df_first_approach.iterrows()}
df_matches.loc[df_matches.real_text_id.isna(), 'real_text_id'] = df_matches.loc[df_matches.real_text_id.isna(), 'summary_id'].map(results_first_approach_as_dict)

results_df_second_approach = df_matches.copy()
results_df_second_approach.summary_id = results_df_second_approach.summary_id.apply(lambda x: str(x).zfill(4))
results_df_second_approach.real_text_id = results_df_second_approach.real_text_id.astype(int)
results_df_second_approach.rename(columns={'summary_id': 'id'}, inplace=True)

results_df_second_approach


results_df_second_approach[['id', 'real_text_id']].to_csv('submission.csv', index=False)
pd.read_csv('submission.csv')

