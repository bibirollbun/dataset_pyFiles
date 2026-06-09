import os
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


import numpy as np
import pandas as pd
import os

input_path = "/kaggle/input"

if os.path.exists(input_path):
    print("Input directory found.")
else:
    print("Input directory NOT found. Please upload your files.")
    
print("Environment check complete. Ready to download NCERT dataset.")


import os
import shutil

src_dir = "/kaggle/input/ncert-books-class-6-to-class-12"
dst_dir = "ncert_pdfs"

os.makedirs(dst_dir, exist_ok=True)

count = 0

print("Copying all PDFs into ncert_pdfs/ ...")

for root, _, files in os.walk(src_dir):
    for f in files:
        if f.lower().endswith(".pdf"):
            src_path = os.path.join(root, f)
            dst_path = os.path.join(dst_dir, f)
            shutil.copy(src_path, dst_path)
            count += 1

print(f"Copied {count} PDF files into ncert_pdfs/")


import shutil
import os

input_dir = "/kaggle/input/input-data"
files_to_copy = [
    "chunks_index.csv",
    "ncert_embeddings.npy",
]

folders_to_copy = [
    "ncert_chunks",
    "ncert_text",
]

# Copy files
for f in files_to_copy:
    src = os.path.join(input_dir, f)
    dst = os.path.join("/kaggle/working", f)
    if os.path.exists(src):
        shutil.copy(src, dst)
        print("Copied file:", f)

# Copy folders
for folder in folders_to_copy:
    src = os.path.join(input_dir, folder)
    dst = os.path.join("/kaggle/working", folder)
    if os.path.exists(src):
        shutil.copytree(src, dst, dirs_exist_ok=True)
        print("Copied folder:", folder)

print("\nAll required data restored to working directory!")


DATA_ROOT = "/kaggle/working"

CHUNKS_FOLDER = f"{DATA_ROOT}/ncert_chunks"
TEXT_FOLDER = f"{DATA_ROOT}/ncert_text"
PDF_FOLDER = f"{DATA_ROOT}/ncert_pdfs"

CHUNKS_INDEX_PATH = f"{DATA_ROOT}/chunks_index.csv"
EMBEDDINGS_PATH = f"{DATA_ROOT}/ncert_embeddings.npy"

print("Folders loaded:")
print(os.listdir(DATA_ROOT))


import pandas as pd
import numpy as np

chunks_index = pd.read_csv(CHUNKS_INDEX_PATH)
print("Index shape:", chunks_index.shape)
chunks_index.head()


embeddings = np.load(EMBEDDINGS_PATH)
print("Embeddings shape:", embeddings.shape)


import os

DATA_ROOT = "/kaggle/working"         
CHUNKS_FOLDER = os.path.join(DATA_ROOT, "ncert_chunks")

chunk_texts = []

for _, row in chunks_index.iterrows():
    rel_path = row["chunk_path"]    
    
    file_path = os.path.join(DATA_ROOT, rel_path)

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            txt = f.read()
    except FileNotFoundError:
        txt = ""
    chunk_texts.append(txt)

chunks_df = chunks_index.copy()
chunks_df["text"] = chunk_texts

print("Chunk DataFrame:", chunks_df.shape)
chunks_df.head()



from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

chunks_df["text"] = chunks_df["text"].fillna("")

tfidf_vectorizer = TfidfVectorizer(
    stop_words="english",
    max_df=0.85,
    min_df=2,
    ngram_range=(1, 2)
)

tfidf_matrix = tfidf_vectorizer.fit_transform(chunks_df["text"].tolist())
print("TF-IDF matrix shape:", tfidf_matrix.shape)


def retrieve_chunks(query, top_k=5):
    qv = tfidf_vectorizer.transform([query])
    sims = cosine_similarity(qv, tfidf_matrix)[0]
    top_idx = sims.argsort()[::-1][:top_k]
    
    results = chunks_df.iloc[top_idx].copy()
    results["score"] = sims[top_idx]
    return results

test_results = retrieve_chunks("photosynthesis class 10", top_k=3)
test_results[["chunk_id", "source_file", "score"]]


import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer as SentTfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class SummarizerAgent:
    def __init__(self, max_sentences=5, alpha=0.7):
        """
        max_sentences: number of sentences in summary
        alpha: tradeoff for MMR (importance vs diversity)
        """
        self.max_sentences = max_sentences
        self.alpha = alpha

        self.banned_substrings = [
            "Reprint", "MATHEMATICS", "Example", "TABLE", "Table ",
            "Chapter", "CHAPTER", "Fig.", "Figure", "EXERCISE"
        ]

    def _split_sentences(self, text):
        sentences = re.split(r'(?<=[.!?])\s+', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 0]
        return sentences

    def _is_noisy(self, s: str) -> bool:
        low = s.lower()

        digits = sum(c.isdigit() for c in s)
        if digits > len(s) * 0.4: 
            return True

        for bad in self.banned_substrings:
            if bad.lower() in low:
                return True

        if len(s) < 30 or len(s) > 300:
            return True

        return False

    def _filter_sentences(self, sentences):
        clean = []
        for s in sentences:
            if not self._is_noisy(s):
                clean.append(s)
        return clean

    def run(self, text, max_sentences=None, topic_keywords=None):
        """
        topic_keywords: optional list of lowercase words to slightly boost
                        sentences that mention them (e.g. ['photosynthesis','chlorophyll'])
        """
        max_sents = max_sentences or self.max_sentences

        sentences = self._split_sentences(text)
        sentences = self._filter_sentences(sentences)

        if not sentences:
            return ""
        if len(sentences) <= max_sents:
            return " ".join(sentences)

        vectorizer = SentTfidfVectorizer(stop_words="english")
        tfidf = vectorizer.fit_transform(sentences)
        base_scores = tfidf.sum(axis=1).A.ravel()

        if topic_keywords:
            topic_keywords = [k.lower() for k in topic_keywords if len(k) > 2]
            bonuses = []
            for s in sentences:
                low = s.lower()
                hit = any(k in low for k in topic_keywords)
                bonuses.append(0.5 if hit else 0.0)
            bonuses = np.array(bonuses)
            scores = base_scores + bonuses
        else:
            scores = base_scores

        sim_matrix = cosine_similarity(tfidf)

        selected_idx = []
        first = int(np.argmax(scores))
        selected_idx.append(first)

        while len(selected_idx) < max_sents and len(selected_idx) < len(sentences):
            best_idx = None
            best_mmr = -1e9

            for i in range(len(sentences)):
                if i in selected_idx:
                    continue
                relevance = scores[i]
                diversity = max(sim_matrix[i, j] for j in selected_idx)
                mmr = self.alpha * relevance - (1 - self.alpha) * diversity
                if mmr > best_mmr:
                    best_mmr = mmr
                    best_idx = i

            if best_idx is None:
                break
            selected_idx.append(best_idx)

        selected_idx = sorted(selected_idx)
        selected = [sentences[i] for i in selected_idx]
        return " ".join(selected)

summarizer_agent = SummarizerAgent(max_sentences=7, alpha=0.7)


class RetrieverAgent:
    def __init__(self, df, vectorizer, matrix):
        self.df = df
        self.vectorizer = vectorizer
        self.matrix = matrix

    def run(self, query, top_k=5):
        qv = self.vectorizer.transform([query])
        sims = cosine_similarity(qv, self.matrix)[0]
        top_idx = sims.argsort()[::-1][:top_k]

        results = self.df.iloc[top_idx].copy()
        results["score"] = sims[top_idx]
        return results

retriever_agent = RetrieverAgent(chunks_df, tfidf_vectorizer, tfidf_matrix)


def strict_topic_filter(df, topic_query):
    """
    Strictly filters chunks by:
    - subject (science/biology)
    - page keywords matching topic
    - removing maths/SST/etc. sections
    """
    topic_words = [w.lower() for w in topic_query.split() if len(w) > 2]

    science_words = [
        "science", "biology", "plant", "leaves", "cell", "photosynthesis",
        "chlorophyll", "sunlight", "carbon dioxide", "glucose", "class 10",
        "ncert", "chapter", "life", "organism", "plants", "higher plants"
    ]

    banned_subjects = [
        "math", "mathematics", "algebra", "geometry",
        "history", "civics", "geography",
        "english", "grammar", "literature"
    ]

    filtered_rows = []

    for _, row in df.iterrows():
        text = str(row["text"]).lower()
        fname = str(row["source_file"]).lower()
        path = str(row["chunk_path"]).lower()

        if any(bad in fname or bad in path for bad in banned_subjects):
            continue

        if any(bad in text for bad in ["frequency", "polygon", "table", "class marks", "data"]):
            continue

        if not any(w in text for w in science_words):
            continue

        if not any(t in text for t in topic_words):
            continue

        filtered_rows.append(row)

    return pd.DataFrame(filtered_rows)


def retrieve_clean(topic_query, top_k=5):
    """
    Full clean retriever: 
    - TF-IDF retrieval
    - Strict filter
    """
    raw = retriever_agent.run(topic_query, top_k=30)
    raw_df = raw.copy()

    clean_df = strict_topic_filter(raw_df, topic_query)

    if clean_df.empty:
        return raw_df.head(top_k)

    return clean_df.head(top_k)


import re
import random

class PatternQuizAgent:
    def __init__(self, num_questions=5, seed=42):
        self.num_questions = num_questions
        random.seed(seed)
        self.banned_tokens = {
            "it", "this", "that", "they", "them", "these", "those",
            "there", "here", "also", "some", "many", "which", "what",
            "when", "where", "who", "whom", "whose", "on", "in", "at",
            "if", "while", "because", "and", "or"
        }
        self.generic_distractors = {
            "hydrochloric acid", "sodium hydroxide", "carbon dioxide",
            "chlorophyll", "evaporation", "osmosis", "photosynthesis",
            "neutralisation", "distillation", "refraction"
        }

    def _candidate_sentences(self, text):
        sentences = re.split(r'(?<=[.!?])\s+', text)
        cleaned = []
        for s in sentences:
            s = s.strip()
            # skip questions and very short/long lines
            if "?" in s:
                continue
            if 40 <= len(s) <= 220:
                cleaned.append(s)
        return cleaned

    def _tokenize_subject(self, subject: str):
        return re.findall(r"[A-Za-z][A-Za-z\-]*", subject)

    def _is_good_subject(self, subject: str) -> bool:
        subj = subject.strip()
        if not subj:
            return False

        tokens = self._tokenize_subject(subj)
        if not tokens:
            return False

        if len(tokens) > 3:
            return False

        if not tokens[0][0].isupper():
            return False

        for t in tokens:
            if t.lower() in self.banned_tokens:
                return False

        if len(tokens[0]) <= 2 and len(tokens) == 1:
            return False

        return True

    def _front_is_pattern(self, sentence):
        """
        Look for 'X is ...' / 'X are ...', blank X if X looks like a concept.
        """
        s = re.sub(r'\s+', ' ', sentence.strip())
        m = re.match(r'(.{3,60}?)\s+(is|are)\b(.*)', s)
        if not m:
            return None, None

        subject = m.group(1).strip()
        rest = m.group(3).strip()
        if not rest:
            return None, None

        if not self._is_good_subject(subject):
            return None, None

        q_sentence = f"____ {m.group(2)} {rest}".strip()
        return subject, q_sentence

    def _make_question(self, sentence):
        ans, q = self._front_is_pattern(sentence)
        if ans and q:
            return ans, q
        return None, None

    def create_mcqs(self, text):
        sentences = self._candidate_sentences(text)
        q_pool = []

        for s in sentences:
            ans, q_sent = self._make_question(s)
            if ans and q_sent:
                q_pool.append((q_sent, ans))

        if not q_pool:
            return []

        random.shuffle(q_pool)
        q_pool = q_pool[:min(self.num_questions, len(q_pool))]

        all_answers = [a for _, a in q_pool]
        mcqs = []

        for i, (q_sent, ans) in enumerate(q_pool, start=1):
            distractors = [a for a in all_answers if a != ans]
            random.shuffle(distractors)
            distractors = distractors[:3]

            for d in self.generic_distractors:
                if len(distractors) >= 3:
                    break
                if d != ans and d not in distractors:
                    distractors.append(d)

            options = distractors + [ans]
            random.shuffle(options)
            correct_index = options.index(ans)

            mcqs.append({
                "id": i,
                "question": q_sent,
                "options": options,
                "correct_option_index": correct_index,
                "answer": ans
            })

        return mcqs

quiz_agent = PatternQuizAgent(num_questions=5)


topic_query = "acids bases and salts class 10"

retrieved = retriever_agent.run(topic_query, top_k=5)
combined_text = "\n\n".join(retrieved["text"].tolist())

mcqs = quiz_agent.create_mcqs(combined_text)
mcqs


class WeakAreaAgent:
    def __init__(self):
        self.stats = {}  

    def update(self, topic, correct):
        stat = self.stats.get(topic, {"correct": 0, "total": 0})
        stat["total"] += 1
        if correct:
            stat["correct"] += 1
        self.stats[topic] = stat

    def get_weak_topics(self, threshold=0.7, min_questions=2):
        weak = []
        for topic, s in self.stats.items():
            if s["total"] >= min_questions:
                acc = s["correct"] / s["total"]
                if acc < threshold:
                    weak.append((topic, acc))
        return sorted(weak, key=lambda x: x[1])  


class PlannerAgent:
    def run(self, weak_topics, days=7):
        if not weak_topics:
            return ["Review all studied topics for the next 7 days."]

        plan = []
        inv_acc = np.array([1 - acc for _, acc in weak_topics])
        weights = inv_acc / inv_acc.sum()
        topic_days = np.maximum(np.round(weights * days).astype(int), 1)

        day = 1
        for (topic, acc), d in zip(weak_topics, topic_days):
            for _ in range(d):
                if day > days:
                    break
                plan.append(f"Day {day}: Focus on '{topic}' (accuracy: {acc:.0%})")
                day += 1

        return plan[:days]

weak_agent = WeakAreaAgent()
planner_agent = PlannerAgent()


def run_study_session(query, top_k=5, topic_name=None):
    """
    query: what the student types (e.g. 'acids bases and salts class 10')
    topic_name: how we label this topic in the weak-area memory.
    """
    topic = topic_name or query

    print(f" Searching NCERT for: {query}\n")
    retrieved = retrieve_clean(topic_query, top_k=5)

    combined_text = "\n\n".join(retrieved["text"].tolist())

    print(" Summary:\n")
    summary = summarizer_agent.run(combined_text, max_sentences=7)
    print(summary)

    print("\n Quiz Time!\n")
    questions = quiz_agent.create_mcqs(combined_text)

    if not questions:
        print("Could not generate quiz questions for this topic. Try a different query.")
        return

    correct_count = 0

    for q in questions:
        print(f"Q{q['id']}. {q['question']}")
        for idx, opt in enumerate(q["options"]):
            print(f"  {idx+1}. {opt}")
            
            ans = 1  # always choose option 1
            print(f"Your answer (1-4): {ans}")

        if ans - 1 == q["correct_option_index"]:
            print(" Correct!\n")
            correct = True
            correct_count += 1
        else:
            correct = False
            correct_opt = q["options"][q["correct_option_index"]]
            print(f" Incorrect. Correct answer: {correct_opt}\n")

        weak_agent.update(topic, correct)

    print(f"Your score: {correct_count}/{len(questions)}")


run_study_session(
    query="acids bases and salts class 10",
    top_k=5,
    topic_name="Acids, Bases and Salts – Class 10"
)


def show_weak_topics_and_plan(days=7):
    weak = weak_agent.get_weak_topics()

    if not weak:
        print("No weak topics yet – complete at least one quiz session first!")
        return

    print(" Weak Topics (low accuracy first):\n")
    for topic, acc in weak:
        print(f"- {topic}: {acc*100:.1f}% accuracy")

    print("\n Suggested Study Plan:\n")
    plan = planner_agent.run(weak, days=days)
    for line in plan:
        print(line)


show_weak_topics_and_plan(days=5)


class MindMapAgent:
    def run(self, title, key_points):
        """
        key_points: list of strings (extracted main ideas)
        """
        tree = [f"[{title}]"]
        for point in key_points:
            tree.append(f" ├─ {point}")
        return "\n".join(tree)

mindmap_agent = MindMapAgent()


def extract_key_points(summary, max_points=5):
    sentences = re.split(r'(?<=[.!?])\s+', summary)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 0]
    return sentences[:max_points]


def final_demo(topic_query, topic_label=None):
    topic = topic_label or topic_query

    print("------------------------------------------")
    print(f"FINAL DEMO — Topic: {topic}")
    print("------------------------------------------\n")

    print("Retrieving relevant NCERT content...\n")
    retrieved = retrieve_clean(topic_query, top_k=5)
    combined_text = "\n\n".join(retrieved["text"].tolist())

    print("Summary:\n")
    topic_keywords = [w.lower() for w in topic_query.split()]
    summary = summarizer_agent.run(
        combined_text,
        max_sentences=7,
        topic_keywords=topic_keywords
    )
    print(summary)

    print("\nQuiz Time!\n")
    questions = quiz_agent.create_mcqs(combined_text)

    if not questions:
        print("Could not generate quiz questions for this topic. Try a different query.")
        return

    correct_count = 0

    for q in questions:
        print(f"Q{q['id']}. {q['question']}")
        for idx, opt in enumerate(q["options"]):
            print(f"  {idx+1}. {opt}")
            ans = 1  # always choose option 1
            print(f"Your answer (1-4): {ans}")

        if ans - 1 == q["correct_option_index"]:
            print("Correct!\n")
            weak_agent.update(topic, True)
            correct_count += 1
        else:
            correct_opt = q["options"][q["correct_option_index"]]
            print(f"Wrong. Correct answer: {correct_opt}\n")
            weak_agent.update(topic, False)

    print(f"Your Score: {correct_count}/{len(questions)}\n")

    print("Weak Topics:\n")
    weak = weak_agent.get_weak_topics()
    if weak:
        for t, acc in weak:
            print(f" - {t}: {acc*100:.1f}% accuracy")
    else:
        print("No weak topics yet!")

    print("\nStudy Plan:\n")
    plan = planner_agent.run(weak_agent.get_weak_topics(), days=7)
    for line in plan:
        print(line)

    print("\nMind Map:\n")
    key_points = extract_key_points(summary)
    mindmap = mindmap_agent.run(topic, key_points)
    print(mindmap)

    print("\n Demo complete!")


topic_query = "acids bases and salts class 10"
retrieved = retriever_agent.run(topic_query, top_k=5)
combined_text = "\n\n".join(retrieved["text"].tolist())

mcqs = quiz_agent.create_mcqs(combined_text)
mcqs


final_demo("photosynthesis", "Photosynthesis")

