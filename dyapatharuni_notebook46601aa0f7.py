import argparse
import os
import json
import csv
import random
import textwrap
from typing import List, Tuple, Dict, Optional

# -- Dependencies: nltk, scikit-learn, pandas optional, genanki optional
try:
    import nltk
    from nltk.tokenize import sent_tokenize
except Exception:
    nltk = None

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
except Exception:
    TfidfVectorizer = None

# Optional: genanki for Anki deck export
try:
    import genanki
except Exception:
    genanki = None

# Optional: OpenAI for improved Q/A (user may set OPENAI_API_KEY)
try:
    import openai
except Exception:
    openai = None

# Utilities -------------------------------------------------------------------

def ensure_nltk():
    global nltk
    if nltk is None:
        raise RuntimeError("nltk is required for sentence tokenization. Install with: pip install nltk")
    # download punkt if missing
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        print("Downloading NLTK punkt tokenizer (one-time)...")
        nltk.download("punkt")

def read_input_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def simple_sentences(text: str) -> List[str]:
    ensure_nltk()
    sents = [s.strip() for s in sent_tokenize(text) if s.strip()]
    return sents

# Key-phrase extraction (TF-IDF over sentences) --------------------------------

def extract_key_phrases_tfidf(text: str, top_n: int = 10) -> List[str]:
    """
    Extract candidate phrases by TF-IDF (1-2 grams) across the document sentences.
    Works reasonably well for medium-length text.
    """
    if TfidfVectorizer is None:
        raise RuntimeError("scikit-learn is required for TF-IDF. Install with: pip install scikit-learn")

    sents = simple_sentences(text)
    if not sents:
        return []

    # Vectorize sentences as documents to find n-grams that score highly overall
    vect = TfidfVectorizer(ngram_range=(1,2), max_features=500, stop_words="english")
    X = vect.fit_transform(sents)  # shape: (n_sents, n_terms)
    # Sum scores per feature across sentences
    scores = X.sum(axis=0).A1  # array shape (n_features,)
    features = vect.get_feature_names_out()
    ranked = sorted(zip(features, scores), key=lambda x: x[1], reverse=True)
    phrases = [feat for feat, sc in ranked[:top_n]]
    # clean/normalize: remove purely stop words etc.
    uniq = []
    for p in phrases:
        if p.lower() not in uniq:
            uniq.append(p)
    return uniq

# Generate flashcards and simple answers ---------------------------------------

def find_sentence_containing(term: str, sents: List[str]) -> Optional[str]:
    term_low = term.lower()
    for s in sents:
        if term_low in s.lower():
            return s
    return None

def make_flashcards_from_text(text: str, n_cards: int = 12) -> List[Dict]:
    sents = simple_sentences(text)
    key_phrases = extract_key_phrases_tfidf(text, top_n=max(20, n_cards*2))
    # choose up to n_cards topics
    chosen = key_phrases[:n_cards]
    flashcards = []
    for term in chosen:
        answer_sent = find_sentence_containing(term, sents)
        if answer_sent:
            answer = answer_sent
        else:
            answer = f"Definition or explanation of '{term}'."
        q = f"What is {term}?"
        flashcards.append({"question": q, "answer": answer, "tags": term})
    return flashcards

# Quiz generation (MCQ, T/F, Fill-in) ------------------------------------------

def generate_mcqs_from_flashcards(flashcards: List[Dict], n_options: int = 4) -> List[Dict]:
    mcqs = []
    answers_texts = [fc["answer"] for fc in flashcards]
    for i, fc in enumerate(flashcards):
        correct = fc["answer"]
        # distractors: pick random other answers (or truncated phrases) ensuring uniqueness
        distractors = []
        pool = [a for j,a in enumerate(answers_texts) if j != i]
        random.shuffle(pool)
        for p in pool:
            if len(distractors) >= n_options-1:
                break
            if p not in distractors and p != correct:
                distractors.append(p)
        # if not enough distractors, fabricate short distractors
        while len(distractors) < n_options-1:
            distractors.append("A plausible-but-incorrect option.")
        options = distractors + [correct]
        random.shuffle(options)
        mcq = {
            "question": f"Which of the following best describes: {flashcards[i]['tags']}?",
            "options": options,
            "answer_index": options.index(correct),
            "explanation": correct
        }
        mcqs.append(mcq)
    return mcqs

def generate_true_false_from_flashcards(flashcards: List[Dict], max_q: int = 10) -> List[Dict]:
    tfs = []
    for i, fc in enumerate(flashcards[:max_q]):
        # create a statement from the answer; sometimes invert to false
        statement = fc["answer"]
        # randomly flip some to false and craft negative statement
        is_true = random.random() > 0.35
        if not is_true:
            # naive negation: prepend "False: " or add "not" if possible (best effort)
            if statement.lower().startswith("is "):
                statement = statement.replace("is ", "is not ", 1)
            else:
                statement = "Not exactly: " + statement
        tfs.append({"statement": statement, "is_true": is_true, "source": fc["tags"]})
    return tfs

def generate_fill_blanks(flashcards: List[Dict], max_q: int = 10) -> List[Dict]:
    fills = []
    for fc in flashcards[:max_q]:
        ans = fc["tags"]
        sent = fc["answer"]
        if ans.lower() in sent.lower():
            blanked = sent.lower().replace(ans.lower(), "_____")
            # restore capitalization if needed
            fills.append({"prompt": blanked, "answer": ans})
        else:
            # fallback: short answer prompt
            fills.append({"prompt": f"Define: {ans}", "answer": ans})
    return fills

# Exporters -------------------------------------------------------------------

def save_flashcards_csv(flashcards: List[Dict], filename: str):
    fieldnames = ["question", "answer", "tags"]
    with open(filename, "w", newline='', encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for fc in flashcards:
            writer.writerow({k: fc.get(k, "") for k in fieldnames})

def save_quiz_json(mcqs: List[Dict], tfs: List[Dict], fills: List[Dict], filename: str):
    data = {"mcqs": mcqs, "true_false": tfs, "fill_in": fills}
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def create_anki_deck(flashcards: List[Dict], deck_name: str, filename: str):
    if genanki is None:
        raise RuntimeError("genanki not installed. Install with: pip install genanki")
    deck_id = abs(hash(deck_name)) % (10**10)
    model = genanki.Model(
        1607392319,
        'Simple Model',
        fields=[
            {'name': 'Question'},
            {'name': 'Answer'},
        ],
        templates=[
            {
                'name': 'Card 1',
                'qfmt': '{{Question}}',
                'afmt': '{{FrontSide}}<hr id="answer">{{Answer}}',
            },
        ])
    deck = genanki.Deck(deck_id, deck_name)
    for i, fc in enumerate(flashcards):
        note = genanki.Note(
            model=model,
            fields=[fc['question'], fc['answer']],
            guid=str(deck_id) + str(i)
        )
        deck.add_note(note)
    package = genanki.Package(deck)
    package.write_to_file(filename)

# Optional: OpenAI-enhanced question rewriting --------------------------------

def openai_rewrite_question(question: str, answer: str, api_key: Optional[str]) -> Tuple[str,str]:
    """
    Use OpenAI to rephrase question & answer to be better. Optional: if no API, return originals.
    """
    if openai is None or not api_key:
        return question, answer
    openai.api_key = api_key
    prompt = (
        "You are a helpful assistant that rewrites short educational flashcard Q&A to be clear and concise.\n\n"
        f"Q: {question}\nA: {answer}\n\n"
        "Return a JSON object with keys 'question' and 'answer' only."
    )
    try:
        resp = openai.ChatCompletion.create(
            model="gpt-4o-mini",  # attempt modern model; if unavailable, user's key may fail - still optional
            messages=[{"role":"user","content":prompt}],
            max_tokens=200,
            temperature=0.2,
        )
        txt = resp['choices'][0]['message']['content']
        # Expect JSON — try to parse
        cleaned = txt.strip()
        if cleaned.startswith("{"):
            parsed = json.loads(cleaned)
            return parsed.get("question", question), parsed.get("answer", answer)
        # fallback: return original
        return question, answer
    except Exception as e:
        # quietly fallback
        return question, answer

# CLI -------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="StudyMate — flashcard & quiz generator")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--input-file", "-i", help="Path to text file to use")
    group.add_argument("--text", "-t", help="Short text or topic provided inline")
    parser.add_argument("--out-prefix", "-o", default="study_output", help="Output file prefix")
    parser.add_argument("--limit", "-n", type=int, default=12, help="Number of flashcards to generate")
    parser.add_argument("--use-openai", action="store_true", help="If set and OPENAI_API_KEY is present, use OpenAI to polish Q&A")
    parser.add_argument("--create-anki", action="store_true", help="Create .apkg file (requires genanki)")
    args = parser.parse_args()

    text = ""
    if args.input_file:
        text = read_input_file(args.input_file)
    else:
        text = args.text

    if not text.strip():
        print("No input text provided.")
        return

    # Generate flashcards
    print("Extracting flashcards...")
    flashcards = make_flashcards_from_text(text, n_cards=args.limit)

    # Optionally polish with OpenAI (best-effort)
    api_key = os.getenv("OPENAI_API_KEY") if args.use_openai else None
    if args.use_openai and api_key and openai is not None:
        print("Polishing Q/A with OpenAI (best-effort)...")
        for fc in flashcards:
            q, a = openai_rewrite_question(fc["question"], fc["answer"], api_key)
            fc["question"], fc["answer"] = q, a

    # Generate quizzes
    print("Generating MCQs, True/False, and fill-in-the-blanks...")
    mcqs = generate_mcqs_from_flashcards(flashcards)
    tfs = generate_true_false_from_flashcards(flashcards, max_q=min(10, len(flashcards)))
    fills = generate_fill_blanks(flashcards, max_q=min(10, len(flashcards)))

    # Save outputs
    flash_csv = f"{args.out_prefix}_flashcards.csv"
    quiz_json = f"{args.out_prefix}_quiz.json"
    save_flashcards_csv(flashcards, flash_csv)
    save_quiz_json(mcqs, tfs, fills, quiz_json)
    print(f"Saved flashcards to: {flash_csv}")
    print(f"Saved quiz (MCQs + others) to: {quiz_json}")

    if args.create_anki:
        if genanki is None:
            print("genanki not installed; skipping Anki export. Install with 'pip install genanki'.")
        else:
            apkg = f"{args.out_prefix}.apkg"
            create_anki_deck(flashcards, deck_name=args.out_prefix, filename=apkg)
            print(f"Created Anki deck: {apkg}")

    # Print a short sample to console
    print("\nSample flashcards (first 3):")
    for fc in flashcards[:3]:
        print(f"- Q: {fc['question']}")
        print(f"  A: {textwrap.shorten(fc['answer'], width=180)}\n")

    print("Done.")

if __name__ == "__main__":
    main()

