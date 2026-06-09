# install bigframe
%pip install --upgrade bigframes google-cloud-automl google-cloud-translate google-ai-generativelanguage tensorflow 
# install pypdf2
%pip install pypdf2


from IPython.display import Image, display, HTML

display(Image(filename="/kaggle/input/imgs-graphs/BigQuery AI Competition .png"))


# get secrets
from kaggle_secrets import UserSecretsClient

user_secrets = UserSecretsClient()
user_credential = user_secrets.get_gcloud_credential()
user_secrets.set_tensorflow_credential(user_credential)



import os
from PyPDF2 import PdfReader
import pandas as pd
pd.reset_option('display.max_colwidth')  # ensure default truncation behavior in displays
import bigframes  # imported for later cells (BigQuery/BigFrames usage)
from IPython.display import display

# --- Configuration ---
# Kaggle input folder containing the PDF library (mounted read-only)
pdf_dir = "/kaggle/input/llm-generated-text-detection"

# --- Data structure to hold results ---
# We will map: {filename_without_extension: full_text}
pdf_dict = {}

# --- Iterate over PDFs deterministically (sorted) ---
for file in sorted(os.listdir(pdf_dir)):
    if not file.lower().endswith(".pdf"):
        continue  # skip non-PDFs

    file_path = os.path.join(pdf_dir, file)

    # Try to read the PDF; if it fails, skip with a minimal message
    try:
        reader = PdfReader(file_path)
    except Exception as e:
        print(f"[WARN] Could not open {file}: {e}")
        continue

    # Collect page texts; some pages may return None (e.g., images or parsing issues)
    page_texts = []
    for page in reader.pages:
        try:
            txt = page.extract_text()
        except Exception as e:
            txt = None  # on extraction error, treat as empty
        if txt:
            # Normalize whitespace a bit to avoid excessive newlines/tabs
            page_texts.append(" ".join(txt.split()))

    # Join all extracted text for this PDF
    full_text = " ".join(page_texts)

    # Use filename (without extension) as the dictionary key
    key = os.path.splitext(file)[0]
    pdf_dict[key] = full_text

# --- Quick sanity checks / preview ---
pdf_keys = list(pdf_dict.keys())

print(f"Loaded {len(pdf_dict)} PDFs\n")
# Show keys (paper identifiers); keep it compact if many
for k in pdf_keys:
    print(k)

# Show a short snippet from the first document (if any)
if pdf_keys:
    first_key = pdf_keys[0]
    print("\nText from first pdf:\n")
    # Display the first 400 characters for a quick look
    print(pdf_dict[first_key][:400])
else:
    print("No PDFs were parsed. Check the input path or file permissions.")


import bigframes._config
import bigframes.pandas as bpd


bpd.options.bigquery.location = "US"
# Set to your GCP project ID.
bpd.options.bigquery.project = "rising-beach-319705"


import pandas as pd
from bigframes.ml.llm import GeminiTextGenerator


# Make a small local pandas DF, then lift to BigFrames.
rows = [{"title": k, "text": v} for k, v in pdf_dict.items()]
pd_df = pd.DataFrame(rows)
df = bpd.DataFrame(pd_df)  # BigFrames (BigQuery-backed)

# A compact extraction instruction.
# Strict JSON fields that map to columns declared in output_schema.
system_preamble = (
    """
        You are an information extraction system for academic papers. 
    From the given full paper text, extract ONLY the following fields as a valid JSON object:
    
    {
      "abstract": "string (<=1200 characters, no newlines)",
      "authors": "comma-separated author full names",
      "year": "4-digit year of publication (e.g., 2023)"
    }
    
    Do not return explanations, commentary, or anything other than the JSON object.
    Ensure all 3 keys are present.
    """
)

# Instantiate Gemini model
model = GeminiTextGenerator(model_name="gemini-2.0-flash-001")

# Use predict with a prompt made of [preamble, the paper text].
# NOTE: output_schema controls the structured columns that come back.
result = model.predict(
    df[["title", "text"]],
    prompt=[system_preamble, df["text"]],
    temperature=0.0,
    max_output_tokens=2048,
    output_schema={
        "abstract": "STRING",
        "authors": "STRING",   # return comma-separated; split later if you want ARRAY
        "year": "INT64",
    },
)

# Stitch title back in
# BigFrames supports SQL-ish ops; easiest is to keep the original df's title col and join by index.
summary_df = df[["title"]].join(result[["abstract", "authors", "year"]])

with pd.option_context("display.max_colwidth", None):
    display(summary_df.tail(2))


import re 


# Next let's chunk the research papers so we can build a semantic search engine
def chunk_text(text: str, chunk_size=10_000, stride=500):
    chunks = []
    i = 0
    while i < len(text):
        chunk = text[i:i + chunk_size]
        chunks.append(chunk)
        if i + chunk_size >= len(text):
            break
        i += stride
    return chunks

chunk_data = []

for title, text in pdf_dict.items():    
    # only extract text prior to references section
    split_text = re.split(r'references\n', text, flags=re.IGNORECASE)
    if len(split_text) > 1:
        main_body, references = split_text[0], split_text[1]
    else:
        main_body, references = text, ""
    chunks = chunk_text(main_body, chunk_size=5_000, stride=500)
    for i, chunk in enumerate(chunks):
        chunk_data.append({
            "title": title,
            "chunk_index": i,
            "text_chunk": chunk
        })

df_chunks = pd.DataFrame(chunk_data)
print(f"{len(df_chunks)} total chunks created")


from bigframes.ml.llm import TextEmbeddingGenerator


# Convert pandas DF to BigFrames
bf_chunks = bpd.DataFrame(df_chunks)

# Choose embedding model: gemini-embedding-001
embedder = TextEmbeddingGenerator(model_name="text-embedding-005")

# Generate embeddings for each chunk
# Output will include a vector column (e.g. 'embedding')
embeddings = embedder.predict(bf_chunks[["text_chunk"]])

# Join metadata back
bf_with_embeddings = bf_chunks[["title", "chunk_index", "text_chunk"]].join(embeddings)

# write to BigQuery
bf_with_embeddings.to_gbq("rising-beach-319705.big_tables_demo.chunk_embeddings", if_exists="replace")


import bigframes.bigquery as bbq


def search_chunks_via_vector_index(
    query: str,
    embedded_table: str,       # BigQuery table, e.g. "project.dataset.embeddings"
    top_k: int = 10,
    embedding_col: str = "ml_generate_embedding_result",
    query_col: str = "embedding",
    distance_type: str = "cosine",
    model_name: str = "text-embedding-005"
):
    """
    Performs vector search in BigQuery using BigFrames.

    Returns a DataFrame with the search query and top_k closest rows:
    columns: original columns + 'distance'
    """
    # Step 1: Embed the query
    query_df = pg = pd.DataFrame({query_col: [query]})
    query_bf = bpd.DataFrame(query_df)

    embedder = TextEmbeddingGenerator(model_name=model_name)
    embedded_query = embedder.predict(query_bf[[query_col]])
    query_vector = embedded_query["ml_generate_embedding_result"].iloc[0]

    # Step 2: Use vector_search
    query_emb_bf = bpd.DataFrame(pd.DataFrame({query_col: [query_vector]}))
    results = bbq.vector_search(
        base_table=embedded_table,
        column_to_search=embedding_col,
        query=query_emb_bf,
        query_column_to_search=query_col,
        distance_type=distance_type
    )
    return results.sort_values("distance", ascending=True).head(top_k)


question = "what are the biggest challenges in detecting llm generated text?"

top_chunks = search_chunks_via_vector_index(
    question,
    "rising-beach-319705.big_tables_demo.chunk_embeddings",
    10
)


top_chunks.head(7)


def merge_chunks_by_title(top_chunks, text_col="text_chunk", sep=" -- "):
    """
    Group a DataFrame by 'title' and merge the text chunks into a single string.
    
    Parameters:
        top_chunks (bpd.DataFrame or pd.DataFrame): Top N chunks from vector search.
        text_col (str): Column containing text chunks.
        sep (str): Separator used to concatenate chunks.

    Returns:
        pd.DataFrame: DataFrame with columns: title, merged_text
    """
    if hasattr(top_chunks, "to_pandas"):
        df = top_chunks.to_pandas()
    else:
        df = top_chunks

    grouped = df.groupby("title").agg({
        text_col: lambda x: sep.join(str(chunk) for chunk in x),
        "distance": "min",  # keep minimum distance for reference
    }).reset_index()

    grouped = grouped.rename(columns={text_col: "merged_text", "ml_generate_embedding_result": "minimum_distance"})
    return grouped


def question_papers_with_llm(papers_df, question: str):
    """
    Ask Gemini whether each paper answers a given question, and if so, what is the answer.

    Parameters:
        papers_df (pd.DataFrame): Must contain 'title' and 'merged_text'.
        question (str): The research question you're asking.

    Returns:
        pd.DataFrame: With columns: title, is_relevant, extracted_answer
    """

    # Build structured prompt
    system_prompt = (
        "You are a scientific assistant reading academic papers. "
        "For each paper, answer:\n"
        "1. Does this paper contain a relevant answer to the following question?\n"
        f"   \"{question}\"\n"
        "2. If relevant, what is the answer in your own words (max 50 words)?\n"
        "Reply in this JSON format:\n"
        "{\n"
        "  \"answer\": \"Add summary if true, else empty string\"\n"
        "  \"is_relevant\": true/false,\n"
        
        "}"
        "Example:\n"
        "{\n"
        "  \"answer\":\"This paper introduces a novel method for...\"\n"
        "  \"is_relevant\":True,\n"
        "}"
    )

    # Prepare BigFrames DF
    bf = bpd.DataFrame(papers_df[["title", "merged_text", "distance"]])

    # Call Gemini
    model = GeminiTextGenerator()
    output = model.predict(
        bf[["title", "merged_text"]],
        prompt=[system_prompt, bf["merged_text"]],
        output_schema={
            "answer": "STRING",
            "is_relevant": "BOOL"
        },
        temperature=0.0,
        max_output_tokens=512
    )



    # Merge outputs back with title
    return bf[["title", "distance"]].join(output[["is_relevant", "answer"]])


merged_chunks = merge_chunks_by_title(top_chunks)


relevant_papers = question_papers_with_llm(merged_chunks, question)


# Join on title (inner join keeps only matching titles)
final_df = summary_df.merge(relevant_papers, on="title", how="inner")

final_df = final_df[[
    "title", "year", "authors", "abstract", "is_relevant", "answer", "distance"
]]

# Display nicely without truncation
with pd.option_context('display.max_colwidth', None):
    display(HTML(f"<h3><strong>Question: {question}</strong></h3>"))
    display(final_df)


from IPython.display import display, HTML



def get_relevant_papers(question: str, 
                        top_k: int = 10,
                        chunk_table_path: str = "rising-beach-319705.big_tables_demo.chunk_embeddings", 
                        ) -> pd.DataFrame:
    """
    Retrieve relevant papers for a given question using vector search and Gemini.

    Parameters:
        question (str): Research question.
        chunk_table_path (str): BigQuery table with chunk embeddings.
        top_k (int): Number of chunks to retrieve.

    Returns:
        pd.DataFrame: DataFrame with columns ['title', 'is_relevant', 'answer']
    """
    top_chunks = search_chunks_via_vector_index(
        question, chunk_table_path, top_k
    )

    merged_chunks = merge_chunks_by_title(top_chunks)

    return question_papers_with_llm(merged_chunks, question)



def show_relevant_papers(
    question: str, 
    relevant_papers,     # BigFrames DataFrame
    summary_bf,          # BigFrames DataFrame
    how: str = "inner"
):
    """
    Merge relevance results with paper metadata (BigFrames) and display nicely.

    Parameters:
        question (str): The question asked.
        relevant_papers (bpd.DataFrame): LLM relevance results (title, is_relevant, answer, distance).
        summary_bf (bpd.DataFrame): Paper metadata (title, authors, year, abstract).
        how (str): Join type ('inner' by default).
    """
    # Perform BigFrames join
    merged_bf = summary_bf.merge(relevant_papers, on="title", how=how)

    # Pick a nice column order if columns exist
    cols = [c for c in [
        "title", "year", "authors", "abstract",
        "is_relevant", "answer", "min_distance", "distance"
    ] if c in merged_bf.columns]
    other_cols = [c for c in merged_bf.columns if c not in cols]
    merged_bf = merged_bf[cols + other_cols]

    # Display question + merged BigFrames table
    with pd.option_context('display.max_colwidth', None):
        display(HTML(f"<h3>Question:</h3><h4><b>{question}</h4></p>"))
        display(merged_bf)




question = "Can large language models be trained to detect LLM generated text?"

relevant_papers = get_relevant_papers(question, top_k=10)


show_relevant_papers(question, relevant_papers, summary_df)

