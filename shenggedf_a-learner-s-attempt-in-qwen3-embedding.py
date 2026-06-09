# URLs
url_re = re.compile(r"(?i)\b(?:https?://|www\.)\S+")
t = url_re.sub("<URL>", t)


# prompt template for Qwen3Embedding
EMBEDDING_MODEL_QUERY = "Instruct: Retrieve relevant passages for this query\nQuery: "
EMBEDDING_MODEL_PASSAGE = "Instruct: Encode this passage for retrieval\nPassage: "




