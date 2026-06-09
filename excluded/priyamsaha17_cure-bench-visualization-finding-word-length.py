import pandas as pd
import json

# File paths
input_file = "/kaggle/input/cure-bench/curebench_valset_pharse1.jsonl"
output_file = "/kaggle/working/Validation Set.csv"

rows = []

with open(input_file, "r", encoding="utf-8") as f:
    for line in f:
        obj = json.loads(line.strip())
        q_text = obj.get("question", "")
        options = obj.get("options", None)
        correct_answer = obj.get("correct_answer", "")

        # If options exist, append them neatly to the question text
        if options:
            formatted_options = "\n".join([f"{k}: {v}" for k, v in options.items()])
            q_text = f"{q_text}\n\nOptions:\n{formatted_options}"

        rows.append({
            "id": obj.get("id"),
            "question_type": obj.get("question_type"),
            "question": q_text,
            "correct_answer": correct_answer
        })

# Convert to DataFrame
df = pd.DataFrame(rows)

# Save as CSV
df.to_csv(output_file, index=False, encoding="utf-8")

print(f"âœ… Saved {len(df)} questions to {output_file}")


import pandas as pd

# Show full content in each cell
pd.set_option('display.max_colwidth', None)

# Show all columns
pd.set_option('display.max_columns', None)

# Show all rows (optional â€” use with care for large dataframes)
pd.set_option('display.max_rows', None)

# Now display
df.head()


import pandas as pd
import json

# File paths
input_file = "/kaggle/input/cure-bench/curebench_testset_phase1.jsonl"
output_file = "/kaggle/working/Phase 1 - Test Set.csv"

rows = []

with open(input_file, "r", encoding="utf-8") as f:
    for line in f:
        obj = json.loads(line.strip())
        q_text = obj.get("question", "")
        options = obj.get("options", None)

        # If options exist, append them neatly to the question text
        if options:
            formatted_options = "\n".join([f"{k}: {v}" for k, v in options.items()])
            q_text = f"{q_text}\n\nOptions:\n{formatted_options}"

        rows.append({
            "id": obj.get("id"),
            "question_type": obj.get("question_type"),
            "question": q_text
        })

# Convert to DataFrame
df = pd.DataFrame(rows)

# Save as CSV
df.to_csv(output_file, index=False, encoding="utf-8")

print(f"âœ… Saved {len(df)} questions to {output_file}")


import pandas as pd

# Show full content in each cell
pd.set_option('display.max_colwidth', None)

# Show all columns
pd.set_option('display.max_columns', None)

# Show all rows (optional â€” use with care for large dataframes)
pd.set_option('display.max_rows', None)

# Now display
df.head()


import pandas as pd
import json

# File paths
input_file = "/kaggle/input/cure-bench/curebench_testset_phase2.jsonl"
output_file = "/kaggle/working/Phase 2 - Test Set.csv"

rows = []

with open(input_file, "r", encoding="utf-8") as f:
    for line in f:
        obj = json.loads(line.strip())
        q_text = obj.get("question", "")
        options = obj.get("options", None)

        # If options exist, append them neatly to the question text
        if options:
            formatted_options = "\n".join([f"{k}: {v}" for k, v in options.items()])
            q_text = f"{q_text}\n\nOptions:\n{formatted_options}"

        rows.append({
            "id": obj.get("id"),
            "question_type": obj.get("question_type"),
            "question": q_text
        })

# Convert to DataFrame
df = pd.DataFrame(rows)

# Save as CSV
df.to_csv(output_file, index=False, encoding="utf-8")

print(f"âœ… Saved {len(df)} questions to {output_file}")


import pandas as pd

# Show full content in each cell
pd.set_option('display.max_colwidth', None)

# Show all columns
pd.set_option('display.max_columns', None)

# Show all rows (optional â€” use with care for large dataframes)
pd.set_option('display.max_rows', None)

# Now display
df.head()


import pandas as pd

# ---------------------------
# Load CSV
# ---------------------------
df = pd.read_csv("/kaggle/working/Validation Set.csv")

# Ensure all text are strings
df["question"] = df["question"].astype(str)

# ---------------------------
# Compute word lengths
# ---------------------------
df["word_length"] = df["question"].apply(lambda x: len(x.split()))

# ---------------------------
# Find maximum length
# ---------------------------
max_len = df["word_length"].max()
max_idx = df["word_length"].idxmax()
longest_q = df.loc[max_idx, "question"]

print(f"ğŸ“� Maximum word length: {max_len}")
print(f"ğŸ†” Row index with maximum length: {max_idx}\n")
print("ğŸ§¾ Longest question (full text):\n")
print(longest_q)


import pandas as pd

# ---------------------------
# Load CSV
# ---------------------------
df = pd.read_csv("/kaggle/working/Phase 1 - Test Set.csv")

# Ensure all text are strings
df["question"] = df["question"].astype(str)

# ---------------------------
# Compute word lengths
# ---------------------------
df["word_length"] = df["question"].apply(lambda x: len(x.split()))

# ---------------------------
# Find maximum length
# ---------------------------
max_len = df["word_length"].max()
max_idx = df["word_length"].idxmax()
longest_q = df.loc[max_idx, "question"]

print(f"ğŸ“� Maximum word length: {max_len}")
print(f"ğŸ†” Row index with maximum length: {max_idx}\n")
print("ğŸ§¾ Longest question (full text):\n")
print(longest_q)


import pandas as pd

# ---------------------------
# Load CSV
# ---------------------------
df = pd.read_csv("/kaggle/working/Phase 2 - Test Set.csv")

# Ensure all text are strings
df["question"] = df["question"].astype(str)

# ---------------------------
# Compute word lengths
# ---------------------------
df["word_length"] = df["question"].apply(lambda x: len(x.split()))

# ---------------------------
# Find maximum length
# ---------------------------
max_len = df["word_length"].max()
max_idx = df["word_length"].idxmax()
longest_q = df.loc[max_idx, "question"]

print(f"ğŸ“� Maximum word length: {max_len}")
print(f"ğŸ†” Row index with maximum length: {max_idx}\n")
print("ğŸ§¾ Longest question (full text):\n")
print(longest_q)

