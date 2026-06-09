%%time
from pathlib import Path
import pandas as pd

TRAIN_ANNOTATION_DIR = Path("../input/MABe-mouse-behavior-detection/train_annotation")
TRAIN_TRACKING_DIR = Path("../input/MABe-mouse-behavior-detection/train_tracking")

train_files = list(TRAIN_ANNOTATION_DIR.glob("**/*.parquet"))
track_files = list(TRAIN_TRACKING_DIR.glob("**/*.parquet"))

print(f"âœ… Found {len(train_files)} annotation files")
print(f"âœ… Found {len(track_files)} tracking files")

sample = pd.read_parquet(train_files[0])
display(sample.head())


%%time
from collections import Counter
import matplotlib.pyplot as plt

# Gather all actions
all_actions = []
for f in train_files:
    df = pd.read_parquet(f)
    all_actions.extend(df['action'].tolist())

action_counts = Counter(all_actions)
actions, counts = zip(*action_counts.most_common())

plt.figure(figsize=(12,5))
plt.bar(actions, counts)
plt.xticks(rotation=45, ha='right')
plt.title("Action Frequency Distribution in Training Set")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

print(f"âœ… Total unique actions: {len(actions)}")
print(f"Most frequent: {actions[0]} ({counts[0]} samples)")
print(f"Least frequent: {actions[-1]} ({counts[-1]} samples)")


%%time
import seaborn as sns

lab_action_data = []
for f in train_files:
    lab = f.parent.name
    df = pd.read_parquet(f)
    action_counts = df['action'].value_counts()
    for act, count in action_counts.items():
        lab_action_data.append((lab, act, count))

lab_df = pd.DataFrame(lab_action_data, columns=["lab", "action", "count"])

plt.figure(figsize=(12,6))
sns.barplot(data=lab_df, x="action", y="count", hue="lab", estimator=sum)
plt.xticks(rotation=45, ha='right')
plt.title("Action Distribution Across Labs")
plt.tight_layout()
plt.show()


%%time
video_lengths = []
for f in train_files:
    df = pd.read_parquet(f)
    video_lengths.append(df['stop_frame'].max())

plt.figure(figsize=(8,4))
plt.hist(video_lengths, bins=40)
plt.xlabel("Frames per video")
plt.ylabel("Count")
plt.title("Distribution of Video Lengths")
plt.tight_layout()
plt.show()


%%time
tracking_summary = []
for f in track_files[:10]:
    df = pd.read_parquet(f)
    tracking_summary.append({
        "file": f.stem,
        "bodyparts": df['bodypart'].nunique(),
        "frames": df['video_frame'].nunique()
    })

tracking_df = pd.DataFrame(tracking_summary)
display(tracking_df)


%%time
from collections import Counter

pairs = []
for f in train_files:
    df = pd.read_parquet(f)
    pairs.extend(zip(df['agent_id'], df['target_id']))

pair_counts = Counter(pairs)
top_pairs = pair_counts.most_common(10)

print("Top 10 agent-target pairs:")
for (a, t), c in top_pairs:
    print(f"  {a} â†’ {t}: {c}")


%%time
TRACKING_DIR = Path("../input/MABe-mouse-behavior-detection/train_tracking")
tracking_files = list(TRACKING_DIR.glob("**/*.parquet"))

for f in tracking_files[:5]:  # just first 5 for quick check
    df = pd.read_parquet(f)
    print(f"{f.stem}: {df['bodypart'].nunique()} body parts tracked")


%%time
from collections import Counter
from pathlib import Path
import pandas as pd

TRAIN_ANNOTATION_DIR = Path("../input/MABe-mouse-behavior-detection/train_annotation")
train_files = list(TRAIN_ANNOTATION_DIR.glob("**/*.parquet"))

global_counts = Counter()
lab_major_action = {}

for f in train_files:
    lab = f.parent.name
    df = pd.read_parquet(f)
    counts = Counter(df['action'])
    global_counts.update(counts)
    lab_major_action[lab] = counts.most_common(1)[0][0]

global_top3 = [a for a, _ in global_counts.most_common(3)]
print("Top 3 global actions:", global_top3)
print("Per-lab most frequent actions:", lab_major_action)


%%time
from tqdm.auto import tqdm
import random

TEST_TRACKING_DIR = Path("../input/MABe-mouse-behavior-detection/test_tracking")
test_files = list(TEST_TRACKING_DIR.glob("**/*.parquet"))
print(f"ğŸ”� Found {len(test_files)} test tracking files")

rows = []
row_id = 0

for f in tqdm(test_files, desc="Generating V2 predictions"):
    df = pd.read_parquet(f)
    video_id = int(f.stem)
    
    # infer lab name if directory structure allows
    lab = f.parent.name
    if lab in lab_major_action:
        predicted_action = lab_major_action[lab]
    else:
        predicted_action = random.choice(global_top3)
    
    mice_ids = df['mouse_id'].unique()
    start_frame = int(df['video_frame'].min())
    stop_frame = int(df['video_frame'].max())
    
    for agent in mice_ids:
        for target in mice_ids:
            t_id = "mouseself" if agent == target else f"mouse{target}"
            a_id = f"mouse{agent}"
            rows.append([row_id, video_id, a_id, t_id, predicted_action, start_frame, stop_frame])
            row_id += 1


%%time
submission_df = pd.DataFrame(rows, columns=[
    "row_id", "video_id", "agent_id", "target_id", "action", "start_frame", "stop_frame"
])
submission_df.to_csv("submission.csv", index=False)
print(f"âœ… Baseline submission.csv written: {submission_df.shape[0]} rows")


%%time
import pandas as pd

# Load the submission file
submission = pd.read_csv('submission.csv')

# Display the first few rows
submission.head()  

