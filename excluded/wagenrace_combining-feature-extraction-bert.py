# ðŸ“¦ Import needed package
import regex as re
from typing import Tuple

import pandas as pd
from pathlib import Path


training_data = pd.read_csv(
    r"/kaggle/input/fake-or-real-the-impostor-hunt/data/train.csv"
)

for i, row in training_data.iterrows():
    id = int(row.id)
    real_text_id = row.real_text_id
    fake_text_id = 1 if real_text_id == 2 else 2

    # Get file paths to text
    files_path = Path(
        rf"/kaggle/input/fake-or-real-the-impostor-hunt/data/train/article_{str(id).zfill(4)}"
    )
    real_text_path = files_path / f"file_{real_text_id}.txt"
    fake_text_path = files_path / f"file_{fake_text_id}.txt"

    # Load texts
    real_text = real_text_path.read_text()
    fake_text = fake_text_path.read_text()
    training_data.loc[i, "real_text"] = real_text
    training_data.loc[i, "fake_text"] = fake_text

training_data.head(10)


def count_word(text: str, word: str) -> int:
    return len(re.findall(word.lower(), text.lower()))

def repeats_word_three_times(text: str) -> Tuple[bool, list]:
    repeating_phrases = re.findall(r"([^\w].{4,})\1+", text.lower())
    if len(repeating_phrases) > 0:
        return True, repeating_phrases
    else:
        return False, []

def count_none_latin_letters(text):
    # Search for things that are NOT
    # \p{Latin} Latin letters
    # \s empty spaces
    # \p{S} Symbols
    # \p{P} Punitions
    # \p{N} Numbers
    # \p{Greek} greek letters (boy do scientists love themselves some greek letters)
    # \Âµ for some reason Âµ is not part of \p{Greek}? Weird
    return len(re.findall("[^\p{Latin}\s\p{S}\p{P}\p{N}\p{Greek}\Âµ]+", text))



def get_the_real(text1: str, text2: str) -> Tuple[int, str]:
    # Empty strings are fake
    if len(text1) == 0:
        return 2, "Empty"
    if len(text2) == 0:
        return 1, "Empty"

    # Did you use weird letters
    # If both are the same we continue
    count1 = count_none_latin_letters(text1)
    count2 = count_none_latin_letters(text2)
    if count1 > count2:
        return 2, "None Latin"
    if count2 > count1:
        return 1, "None Latin"

    # China
    china_1 = count_word(text1, "china")
    china_2 = count_word(text2, "china")
    if china_1 > china_2 and china_1 > 2:
        return 2, "china"
    if china_2 > china_1 and china_2 > 2:
        return 1, "china"

    # Dino
    dino_1 = count_word(text1, "dinosaur")
    dino_2 = count_word(text2, "dinosaur")
    if dino_1 > dino_2 and dino_1 > 2:
        return 2, "dinosaur"
    if dino_2 > dino_1 and dino_2 > 2:
        return 1, "dinosaur"

    # Music
    music_1 = count_word(text1, "music")
    music_2 = count_word(text2, "music")
    if music_1 > music_2 and music_1 > 2:
        return 2, "music"
    if music_2 > music_1 and music_2 > 2:
        return 1, "music"

    # AddTagHelper
    AddTagHelper_1 = count_word(text1, "AddTagHelper")
    AddTagHelper_2 = count_word(text2, "AddTagHelper")
    if AddTagHelper_1 > AddTagHelper_2:
        return 2, "AddTagHelper"
    if AddTagHelper_2 > AddTagHelper_1:
        return 1, "AddTagHelper"

    # Repeating words
    # If you repeat a word and it is the most repeated
    repeats_1 = repeats_word_three_times(text1)
    repeats_2 = repeats_word_three_times(text2)
    if repeats_1[0] and not repeats_2[0]:
        return 2, f"repeats {repeats_1[1]}"
    if repeats_2[0] and not repeats_1[0]:
        return 1, f"repeats {repeats_2[1]}"

    # No clue? You get a zero
    return 0, "No Clue"


submission_features = pd.DataFrame(columns=["id", "real_text_id", "reason"])
test_path_base = Path(r"/kaggle/input/fake-or-real-the-impostor-hunt/data/test")
for test_path in test_path_base.glob("**/article_*"):
    text_1 = (test_path / "file_1.txt").read_text()
    text_2 = (test_path / "file_2.txt").read_text()
    article_id = int(re.findall("\d+", test_path.name)[0])
    real_id, reason = get_the_real(text_1, text_2)

    submission_features = pd.concat(
        [pd.DataFrame([{"id": article_id, "real_text_id": real_id, "reason": reason}]), submission_features]
    )

print(
    f"Submissions without predictions: {(submission_features.real_text_id == 0).sum() / len(submission_features) * 100:.1f}%"
)

# Replace unknown with 1
submission_features = submission_features.sort_values(by="id")
submission_features = submission_features.reset_index()
submission_features.head()


# Loading the results of Bert
submission_bert_path = Path("/kaggle/input/0-87759-fake-or-real-bert-pca-randomforest/submission.csv")
submission_bert = pd.read_csv(submission_bert_path)


# Lets find-out where the features and Bert disagrees
disagrement_rows = (submission_features.real_text_id != 0) & (submission_features.real_text_id != submission_bert.real_text_id)
submission_features[disagrement_rows]


agrement_rows = (submission_features.real_text_id != 0) & (submission_features.real_text_id == submission_bert.real_text_id)
print(f"When a prediction is made Bert and Feature agree {agrement_rows.sum()/(agrement_rows.sum() + disagrement_rows.sum()) * 100:.1f}% of the time")


# Make submission
submission_bert.loc[disagrement_rows] = submission_features[disagrement_rows]
submission_bert.to_csv("submission.csv", index=False)
submission_bert[disagrement_rows].head()

