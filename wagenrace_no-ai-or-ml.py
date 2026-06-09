# ğŸ“¦ Import needed package
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

for i, row in training_data.iterrows():
    # Count real
    training_data.loc[i, "real_china_count"] = count_word(row.real_text, "china")
    training_data.loc[i, "real_dino_count"] = count_word(row.real_text, "dinosaur")

    # Count fake
    training_data.loc[i, "fake_china_count"] = count_word(row.fake_text, "china")
    training_data.loc[i, "fake_dino_count"] = count_word(row.fake_text, "dinosaur")

training_data[training_data.fake_china_count > 0].head(10)


more_china_real = (
    training_data.real_china_count > training_data.fake_china_count
).sum()
more_china_fake = (
    training_data.real_china_count < training_data.fake_china_count
).sum()
print(
    f"{more_china_real} real have more China, {more_china_fake} fake have more china. The rest is equal"
)

more_dino_real = (training_data.real_dino_count > training_data.fake_dino_count).sum()
more_dino_fake = (training_data.real_dino_count < training_data.fake_dino_count).sum()
print(
    f"{more_dino_real} real have more dino, {more_dino_fake} fake have more dino. The rest is equal"
)


def repeats_word_three_times(text: str) -> Tuple[bool, list]:
    repeating_phrases = re.findall(r"([^\w].{4,})\1+", text.lower())
    if len(repeating_phrases) > 0:
        return True, repeating_phrases
    else:
        return False, []


for i, row in training_data.iterrows():
    real_repeats, phrases = repeats_word_three_times(row.real_text)
    training_data.loc[i, "real_repeat"] = real_repeats
    if real_repeats:
        print(row.id, row.real_text_id, phrases)
    training_data.loc[i, "fake_repeat"] = repeats_word_three_times(row.fake_text)[0]

real_repeat_3 = (training_data.real_repeat).sum()
fake_repeat_3 = (training_data.fake_repeat).sum()
print(
    f"{real_repeat_3} real have to much repeat, {fake_repeat_3} fake have to much repeat. The rest does not"
)

# If you have 1 repeat it can be normal English but 2 gets weirds
training_data[(training_data.real_repeat)].head(5)


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


for i, row in training_data.iterrows():
    training_data.loc[i, "real_none_latin_count"] = count_none_latin_letters(
        row.real_text
    )
    training_data.loc[i, "fake_none_latin_count"] = count_none_latin_letters(
        row.fake_text
    )

print(
    f"Reals with more none latin: {(training_data.real_none_latin_count > training_data.fake_none_latin_count).sum()}"
)
print(
    f"Fakes with more none latin: {(training_data.fake_none_latin_count > training_data.real_none_latin_count).sum()}"
)
print(
    f"Number of reals with a none latin character: {len(training_data[training_data.fake_none_latin_count > 0])}"
)
training_data[training_data.fake_none_latin_count > 0][
    ["real_text", "fake_text", "real_none_latin_count", "fake_none_latin_count"]
].head()


print("real text that are empty", (training_data.real_text == "").sum())
print("fake text that are empty", (training_data.fake_text == "").sum())


def get_the_real(text1: str, text2: str) -> int:
    # Empty strings are fake
    if len(text1) == 0:
        return 2
    if len(text2) == 0:
        return 1

    # Did you use weird letters
    # If both are the same we continue
    count1 = count_none_latin_letters(text1)
    count2 = count_none_latin_letters(text2)
    if count1 > count2:
        return 2
    if count2 > count1:
        return 1

    # China
    china_1 = count_word(text1, "china")
    china_2 = count_word(text2, "china")
    if china_1 > china_2 and china_1 > 2:
        return 2
    if china_2 > china_1 and china_2 > 2:
        return 1

    # Dino
    dino_1 = count_word(text1, "dinosaur")
    dino_2 = count_word(text2, "dinosaur")
    if dino_1 > dino_2 and dino_1 > 2:
        return 2
    if dino_2 > dino_1 and dino_2 > 2:
        return 1

    # Music
    music_1 = count_word(text1, "music")
    music_2 = count_word(text2, "music")
    if music_1 > music_2 and music_1 > 2:
        return 2
    if music_2 > music_1 and music_2 > 2:
        return 1

    # AddTagHelper
    AddTagHelper_1 = count_word(text1, "AddTagHelper")
    AddTagHelper_2 = count_word(text2, "AddTagHelper")
    if AddTagHelper_1 > AddTagHelper_2:
        return 2
    if AddTagHelper_2 > AddTagHelper_1:
        return 1

    # Repeating words
    # If you repeat a word more then 3 and it is the most repeated
    repeats_1 = repeats_word_three_times(text1)
    repeats_2 = repeats_word_three_times(text2)
    if repeats_1[0] and not repeats_2[0]:
        print("repeated word", repeats_1[1])
        return 2
    if repeats_2[0] and not repeats_1[0]:
        print("repeated word", repeats_2[1])
        return 1

    # No clue? You get a zero
    return 0


# Test is on training data
for i, row in training_data.iterrows():
    training_data.loc[i, "prediction"] = get_the_real(row.real_text, row.fake_text)

unknowns = (training_data["prediction"] == 0).sum()
corrects = (training_data["prediction"] == 1).sum()
incorrects = (training_data["prediction"] == 2).sum()

print(f"correct: {corrects} | incorrect: {incorrects} | unknown: {unknowns}")

training_data[training_data["prediction"] == 2]


submission = pd.DataFrame(columns=["id", "real_text_id"])
test_path_base = Path(r"/kaggle/input/fake-or-real-the-impostor-hunt/data/test")
for test_path in test_path_base.glob("**/article_*"):
    text_1 = (test_path / "file_1.txt").read_text()
    text_2 = (test_path / "file_2.txt").read_text()
    article_id = int(re.findall("\d+", test_path.name)[0])
    real_id = get_the_real(text_1, text_2)

    submission = pd.concat(
        [pd.DataFrame([{"id": article_id, "real_text_id": real_id}]), submission]
    )

print(
    f"Submissions without predictions: {(submission.real_text_id == 0).sum() / len(submission) * 100:.1f}%"
)

# Replace unknown with 1
submission.loc[submission.real_text_id == 0, "real_text_id"] = 2
submission = submission.sort_values(by="id")
submission.to_csv("submission.csv", index=False)
submission

