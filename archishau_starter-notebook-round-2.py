# âš”ï¸� The Duelist's Prophecy âš”ï¸�
#
# Welcome, Duelist. Your journey has led you to the heart of this world's conflict:
# the honor-bound arena where legends are forged and tragically broken.
#
# Now that you have a team, you must guide them through the perilous path of combat,
# where every victory brings glory and every defeat is an irreversible farewell.
# Your predictive skills are the only thing standing between your PokÃ©mon and eternal silence.
#
# Will you be able to predict the outcome of future matches?
# Can you become a seer of combat, gazing into the future to protect those you lead?
#
# To aid you, two ancient and powerful sources of knowledge have been entrusted:
#
# ğŸ“œ The Codex of Beings (pokemons.csv): the essence of every known PokÃ©mon â€”
#     their characteristics, strengths, and weaknesses.
#     The first column marks each creature's unique identifier.
#
# âš”ï¸� The Chronicle of Duels (battles.csv): a grim record of past combats,
#     listing the two combatants and, most importantly, the victor.
#
# A sacred rule governs this arena:
#   The PokÃ©mon listed first in each battle is the one who strikes first.
#   This honor of the first blow can shift the tide of battle in an instant.
#
# ğŸ�¯ Your Goal:
# Forge a Machine Learning model â€” a crystal ball of code ğŸ”® â€”
# able to consult these ancient texts and prophesy the result of future battles.
# You will then face the Gauntlet of Fate (test.csv), a scroll of duels yet to come.



# ğŸ› ï¸� Every Duelist needs their weapons.
# Here you must summon libraries of knowledge to process scrolls, craft features,
# and forge your crystal ball of foresight.
#
# Hint: Pandas and NumPy for scroll reading, Scikit-learn for prophecy.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# Optional: silence the whispers of unnecessary warnings ğŸ•¯ï¸�
import warnings
warnings.filterwarnings("ignore")



# ğŸ“œ The Codex of Beings and the Chronicle of Duels lie before you.
# Unroll them and glimpse their secrets...

pokemons = pd.read_csv("/kaggle/input/innovision-2025-round-two/pokemons.csv")
battles = pd.read_csv("/kaggle/input/innovision-2025-round-two/battles.csv")

pokemons.head(), battles.head()



# ğŸ‘�ï¸� A wise Duelist studies the scrolls before acting.
# Peek into their structure: how many entries, what columns, what mysteries?



# âš”ï¸� In battle, knowledge must be attached to each fighter.
# Each combatant must carry their attributes from the Codex into the Chronicle.
#
# Task for the Seer:
#  â€¢ Merge first_pokemon with its Codex attributes.
#  â€¢ Merge second_pokemon with its Codex attributes.
#  â€¢ Remember: the first_pokemon always strikes first.
#
# (Only the brave may attempt clever feature crafting here...)

# Example skeleton (to be completed):
# battles = battles.merge(pokemons, left_on="First_pokemon", right_on="id", suffixes=("", "_first"))
# battles = battles.merge(pokemons, left_on="Second_pokemon", right_on="id", suffixes=("", "_second"))



# ğŸ› ï¸� Raw stats are powerful, but differences reveal the true advantage.
# Suggested incantations:
#   attack_diff   = Attack_first - Attack_second
#   defense_diff  = Defense_first - Defense_second
#   speed_diff    = Speed_first - Speed_second
#
# Add more if your foresight allows... but beware of overcomplication.

# Example skeleton:
# X["attack_diff"] = battles["Attack_first"] - battles["Attack_second"]
# X["defense_diff"] = ...
# X["speed_diff"] = ...

# The chosen prophecy target:
# y = (battles["Winner"] == battles["First_pokemon"]).astype(int)



# âœ‚ï¸� The Chronicle must be divided:
# One part for training your foresight, one part to test your wisdom.



# ğŸ”® Time to summon the prophecy engine.
# Choose wisely, for the spirit you invoke will shape the clarity of your visions.



# Your crystal ball must be judged for its clarity.
# Find out how often your foresight is true
# And further your vision by revealing where your visions falter.
#
# Study it well â€” are certain fates harder to predict?


# âš”ï¸� The Gauntlet of Fate (test.csv) has arrived.
# Use the same rituals as you have before.

test = pd.read_csv("/kaggle/input/innovision-2025-round-two/test.csv")

# -- Merge and feature craft here, mirroring your training steps --
# X_test = ...

test_preds = model.predict(X_test)

submission = pd.DataFrame({
    "id": test["id"],
    "winner": test_preds
})

submission.to_csv("submission.csv", index=False)
print("âœ¨ Prophecies etched into submission.csv")


