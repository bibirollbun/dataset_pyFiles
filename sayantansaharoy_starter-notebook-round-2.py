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

from sklearn import preprocessing
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings("ignore")
le = preprocessing.LabelEncoder()


# ğŸ“œ The Codex of Beings and the Chronicle of Duels lie before you.
# Unroll them and glimpse their secrets...

pokemons = pd.read_csv("/kaggle/input/innovision-2025-round-two/pokemons.csv")
battles = pd.read_csv("/kaggle/input/innovision-2025-round-two/battles.csv")

pokemons.info()
battles.info()

pokemons.head(), battles.head()


b1 = battles
for index,row in b1.iterrows():
    if row['First_pokemon'] == row['Winner']:
        row['Winner'] = 0
    elif row['Second_pokemon'] == row['Winner']:
        row['Winner'] = 1
    else:
        row['Winner'] = 'nan'

b1.info()
b1.head()



p1=pokemons

p1=p1.drop(['Name'],axis=1)

p1 = p1.fillna("None")

p1.head()


merged1 = b1.merge(p1, left_on="First_pokemon", right_on="#", suffixes=("", "_first"))

merged1 = merged1.merge(p1, left_on="Second_pokemon", right_on="#", suffixes=("", "_second"))

merged1 = merged1.drop(['First_pokemon','Second_pokemon'],axis=1)

merged1.info()


X=merged1.drop(['Winner'],axis=1)
y=merged1['Winner']

X.info()
y.info()


# ğŸ‘�ï¸� A wise Duelist studies the scrolls before acting.
# Peek into their structure: how many entries, what columns, what mysteries?
diff=['HP','Attack','Defense','Sp. Atk','Sp. Def','Speed']

for column in diff:
    X[column+"_diff"] = X[column] - X[column+"_second"]

cat=['Type 1','Type 2','Type 1_second','Type 2_second']
for column in cat:
    X[column] = le.fit_transform(X[column])

X.head()


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print("âœ… Split complete:")
print("Training set:", X_train.shape, y_train.shape)
print("Test set:", X_test.shape, y_test.shape)


model=RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train,y_train)


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


# âš”ï¸� Testing the blade against known foes
y_train_pred = model.predict(X_train)

# Compute Accuracy
accuracy = accuracy_score(y_train, y_train_pred)
print("Accuracy on training set:", accuracy,"\n\n")

# Optional: confusion matrix
cm = confusion_matrix(y_train, y_train_pred)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=[0,1], yticklabels=[0,1])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("âš”ï¸� Confusion Matrix - Training Set")
plt.show()




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
test = pd.read_csv("/kaggle/input/innovision-2025-round-two/test.csv")

test = test.merge(p1, left_on="First_pokemon", right_on="#", suffixes=("", "_first"))

test = test.merge(p1, left_on="Second_pokemon", right_on="#", suffixes=("", "_second"))

t = test.drop(['First_pokemon','Second_pokemon','id'],axis=1)

diff=['HP','Attack','Defense','Sp. Atk','Sp. Def','Speed']

for column in diff:
    t[column+"_diff"] = t[column] - t[column+"_second"]

cat=['Type 1','Type 2','Type 1_second','Type 2_second']
for column in cat:
    t[column] = le.fit_transform(t[column])


t.info()


# âš”ï¸� The Gauntlet of Fate (test.csv) has arrived.
# Use the same rituals as you have before.
# -- Merge and feature craft here, mirroring your training steps --
# X_test = ...

test_preds = model.predict(t)



submission = pd.DataFrame({
    "id": test['id'],
    "First_pokemon":test["First_pokemon"],
    "Second_pokemon":test["Second_pokemon"],
    "Winner": test_preds
})

for index,row in submission.iterrows():
    if row['Winner']==0:
        row['Winner']=row['First_pokemon']
    elif row['Winner']==1:
        row['Winner']=row['Second_pokemon']

submission = submission.drop(["First_pokemon","Second_pokemon"],axis=1)

submission.to_csv("submission.csv", index=False)
print("âœ¨ Prophecies etched into submission.csv")


