# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

def detect_competition_path():
    """
    è‡ªå‹•å�µæ¸¬ Kaggle /kaggle/input/ ä¸‹çš„æ¯”è³½è³‡æ–™å¤¾
    å›�å‚³æ¯”è³½è³‡æ–™å¤¾è·¯å¾‘
    """
    input_dir = "/kaggle/input"
    dirs = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]
    if not dirs:
        raise FileNotFoundError("â�Œ æ²’æœ‰æ‰¾åˆ°ä»»ä½•æ¯”è³½è³‡æ–™å¤¾ï¼Œè«‹ç¢ºèª� /kaggle/input/ ä¸‹æœ‰è³‡æ–™")
    # å¦‚æ�œæœ‰å¤šå€‹è³‡æ–™å¤¾ï¼Œå�–ç¬¬ä¸€å€‹
    comp_path = os.path.join(input_dir, dirs[0])
    print(f"ğŸ“‚ ä½¿ç”¨è³‡æ–™å¤¾: {comp_path}")
    return comp_path

def load_data(base_path):
    """
    è¼‰å…¥æ¯”è³½æ��ä¾›çš„è³‡æ–™
    """
    train_seq_path = os.path.join(base_path, "train_sequences.csv")
    train_lbl_path = os.path.join(base_path, "train_labels.csv")
    test_seq_path  = os.path.join(base_path, "test_sequences.csv")

    if not (os.path.exists(train_seq_path) and os.path.exists(train_lbl_path) and os.path.exists(test_seq_path)):
        raise FileNotFoundError(f"â�Œ æ‰¾ä¸�åˆ°æ¯”è³½è³‡æ–™ï¼Œè«‹ç¢ºèª� {base_path} ä¸‹æœ‰ train/test CSV æª”æ¡ˆ")

    # è¼‰å…¥è³‡æ–™
    train_seq = pd.read_csv(train_seq_path)
    train_lbl = pd.read_csv(train_lbl_path)
    test_df   = pd.read_csv(test_seq_path)

    # å�ˆä½µè¨“ç·´è³‡æ–™
    train_df = train_seq.merge(train_lbl, left_on="id", right_on="ID", how="inner")
    if "ID" in train_df.columns:
        train_df = train_df.drop("ID", axis=1)

    # ç‰¹å¾µèˆ‡æ¨™ç±¤
    X_train = train_df.drop(["id", "label"], axis=1, errors="ignore")
    y_train = train_df["label"]

    # æ¸¬è©¦è³‡æ–™
    test_ids = test_df["id"]
    X_test = test_df.drop("id", axis=1, errors="ignore")

    return X_train, y_train, X_test, test_ids

def main():
    # å�µæ¸¬æ¯”è³½è³‡æ–™å¤¾
    base_path = detect_competition_path()

    # è¼‰å…¥è³‡æ–™
    X_train, y_train, X_test, test_ids = load_data(base_path)

    # åˆ‡åˆ†è¨“ç·´/é©—è­‰é›†
    X_train_split, X_val, y_train_split, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )

    # å»ºç«‹æ¨¡å�‹
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train_split, y_train_split)

    # é©—è­‰
    val_score = model.score(X_val, y_val)
    print(f"ğŸ“Š Validation accuracy: {val_score:.4f}")

    # é �æ¸¬æ¸¬è©¦é›†
    predictions = model.predict(X_test)

    # âš ï¸� ä¿®æ”¹é€™è£¡æˆ�æ¯”è³½è¦�æ±‚çš„æ¬„ä½�å��ç¨± (label / target / prediction)
    submission = pd.DataFrame({
        "id": test_ids,
        "label": predictions   # é �è¨­ç”¨ "label"
    })
    
    submission.to_csv("submission.csv", index=False)
    print("âœ… submission.csv å·²ç”¢ç”Ÿ")
    print(submission.head())

if __name__ == "__main__":
    main()


