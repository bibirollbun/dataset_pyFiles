import pandas as pd
import numpy as np
import re
import string
import warnings
import os
warnings.filterwarnings('ignore')

# ==================== Kaggle ç�¯å¢ƒé…�ç½® ====================
INPUT_DIR = "/kaggle/input/tweet-sentiment-extraction"  # æ¯”èµ›æ•°æ�®ç›®å½•
OUTPUT_DIR = "/kaggle/working/"  # è¾“å‡ºç›®å½•

# æƒ…æ„Ÿå…³é”®è¯�è¯�å…¸ï¼ˆæ ¸å¿ƒè§„åˆ™ï¼‰
POSITIVE_KEYWORDS = {
    'good', 'great', 'best', 'happy', 'love', 'liked', 'like', 'awesome', 'amazing', 
    'fantastic', 'perfect', 'excited', 'joy', 'smile', 'win', 'success', 'proud', 
    'blessed', 'grateful', 'nice', 'beautiful', 'excellent', 'wonderful', 'cool'
}

NEGATIVE_KEYWORDS = {
    'bad', 'worst', 'sad', 'hate', 'dislike', 'terrible', 'awful', 'horrible', 
    'angry', 'mad', 'upset', 'cry', 'lose', 'failure', 'sick', 'hurt', 'annoyed',
    'frustrated', 'disappointed', 'boring', 'ugly', 'stupid', 'hated', 'annoying'
}

# æ–‡æœ¬æ¸…æ´—å‡½æ•°
def clean_text(text):
    """æ¸…æ´—æ–‡æœ¬ï¼šç§»é™¤æ ‡ç‚¹ã€�å°�å†™åŒ–ã€�å�»å�œç”¨è¯�"""
    # ç§»é™¤URL
    text = re.sub(r'http\S+|www.\S+', '', text)
    # ç§»é™¤@ç”¨æˆ·
    text = re.sub(r'@\w+', '', text)
    # ç§»é™¤æ ‡ç‚¹
    text = text.translate(str.maketrans('', '', string.punctuation))
    # å°�å†™åŒ–
    text = text.lower().strip()
    # ç§»é™¤å¤šä½™ç©ºæ ¼
    text = re.sub(r'\s+', ' ', text)
    return text

# æ ¸å¿ƒæŠ½å�–å‡½æ•°
def extract_selected_text(text, sentiment):
    """æ ¹æ�®æƒ…æ„Ÿç±»å�‹æŠ½å�–å¯¹åº”æ–‡æœ¬"""
    if pd.isna(text) or text.strip() == "":
        return ""
    
    original_text = text.strip()
    clean_txt = clean_text(original_text)
    words = clean_txt.split()
    
    # 1. ä¸­æ€§æƒ…æ„Ÿï¼šç›´æ�¥è¿”å›�å�Ÿæ–‡æœ¬
    if sentiment == "neutral":
        return original_text
    
    # 2. ç§¯æ��æƒ…æ„Ÿï¼šæ��å�–ç§¯æ��å…³é”®è¯�ç›¸å…³ç‰‡æ®µ
    elif sentiment == "positive":
        # æ‰¾æ‰€æœ‰ç§¯æ��å…³é”®è¯�çš„ä½�ç½®
        positive_indices = []
        for i, word in enumerate(words):
            if word in POSITIVE_KEYWORDS:
                positive_indices.append(i)
        
        if not positive_indices:
            return original_text  # æ— å…³é”®è¯�è¿”å›�å�Ÿæ–‡
        
        # æ��å�–åŒ…å�«å…³é”®è¯�çš„è¿�ç»­ç‰‡æ®µ
        start_idx = max(0, positive_indices[0] - 1)
        end_idx = min(len(words)-1, positive_indices[-1] + 1)
        selected_words = words[start_idx:end_idx+1]
        
        # æ˜ å°„å›�å�Ÿæ–‡æœ¬ï¼ˆåŒ¹é…�æœ€æ�¥è¿‘çš„ç‰‡æ®µï¼‰
        selected_phrase = ' '.join(selected_words)
        # åœ¨å�Ÿæ–‡æœ¬ä¸­æ‰¾æœ€æ�¥è¿‘çš„åŒ¹é…�
        if selected_phrase in original_text.lower():
            # æ‰¾åˆ°å�Ÿæ–‡æœ¬ä¸­çš„ä½�ç½®
            match = re.search(re.escape(selected_phrase), original_text.lower())
            if match:
                return original_text[match.start():match.end()]
        return original_text
    
    # 3. æ¶ˆæ��æƒ…æ„Ÿï¼šæ��å�–æ¶ˆæ��å…³é”®è¯�ç›¸å…³ç‰‡æ®µ
    elif sentiment == "negative":
        # æ‰¾æ‰€æœ‰æ¶ˆæ��å…³é”®è¯�çš„ä½�ç½®
        negative_indices = []
        for i, word in enumerate(words):
            if word in NEGATIVE_KEYWORDS:
                negative_indices.append(i)
        
        if not negative_indices:
            return original_text  # æ— å…³é”®è¯�è¿”å›�å�Ÿæ–‡
        
        # æ��å�–åŒ…å�«å…³é”®è¯�çš„è¿�ç»­ç‰‡æ®µ
        start_idx = max(0, negative_indices[0] - 1)
        end_idx = min(len(words)-1, negative_indices[-1] + 1)
        selected_words = words[start_idx:end_idx+1]
        
        # æ˜ å°„å›�å�Ÿæ–‡æœ¬
        selected_phrase = ' '.join(selected_words)
        if selected_phrase in original_text.lower():
            match = re.search(re.escape(selected_phrase), original_text.lower())
            if match:
                return original_text[match.start():match.end()]
        return original_text
    
    # é»˜è®¤è¿”å›�å�Ÿæ–‡
    return original_text

# ==================== æ•°æ�®åŠ è½½ ====================
def load_data():
    """åŠ è½½Kaggleæ¯”èµ›æ•°æ�®"""
    try:
        # ä¼˜å…ˆåŠ è½½Kaggleè¾“å…¥ç›®å½•æ•°æ�®
        train_path = os.path.join(INPUT_DIR, "train.csv")
        test_path = os.path.join(INPUT_DIR, "test.csv")
        train_df = pd.read_csv(train_path) if os.path.exists(train_path) else pd.DataFrame()
        test_df = pd.read_csv(test_path) if os.path.exists(test_path) else pd.DataFrame()
        print(f"åŠ è½½æ•°æ�®æˆ�åŠŸ - è®­ç»ƒé›†: {len(train_df)}, æµ‹è¯•é›†: {len(test_df)}")
    except Exception as e:
        print(f"åŠ è½½Kaggleæ•°æ�®å¤±è´¥: {e}ï¼Œå°�è¯•æœ¬åœ°è·¯å¾„")
        train_df = pd.read_csv("train.csv") if os.path.exists("train.csv") else pd.DataFrame()
        test_df = pd.read_csv("test.csv") if os.path.exists("test.csv") else pd.DataFrame()
    
    # æ•°æ�®æ¸…æ´—
    train_df = train_df.fillna({"text": "", "sentiment": "neutral", "selected_text": ""})
    test_df = test_df.fillna({"text": "", "sentiment": "neutral"})
    return train_df, test_df

# ==================== è¯„ä¼°å‡½æ•°ï¼ˆå�¯é€‰ï¼‰ ====================
def jaccard_similarity(str1, str2):
    """è®¡ç®—Jaccardç›¸ä¼¼åº¦ï¼ˆè¯„ä¼°æ•ˆæ�œï¼‰"""
    set1 = set(str1.split())
    set2 = set(str2.split())
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union != 0 else 0

def evaluate_train_set(train_df):
    """è¯„ä¼°è®­ç»ƒé›†æŠ½å�–æ•ˆæ�œ"""
    print("\nå¼€å§‹è¯„ä¼°è®­ç»ƒé›†æŠ½å�–æ•ˆæ�œ...")
    jaccard_scores = []
    for idx, row in train_df.iterrows():
        true_text = row["selected_text"].strip()
        pred_text = extract_selected_text(row["text"], row["sentiment"])
        jaccard_scores.append(jaccard_similarity(true_text, pred_text))
    
    avg_jaccard = np.mean(jaccard_scores)
    print(f"è®­ç»ƒé›†å¹³å�‡Jaccardç›¸ä¼¼åº¦: {avg_jaccard:.4f}")
    return avg_jaccard

# ==================== ä¸»æµ�ç¨‹ ====================
if __name__ == "__main__":
    # 1. åŠ è½½æ•°æ�®
    train_df, test_df = load_data()
    
    # 2. è¯„ä¼°è®­ç»ƒé›†æ•ˆæ�œï¼ˆå�¯é€‰ï¼‰
    if len(train_df) > 0:
        evaluate_train_set(train_df)
    
    # 3. å¤„ç�†æµ‹è¯•é›†
    print("\nå¼€å§‹å¤„ç�†æµ‹è¯•é›†...")
    test_df["selected_text"] = test_df.apply(
        lambda row: extract_selected_text(row["text"], row["sentiment"]),
        axis=1
    )
    
    # 4. ç”Ÿæˆ�æ��äº¤æ–‡ä»¶
    submission_df = test_df[["textID", "selected_text"]].copy()
    submission_path = os.path.join(OUTPUT_DIR, "submission.csv")
    submission_df.to_csv(submission_path, index=False)
    
    # è¾“å‡ºç»“æ�œ
    print(f"\nâœ… æ��äº¤æ–‡ä»¶ç”Ÿæˆ�å®Œæˆ�ï¼�")
    print(f"ğŸ“� æ–‡ä»¶è·¯å¾„: {submission_path}")
    print(f"ğŸ“„ æ��äº¤æ–‡ä»¶é¢„è§ˆ:")
    print(submission_df.head(10))
    
    # éªŒè¯�æ–‡ä»¶å®Œæ•´æ€§
    print(f"\nğŸ“Š æ–‡ä»¶ç»Ÿè®¡:")
    print(f"- æ€»æ�¡ç›®æ•°: {len(submission_df)}")
    print(f"- ç©ºå€¼æ•°é‡�: {submission_df['selected_text'].isna().sum()}")
    print(f"- ç©ºæ–‡æœ¬æ•°é‡�: {len(submission_df[submission_df['selected_text'].str.strip() == ''])}")

