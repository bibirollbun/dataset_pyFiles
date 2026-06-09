# é€™å€‹ Python 3 ç’°å¢ƒé �è£�äº†è¨±å¤šå¯¦ç”¨çš„åˆ†æ��å‡½å¼�åº«
# å®ƒæ˜¯ä»¥ kaggle/python Docker æ˜ åƒ�æª”ç‚ºåŸºç¤�æ‰€å®šç¾©ï¼š[https://github.com/kaggle/docker-python](https://github.com/kaggle/docker-python)
# ä¾‹å¦‚ï¼Œä»¥ä¸‹æ˜¯ä¸€äº›è¼‰å…¥çš„å¯¦ç”¨å¥—ä»¶
import os
import numpy as np # ç·šæ€§ä»£æ•¸
import pandas as pd # è³‡æ–™è™•ç�†ã€�CSV æª”æ¡ˆè¼¸å…¥/è¼¸å‡º (ä¾‹å¦‚ pd.read_csv)

# è¼¸å…¥è³‡æ–™æª”æ¡ˆä½�æ–¼å”¯è®€çš„ "../input/" ç›®éŒ„ä¸‹
# ä¾‹å¦‚ï¼ŒåŸ·è¡Œæ­¤è™• (é»�æ“Š "run" æˆ–æŒ‰ä¸‹ Shift+Enter) å°‡æœƒåˆ—å‡ºè¼¸å…¥ç›®éŒ„ä¸‹çš„æ‰€æœ‰æª”æ¡ˆ

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# ä½ å�¯ä»¥å°‡æœ€å¤š 20GB çš„è³‡æ–™å¯«å…¥ç›®å‰�ç›®éŒ„ (/kaggle/working/)ï¼Œç•¶ä½ ä½¿ç”¨ "Save & Run All" å»ºç«‹ç‰ˆæœ¬æ™‚ï¼Œé€™äº›è³‡æ–™æœƒè¢«å„²å­˜ä¸‹ä¾†
# ä½ ä¹Ÿå�¯ä»¥å°‡æš«å­˜æª”æ¡ˆå¯«å…¥ /kaggle/temp/ï¼Œä½†é€™äº›æª”æ¡ˆåœ¨ç›®å‰�å·¥ä½œéš�æ®µçµ�æ�Ÿå¾Œä¸�æœƒè¢«å„²å­˜


# å®šç¾©åŸºæœ¬å�ƒæ•¸
N_MODELS = 45      # æ¨¡å�‹ç¸½æ•¸
N_SAMPLES = 75     # æ¯�å€‹é€šé�“çš„æ¨£æœ¬æ•¸
CHANNELS = ['channel_44', 'channel_45', 'channel_46']  # ä½¿ç”¨ç®¡é�“

# è¨­å®šæª”æ¡ˆè·¯å¾‘
INPUT_DIR = '/kaggle/input/trojan-horse-hunt-in-space'
CLEAN_MODEL_PATH = os.path.join(INPUT_DIR, 'clean_model')
POISONED_MODELS_PATH = os.path.join(INPUT_DIR, 'poisoned_models')
SUBMISSION_PATH = 'submission.csv'

# é™¤éŒ¯æ¨¡å¼�è¨­å®šï¼ˆæœ¬åœ°æ¸¬è©¦ç”¨ï¼‰
DEBUG = False
if DEBUG:
    INPUT_DIR = './data'
    CLEAN_MODEL_PATH = os.path.join(INPUT_DIR, 'clean_model')
    POISONED_MODELS_PATH = os.path.join(INPUT_DIR, 'poisoned_models')
    SUBMISSION_PATH = 'submission.csv'

print("âœ… å�ƒæ•¸è¨­å®šå®Œæˆ�")
print(f"æ¨¡å�‹æ•¸é‡�: {N_MODELS}")
print(f"æ¯�é€šé�“æ¨£æœ¬æ•¸: {N_SAMPLES}")
print(f"é€šé�“åˆ—è¡¨: {CHANNELS}")
print(f"ç¸½ç‰¹å¾µæ•¸: {N_SAMPLES * len(CHANNELS)}")


def create_zero_trigger_submission():
    """
    å»ºç«‹ä»¥é›¶å€¼åˆ�å§‹åŒ–çš„è§¸ç™¼å™¨çŸ©é™£ï¼Œä¸¦ç”¢ç”Ÿç¬¦å�ˆæ��äº¤æ ¼å¼�çš„CSVæª”æ¡ˆ
    
    Returns:
        pandas.DataFrame: æ��äº¤ç”¨çš„è³‡æ–™æ¡†æ�¶
    """
    print("ğŸ“Š é–‹å§‹å»ºç«‹é›¶è§¸ç™¼å™¨æ��äº¤æª”æ¡ˆ...")
    
    # å»ºç«‹é›¶è§¸ç™¼å™¨å�‘é‡�ï¼ˆæ¯�å€‹æ¨¡å�‹çš„è§¸ç™¼å™¨ï¼‰
    zero_trigger = np.zeros(N_SAMPLES * len(CHANNELS))
    print(f"é›¶è§¸ç™¼å™¨å�‘é‡�é•·åº¦: {len(zero_trigger)}")
    
    # ç‚ºæ¯�å€‹æ¨¡å�‹è¤‡è£½ç›¸å�Œçš„é›¶è§¸ç™¼å™¨
    data = np.tile(zero_trigger, (N_MODELS, 1))
    print(f"è³‡æ–™çŸ©é™£å½¢ç‹€: {data.shape}")
    
    # è½‰æ�›ç‚ºè³‡æ–™æ¡†æ�¶
    df = pd.DataFrame(data)
    
    # ç”¢ç”Ÿé€šé�“åˆ¥çš„æ¬„ä½�å��ç¨±
    channel_cols = [
        f"{ch}_{i+1}"
        for ch in CHANNELS
        for i in range(N_SAMPLES)
    ]
    
    print(f"æ¬„ä½�æ•¸é‡�: {len(channel_cols)}")
    print(f"å‰�5å€‹æ¬„ä½�å��ç¨±: {channel_cols[:5]}")
    
    # è¨­å®šæ¬„ä½�å��ç¨±
    df.columns = channel_cols
    
    # åœ¨æœ€å‰�é�¢æ�’å…¥ model_id æ¬„ä½�ï¼ˆå¾�1é–‹å§‹ç·¨è™Ÿï¼‰
    df.insert(0, "model_id", range(1, N_MODELS + 1))
    
    # è¨­å®šç´¢å¼•å¾�1é–‹å§‹
    df.index = df.index + 1
    
    print("âœ… é›¶è§¸ç™¼å™¨è³‡æ–™æ¡†æ�¶å»ºç«‹å®Œæˆ�")
    return df


def save_and_validate_submission(df):
    """
    å„²å­˜æ��äº¤æª”æ¡ˆä¸¦é€²è¡ŒåŸºæœ¬é©—è­‰
    
    Args:
        df (pandas.DataFrame): è¦�å„²å­˜çš„è³‡æ–™æ¡†æ�¶
    """
    print("ğŸ’¾ å„²å­˜æ��äº¤æª”æ¡ˆä¸­...")
    
    # å„²å­˜ç‚ºCSVæª”æ¡ˆ
    df.to_csv(SUBMISSION_PATH, index=False)
    print(f"âœ… æ��äº¤æª”æ¡ˆå·²å„²å­˜: {SUBMISSION_PATH}")
    
    # é©—è­‰æª”æ¡ˆ
    print("\nğŸ”� æª”æ¡ˆé©—è­‰:")
    print(f"è³‡æ–™æ¡†æ�¶å½¢ç‹€: {df.shape}")
    print(f"é �æœŸå½¢ç‹€: ({N_MODELS}, {N_SAMPLES * len(CHANNELS) + 1})")  # +1 æ˜¯å› ç‚º model_id æ¬„ä½�
    
    # æª¢æŸ¥å‰�å¹¾åˆ—
    print(f"\nğŸ“‹ å‰�3åˆ—è³‡æ–™é �è¦½:")
    print(df.head(3))
    
    # æª¢æŸ¥æ˜¯å�¦æœ‰ç¼ºå¤±å€¼
    missing_values = df.isnull().sum().sum()
    print(f"\nç¼ºå¤±å€¼æ•¸é‡�: {missing_values}")
    
    # æª¢æŸ¥æª”æ¡ˆå¤§å°�
    file_size = os.path.getsize(SUBMISSION_PATH)
    print(f"æª”æ¡ˆå¤§å°�: {file_size:,} bytes ({file_size/1024/1024:.2f} MB)")
    
    return True


def main():
    """ä¸»è¦�åŸ·è¡Œå‡½æ•¸"""
    print("ğŸš€ å¤ªç©ºæœ¨é¦¬ç—…æ¯’å�µæ¸¬ç«¶è³½ - é–‹å§‹åŸ·è¡ŒåŸºæº–æ��äº¤ç¨‹å¼�...")
    print("=" * 60)
    
    try:
        # å»ºç«‹é›¶è§¸ç™¼å™¨æ��äº¤æª”æ¡ˆ
        df = create_zero_trigger_submission()
        
        # å„²å­˜ä¸¦é©—è­‰æ��äº¤æª”æ¡ˆ
        save_and_validate_submission(df)
        
        print("\n" + "=" * 60)
        print("ğŸ�‰ ç¨‹å¼�åŸ·è¡Œå®Œæˆ�ï¼�æ��äº¤æª”æ¡ˆå·²æº–å‚™å°±ç·’")
        
        return df
        
    except Exception as e:
        print(f"â�Œ åŸ·è¡Œé��ç¨‹ä¸­ç™¼ç”ŸéŒ¯èª¤: {str(e)}")
        raise


if __name__ == "__main__":
    # åŸ·è¡Œä¸»ç¨‹å¼�
    submission_df = main()
    
    # é¡�å¤–çš„æª¢æŸ¥å’Œçµ±è¨ˆ
    print("\nğŸ“Š æœ€çµ‚çµ±è¨ˆè³‡è¨Š:")
    print(f"â€¢ è™•ç�†çš„æ¨¡å�‹æ•¸é‡�: {len(submission_df)}")
    print(f"â€¢ æ¯�å€‹æ¨¡å�‹çš„ç‰¹å¾µæ•¸: {len(submission_df.columns) - 1}")  # -1 å› ç‚º model_id ä¸�ç®—ç‰¹å¾µ
    print(f"â€¢ è§¸ç™¼å™¨å€¼ç¯„åœ�: [{submission_df.iloc[:, 1:].min().min():.3f}, {submission_df.iloc[:, 1:].max().max():.3f}]")
    print(f"â€¢ æª”æ¡ˆè·¯å¾‘: {SUBMISSION_PATH}")
    
    print("\nğŸ�¯ å�¯ä»¥å°‡æ­¤æª”æ¡ˆæ��äº¤åˆ° Kaggle ç«¶è³½äº†ï¼�")

