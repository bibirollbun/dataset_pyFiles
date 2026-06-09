import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.sparse import csr_matrix
from implicit.als import AlternatingLeastSquares

# Load Data
fname_tran = '../input/h-and-m-personalized-fashion-recommendations/transactions_train.csv'
fname_article = '../input/h-and-m-personalized-fashion-recommendations/articles.csv'

# CSV íŒŒì�¼ ë¡œë“œ
data = pd.read_csv(fname_tran, usecols=['customer_id', 'article_id', 'price'])
articles = pd.read_csv(fname_article)

# ğŸ”¹ article_idë¥¼ ë¬¸ì��ì—´ë¡œ ë³€í™˜ (ë�°ì�´í„° ì�¼ê´€ì„± ìœ ì§€)
data['article_id'] = data['article_id'].astype(str)
articles['article_id'] = articles['article_id'].astype(str)

# ğŸ”¹ ì�´ë¯¸ì§€ ê²½ë¡œ ìƒ�ì„± (H&M ë�°ì�´í„°ì…‹ í�´ë�” êµ¬ì¡° ì �ìš©)
articles['image_path'] = articles['article_id'].apply(lambda x: f"0{x[:2]}/0{x}.jpg")

# ğŸ”¹ ë�°ì�´í„° ê°€ê³µ (ê³ ê°�-ìƒ�í’ˆ ë§¤í•‘)
data['count'] = 1
data = data.groupby(['customer_id', 'article_id'], as_index=False).sum()
user_unique = data['customer_id'].unique()
article_unique = data['article_id'].unique()

# ğŸ”¹ ID ë³€í™˜ (Mapping)
user_to_idx = {v: k for k, v in enumerate(user_unique)}
article_to_idx = {v: k for k, v in enumerate(article_unique)}
idx_to_article = {v: k for k, v in article_to_idx.items()}

# ğŸ”¹ ID ë§¤í•‘ (NaN ë°©ì§€ ë°� í•„í„°ë§�)
data['customer_id'] = data['customer_id'].map(user_to_idx).fillna(-1).astype(int)
data['article_id'] = data['article_id'].map(article_to_idx).fillna(-1).astype(int)
data = data[(data['customer_id'] >= 0) & (data['article_id'] >= 0)]

# ğŸ”¹ CSR Matrix ìƒ�ì„±
num_user = data['customer_id'].nunique()
num_article = data['article_id'].nunique()
csr_data = csr_matrix((data['count'], (data.customer_id, data.article_id)), shape=(num_user, num_article))

# ğŸ”¹ ALS ëª¨ë�¸ í•™ìŠµ
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['MKL_NUM_THREADS'] = '1'

als_model = AlternatingLeastSquares(factors=360, regularization=0.01, use_gpu=True, iterations=5, dtype=np.float32, calculate_training_loss=True)
als_model.fit(csr_data.T)

# ğŸ”¹ ì¶”ì²œ ì‹œìŠ¤í…œ - ì�´ë¯¸ì§€ ê°¤ëŸ¬ë¦¬ í‘œì‹œ
def show_recommendations(input_id, is_user=True, N=10):
    """ ì‚¬ìš©ì��ê°€ êµ¬ë§¤í•  ë§Œí•œ ì¶”ì²œ ì•„ì�´í…œ ë˜�ëŠ” íŠ¹ì • ì œí’ˆê³¼ ìœ ì‚¬í•œ ì œí’ˆì�„ ê°¤ëŸ¬ë¦¬ë¡œ í‘œì‹œ """
    if is_user:
        if input_id not in user_to_idx:
            print("ì‚¬ìš©ì�� IDë¥¼ ì°¾ì�„ ìˆ˜ ì—†ìŠµë‹ˆë‹¤.")
            return
        user_idx = user_to_idx[input_id]
        recommended = als_model.recommend(user_idx, csr_data, N=N)
        recommended_articles = [idx_to_article[i[0]] for i in recommended]
    else:
        if input_id not in article_to_idx:
            print("ìƒ�í’ˆ IDë¥¼ ì°¾ì�„ ìˆ˜ ì—†ìŠµë‹ˆë‹¤.")
            return
        article_idx = article_to_idx[input_id]
        similar_items = als_model.similar_items(article_idx, N=N)
        recommended_articles = [idx_to_article[i[0]] for i in similar_items]
    
    # ì¶”ì²œ ì•„ì�´í…œ ì •ë³´ ê°€ì ¸ì˜¤ê¸°
    recommended_df = articles[articles['article_id'].isin(recommended_articles)]
    
    # ğŸ”¹ ì�´ë¯¸ì§€ ê°¤ëŸ¬ë¦¬ ì¶œë ¥
    fig, axes = plt.subplots(1, len(recommended_df), figsize=(15, 5))
    if len(recommended_df) == 1:
        axes = [axes]  # ë¦¬ìŠ¤íŠ¸ë¡œ ë³€í™˜
    for ax, (_, row) in zip(axes, recommended_df.iterrows()):
        img_path = f"../input/h-and-m-personalized-fashion-recommendations/images/{row['image_path']}"
        
        try:
            img = plt.imread(img_path)
            ax.imshow(img)
        except FileNotFoundError:
            ax.text(0.5, 0.5, "No Image", fontsize=12, ha='center', va='center')

        ax.set_title(row['prod_name'][:15])
        ax.axis("off")
    plt.show()

# ì‚¬ìš© ì˜ˆì‹œ
# show_recommendations('000058a12d5b43e67d225668fa1f8d618c13dc232df0cad8ffe7ad4a1091e318', is_user=True)
# show_recommendations('176209023', is_user=False)


show_recommendations('000058a12d5b43e67d225668fa1f8d618c13dc232df0cad8ffe7ad4a1091e318', is_user=True)
# show_recommendations(176209023, is_user=False)


show_recommendations('176209023', is_user=False)

