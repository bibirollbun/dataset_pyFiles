!pip install -q scikit-learn sentence-transformers


# Tải các thư viện cần thiết
import pandas as pd
import numpy as np
from datetime import timedelta
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')
import sys # DEBUG: Dùng để kiểm tra bộ nhớ

# Thư viện cho Content-based
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

print("Các thư viện đã sẵn sàng.")

# Tải dữ liệu
try:
    transactions = pd.read_csv('/kaggle/input/h-and-m-personalized-fashion-recommendations/transactions_train.csv', dtype={'article_id': str}, parse_dates=['t_dat'])
    articles = pd.read_csv('/kaggle/input/h-and-m-personalized-fashion-recommendations/articles.csv', dtype={'article_id': str})
except FileNotFoundError:
    print("Không tìm thấy file trên Kaggle, vui lòng cập nhật đường dẫn.")

print("Dữ liệu đã tải xong.")
# DEBUG: In ra kích thước và bộ nhớ sử dụng của các DataFrame ban đầu
print(f"Transactions: {transactions.shape[0]:,} dòng, {transactions.shape[1]} cột. Memory: {transactions.memory_usage(deep=True).sum() / 1e9:.2f} GB")
print(f"Articles: {articles.shape[0]:,} dòng, {articles.shape[1]} cột. Memory: {articles.memory_usage(deep=True).sum() / 1e9:.2f} GB")


def build_transactional_item_map(transactions_df, recent_days=30, top_k=50):
    print(f"\n--- Bắt đầu xây dựng Bản đồ Giao dịch (dữ liệu {recent_days} ngày cuối) ---")
    
    # 1. Lọc dữ liệu gần đây
    max_date = transactions_df['t_dat'].max()
    min_date = max_date - timedelta(days=recent_days)
    recent_transactions = transactions_df[transactions_df['t_dat'] >= min_date]
    # DEBUG: Hiển thị số lượng giao dịch được sử dụng
    print(f"# DEBUG: Đang làm việc trên {len(recent_transactions):,} giao dịch từ {min_date.date()} đến {max_date.date()}.")
    
    # 2. Tạo các cặp sản phẩm
    df = recent_transactions[['customer_id', 'article_id']]
    merged_df = pd.merge(df, df, on='customer_id', suffixes=('_left', '_right'))
    # DEBUG: Hiển thị kích thước của DataFrame sau khi tự join (bước tốn tài nguyên nhất)
    print(f"# DEBUG: DataFrame sau khi self-join có {len(merged_df):,} dòng.")
    
    pairs = merged_df[merged_df['article_id_left'] != merged_df['article_id_right']]
    # DEBUG: Hiển thị số lượng cặp sản phẩm hợp lệ
    print(f"# DEBUG: Tìm thấy {len(pairs):,} cặp sản phẩm (đã loại bỏ cặp trùng).")
    
    # 3. Đếm tần suất
    pair_counts = pairs.groupby(['article_id_left', 'article_id_right']).size().reset_index(name='count')
    # DEBUG: Hiển thị số lượng cặp sản phẩm duy nhất
    print(f"# DEBUG: Có {len(pair_counts):,} cặp sản phẩm duy nhất được tìm thấy.")
    
    # 4. Xây dựng bản đồ
    item_map = defaultdict(list)
    pair_counts_sorted = pair_counts.sort_values('count', ascending=False)
    
    print("# DEBUG: Bắt đầu lặp để tạo dictionary tra cứu...")
    for row in pair_counts_sorted.itertuples(index=False):
        source_item = row.article_id_left
        related_item = row.article_id_right
        if len(item_map[source_item]) < top_k:
            item_map[source_item].append(related_item)
            
    # DEBUG: Hiển thị kích thước của bản đồ cuối cùng
    print(f"# DEBUG: Bản đồ giao dịch hoàn thành với {len(item_map)} sản phẩm gốc có gợi ý.")
    print("Xây dựng Bản đồ Giao dịch hoàn tất.")
    return dict(item_map)

# Chạy hàm với debug
transaction_map = build_transactional_item_map(transactions, recent_days=30, top_k=50)


def build_name_similarity_map(articles_df, top_k=50):
    """
    Xây dựng bản đồ item-to-item dựa trên sự tương đồng về tên sản phẩm (TF-IDF).
    *** PHIÊN BẢN TỐI ƯU BỘ NHỚ ***
    """
    print("\n--- Bắt đầu xây dựng Bản đồ Tương đồng Tên (TF-IDF) - Tối ưu hóa ---")
    
    # TfidfVectorizer mặc định đã chuẩn hóa L2, nên tích ma trận sẽ tương đương cosine similarity
    vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2, 5))
    tfidf_matrix = vectorizer.fit_transform(articles_df['prod_name'].fillna(''))
    print(f"# DEBUG: Ma trận TF-IDF thưa có kích thước: {tfidf_matrix.shape}.")

    print("# DEBUG: Đang tính toán ma trận tương đồng (sparse matrix multiplication)...")
    # TỐI ƯU: Thay vì dùng cosine_similarity, ta nhân ma trận thưa với chuyển vị của nó.
    # Đây là cách hiệu quả hơn nhiều về bộ nhớ.
    cosine_sim_sparse = tfidf_matrix * tfidf_matrix.T
    print(f"# DEBUG: Ma trận tương đồng thưa có kích thước: {cosine_sim_sparse.shape}.")
    
    item_map = {}
    article_ids = articles_df['article_id'].tolist()
    
    print("# DEBUG: Bắt đầu lặp để tạo dictionary tra cứu...")
    for idx, article_id in enumerate(article_ids):
        if (idx + 1) % 10000 == 0:
            print(f"# DEBUG: Đã xử lý {idx + 1}/{len(article_ids)} sản phẩm...")
            
        # Lấy một hàng của ma trận thưa
        row = cosine_sim_sparse[idx]
        
        # Chuyển hàng đó thành mảng đặc để sắp xếp
        # .A1 là cách nhanh để chuyển một hàng/cột của ma trận thưa thành mảng numpy 1D
        sim_scores_array = row.toarray().ravel()
        
        # Lấy chỉ số của các phần tử lớn nhất, hiệu quả hơn so với việc sort toàn bộ
        # np.argpartition sẽ tìm top k phần tử lớn nhất mà không cần sort toàn bộ
        top_indices = np.argpartition(sim_scores_array, -(top_k + 1))[-(top_k + 1):]
        
        # Sắp xếp lại chỉ top k+1 phần tử này
        sorted_top_indices = top_indices[np.argsort(sim_scores_array[top_indices])][::-1]
        
        # Bỏ qua phần tử đầu tiên (là chính nó)
        final_top_indices = [i for i in sorted_top_indices if i != idx][:top_k]

        top_articles = [article_ids[i] for i in final_top_indices]
        item_map[article_id] = top_articles
        
    print(f"# DEBUG: Bản đồ tương đồng tên hoàn thành với {len(item_map)} sản phẩm gốc có gợi ý.")
    print("Xây dựng Bản đồ Tương đồng Tên hoàn tất.")
    return item_map


print("Tạo một mẫu articles nhỏ để kiểm tra nhanh pipeline...")
articles_sample_for_tfidf = articles.head(40000) # Chỉ lấy 20k sản phẩm đầu tiên

# Gọi hàm trên DỮ LIỆU MẪU
name_sim_map = build_name_similarity_map(articles_sample_for_tfidf, top_k=50)

# (Tùy chọn) In ra một ví dụ để kiểm tra
sample_article_id_name = '0706016001'
if sample_article_id_name in name_sim_map:
    print(f"\nKiểm tra: Sản phẩm có tên tương tự '{sample_article_id_name}':")
    similar_ids = name_sim_map[sample_article_id_name][:5]
    print(articles[articles['article_id'].isin(similar_ids)][['article_id', 'prod_name']])


def build_embedding_similarity_map(articles_df, top_k=50):
    print("\n--- Bắt đầu xây dựng Bản đồ Tương đồng Embedding ---")
    
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    articles_df['full_desc'] = articles_df.apply(
        lambda row: ' '.join([
            str(row['prod_name']), str(row['product_type_name']), 
            str(row['product_group_name']), str(row['detail_desc'])
        ]).replace('nan', ''), axis=1)
    # DEBUG: In ra một ví dụ về văn bản đầy đủ
    print(f"# DEBUG: Ví dụ về văn bản đầy đủ để tạo embedding:\n'{articles_df['full_desc'].iloc[0]}'")

    print("Đang tạo embeddings... (Thanh tiến trình sẽ hiển thị bên dưới)")
    embeddings = model.encode(articles_df['full_desc'].tolist(), show_progress_bar=True, device='cuda')
    # DEBUG: Hiển thị kích thước của mảng embeddings
    print(f"# DEBUG: Mảng embedding có kích thước: {embeddings.shape} (sản phẩm, chiều embedding).")
    
    print("# DEBUG: Đang tính toán ma trận tương đồng cosine...")
    cosine_sim = cosine_similarity(embeddings, embeddings)
    # DEBUG: Hiển thị kích thước và bộ nhớ của ma trận tương đồng
    print(f"# DEBUG: Ma trận tương đồng có kích thước: {cosine_sim.shape}, Bộ nhớ: {sys.getsizeof(cosine_sim) / 1e9:.2f} GB.")
    
    item_map = {}
    article_ids = articles_df['article_id'].tolist()
    
    print("# DEBUG: Bắt đầu lặp để tạo dictionary tra cứu...")
    for idx, article_id in enumerate(article_ids):
        # DEBUG: Thêm một bộ đếm tiến độ
        if (idx + 1) % 5000 == 0:
            print(f"# DEBUG: Đã xử lý {idx + 1}/{len(article_ids)} sản phẩm...")
            
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        top_indices = [i[0] for i in sim_scores[1:top_k+1]]
        top_articles = [article_ids[i] for i in top_indices]
        item_map[article_id] = top_articles
        
    print(f"# DEBUG: Bản đồ tương đồng embedding hoàn thành với {len(item_map)} sản phẩm gốc có gợi ý.")
    print("Xây dựng Bản đồ Tương đồng Embedding hoàn tất.")
    return item_map

# Chạy với debug trên một mẫu nhỏ
articles_sample_for_embedding = articles.head(40000)
embedding_sim_map = build_embedding_similarity_map(articles_sample_for_embedding, top_k=50)


def generate_combined_candidates_for_user(customer_history, trans_map, name_map, embed_map, num_recent_items=5):
    # 1. Lấy các sản phẩm mua gần đây
    recent_items = customer_history.sort_values('t_dat', ascending=False)['article_id'].unique()[:num_recent_items]
    
    if len(recent_items) == 0:
        print("# DEBUG: Khách hàng này không có lịch sử mua hàng.")
        return []
    
    # DEBUG: In ra các sản phẩm "hạt giống"
    print(f"\n# DEBUG: Tìm ứng viên dựa trên {len(recent_items)} sản phẩm mua gần nhất: {recent_items}")
    
    candidate_set = set()
    
    for item_id in recent_items:
        # DEBUG: Theo dõi số lượng ứng viên từ mỗi nguồn
        trans_cands = trans_map.get(item_id, [])
        name_cands = name_map.get(item_id, [])
        embed_cands = embed_map.get(item_id, [])
        
        candidate_set.update(trans_cands)
        candidate_set.update(name_cands)
        candidate_set.update(embed_cands)
        
        print(f"# DEBUG: Từ sản phẩm '{item_id}': Tìm thấy {len(trans_cands)} (giao dịch), {len(name_cands)} (tên), {len(embed_cands)} (embedding) ứng viên.")
    
    # DEBUG: Hiển thị số lượng ứng viên trước và sau khi lọc
    initial_candidate_count = len(candidate_set)
    
    # 3. Loại bỏ các sản phẩm khách hàng đã mua
    purchased_items = set(customer_history['article_id'].unique())
    final_candidates = list(candidate_set - purchased_items)
    
    print(f"# DEBUG: Tổng số ứng viên ban đầu: {initial_candidate_count}")
    print(f"# DEBUG: Số ứng viên sau khi loại bỏ {len(purchased_items)} sản phẩm đã mua: {len(final_candidates)}")
    
    return final_candidates

# Ví dụ chạy với debug
sample_customer_id = transactions['customer_id'].unique()[200]
customer_purchase_history = transactions[transactions['customer_id'] == sample_customer_id]

candidates = generate_combined_candidates_for_user(
    customer_purchase_history,
    transaction_map,
    name_sim_map,
    embedding_sim_map
)

print(f"\n---> KẾT QUẢ CUỐI CÙNG: Đã tạo ra {len(candidates)} ứng viên duy nhất cho khách hàng {sample_customer_id[:10]}...")
print("Một vài ví dụ:", candidates[:15])


# =============================================================================
# === BƯỚC 0: CÀI ĐẶT THƯ VIỆN ===
# =============================================================================
!pip install -q sentence-transformers tqdm

# =============================================================================
# === BƯỚC 1: TẢI THƯ VIỆN VÀ DỮ LIỆU ===
# =============================================================================
import pandas as pd
import numpy as np
from datetime import timedelta
from collections import defaultdict
import warnings
from tqdm.auto import tqdm

warnings.filterwarnings('ignore')

from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

print("Các thư viện đã sẵn sàng.")

# Tải dữ liệu từ đường dẫn Kaggle
try:
    transactions = pd.read_csv('/kaggle/input/h-and-m-personalized-fashion-recommendations/transactions_train.csv', dtype={'article_id': str}, parse_dates=['t_dat'])
    articles = pd.read_csv('/kaggle/input/h-and-m-personalized-fashion-recommendations/articles.csv', dtype={'article_id': str})
    customers = pd.read_csv('/kaggle/input/h-and-m-personalized-fashion-recommendations/customers.csv')
except FileNotFoundError:
    print("Không tìm thấy file trên Kaggle. Vui lòng kiểm tra lại đường dẫn.")
    # Tạo dữ liệu giả để code chạy không lỗi nếu không tìm thấy file
    transactions = pd.DataFrame()
    articles = pd.DataFrame()
    customers = pd.DataFrame()


print("Dữ liệu đã tải xong.")
print(f"Transactions: {transactions.shape[0]:,} dòng")
print(f"Articles: {articles.shape[0]:,} dòng")
print(f"Customers: {customers.shape[0]:,} dòng")


# =============================================================================
# === BƯỚC 2: CÁC HÀM XÂY DỰNG BẢN ĐỒ RECALL ===
# =============================================================================

def build_transactional_item_map(transactions_df, recent_days=30, top_k=50):
    print(f"\n--- Bắt đầu xây dựng Bản đồ Giao dịch (dữ liệu {recent_days} ngày cuối) ---")
    max_date = transactions_df['t_dat'].max()
    min_date = max_date - timedelta(days=recent_days)
    recent_transactions = transactions_df[transactions_df['t_dat'] >= min_date]
    df = recent_transactions[['customer_id', 'article_id']]
    merged_df = pd.merge(df, df, on='customer_id', suffixes=('_left', '_right'))
    pairs = merged_df[merged_df['article_id_left'] != merged_df['article_id_right']]
    pair_counts = pairs.groupby(['article_id_left', 'article_id_right']).size().reset_index(name='count')
    item_map = defaultdict(list)
    pair_counts_sorted = pair_counts.sort_values('count', ascending=False)
    for row in tqdm(pair_counts_sorted.itertuples(index=False), total=len(pair_counts_sorted), desc="Tạo map giao dịch"):
        source_item, related_item = row.article_id_left, row.article_id_right
        if len(item_map[source_item]) < top_k:
            item_map[source_item].append(related_item)
    print("Xây dựng Bản đồ Giao dịch hoàn tất.")
    return dict(item_map)

def build_name_similarity_map(articles_df, top_k=50):
    print("\n--- Bắt đầu xây dựng Bản đồ Tương đồng Tên (TF-IDF) ---")
    vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2, 5))
    tfidf_matrix = vectorizer.fit_transform(articles_df['prod_name'].fillna(''))
    cosine_sim_sparse = tfidf_matrix * tfidf_matrix.T
    item_map = {}
    article_ids = articles_df['article_id'].tolist()
    for idx in tqdm(range(len(article_ids)), desc="Tạo map tên SP"):
        row = cosine_sim_sparse[idx].toarray().ravel()
        top_indices = np.argpartition(row, -(top_k + 1))[-(top_k + 1):]
        sorted_top_indices = top_indices[np.argsort(row[top_indices])][::-1]
        final_top_indices = [i for i in sorted_top_indices if i != idx][:top_k]
        item_map[article_ids[idx]] = [article_ids[i] for i in final_top_indices]
    print("Xây dựng Bản đồ Tương đồng Tên hoàn tất.")
    return item_map

def build_embedding_similarity_map(articles_df, top_k=50):
    print("\n--- Bắt đầu xây dựng Bản đồ Tương đồng Embedding ---")
    model = SentenceTransformer('all-MiniLM-L6-v2', device='cuda')
    articles_df['full_desc'] = articles_df.apply(lambda row: ' '.join([str(row['prod_name']), str(row['product_type_name']), str(row['product_group_name']), str(row['detail_desc'])]).replace('nan', ''), axis=1)
    embeddings = model.encode(articles_df['full_desc'].tolist(), show_progress_bar=True)
    cosine_sim = cosine_similarity(embeddings, embeddings)
    item_map = {}
    article_ids = articles_df['article_id'].tolist()
    for idx in tqdm(range(len(article_ids)), desc="Tạo map embedding"):
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        top_indices = [i[0] for i in sim_scores[1:top_k+1]]
        item_map[article_ids[idx]] = [article_ids[i] for i in top_indices]
    print("Xây dựng Bản đồ Tương đồng Embedding hoàn tất.")
    return item_map

# =============================================================================
# === BƯỚC 3: XÂY DỰNG CÁC BẢN ĐỒ RECALL THỰC TẾ ===
# =============================================================================

# Chạy trên toàn bộ dữ liệu giao dịch
transaction_map = build_transactional_item_map(transactions)

# Chạy trên mẫu lớn hơn để có kết quả tốt hơn, nhưng vẫn đảm bảo thời gian chạy
# Bạn có thể tăng/giảm con số này tùy vào cấu hình máy
SAMPLE_SIZE = 40000 
articles_sample = articles.head(SAMPLE_SIZE)

name_sim_map = build_name_similarity_map(articles_sample)
embedding_sim_map = build_embedding_similarity_map(articles_sample)


# =============================================================================
# === BƯỚC 4: HÀM TẠO FILE SUBMISSION ===
# =============================================================================

def generate_recall_file(target_customers_df, transactions_df, recall_maps: list, output_filename: str, num_recent_items=5):
    print(f"\n--- Bắt đầu tạo file '{output_filename}' ---")
    
    all_predictions = []
    customer_ids = target_customers_df['customer_id'].unique()
    
    # Tạo map lịch sử mua hàng để tra cứu nhanh
    customer_history_map = transactions_df.groupby('customer_id')['article_id'].apply(list).to_dict()
    # Tạo map lịch sử mua hàng gần đây để tra cứu nhanh
    recent_transactions_df = transactions_df[transactions_df['t_dat'] >= (transactions_df['t_dat'].max() - timedelta(days=30))]
    customer_recent_history_map = recent_transactions_df.groupby('customer_id')['article_id'].apply(lambda x: list(dict.fromkeys(x))).to_dict()

    for customer_id in tqdm(customer_ids, desc=f"Tạo file {output_filename}"):
        
        purchased_items = set(customer_history_map.get(customer_id, []))
        recent_items = customer_recent_history_map.get(customer_id, [])[-num_recent_items:] # Lấy N item cuối cùng
        
        candidate_set = set()
        
        if len(recent_items) > 0:
            for item_id in recent_items:
                for recall_map in recall_maps:
                    candidate_set.update(recall_map.get(item_id, []))
        
        final_candidates = list(candidate_set - purchased_items)
        top_12_preds = final_candidates[:12]
        
        # Thêm các sản phẩm phổ biến nhất để lấp đầy nếu không đủ 12
        if len(top_12_preds) < 12:
            # (Tùy chọn) Có thể thêm logic lấp đầy ở đây, ví dụ: các sản phẩm phổ biến nhất
            pass

        prediction_str = ' '.join(top_12_preds)
        
        all_predictions.append({'customer_id': customer_id, 'prediction': prediction_str})
        
    submission_df = pd.DataFrame(all_predictions)
    # Lưu file vào thư mục output của Kaggle
    submission_df.to_csv(f"/kaggle/working/{output_filename}", index=False)
    
    print(f"Đã tạo thành công file '/kaggle/working/{output_filename}'.")
    print("Xem trước 5 dòng đầu:")
    print(submission_df.head())


# =============================================================================
# === BƯỚC 5: TẠO 2 FILE RECALL RIÊNG BIỆT ===
# =============================================================================

# Lấy danh sách tất cả khách hàng cần dự đoán từ file sample_submission
try:
    sample_submission = pd.read_csv('/kaggle/input/h-and-m-personalized-fashion-recommendations/sample_submission.csv')
    target_customers = sample_submission
except FileNotFoundError:
    print("Không tìm thấy sample_submission.csv, sẽ dùng toàn bộ khách hàng trong customers.csv")
    target_customers = customers


# --- TẠO FILE 1: CHỈ DỰA TRÊN GIAO DỊCH ---
recall_maps_transactional = [transaction_map]
generate_recall_file(
    target_customers,
    transactions,
    recall_maps_transactional,
    "transactional_recall_submission.csv"
)

# --- TẠO FILE 2: CHỈ DỰA TRÊN NỘI DUNG ---
recall_maps_content = [name_sim_map, embedding_sim_map]
generate_recall_file(
    target_customers,
    transactions,
    recall_maps_content,
    "content_based_recall_submission.csv"
)

print("\nHoàn tất! Đã tạo ra 2 file recall trong thư mục /kaggle/working/.")

