# =============================================================================
# === NOTEBOOK #2: RECALL GENERATOR (CLASS VERSION) ===
# =============================================================================

import pandas as pd
import numpy as np
import pickle
import gc
import os
import warnings
from datetime import timedelta
from collections import defaultdict
from tqdm.auto import tqdm

warnings.filterwarnings('ignore')

class HMRecallGenerator:
    def __init__(self, trans_path, sub_path, maps_dir, output_dir='/kaggle/working/'):
        """
        Khởi tạo Generator.
        :param trans_path: Đường dẫn file transactions_train.csv
        :param sub_path: Đường dẫn file sample_submission.csv (chứa list khách hàng cần predict)
        :param maps_dir: Thư mục chứa các file .pkl (Input)
        :param output_dir: Thư mục lưu file .csv (Output)
        """
        self.trans_path = trans_path
        self.sub_path = sub_path
        self.maps_dir = maps_dir
        self.output_dir = output_dir
        
        # Các biến dữ liệu
        self.transactions = None
        self.target_customers = None
        
        # Các biến cache lịch sử (để không phải tính lại nhiều lần)
        self.full_history_map = None # Set các đồ đã mua (để lọc)
        self.recent_history_map = None # List đồ mới mua (để làm seed)
        
        # Kho chứa các maps
        self.loaded_maps = {}

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def load_base_data(self, recent_days=30):
        """Bước 1: Load giao dịch và chuẩn bị lịch sử khách hàng"""
        print("\n--- [1] Đang tải và xử lý dữ liệu gốc ---")
        
        # 1. Load Transactions
        self.transactions = pd.read_csv(self.trans_path, dtype={'article_id': str}, parse_dates=['t_dat'])
        print(f"-> Đã tải {len(self.transactions):,} dòng giao dịch.")
        
        # 2. Load Target Customers
        self.target_customers = pd.read_csv(self.sub_path)
        customer_ids = self.target_customers['customer_id'].unique()
        print(f"-> Số lượng khách hàng cần dự đoán: {len(customer_ids):,}")

        # 3. Pre-compute History (Tối ưu tốc độ)
        print("-> Đang tạo cache lịch sử mua sắm (Seed Items & Filter Items)...")
        
        # a. Full history (để loại bỏ hàng đã mua nếu muốn, hoặc dùng cho logic khác)
        # Dùng set để tra cứu cực nhanh O(1)
        self.full_history_map = self.transactions.groupby('customer_id')['article_id'].apply(set).to_dict()
        
        # b. Recent history (Lấy làm input đầu vào cho gợi ý)
        max_date = self.transactions['t_dat'].max()
        cutoff_date = max_date - timedelta(days=recent_days)
        recent_trans = self.transactions[self.transactions['t_dat'] >= cutoff_date]
        
        # Giữ thứ tự thời gian, loại bỏ trùng lặp (dict.fromkeys)
        self.recent_history_map = recent_trans.groupby('customer_id')['article_id'].apply(lambda x: list(dict.fromkeys(x))).to_dict()
        
        print("-> Hoàn tất chuẩn bị dữ liệu.")
        # Có thể xóa self.transactions nếu RAM quá yếu, nhưng giữ lại để an toàn
        # del self.transactions; gc.collect()

    def load_recall_maps(self, map_filenames):
        """Bước 2: Tải các file .pkl vào bộ nhớ"""
        print(f"\n--- [2] Đang tải các bản đồ gợi ý từ {self.maps_dir} ---")
        
        for name, filename in map_filenames.items():
            path = os.path.join(self.maps_dir, filename)
            if os.path.exists(path):
                print(f"-> Loading: {filename} ...")
                with open(path, 'rb') as f:
                    self.loaded_maps[name] = pickle.load(f)
                print(f"   Done. Size: {len(self.loaded_maps[name])} items.")
            else:
                print(f"!!! CẢNH BÁO: Không tìm thấy file {filename}")

    def _generate_candidates(self, customer_ids, used_map_keys, recent_item_count=50):
        """Hàm nội bộ: Logic cốt lõi tạo candidates"""
        candidates_list = []
        
        # Chỉ sử dụng các map được yêu cầu
        active_maps = {k: v for k, v in self.loaded_maps.items() if k in used_map_keys}
        
        if not active_maps:
            print("!!! Lỗi: Không có map nào được chọn để chạy.")
            return []

        for cust_id in tqdm(customer_ids, desc="Generating Candidates"):
            # 1. Lấy Seed Items (Hàng mới mua)
            seed_items = self.recent_history_map.get(cust_id, [])[-recent_item_count:]
            
            candidate_scores = defaultdict(float)
            
            # 2. Tìm gợi ý từ các Maps
            if seed_items:
                for item in seed_items:
                    for map_name, brain_map in active_maps.items():
                        # brain_map structure: {item_id: [(rec_item, score), ...]}
                        suggestions = brain_map.get(item, [])
                        for rec_item, score in suggestions:
                            # Chiến thuật MAX SCORE: Nếu 1 món được gợi ý bởi nhiều nguồn, lấy điểm cao nhất
                            if score > candidate_scores[rec_item]:
                                candidate_scores[rec_item] = score
            
            # 3. Lọc bỏ hàng đã mua (Optional - Tùy chiến lược, ở đây giữ nguyên logic cũ là lọc)
            owned_items = self.full_history_map.get(cust_id, set())
            
            for art_id, score in candidate_scores.items():
                if art_id not in owned_items:
                    candidates_list.append({
                        'customer_id': cust_id,
                        'article_id': art_id,
                        'score': score
                    })
        
        return candidates_list

    def export_csv(self, map_keys, output_filename):
        """Bước 3: Tạo file CSV từ các map được chọn"""
        print(f"\n--- [3] Exporting: {output_filename} ---")
        print(f"-> Sử dụng các nguồn: {map_keys}")
        
        cust_ids = self.target_customers['customer_id'].unique()
        results = self._generate_candidates(cust_ids, map_keys)
        
        df_output = pd.DataFrame(results)
        save_path = os.path.join(self.output_dir, output_filename)
        df_output.to_csv(save_path, index=False)
        
        print(f"-> Đã lưu: {save_path}")
        print(f"-> Số lượng candidates: {len(df_output):,}")
        
        # Giải phóng bộ nhớ list tạm
        del results, df_output
        gc.collect()

    def clear_memory(self):
        """Hàm dọn dẹp bộ nhớ thủ công"""
        self.loaded_maps = {}
        gc.collect()
        print("-> Đã giải phóng bộ nhớ Maps.")

# =============================================================================
# === CHẠY PIPELINE ===
# =============================================================================

if __name__ == "__main__":
    # Cấu hình đường dẫn
    BASE_PATH = '/kaggle/input/h-and-m-personalized-fashion-recommendations/'
    MAPS_PATH = '/kaggle/input/my-recall-maps/' # <-- Đảm bảo bạn đã Add Data này vào notebook
    
    # 1. Khởi tạo
    generator = HMRecallGenerator(
        trans_path=BASE_PATH + 'transactions_train.csv',
        sub_path=BASE_PATH + 'sample_submission.csv',
        maps_dir=MAPS_PATH
    )
    
    # 2. Load dữ liệu cơ bản (chạy 1 lần)
    generator.load_base_data()
    
    # 3. Load tất cả các maps (hoặc load từng phần nếu RAM yếu)
    map_files = {
        'trans': 'transaction_map_with_scores.pkl', # Hoặc 'transaction_map_decay.pkl' nếu bạn dùng code mới
        'name': 'name_sim_map_with_scores.pkl',
        'emb': 'embedding_sim_map_with_scores.pkl'
    }
    generator.load_recall_maps(map_files)
    
    # 4. Tạo các file Output
    
    # File 1: Chỉ Transaction
    generator.export_csv(
        map_keys=['trans'], 
        output_filename='raw_recall_transactional_with_scores.csv'
    )
    
    # File 2: Chỉ Content (Name + Embedding)
    generator.export_csv(
        map_keys=['name', 'emb'], 
        output_filename='raw_recall_content_based_with_scores.csv'
    )
    
    # File 3: Kết hợp tất cả (Combined)
    generator.export_csv(
        map_keys=['trans', 'name', 'emb'], 
        output_filename='raw_recall_combined_with_scores.csv'
    )
    
    print("\n=== HOÀN TẤT ===")

