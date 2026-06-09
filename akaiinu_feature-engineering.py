# =============================================================================
# === H&M ADVANCED FEATURE ENGINEERING ===
# =============================================================================

import pandas as pd
import numpy as np
import gc
import os
import warnings
from datetime import timedelta

warnings.filterwarnings('ignore')

# Cấu hình hiển thị
pd.set_option('display.max_columns', None)

class HMFeatureEngineering:
    def __init__(self, path, output_dir='/kaggle/working/'):
        self.path = path
        self.output_dir = output_dir
        self.df_final = None # Bảng kết quả cuối cùng
        
    def load_data(self):
        print("--- [1] Loading Data ---")
        # 1. Transactions: Chỉ load dữ liệu gần đây (ví dụ 5 tuần cuối) để tính Feature động
        # Lý do: Hành vi mua sắm cách đây 2 năm thường không còn giá trị dự báo cho hiện tại
        print("Loading Transactions (Last 5 weeks)...")
        df_trans = pd.read_csv(self.path + 'transactions_train.csv', 
                               dtype={'article_id': str, 'sales_channel_id': 'int8'},
                               parse_dates=['t_dat'])
        
        max_date = df_trans['t_dat'].max()
        # Lấy 5 tuần dữ liệu: 4 tuần để tính lịch sử (lags), 1 tuần cuối để làm Target Train
        start_date = max_date - timedelta(days=35)
        self.trans = df_trans[df_trans['t_dat'] >= start_date].copy()
        
        # Thêm cột 'week' đếm ngược (0 là tuần mới nhất)
        self.trans['week'] = (max_date - self.trans['t_dat']).dt.days // 7
        
        # 2. Articles & Customers
        print("Loading Articles & Customers...")
        self.articles = pd.read_csv(self.path + 'articles.csv', dtype={'article_id': str})
        self.customers = pd.read_csv(self.path + 'customers.csv')
        
        del df_trans; gc.collect()

    def create_article_features(self):
        """
        Tạo các đặc trưng về SẢN PHẨM (Items)
        """
        print("\n--- [2] Engineering Article Features ---")
        
        # === A. Static Features (Đặc trưng tĩnh từ bảng Articles) ===
        # Vai trò: Giúp model phân loại hàng hóa (đồ nữ, đồ trẻ em, màu đen, màu đỏ...)
        cols_to_use = ['article_id', 'product_group_name', 'graphical_appearance_name', 
                       'colour_group_name', 'section_name', 'index_group_name']
        
        df_items = self.articles[cols_to_use].copy()
        
        # Label Encoding: Chuyển text sang số để model đọc được
        for col in cols_to_use[1:]:
            df_items[col] = df_items[col].astype('category').cat.codes
            
        # === B. Dynamic Features (Đặc trưng động từ Transactions) ===
        
        # Feature 1: Recent Popularity (Độ phổ biến tuần gần nhất)
        # Vai trò: Model học được "Hàng Hot Trend". Món nào bán nhiều tuần 1 thì khả năng cao bán tiếp tuần 0.
        pop_1w = self.trans[self.trans['week']==1].groupby('article_id').size().reset_index(name='pop_1w')
        df_items = df_items.merge(pop_1w, on='article_id', how='left').fillna(0)
        
        # Feature 2: Popularity Trend (Xu hướng tăng/giảm)
        # Vai trò: Phân biệt hàng đang lên ngôi (Trend dương) và hàng hết thời (Trend âm).
        pop_2w = self.trans[self.trans['week']==2].groupby('article_id').size().reset_index(name='pop_2w')
        df_items = df_items.merge(pop_2w, on='article_id', how='left').fillna(0)
        
        # Công thức: (Tuần 1 - Tuần 2) / (Tuần 2 + 1)
        df_items['sales_trend'] = (df_items['pop_1w'] - df_items['pop_2w']) / (df_items['pop_2w'] + 1)
        
        # Feature 3: Average Price (Giá trung bình của món hàng)
        # Vai trò: Xác định phân khúc sản phẩm (Cao cấp hay Bình dân).
        avg_price = self.trans.groupby('article_id')['price'].mean().reset_index(name='item_avg_price')
        df_items = df_items.merge(avg_price, on='article_id', how='left')
        
        self.df_items = df_items
        print(f"Article Features Created: {df_items.shape}")

    def create_customer_features(self):
        """
        Tạo các đặc trưng về KHÁCH HÀNG (Users) - ĐÃ SỬA LỖI
        """
        print("\n--- [3] Engineering Customer Features ---")
        
        df_users = self.customers[['customer_id', 'age', 'club_member_status', 'FN', 'Active']].copy()
        
        # Xử lý Missing Values
        df_users['FN'] = df_users['FN'].fillna(0)
        df_users['Active'] = df_users['Active'].fillna(0)
        
        # Fill NA tuổi bằng trung bình, sau đó ép kiểu int để sạch dữ liệu
        mean_age = df_users['age'].mean()
        df_users['age'] = df_users['age'].fillna(mean_age).astype(int)
        
        # Feature 4: Age Group (Nhóm tuổi)
        # FIX LỖI: Mở rộng bins từ -1 đến 120 để bắt toàn bộ các giá trị ngoại lai (outliers)
        # Nhóm 0: <25, Nhóm 1: 25-35, Nhóm 2: 35-45, Nhóm 3: 45-60, Nhóm 4: >60
        # Sử dụng .cat.codes: Nếu có giá trị nào lọt ra ngoài, nó sẽ thành -1 (không bị crash)
        df_users['age_group'] = pd.cut(df_users['age'], 
                                       bins=[-1, 25, 35, 45, 60, 120], 
                                       labels=False) # labels=False trả về số nguyên 0,1,2,3,4 luôn
        
        # Đảm bảo không còn NaN nào sót lại (phòng hờ)
        df_users['age_group'] = df_users['age_group'].fillna(-1).astype('int8')
        
        # Feature 5: Spending Power (Sức mua trung bình)
        user_spend = self.trans[self.trans['week'] > 0].groupby('customer_id')['price'].mean().reset_index(name='user_avg_spend')
        df_users = df_users.merge(user_spend, on='customer_id', how='left')
        
        # Fill NA cho khách mới bằng giá trị trung bình toàn sàn
        global_mean = user_spend['user_avg_spend'].mean()
        df_users['user_avg_spend'] = df_users['user_avg_spend'].fillna(global_mean)
        
        self.df_users = df_users
        print(f"Customer Features Created: {df_users.shape}")

    def build_train_dataset(self):
        """
        Bước quan trọng: Ghép tất cả lại và tạo Negative Samples
        """
        print("\n--- [4] Building Training Dataset ---")
        
        # 1. Positive Samples: Các giao dịch THỰC trong tuần 0 (Tuần gần nhất)
        train_pos = self.trans[self.trans['week'] == 0][['customer_id', 'article_id']].copy()
        train_pos['target'] = 1 # Nhãn 1: Đã mua
        
        # 2. Negative Samples: Các giao dịch GIẢ (Khách KHÔNG mua)
        # Vai trò: Giúp Model học được cái gì khách KHÔNG thích. Nếu chỉ đưa toàn số 1, model sẽ học kém.
        # Chiến lược: Random 4 sản phẩm phổ biến gán cho mỗi khách hàng
        popular_items = self.trans['article_id'].value_counts().head(1000).index.tolist()
        
        # Tạo dataframe negative
        neg_inst = []
        unique_users = train_pos['customer_id'].unique()
        
        # (Code tối ưu tốc độ bằng numpy)
        neg_users = np.repeat(unique_users, 4) # Mỗi user tạo 4 mẫu âm
        neg_items = np.random.choice(popular_items, size=len(neg_users))
        
        train_neg = pd.DataFrame({'customer_id': neg_users, 'article_id': neg_items})
        train_neg['target'] = 0 # Nhãn 0: Không mua
        
        # Gộp lại
        self.df_final = pd.concat([train_pos, train_neg], ignore_index=True)
        self.df_final = self.df_final.sample(frac=1).reset_index(drop=True) # Shuffle
        
        print(f"Skeleton created. Shape: {self.df_final.shape} (Positives + Negatives)")

    def add_interaction_features(self):
        """
        Tạo đặc trưng tương tác (User x Item) - Phần quan trọng nhất
        """
        print("\n--- [5] Adding Interaction Features ---")
        
        # Merge thông tin User và Item vào bảng Train
        self.df_final = self.df_final.merge(self.df_users, on='customer_id', how='left')
        self.df_final = self.df_final.merge(self.df_items, on='article_id', how='left')
        
        # Feature 6: Price Sensitivity (Độ lệch giá)
        # Công thức: Giá món hàng - Sức mua trung bình của khách
        # Vai trò: Nếu dương quá lớn (Món hàng đắt hơn nhiều so với thói quen), khả năng mua thấp.
        self.df_final['price_diff'] = self.df_final['item_avg_price'] - self.df_final['user_avg_spend']
        
        # Feature 7: Age Affinity (Độ hợp tuổi)
        # (Tính offline trước: Độ tuổi trung bình của người mua món hàng X là bao nhiêu?)
        item_target_age = self.trans.groupby('article_id').apply(
            lambda x: x.merge(self.customers[['customer_id', 'age']], on='customer_id')['age'].mean()
        ).reset_index(name='item_target_age')
        
        self.df_final = self.df_final.merge(item_target_age, on='article_id', how='left')
        
        # Công thức: |Tuổi khách - Tuổi trung bình của người mua món đó|
        # Vai trò: Nếu khách 50 tuổi, món hàng toàn người 20 tuổi mua -> Khả năng mua thấp (trừ khi mua cho con).
        self.df_final['age_diff'] = abs(self.df_final['age'] - self.df_final['item_target_age'])

    def export(self, filename='rich_features_train.csv'):
        print(f"\n--- [6] Exporting to CSV ---")
        save_path = os.path.join(self.output_dir, filename)
        
        # Chọn các cột quan trọng nhất để xuất
        cols = [
            'customer_id', 'article_id', 'target', # Key info
            'pop_1w', 'sales_trend', # Trend info
            'product_group_name', 'colour_group_name', # Item Categories
            'age', 'user_avg_spend', 'FN', # User info
            'price_diff', 'age_diff' # Interactions (Crucial)
        ]
        
        # Clean up NaNs created by merges
        self.df_final = self.df_final.fillna(0)
        
        self.df_final[cols].to_csv(save_path, index=False)
        print(f"File saved: {save_path}")
        print(f"Rows: {len(self.df_final)}")
        print("Feature Descriptions for Model:")
        print("1. pop_1w: Độ phổ biến tuần gần nhất (Bắt trend).")
        print("2. sales_trend: Tốc độ tăng trưởng doanh số (Phân biệt hàng Hot/Lỗi mốt).")
        print("3. price_diff: Sự chênh lệch giữa giá sản phẩm và thói quen chi tiêu của khách.")
        print("4. age_diff: Sự phù hợp giữa tuổi khách và độ tuổi mục tiêu của sản phẩm.")

# =============================================================================
# RUN
# =============================================================================
if __name__ == "__main__":
    PATH = '/kaggle/input/h-and-m-personalized-fashion-recommendations/'
    
    fe = HMFeatureEngineering(PATH)
    fe.load_data()
    fe.create_article_features()
    fe.create_customer_features()
    fe.build_train_dataset()
    fe.add_interaction_features()
    fe.export('hm_rich_features_train.csv')

