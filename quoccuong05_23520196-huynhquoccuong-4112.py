# Cài đặt các thư viện cần thiết
!pip install -U scikit-learn imbalanced-learn

# Nhập các thư viện
import pandas as pd
from IPython.display import FileLink, display

# Hàm tiền xử lý dữ liệu
def preprocess_data(df: pd.DataFrame,
                    order_date_col: str = 'Order date',
                    vsd_col: str = 'VSD') -> pd.DataFrame:
    """
    - Chuyển đổi cột ngày thành định dạng datetime.
    - Loại bỏ các hàng có ngày không hợp lệ.
    """
    df = df.copy()
    df[order_date_col] = pd.to_datetime(df[order_date_col], errors='coerce')
    df[vsd_col]        = pd.to_datetime(df[vsd_col], errors='coerce')
    df = df.dropna(subset=[order_date_col, vsd_col])
    return df

# Hàm gắn nhãn trễ giao hàng
def assign_delay_label(df: pd.DataFrame,
                       order_date_col: str = 'Order date',
                       vsd_col: str = 'VSD',
                       threshold: int = 7) -> pd.DataFrame:
    """
    - Tính delay_days = VSD - Order date.
    - Gắn nhãn: 1 nếu delay_days > ngưỡng, ngược lại là 0.
    """
    df = df.copy()
    df['delay_days'] = (df[vsd_col] - df[order_date_col]).dt.days
    df['label']      = (df['delay_days'] > threshold).astype(int)
    return df

# Đọc dữ liệu từ thư mục đầu vào của Kaggle
input_path = '/kaggle/input/pilot-10/PILOT_10.csv'
df_raw = pd.read_csv(input_path)

# Kiểm tra trùng lặp ID trong dữ liệu gốc
if df_raw['ID'].duplicated().any():
    print("Cảnh báo: Tìm thấy ID trùng lặp trong dữ liệu đầu vào.")
else:
    print("Tất cả ID trong dữ liệu đầu vào là duy nhất.")

# Tiền xử lý và gắn nhãn dữ liệu
df_clean = preprocess_data(df_raw, order_date_col='Order date', vsd_col='VSD')
print(f"Số hàng sau khi tiền xử lý: {len(df_clean)}")

df_labeled = assign_delay_label(df_clean, order_date_col='Order date', vsd_col='VSD', threshold=7)

# Chuẩn bị DataFrame đầu ra với cột 'ID' duy nhất
out = pd.DataFrame({
    'ID': df_labeled['ID'],  # Sử dụng cột 'ID' duy nhất
    'label': df_labeled['label']
})

# Kiểm tra trùng lặp ID trong đầu ra
if out['ID'].duplicated().any():
    print("Cảnh báo: Tìm thấy ID trùng lặp sau khi xử lý.")
else:
    print("Không có ID trùng lặp sau khi xử lý.")

# Kiểm tra số hàng
expected_rows = 125516
if len(out) != expected_rows:
    print(f"Lỗi: Đầu ra có {len(out)} hàng, kỳ vọng là {expected_rows}.")
else:
    print(f"Đầu ra có {len(out)} hàng - khớp với yêu cầu.")

# Lưu file đầu ra vào thư mục làm việc của Kaggle
output_path = '/kaggle/working/PILOT_10_labeled.csv'
out.to_csv(output_path, index=False)
print(f"File đã được lưu: {output_path}")


# Hiển thị vài hàng đầu tiên để kiểm tra
print(out.head())


from IPython.display import FileLink, display
display(FileLink('PILOT_10_labeled.csv'))


