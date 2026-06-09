import pandas as pd
import numpy as np
import os

def ensemble_submissions(file_paths, output_dir='.', weights=None):
    """
    將多個 submission CSV 檔案進行整合，可選擇計算加權平均。

    Args:
        file_paths (list): 一個包含所有 submission.csv 檔案路徑的列表。
        output_dir (str): 輸出結果檔案的儲存目錄。
        weights (list, optional): 一個與 file_paths 對應的權重列表。
                                  若為 None，則計算簡單平均。預設為 None。
    """
    if not file_paths:
        print("錯誤：檔案路徑列表是空的。")
        return

    # --- 新增：權重驗證 ---
    if weights is not None:
        if len(weights) != len(file_paths):
            print(f"錯誤：權重列表的長度 ({len(weights)}) 與檔案列表的長度 ({len(file_paths)}) 不符。")
            return
        print(f"開始進行加權整合，使用的權重為: {weights}")
    else:
        print(f"開始進行簡單平均整合（未提供權重）。")

    # --- 步驟 1: 讀取所有 CSV 檔案，並將 'oid_ypos' 設為索引 ---
    try:
        print("正在讀取 CSV 檔案...")
        dataframes = [pd.read_csv(f, index_col='oid_ypos') for f in file_paths]
    except FileNotFoundError as e:
        print(f"錯誤：找不到檔案 -> {e}。請檢查您的檔案路徑是否正確。")
        return
    except KeyError:
        print("錯誤：某個檔案中找不到 'oid_ypos' 欄位。請確保所有 CSV 都有此欄位。")
        return

    # --- 步驟 2: 資料對齊與堆疊 ---
    print("正在對齊資料...")
    master_index = dataframes[0].index
    master_columns = dataframes[0].columns
    aligned_data = [df.reindex(master_index).values for df in dataframes]
    data_array = np.stack(aligned_data, axis=0).transpose(1, 2, 0)
    print("資料堆疊完成。 陣列維度 (列數, 欄數, 檔案數):", data_array.shape)

    # --- 步驟 3: 計算平均值 (加權或簡單) 與中位數 ---
    
    # 計算平均值
    if weights is not None:
        print("正在計算加權平均值...")
        # 使用 np.average 進行加權計算
        mean_data = np.average(data_array, axis=2, weights=weights)
        mean_output_filename = 'ensemble_weighted_mean.csv'
    else:
        print("正在計算簡單平均值...")
        mean_data = np.mean(data_array, axis=2)
        mean_output_filename = 'ensemble_mean.csv'
        
    # 計算中位數 (中位數沒有加權的概念，維持原樣)
    print("正在計算中位數...")
    median_data = np.median(data_array, axis=2)

    # --- 步驟 4: 建立結果 DataFrame 並儲存為 CSV 檔案 ---
    # 平均值結果
    df_mean = pd.DataFrame(mean_data, index=master_index, columns=master_columns)
    mean_output_path = os.path.join(output_dir, 'submission.csv')
    df_mean.to_csv(mean_output_path)
    print(f"平均值整合結果已儲存至: {mean_output_path}")

    # # 中位數結果
    # df_median = pd.DataFrame(median_data, index=master_index, columns=master_columns)


submission_files = [
    "/kaggle/input/yale-fwi-best-submission-csv-file/CV07286_TRY8V16_TRY8V22ensTRY8V21_Top4Methods.csv",
    "/kaggle/input/best-yale-submission/submission.csv",
]


#file_weights = [0.5, 0.5]
file_weights = [0.4, 0.6]

ensemble_submissions(submission_files, weights=file_weights)















