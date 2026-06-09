# import pandas as pd


# path = '/kaggle/input/drw-quatro-2/'

# #df1 = pd.read_csv(path+ 'Means_5_files - 07-06-2025.csv' ) # Top.5 - 07.06, Lb=0.10727
# #df2 = pd.read_csv(path+ 'Means_5_files - 11-06-2025.csv' ) # Top.5 - 11.06, Lb=0.11026
# #df3 = pd.read_csv(path+ 'Means_4_files (40-30-20-10).csv') # Top.4 - 12.06, Lb=0.11427
# df4 = pd.read_csv(path+ 'Means_4_files (25-25-25-25).csv') # Top.4 - 12.06, Lb=0.11471

# df = df4

# df.to_csv("submission.csv", index=False)

# df


import os
import csv
from pathlib import Path

# è®¾ç½®ç›®å½•å’Œæ–‡ä»¶è·¯å¾„
# csv_file1 = '/kaggle/input/results/submission-11033.csv'
# csv_file2 = '/kaggle/input/results/submission-11275.csv'
# csv_file3 = '/kaggle/input/results/submission-11356.csv'
# csv_file4 = '/kaggle/input/results/submission-11379.csv'
# csv_file5 = '/kaggle/input/results/submission-11471.csv'
csv_file4 = '/kaggle/input/results/submission-11914.csv'
csv_file5 = '/kaggle/input/drw-quatro-2/17-06-2025--2_public_subm__schema_weights_(998-002).csv'
csv_file6 = '/kaggle/input/results/submission-11591.csv'
output_file = 'submission.csv'

# è�·å�–ç›®å½•ä¸‹æ‰€æœ‰çš„CSVæ–‡ä»¶
csv_files = [csv_file4, csv_file5, csv_file6]

# å�ªå¤„ç�†å‰�4ä¸ªCSVæ–‡ä»¶
csv_files = csv_files[:3]
weights = [0.9, 0.05, 0.05]

# å­˜å‚¨æ‰€æœ‰æ–‡ä»¶çš„ç¬¬äºŒåˆ—æ•°æ�®
all_second_columns = []
header = None

# è¯»å�–æ¯�ä¸ªCSVæ–‡ä»¶çš„ç¬¬äºŒåˆ—
for file_path in csv_files:
    with open(file_path, 'r', newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        # è¯»å�–æ ‡é¢˜è¡Œ(å�ªè¯»å�–ä¸€æ¬¡)
        if header is None:
            header = next(reader)
        else:
            next(reader)  # è·³è¿‡æ ‡é¢˜è¡Œ

        # è¯»å�–ç¬¬äºŒåˆ—æ•°æ�®
        second_column = [row[1] for row in reader if len(row) > 1]
        all_second_columns.append(second_column)

# ç¡®ä¿�æ‰€æœ‰æ–‡ä»¶çš„ç¬¬äºŒåˆ—é•¿åº¦ç›¸å�Œ
lengths = [len(col) for col in all_second_columns]
if len(set(lengths)) > 1:
    print("è­¦å‘Šï¼šCSVæ–‡ä»¶çš„ç¬¬äºŒåˆ—è¡Œæ•°ä¸�ä¸€è‡´ï¼Œå°†ä»¥æœ€çŸ­çš„ä¸ºå‡†")
    min_length = min(lengths)
    all_second_columns = [col[:min_length] for col in all_second_columns]

# è®¡ç®—å¹³å�‡å€¼
weighted_averages = []
for i in range(len(all_second_columns[0])):
    weighted_sum = 0.0
    for j in range(len(all_second_columns)):
        try:
            value = float(all_second_columns[j][i])
            weighted_sum += value * weights[j]
        except ValueError:
            weighted_sum += 0.0  # å¦‚æ�œè½¬æ�¢å¤±è´¥ï¼Œä½¿ç”¨0ä½œä¸ºé»˜è®¤å€¼
    
    weighted_averages.append(weighted_sum)

# è¯»å�–ç¬¬ä¸€ä¸ªæ–‡ä»¶çš„ç»“æ�„ä½œä¸ºæ¨¡æ�¿
template_file = csv_files[0]
with open(template_file, 'r', newline='', encoding='utf-8') as csvfile:
    reader = csv.reader(csvfile)
    template_rows = list(reader)

# åˆ›å»ºç»“æ�œè¡Œ - ä½¿ç”¨ç¬¬ä¸€ä¸ªæ–‡ä»¶çš„ç»“æ�„ï¼Œä½†æ›¿æ�¢ç¬¬äºŒåˆ—
result_rows = []
result_rows.append(template_rows[0])  # æ·»åŠ æ ‡é¢˜è¡Œ

for i in range(len(weighted_averages)):
    if i + 1 < len(template_rows):  # ç¡®ä¿�ä¸�è¶…å‡ºèŒƒå›´
        new_row = template_rows[i + 1].copy()
        if len(new_row) > 1:
            new_row[1] = str(weighted_averages[i])
        result_rows.append(new_row)
    else:
        # å¦‚æ�œæ¨¡æ�¿æ–‡ä»¶è¡Œæ•°ä¸�è¶³ï¼Œåˆ›å»ºæ–°è¡Œ
        new_row = [''] * len(header)
        new_row[1] = str(weighted_averages[i])
        result_rows.append(new_row)

# å†™å…¥ç»“æ�œæ–‡ä»¶
with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(result_rows)

print(f"å¤„ç�†å®Œæˆ�ï¼Œç»“æ�œå·²ä¿�å­˜åˆ° {output_file}")


result_rows[:5]


# Arhiv

# Next 5 public notebook - 08.06.2025 - their average - LB =

# - 0.10152 - v22 - [Ensemble](https://www.kaggle.com/code/migrantworkerdatahub/ensemble) - [Migrant Worker Data Hub](https://www.kaggle.com/migrantworkerdatahub)
# - 0.10145 - v01 - [ğŸ§ª Experimental](https://www.kaggle.com/code/sadettinamilverdil/experimental/notebook) - [Sadettin Å�amil Verdil](https://www.kaggle.com/sadettinamilverdil)
# - 0.09891 - v12 - [DRW Remix II](https://www.kaggle.com/code/taylorsamarel/drw-remix-ii/notebook) - [Taylor S. Amarel](https://www.kaggle.com/taylorsamarel)
# - 0.10001 - v05 - [Remix of Top Notebooks](https://www.kaggle.com/code/charleskluis/remix-of-top-notebooks/notebook) - [Charles K Luis](https://www.kaggle.com/charleskluis)
# - 0.09852 - v01 - [ğŸ¤·â€�â™‚ï¸� YatÄ±rÄ±m Tavsiyesi DeÄŸildir](https://www.kaggle.com/code/sadettinamilverdil/yat-r-m-tavsiyesi-de-ildir/notebook) - [Sadettin Å�amil Verdil](https://www.kaggle.com/sadettinamilverdil)

# Next 5 public notebook - 08.06.2025 - their average - LB =

# - 0.09852 - v02 - [Custom Greedy Recursive Feature Addition](https://www.kaggle.com/code/verifyitisyou/custom-greedy-recursive-feature-addition) - []()
# - 0.09837 - v15 - [DRW Crypto Market Prediction](https://www.kaggle.com/code/ayushchandramaurya/drw-crypto-market-prediction) - []()
# - 0.09819 - v02 - []() - []()
# - 0.09727 - v01 - [DRW Remix VII](https://www.kaggle.com/code/taylorsamarel/drw-remix-vii) - []()
# - 0.09609 - v01 - []() - []() 

# Next 5 public notebook - 08.06.2025 - their average - LB =

