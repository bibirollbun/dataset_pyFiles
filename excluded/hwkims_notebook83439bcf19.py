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


import pandas as pd
import os

# --- 설정 ---
# Kaggle Notebook 환경에서는 데이터가 보통 /kaggle/input/ 폴더 하위에 마운트됩니다.
# 대회 데이터 폴더 이름을 확인하세요 (예: 'heart-disease-prediction-dataquest').
competition_data_dir = '/kaggle/input/heart-disease-prediction-dataquest'
sample_submission_filename = 'sample_submission.csv'
output_submission_filename = 'submission.csv' # Kaggle이 제출 파일로 인식하는 기본 이름

# --- 파일 경로 조합 ---
sample_submission_path = os.path.join(competition_data_dir, sample_submission_filename)

# --- 샘플 제출 파일 읽기 ---
try:
    print(f"'{sample_submission_path}' 파일을 읽는 중...")
    # sample_submission.csv 파일을 pandas DataFrame으로 읽어옵니다.
    submission_df = pd.read_csv(sample_submission_path)
    print("파일 읽기 완료.")
    print("\n샘플 제출 파일 내용 (상위 5개 행):")
    print(submission_df.head())

    # --- 제출 파일 생성 ---
    # 읽어온 DataFrame을 'submission.csv' 이름으로 저장합니다.
    # index=False 옵션: DataFrame의 인덱스를 CSV 파일에 포함하지 않도록 합니다.
    #                  Kaggle 제출 형식은 보통 인덱스 열이 필요 없습니다.
    print(f"\n'{output_submission_filename}' 이름으로 제출 파일 생성 중...")
    submission_df.to_csv(output_submission_filename, index=False)
    print(f"'{output_submission_filename}' 파일 생성 완료!")
    print("이 노트북을 커밋(Commit)한 후, Output 섹션에서 생성된 파일을 제출할 수 있습니다.")

except FileNotFoundError:
    print(f"오류: '{sample_submission_path}' 파일을 찾을 수 없습니다.")
    print("Kaggle 데이터셋이 노트북에 올바르게 추가되었는지 확인하세요.")
    print("데이터셋 추가 방법: 노트북 편집 화면 우측의 'Add Data' 버튼 클릭 -> 'Competition Data' 검색 및 추가")

except Exception as e:
    print(f"파일 처리 중 오류 발생: {e}")

