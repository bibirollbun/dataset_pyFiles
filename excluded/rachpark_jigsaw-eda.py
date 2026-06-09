import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정 (Windows)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False


IS_KAGGLE = True
TRAIN_DATA = "/kaggle/input/jigsaw-agile-community-rules/train.csv"


def load_data():
    """데이터 로드 및 기본 정보 확인"""
    print("=== 1. 데이터 로드 및 기본 정보 확인 ===")
    
    # 데이터 로드
    df = pd.read_csv(TRAIN_DATA)
    print(f"데이터셋 크기: {df.shape}")
    print(f"컬럼 수: {df.shape[1]}")
    print(f"행 수: {df.shape[0]}")
    print()
    
    # 데이터 기본 정보 확인
    print("=== 데이터 기본 정보 ===")
    display(df.info())
    print()
    
    # 처음 5행 확인
    print("=== 처음 5행 ===")
    display(df.head())
    print()
    
    # 컬럼명 확인
    print("=== 컬럼명 ===")
    for i, col in enumerate(df.columns):
        print(f"{i+1}. {col}")
    print()
    
    return df


def basic_statistics(df):
    """기본 통계 분석"""
    print("=== 2. 데이터 통계 분석 ===")
    
    # 기술 통계
    print("=== 기술 통계 ===")
    display(df.describe())
    print()
    
    # 결측값 확인
    print("=== 결측값 확인 ===")
    missing_data = df.isnull().sum()
    missing_percent = (missing_data / len(df)) * 100
    missing_df = pd.DataFrame({
        'Missing Values': missing_data,
        'Missing Percentage': missing_percent
    })
    print(missing_df)
    print()
    
    # 중복값 확인
    print(f"중복된 행 수: {df.duplicated().sum()}")
    print(f"전체 행 대비 중복 비율: {(df.duplicated().sum() / len(df)) * 100:.2f}%")
    print()


def target_analysis(df):
    """타겟 변수 분석 (rule_violation)"""
    print("=== 3. 타겟 변수 분석 (rule_violation) ===")
    
    # 타겟 변수 분포 확인
    print("=== 규칙 위반 분포 ===")
    violation_counts = df['rule_violation'].value_counts()
    violation_percentages = df['rule_violation'].value_counts(normalize=True) * 100
    
    print(f"규칙 위반 (1): {violation_counts[1]} ({violation_percentages[1]:.2f}%)")
    print(f"규칙 준수 (0): {violation_counts[0]} ({violation_percentages[0]:.2f}%)")
    print(f"전체: {len(df)}")
    print()
    
    # 타겟 변수 분포 시각화
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    df['rule_violation'].value_counts().plot(kind='bar', color=['lightblue', 'lightcoral'])
    plt.title('Rule Violation Frequency')
    plt.xlabel('Rule Violation')
    plt.ylabel('Frequency')
    plt.xticks([0, 1], ['Not Violate (0)', 'Violate (1)'], rotation=0)
    
    plt.subplot(1, 2, 2)
    plt.pie(violation_counts.values, labels=['Not Violate (0)', 'Violate (1)'], 
            autopct='%1.1f%%', colors=['lightblue', 'lightcoral'])
    plt.title('Rule Violation Ratio')
    
    plt.tight_layout()
    # plt.savefig('rule_violation_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()
    # print("그래프가 'rule_violation_distribution.png'로 저장되었습니다.")
    print()


def rule_analysis(df):
    """규칙(rule) 분석"""
    print("=== 4. 규칙(rule) 분석 ===")
    
    # 규칙 종류 분석
    print("=== 규칙 종류 분석 ===")
    rule_counts = df['rule'].value_counts()
    print(f"총 규칙 종류: {len(rule_counts)}")
    print("\n상위 10개 규칙:")
    print(rule_counts.head(10))
    print()
    
    # 규칙별 위반률 분석
    rule_violation_rate = df.groupby('rule')['rule_violation'].agg(['count', 'sum', 'mean']).round(3)
    rule_violation_rate.columns = ['총 게시물', '위반 수', '위반률']
    rule_violation_rate = rule_violation_rate.sort_values('위반률', ascending=False)
    
    print("=== 규칙별 위반률 (상위 10개) ===")
    print(rule_violation_rate.head(10))
    print()
    
    # 상위 10개 규칙의 위반률 시각화
    top_rules = rule_violation_rate.head(10)
    
    plt.figure(figsize=(15, 8))
    bars = plt.bar(range(len(top_rules)), top_rules['위반률'], 
                   color=['red' if x > 0.5 else 'orange' if x > 0.3 else 'green' for x in top_rules['위반률']])
    
    plt.title('Violation Ratio for Top 10 Rules', fontsize=16)
    plt.xlabel('Rule')
    plt.ylabel('Violation Ratio')
    plt.xticks(range(len(top_rules)), [rule[:50] + '...' if len(rule) > 50 else rule for rule in top_rules.index], 
               rotation=45, ha='right')
    plt.ylim(0, 1)
    
    # 값 표시
    for i, (bar, rate) in enumerate(zip(bars, top_rules['위반률'])):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                 f'{rate:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    # plt.savefig('top_rules_violation_rate.png', dpi=300, bbox_inches='tight')
    plt.show()
    # print("그래프가 'top_rules_violation_rate.png'로 저장되었습니다.")
    print()


def subreddit_analysis(df):
    """서브레딧(subreddit) 분석"""
    print("=== 5. 서브레딧(subreddit) 분석 ===")
    
    # 서브레딧 분석
    print("=== 서브레딧 분석 ===")
    subreddit_counts = df['subreddit'].value_counts()
    print(f"총 서브레딧 수: {len(subreddit_counts)}")
    print("\n상위 20개 서브레딧:")
    print(subreddit_counts.head(20))
    print()
    
    # 서브레딧별 위반률 분석
    subreddit_violation_rate = df.groupby('subreddit')['rule_violation'].agg(['count', 'sum', 'mean']).round(3)
    subreddit_violation_rate.columns = ['총 게시물', '위반 수', '위반률']
    subreddit_violation_rate = subreddit_violation_rate.sort_values('위반률', ascending=False)
    
    print("=== 서브레딧별 위반률 (상위 20개) ===")
    print(subreddit_violation_rate.head(20))
    print()
    
    # 상위 20개 서브레딧의 위반률 시각화
    top_subreddits = subreddit_violation_rate.head(20)
    
    plt.figure(figsize=(16, 10))
    bars = plt.bar(range(len(top_subreddits)), top_subreddits['위반률'], 
                   color=['red' if x > 0.7 else 'orange' if x > 0.5 else 'yellow' if x > 0.3 else 'green' 
                          for x in top_subreddits['위반률']])
    
    plt.title('Violation Ratio of Top 20 Subreddit', fontsize=16)
    plt.xlabel('Subreddit')
    plt.ylabel('Violation Ratio')
    plt.xticks(range(len(top_subreddits)), top_subreddits.index, rotation=45, ha='right')
    plt.ylim(0, 1)
    
    # 값 표시
    for i, (bar, rate) in enumerate(zip(bars, top_subreddits['위반률'])):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                 f'{rate:.3f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    # plt.savefig('top_subreddits_violation_rate.png', dpi=300, bbox_inches='tight')
    plt.show()
    # print("그래프가 'top_subreddits_violation_rate.png'로 저장되었습니다.")
    print()


def text_analysis(df):
    """텍스트 데이터 분석 (body)"""
    print("=== 6. 텍스트 데이터 분석 (body) ===")
    
    # 텍스트 길이 분석
    df['body_length'] = df['body'].str.len()
    df['body_word_count'] = df['body'].str.split().str.len()
    
    print("=== 텍스트 길이 통계 ===")
    print(f"평균 문자 수: {df['body_length'].mean():.2f}")
    print(f"평균 단어 수: {df['body_word_count'].mean():.2f}")
    print(f"최대 문자 수: {df['body_length'].max()}")
    print(f"최대 단어 수: {df['body_word_count'].max()}")
    print(f"최소 문자 수: {df['body_length'].min()}")
    print(f"최소 단어 수: {df['body_word_count'].min()}")
    print()
    
    # 텍스트 길이 분포 시각화
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 문자 수 분포
    axes[0, 0].hist(df['body_length'], bins=50, alpha=0.7, color='skyblue')
    axes[0, 0].set_title('Character Length Frequency')
    axes[0, 0].set_xlabel('Charactor Length')
    axes[0, 0].set_ylabel('Frequency')
    
    # 단어 수 분포
    axes[0, 1].hist(df['body_word_count'], bins=50, alpha=0.7, color='lightgreen')
    axes[0, 1].set_title('Word Length Frequency')
    axes[0, 1].set_xlabel('Word Length')
    axes[0, 1].set_ylabel('Frequency')
    
    # 규칙 위반별 문자 수 분포
    df[df['rule_violation'] == 0]['body_length'].hist(ax=axes[1, 0], bins=30, alpha=0.7, 
                                                       color='lightblue', label='규칙 준수')
    df[df['rule_violation'] == 1]['body_length'].hist(ax=axes[1, 0], bins=30, alpha=0.7, 
                                                       color='lightcoral', label='규칙 위반')
    axes[1, 0].set_title('Character Length Frequency by Violation')
    axes[1, 0].set_xlabel('Charactor Length')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].legend()
    
    # 규칙 위반별 단어 수 분포
    df[df['rule_violation'] == 1]['body_word_count'].hist(ax=axes[1, 1], bins=30, alpha=0.7, 
                                                           color='lightcoral', label='규칙 위반')
    df[df['rule_violation'] == 0]['body_word_count'].hist(ax=axes[1, 1], bins=30, alpha=0.7, 
                                                           color='lightblue', label='규칙 준수')
    axes[1, 1].set_title('Word Length Frequency by Violation')
    axes[1, 1].set_xlabel('Word Length')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].legend()
    
    plt.tight_layout()
    # plt.savefig('text_length_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()
    # print("그래프가 'text_length_distribution.png'로 저장되었습니다.")
    print()
    
    # 텍스트 길이와 규칙 위반의 상관관계
    print("=== 텍스트 길이와 규칙 위반 상관관계 ===")
    correlation_length = df['body_length'].corr(df['rule_violation'])
    correlation_words = df['body_word_count'].corr(df['rule_violation'])
    
    print(f"문자 수와 규칙 위반 상관계수: {correlation_length:.4f}")
    print(f"단어 수와 규칙 위반 상관계수: {correlation_words:.4f}")
    
    # 규칙 위반별 평균 텍스트 길이
    print("\n=== 규칙 위반별 평균 텍스트 길이 ===")
    violation_length_stats = df.groupby('rule_violation').agg({
        'body_length': ['mean', 'std'],
        'body_word_count': ['mean', 'std']
    }).round(2)
    print(violation_length_stats)
    print()


def example_analysis(df):
    """예시 데이터 분석"""
    print("=== 7. 예시 데이터 분석 ===")
    
    # 규칙 위반 예시 확인
    print("=== 규칙 위반 예시 ===")
    violation_examples = df[df['rule_violation'] == 1][['body', 'rule', 'subreddit']].head(5)
    for idx, row in violation_examples.iterrows():
        print(f"\n--- 예시 {idx} ---")
        print(f"서브레딧: {row['subreddit']}")
        print(f"규칙: {row['rule']}")
        print(f"내용: {row['body'][:200]}..." if len(row['body']) > 200 else f"내용: {row['body']}")
    print()
    
    # 규칙 준수 예시 확인
    print("=== 규칙 준수 예시 ===")
    compliance_examples = df[df['rule_violation'] == 0][['body', 'rule', 'subreddit']].head(5)
    for idx, row in compliance_examples.iterrows():
        print(f"\n--- 예시 {idx} ---")
        print(f"서브레딧: {row['subreddit']}")
        print(f"규칙: {row['rule']}")
        print(f"내용: {row['body'][:200]}..." if len(row['body']) > 200 else f"내용: {row['body']}")
    print()


def summary_and_insights(df):
    """데이터 품질 및 인사이트 요약"""
    print("=== 8. 데이터 품질 및 인사이트 요약 ===")
    
    # 데이터 품질 요약
    print("=== 데이터 품질 요약 ===")
    print(f"1. 데이터셋 크기: {df.shape[0]}행 x {df.shape[1]}열")
    print(f"2. 결측값: {df.isnull().sum().sum()}개")
    print(f"3. 중복값: {df.duplicated().sum()}개")
    print(f"4. 규칙 위반 비율: {(df['rule_violation'].sum() / len(df)) * 100:.2f}%")
    print(f"5. 고유 규칙 수: {df['rule'].nunique()}개")
    print(f"6. 고유 서브레딧 수: {df['subreddit'].nunique()}개")
    print(f"7. 평균 텍스트 길이: {df['body_length'].mean():.1f}자")
    print(f"8. 평균 단어 수: {df['body_word_count'].mean():.1f}개")
    print()


def main():
    """메인 함수"""
    print("Reddit Community Rules Classification - Train Data Analysis")
    print("=" * 60)
    
    try:
        # 1. 데이터 로드
        df = load_data()
        
        # 2. 기본 통계
        basic_statistics(df)
        
        # 3. 타겟 변수 분석
        target_analysis(df)
        
        # 4. 규칙 분석
        rule_analysis(df)
        
        # 5. 서브레딧 분석
        subreddit_analysis(df)
        
        # 6. 텍스트 분석
        text_analysis(df)
        
        # 7. 예시 데이터 분석
        example_analysis(df)
        
        # 8. 요약 및 인사이트
        summary_and_insights(df)
        
    except Exception as e:
        print(f"오류가 발생했습니다: {e}")
        import traceback
        traceback.print_exc()


main()

