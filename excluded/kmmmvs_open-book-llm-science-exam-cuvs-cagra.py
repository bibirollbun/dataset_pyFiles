!pip install -U /kaggle/input/faiss-cpu-173/faiss_cpu-1.7.3-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl


!pip install "huggingface_hub<0.26.0" # Install an older version of huggingface_hub


!pip install -U sentence-transformers


#!pip install -U /kaggle/input/blingfire-018/blingfire-0.1.8-py3-none-any.whl
!pip install --upgrade blingfire


#1. 라이브러리 임포트 및 환경 설정
import os
import gc
import cudf as pd
import numpy as np
import re
from tqdm.auto import tqdm
import blingfire as bf
from collections.abc import Iterable
import faiss
from faiss import write_index, read_index
from sentence_transformers import SentenceTransformer
import warnings
import cupy as cp
from cuvs.neighbors import cagra

warnings.filterwarnings("ignore")

# 설정
MODEL = '/kaggle/input/sentencetransformers-allminilml6v2/sentence-transformers_all-MiniLM-L6-v2'
DEVICE = 0
MAX_LENGTH = 384
BATCH_SIZE = 16
WIKI_PATH = "/kaggle/input/wikipedia-20230701"
NUM_SENTENCES_INCLUDE = 3


# 텍스트 문장 분할 함수 정의
def process_documents(documents: Iterable[str],
                      document_ids: Iterable,
                      split_sentences: bool = True,
                      filter_len: int = 3,
                      disable_progress_bar: bool = False) -> pd.DataFrame:
    """
    EMR에서 문서를 처리하는 주요 도우미 함수입니다.

    :param documents: 문자열인 문서를 포함하는 반복 가능 객체
    :param document_ids: 문서 고유 식별자를 포함하는 반복 가능 객체
    :param split_sentences: 섹션을 문장으로 더 분할할지 여부를 결정하는 플래그
    :param filter_len: 문장의 최소 문자 길이(그렇지 않으면 필터링)
    :param disable_progress_bar: tqdm 진행률 표시줄을 비활성화하는 플래그
    :return: `document_id`, `text`, `section`, `offset` 열을 포함하는 Pandas DataFrame
    """

    df = sectionize_documents(documents, document_ids, disable_progress_bar)

    if split_sentences:
        # cuDF Series를 Python 리스트로 변환하여 sentencize에 전달합니다.
        df = sentencize(df.text.to_arrow().to_pylist(), 
                          df.document_id.to_arrow().to_pylist(),
                          df.offset.to_arrow().to_pylist(),
                          filter_len,
                          disable_progress_bar)
    return df


def sectionize_documents(documents: Iterable[str],
                           document_ids: Iterable,
                           disable_progress_bar: bool = False) -> pd.DataFrame:
    """
    이미징 보고서의 섹션을 가져오고 선택한 섹션만 반환합니다(기본값은 FINDINGS, IMPRESSION 및 ADDENDUM).

    :param documents: 문자열인 문서를 포함하는 반복 가능 객체
    :param document_ids: 문서 고유 식별자를 포함하는 반복 가능 객체
    :param disable_progress_bar: tqdm 진행률 표시줄을 비활성화하는 플래그
    :return: `document_id`, `text`, `offset` 열을 포함하는 Pandas DataFrame
    """
    processed_documents = []
    for document_id, document in tqdm(zip(document_ids, documents), total=len(documents), disable=disable_progress_bar):
        row = {}
        text, start, end = (document, 0, len(document))
        row['document_id'] = document_id
        row['text'] = text
        row['offset'] = (start, end)

        processed_documents.append(row)

    _df = pd.DataFrame(processed_documents)
    if _df.shape[0] > 0:
        return _df.sort_values(['document_id', 'offset']).reset_index(drop=True)
    else:
        return _df


def sentencize(documents: Iterable[str],
                document_ids: Iterable,
                offsets: Iterable[tuple[int, int]],
                filter_len: int = 3,
                disable_progress_bar: bool = False) -> pd.DataFrame:
    """
    문서를 문장으로 분할합니다. `sectionize_documents`와 함께 사용하여 문서를 더 관리하기 쉬운 조각으로 분할할 수 있습니다.
    분할 후 문장을 원본 문서의 위치와 일치시킬 수 있도록 오프셋을 사용합니다.

    :param documents: 문자열인 문서를 포함하는 반복 가능 객체
    :param document_ids: 문서 고유 식별자를 포함하는 반복 가능 객체
    :param offsets: 시작 및 끝 인덱스의 반복 가능 튜플
    :param filter_len: 문장의 최소 문자 길이(그렇지 않으면 필터링)
    :return: `document_id`, `text`, `section`, `offset` 열을 포함하는 Pandas DataFrame
    """

    document_sentences = []
    for document, document_id, offset in tqdm(zip(documents, document_ids, offsets), total=len(documents), disable=disable_progress_bar):
        try:
            _, sentence_offsets = bf.text_to_sentences_and_offsets(document)
            for o in sentence_offsets:
                if o[1]-o[0] > filter_len:
                    sentence = document[o[0]:o[1]]
                    abs_offsets = (o[0]+offset[0], o[1]+offset[0])
                    row = {}
                    row['document_id'] = document_id
                    row['text'] = sentence
                    row['offset'] = abs_offsets
                    document_sentences.append(row)
        except:
            continue
    return pd.DataFrame(document_sentences)


#2. 데이터 로딩 (위키피디아, 훈련 데이터)
# 데이터 로드
wiki_files = os.listdir(WIKI_PATH)
trn = pd.read_csv("/kaggle/input/kaggle-llm-science-exam/train.csv")

# 모델 초기화
model = SentenceTransformer(MODEL, device='cuda')
model.max_seq_length = MAX_LENGTH
model = model.half()

# 위키피디아 인덱스 로드
sentence_index = read_index("/kaggle/input/wikipedia-2023-07-faiss-index/wikipedia_202307.index")

# 프롬프트 임베딩
prompt_embeddings = model.encode(trn.prompt.to_pandas().values, batch_size=BATCH_SIZE, device=DEVICE, show_progress_bar=True, convert_to_tensor=True, normalize_embeddings=True).half()
prompt_embeddings = prompt_embeddings.detach().cpu().numpy()
_ = gc.collect()

# 상위 3개 페이지 검색
search_score, search_index = sentence_index.search(prompt_embeddings, 3)

del sentence_index
del prompt_embeddings
_ = gc.collect()


#3. 프롬프트(prompt) 임베딩 생성 및 관련 문서 검색
# 위키피디아 인덱스 파일 로드
df = pd.read_parquet("/kaggle/input/wikipedia-20230701/wiki_2023_index.parquet", columns=['id', 'file'])

# 기사 및 관련 파일 위치 가져오기
wikipedia_file_data = []
for i, (scr, idx) in tqdm(enumerate(zip(search_score, search_index)), total=len(search_score)):
    scr_idx = idx
    _df = df.loc[scr_idx].copy()
    _df['prompt_id'] = i
    wikipedia_file_data.append(_df)

wikipedia_file_data = pd.concat(wikipedia_file_data).reset_index(drop=True)
wikipedia_file_data = wikipedia_file_data[['id', 'prompt_id', 'file']].drop_duplicates().sort_values(['file', 'id']).reset_index(drop=True)

del df
_ = gc.collect()


#4. 검색된 문서의 텍스트 로딩
# 전체 텍스트 데이터 가져오기
wiki_text_data = []
for file in tqdm(wikipedia_file_data.file.unique().to_arrow().to_pylist(), total=len(wikipedia_file_data.file.unique())):
    _id = [str(i) for i in wikipedia_file_data[wikipedia_file_data['file']==file]['id'].to_arrow().to_pylist()]
    _df = pd.read_parquet(f"{WIKI_PATH}/{file}", columns=['id', 'text'])
    _df = _df[_df['id'].isin(_id)]
    wiki_text_data.append(_df)
    _ = gc.collect()

wiki_text_data = pd.concat(wiki_text_data).drop_duplicates().reset_index(drop=True)
_ = gc.collect()


#5. 문서를 문장 단위로 분할하고 임베딩 생성
# 위키피디아 문서를 문장으로 분할
processed_wiki_text_data = process_documents(wiki_text_data.text.to_arrow().to_pylist(), wiki_text_data.id.to_arrow().to_pylist())

# 문장 임베딩
wiki_data_embeddings = model.encode(processed_wiki_text_data.text.to_arrow().to_pylist(), batch_size=BATCH_SIZE, device=DEVICE, show_progress_bar=True, convert_to_tensor=True, normalize_embeddings=True).half()
wiki_data_embeddings = wiki_data_embeddings.detach().cpu().numpy()
_ = gc.collect()


#6. 질문과 선택지 결합 후 임베딩 생성
# 답변 결합
trn['answer_all'] = trn['A'].str.cat([trn['B'], trn['C'], trn['D'], trn['E']], sep=" ")
trn['prompt_answer_stem'] = trn['prompt'] + " " + trn['answer_all']

# 질문 임베딩
question_embeddings = model.encode(trn['prompt_answer_stem'].to_arrow().to_pylist(), batch_size=BATCH_SIZE, device=DEVICE, show_progress_bar=True, convert_to_tensor=True, normalize_embeddings=True).half()
question_embeddings = question_embeddings.detach().cpu().numpy()



#7. GPU 기반 유사도 검색을 통한 컨텍스트 추출
# 컨텍스트 검색 및 결합
prompt_contexts = []
contexts = []

# 위키 데이터 임베딩을 GPU에 올리고 인덱스 생성
index_params = cagra.IndexParams()
wiki_data_embeddings = cp.asarray(wiki_data_embeddings, dtype=cp.float32)
wiki_index = cagra.build(index_params, wiki_data_embeddings)

trn_pd = trn.to_pandas()

for r in trn_pd.itertuples():
    prompt_context = ""
    prompt_id = r.id
    context = ""

    if prompt_id >= len(question_embeddings):
        print(f"prompt_id {prompt_id} 가 question_embeddings 범위를 벗어납니다. 건너뜁니다.")
        continue
    
    query_vector = cp.asarray(question_embeddings[prompt_id], dtype=cp.float32).reshape(1, -1)
    
    search_params = cagra.SearchParams(max_queries=100, itopk_size=64)
    
    try:
        distances, indices = cagra.search(search_params, wiki_index, query_vector, NUM_SENTENCES_INCLUDE)
    except Exception as e:
        print(f"검색 실행 오류 (prompt_id {prompt_id}): {e}")
        continue
    
    try:
        prompt_text = trn_pd.at[prompt_id, 'prompt']
        choice_a = trn_pd.at[prompt_id, 'A']
        choice_b = trn_pd.at[prompt_id, 'B']
        choice_c = trn_pd.at[prompt_id, 'C']
        choice_d = trn_pd.at[prompt_id, 'D']
        choice_e = trn_pd.at[prompt_id, 'E']
    except Exception as e:
        print(f"질문 혹은 선택지 접근 실패 (prompt_id {prompt_id}): {e}")
        continue
    
    prompt_context += f"Question: {prompt_text}\n"
    prompt_context += "Choices:\n"
    prompt_context += f"(A) {choice_a}\n"
    prompt_context += f"(B) {choice_b}\n"
    prompt_context += f"(C) {choice_c}\n"
    prompt_context += f"(D) {choice_d}\n"
    prompt_context += f"(E) {choice_e}\n"
    
    if indices.shape[0] > 0:
        prompt_context += "Context:\n"
        distances_cpu = cp.asnumpy(distances)
        indices_cpu = cp.asnumpy(indices)
        for candidate_idx, candidate_distance in zip(indices_cpu[0], distances_cpu[0]):
            if candidate_distance < 2:
                if candidate_idx < processed_wiki_text_data.shape[0]:
                    candidate_text = processed_wiki_text_data['text'].iloc[candidate_idx]
                    context += "[*] " + candidate_text + "\n"
                else:
                    print(f"prompt_id {prompt_id}: 후보 인덱스 {candidate_idx} 가 processed_wiki_text_data 범위를 벗어났습니다.")
        prompt_context += context
    
    contexts.append(context)
    prompt_contexts.append(prompt_context)


#8. 컨텍스트를 훈련 데이터에 추가하고 저장
trn['context'] = contexts
trn.to_csv("./train_context.csv", index=False)



#9. 결과 확인
# 결과 출력 (상위 10개 질문)
for i, p in enumerate(prompt_contexts[:10]):
    print(f"Question {i}")
    print(p)
    print()





