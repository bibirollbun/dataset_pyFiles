!pip install lightfm
!pip install datetime

################################################
import numpy as np
import pandas as pd
import os
################################################


# Library for EDA
import matplotlib.pyplot as plt
import seaborn as sb
from wordcloud import WordCloud


# LightFM import ###############################

from lightfm.data import Dataset
from lightfm import LightFM
from lightfm import cross_validation
from lightfm.evaluation import precision_at_k
from lightfm.evaluation import auc_score

# Import re for text cleaning ##################
import re
from datetime import datetime, timedelta


# ignore panadas warning #######################
import warnings
warnings.filterwarnings('ignore')




data_folder = "../input/data-science-for-good-careervillage/"
for file in os.listdir(data_folder):
    print(file)


df_professionals = pd.read_csv(os.path.join(data_folder, 'professionals.csv'), parse_dates=['professionals_date_joined'])
df_tag_users = pd.read_csv(os.path.join(data_folder, 'tag_users.csv'))
df_students = pd.read_csv(os.path.join(data_folder, 'students.csv'), parse_dates=['students_date_joined'])
df_tag_questions = pd.read_csv(os.path.join(data_folder, 'tag_questions.csv'))
df_groups = pd.read_csv(os.path.join(data_folder, 'groups.csv'))
df_emails = pd.read_csv(os.path.join(data_folder, 'emails.csv'))
df_group_memberships = pd.read_csv(os.path.join(data_folder, 'group_memberships.csv'))
df_answers = pd.read_csv(os.path.join(data_folder, 'answers.csv'), parse_dates=['answers_date_added'])
df_comments = pd.read_csv(os.path.join(data_folder, 'comments.csv'))
df_matches = pd.read_csv(os.path.join(data_folder, 'matches.csv'))
df_tags = pd.read_csv(os.path.join(data_folder, 'tags.csv'))
df_questions = pd.read_csv(os.path.join(data_folder, 'questions.csv'), parse_dates = ['questions_date_added'])
df_school_memberships = pd.read_csv(os.path.join(data_folder, 'school_memberships.csv'))
df_answer_scores = pd.read_csv(os.path.join(data_folder, 'answer_scores.csv'))
df_question_scores = pd.read_csv(os.path.join(data_folder, 'question_scores.csv'))


all_tables = [df_professionals, df_tag_users, df_students, df_tag_questions, df_groups, df_emails, 
              df_group_memberships, df_answers, df_comments, df_matches, df_tags, df_questions, df_school_memberships,
             df_question_scores, df_answer_scores]
              
names = ['professionals', 'tag_users', 'students', 'tag_questions', 'groups', 'emails', 
         'group_memberships', 'answers', 'comments', 'matches', 'tags', 'questions', 'school_memberships',
         'question_scores', 'answer_scores']

for name, df in zip(names, all_tables):
    print(f"df_{name}: {df.columns.tolist()}")


df_answer_scores.head()


df_answers.head()


df_comments.head()


df_professionals.head()


df_matches.head()


df_tag_users.head()


df_groups.head()


df_school_memberships.head()


df_group_memberships.head()


df_emails.head()


df_questions.head()


df_tags.head()


df_tag_questions.head()


df_question_scores.head()


df_students.head()


def generate_int_id(dataframe, id_col_name):
    """
    Lightfm은 사용자 및 아이템 항목을 분석할 때 고유값 번호를 매기는데,
    현재 데이터 셋에는 고유값이 없기 때문에 새로 생성해 줘야 함.
    고유한 정수 아이디를 생성 (for users, questions and answers)

    Parameters
    ----------
    dataframe: Dataframe
        Pandas Dataframe for Users or Q&A. 
    id_col_name : String 
        New integer id's column name.
        
    Returns
    -------
    Dataframe
        Updated dataframe containing new id column 
    """
    new_dataframe=dataframe.assign(
        int_id_col_name=np.arange(len(dataframe))
        ).reset_index(drop=True)
    return new_dataframe.rename(columns={'int_id_col_name': id_col_name})



def create_features(dataframe, features_name, id_col_name):
    """
    lightfm에 투입하기 위한 feature를 생성 (변환)

    Parameters
    __________
    
    dataframe: Dataframe
        Pandas Dataframe which contains features
    
    feature_name: List
        List of feature columns name available in dataframe

    id_col_name: String
        Column name which contains id of the question or answer that the features will map to
        There are two possible values for this variable.
            1. questions_id_num
            2. professionals_id_num


    Return
    ______
    
    Pandas Series
        A pandas series containing process features
        that are ready for feed into lightfm.
        The format of each value will be
        (user_id, ['feature_1', 'feature_2', 'feature_3'])

        EX) (1, ['military', 'army', '5'])

    """

    features = dataframe[features_name].apply(
        lambda x: ','.join(x.map(str)), axis=1)
    features = features.str.split(',')
    features = list(zip(dataframe[id_col_name], features))
    return features


def generate_feature_list(dataframe, features_name):
    """
    맵핑을 위한 특성을 생성 (변환)
    특정 칼럼들을 하나의 리스트로 변환하는 것

    Generate features list for mapping 

    Parameters
    ----------
    dataframe: Dataframe
        Pandas Dataframe for Users or Q&A. 
    features_name : List
        List of feature columns name avaiable in dataframe. 
        
    Returns
    -------
    List of all features for mapping 
    """
    features = dataframe[features_name].apply(
        lambda x: ','.join(x.map(str)), axis=1)
    features = features.str.split(',')
    features = features.apply(pd.Series).stack().reset_index(drop=True)
    return features



def calculate_auc_score(lightfm_model, interactions_matrix, 
                        question_features, professional_features): 
    """
    ROC AUC 기법으로 최종 LightFM 모델의 성능 평가 
    perfect score는 1.0

    Parameters
    ----------
    lightfm_model: LightFM model 
        A fitted lightfm model 
    interactions_matrix : 
        A lightfm interactions matrix 
    question_features, professional_features: 
        Lightfm features 
        
    Returns
    -------
    String containing AUC score 
    """
    ##auc_score: lightfm 라이브러리에 있는 함수
    score = auc_score( 
        lightfm_model, interactions_matrix, 
        item_features=question_features, 
        user_features=professional_features, 
        num_threads=4).mean()
    return score


df_professionals = generate_int_id(df_professionals, 'professionals_id_num')
df_students = generate_int_id(df_students, 'students_id_num')
df_questions = generate_int_id(df_questions, 'questions_id_num')
df_answers = generate_int_id(df_answers, 'answers_id_num')


###########################
# 데이터셋 병합
###########################

# 결측치 거거
df_tags = df_tags.dropna()
df_tags['tags_tag_name'] = df_tags['tags_tag_name'].str.replace('#', '')


# tag_questions + tags name
# group all tags for each question into single rows
df_tags_question = df_tag_questions.merge(
    df_tags, how='inner',
    left_on='tag_questions_tag_id', right_on='tags_tag_id')
df_tags_question = df_tags_question.groupby(
    ['tag_questions_question_id'])['tags_tag_name'].apply(
        ','.join).reset_index()
df_tags_question = df_tags_question.rename(columns={'tags_tag_name': 'questions_tag_name'})

# tag_users + tags name 
# group all tags for each user into single rows
# rename the tag column name 
df_tags_pro = df_tag_users.merge(
    df_tags, how='inner',
    left_on='tag_users_tag_id', right_on='tags_tag_id')
df_tags_pro = df_tags_pro.groupby(
    ['tag_users_user_id'])['tags_tag_name'].apply(
        ','.join).reset_index()
df_tags_pro = df_tags_pro.rename(columns={'tags_tag_name': 'professionals_tag_name'})


# professionals & questions tags + main merge_dataset 
df_questions = df_questions.merge(
    df_tags_question, how='left',
    left_on='questions_id', right_on='tag_questions_question_id')
df_professionals = df_professionals.merge(
    df_tags_pro, how='left',
    left_on='professionals_id', right_on='tag_users_user_id')

# questions + scores 
df_questions = df_questions.merge(
    df_question_scores, how='left',
    left_on='questions_id', right_on='id')
# merge questions with students 
df_questions = df_questions.merge(
    df_students, how='left',
    left_on='questions_author_id', right_on='students_id')



# (answers + questions) + (professionals & questions score) 
df_merge = df_answers.merge(
    df_questions, how='inner',
    left_on='answers_question_id', right_on='questions_id')
df_merge = df_merge.merge(
    df_professionals, how='inner',
    left_on='answers_author_id', right_on='professionals_id')
df_merge = df_merge.merge(
    df_question_scores, how='inner',
    left_on='questions_id', right_on='id')


#######################
# 가중치 계산을 위한 features 생성
# 인터랙션 매트릭스에 사용

df_merge['num_of_ans_by_professional'] = df_merge.groupby(['answers_author_id'])['questions_id'].transform('count')
df_merge['num_ans_per_ques'] = df_merge.groupby(['questions_id'])['answers_id'].transform('count')
df_merge['num_tags_professional'] = df_merge['professionals_tag_name'].str.split(",").str.len()
df_merge['num_tags_question'] = df_merge['questions_tag_name'].str.split(",").str.len()


print("Maximum number of answer per question : " + str(df_merge['num_ans_per_ques'].max()))
print("Maximum number of tags per professional : " + str(df_merge['num_tags_professional'].max()))
print("Maximum number of tags per question : " + str(df_merge['num_tags_question'].max()))


########################
# 전문가의 태그 데이터를 답변된 질문의 태그와 연결 
########################

# select professionals answered questions tags 
# and stored as a dataframe
professionals_prev_ans_tags = df_merge[['professionals_id', 'questions_tag_name']]

# drop null values from that 
professionals_prev_ans_tags = professionals_prev_ans_tags.dropna()

# because professsionals answers multiple questions, 
# we group all of tags of each user into single row 
professionals_prev_ans_tags = professionals_prev_ans_tags.groupby(
    ['professionals_id'])['questions_tag_name'].apply(
        ','.join).reset_index()

# drop duplicates tags from each professionals rows
professionals_prev_ans_tags['questions_tag_name'] = (
    professionals_prev_ans_tags['questions_tag_name'].str.split(',').apply(set).str.join(','))

# finally merge the dataframe with professionals dataframe 
df_professionals = df_professionals.merge(professionals_prev_ans_tags, how='left', on='professionals_id')

# join professionals tags and their answered tags 
# we replace nan values with ""
df_professionals['professional_all_tags'] = (
    df_professionals[['professionals_tag_name', 'questions_tag_name']].apply(
        lambda x: ','.join(x.dropna()),
        axis=1))


######### 결측치 제어 #########

df_questions['score'] = df_questions['score'].fillna(0)
df_questions['score'] = df_questions['score'].astype(int)
df_questions['questions_tag_name'] = df_questions['questions_tag_name'].fillna('No Tag')

# remove duplicates tags from each questions 
df_questions['questions_tag_name'] = df_questions['questions_tag_name'].str.split(',').apply(set).str.join(',')


# fill nan with 'No Tag' if any 
df_professionals['professional_all_tags'] = df_professionals['professional_all_tags'].fillna('No Tag')

# replace "" with "No Tag", because previously we replace nan with ""
df_professionals['professional_all_tags'] = df_professionals['professional_all_tags'].replace('', 'No Tag')
df_professionals['professionals_location'] = df_professionals['professionals_location'].fillna('No Location')
df_professionals['professionals_industry'] = df_professionals['professionals_industry'].fillna('No Industry')

# remove duplicates tags from each professionals 
df_professionals['professional_all_tags'] = df_professionals['professional_all_tags'].str.split(',').apply(set).str.join(',')


# remove some null values from df_merge
df_merge['num_ans_per_ques']  = df_merge['num_ans_per_ques'].fillna(0)
df_merge['num_tags_professional'] = df_merge['num_tags_professional'].fillna(0)
df_merge['num_tags_question'] = df_merge['num_tags_question'].fillna(0)


pd.set_option('display.max_columns', None)
df_merge.head()


# 맵핑을 위한 features list 생성

question_feature_list = generate_feature_list(
    df_questions,
    ['questions_tag_name'])

professional_feature_list = generate_feature_list(
    df_professionals,
    ['professional_all_tags'])


question_feature_list


professional_feature_list


# 가중치 계산 
df_merge['total_weights'] = 1 / (
    df_merge['num_ans_per_ques'])


# 가중치 투입
df_questions['question_features'] = create_features(
    df_questions, ['questions_tag_name'], 
    'questions_id_num')

df_professionals['professional_features'] = create_features(
    df_professionals,
    ['professional_all_tags'],
    'professionals_id_num')


########################
# Dataset building for lightfm
########################

# 데이터셋 변수 정의 후 LightFM 모델 매핑
# unique professionals & questions ids
# & item and professional feature

dataset = Dataset()
dataset.fit(
    set(df_professionals['professionals_id_num']), 
    set(df_questions['questions_id_num']),
    item_features=question_feature_list, 
    user_features=professional_feature_list)


#전문가와 질문 간의 상호작용 매트릭스 구축.
#전문가 ID와 질문 ID를 튜플로 전달
#예: pd.Series((pro_id, question_id), (pro_id, question_id))
#LightFM의 내장 메서드를 사용하여 인터랙션 매트릭스를 생성

    df_merge['author_question_id_tuple'] = list(zip(
    df_merge.professionals_id_num, df_merge.questions_id_num, df_merge.total_weights))

interactions, weights = dataset.build_interactions(
    df_merge['author_question_id_tuple'])



#LightFM이 이해할 수 있는 방식으로 질문과 전문가의 feature 구축
#LightFM의 내장 메서드를 사용하여 질문과 전문가의 특징 생성
    
questions_features = dataset.build_item_features(
    df_questions['question_features'])

professional_features = dataset.build_user_features(
    df_professionals['professional_features'])


#no_components : 숫자가 클 수록 더 많은 패턴을 학습할 수 있음
    # 과적합이 발생할 수 있으니 적절히 조절해야 함
#loss는 추천 시스템에서 자주 쓰이는 손실 함수를 정의하는 것
#'warp'은 실제 상호작용을 더 많이 한 항목을 우선 배치


model = LightFM(
    no_components=150,
    learning_rate=0.05,
    loss='warp',
    random_state=2019)

model.fit(
    interactions,
    item_features=questions_features,
    user_features=professional_features, sample_weight=weights,
    epochs=5, num_threads=4, verbose=True)


calculate_auc_score(model, interactions, questions_features, professional_features)



# display_html: 여러 개의 DF을 가로로 나란히 출력하게 해 줌
# display_html()을 사용하여 HTML 테이블 형식으로 출력
# *: 여러 개의 인자를 동시에 받을 수 있게 함
###############################################################

from IPython.display import display_html
def display_side_by_side(*args):
    html_str=''
    for df in args:
        html_str+=df.to_html()
    display_html(html_str.replace('table','table style="display:inline"'),raw=True)

###############################################################
# 특정 전문가에게 특정 질문을 추천하는 함수 정의


def recommend_questions(professional_ids):
     
    for professional in professional_ids:
        
        # 이전에 답변한 질문 제목을 출력 (특정 프로페서녈 아이디에 해당되는 행 중 최근 3개, 그리고 그 중 질문 아이디 값만 )
        previous_q_id_num = df_merge.loc[df_merge['professionals_id_num'] == professional][:3]['questions_id_num']
        df_previous_questions = df_questions.loc[df_questions['questions_id_num'].isin(previous_q_id_num)]

        #전문가 + 최근 질문 출력
        print('Professional Id (' + str(professional) + "): Previous Answered Questions")
        display_side_by_side(
            df_previous_questions[['questions_title', 'question_features']],
            df_professionals.loc[df_professionals.professionals_id_num == professional][['professionals_id_num','professionals_tag_name']])
        
        # 이미 답변한 질문은 제외되게끔 분류
        discard_qu_id = df_previous_questions['questions_id_num'].values.tolist()

        # discard_qu_id 목록에 포함되지 않은 질문들만 남김
        # 즉, 전문가가 이미 답변한 질문을 제외하고 새로운 질문들만 저장

        df_use_for_prediction = df_questions.loc[~df_questions['questions_id_num'].isin(discard_qu_id)]
        questions_id_for_predict = df_use_for_prediction['questions_id_num'].values.tolist()

        # 각 질문이 해당 전문가에게 적절한 정도를 나타내는 점수
        # 점수가 높을수록 추천할 확률이 높음
        # 가장 높은 점수를 받은 8개의 후보를 추천 후보로 선택
        
        scores = model.predict(
            professional,
            questions_id_for_predict,
            item_features=questions_features,
            user_features=professional_features)
        
        df_use_for_prediction['scores'] = scores
        df_use_for_prediction = df_use_for_prediction.sort_values(by='scores', ascending=False)[:8]
        print('Professional Id (' + str(professional) + "): Recommended Questions: ")
        display(df_use_for_prediction[['questions_title', 'question_features']])


recommend_questions([1200 ,19897, 3])


df_professionals = pd.read_csv(os.path.join(data_folder, 'professionals.csv'), parse_dates=['professionals_date_joined'])
df_tag_users = pd.read_csv(os.path.join(data_folder, 'tag_users.csv'))
df_students = pd.read_csv(os.path.join(data_folder, 'students.csv'), parse_dates=['students_date_joined'])
df_tag_questions = pd.read_csv(os.path.join(data_folder, 'tag_questions.csv'))
df_groups = pd.read_csv(os.path.join(data_folder, 'groups.csv'))
df_emails = pd.read_csv(os.path.join(data_folder, 'emails.csv'))
df_group_memberships = pd.read_csv(os.path.join(data_folder, 'group_memberships.csv'))
df_answers = pd.read_csv(os.path.join(data_folder, 'answers.csv'), parse_dates=['answers_date_added'])
df_comments = pd.read_csv(os.path.join(data_folder, 'comments.csv'))
df_matches = pd.read_csv(os.path.join(data_folder, 'matches.csv'))
df_tags = pd.read_csv(os.path.join(data_folder, 'tags.csv'))
df_questions = pd.read_csv(os.path.join(data_folder, 'questions.csv'), parse_dates = ['questions_date_added'])
df_school_memberships = pd.read_csv(os.path.join(data_folder, 'school_memberships.csv'))
df_answer_scores = pd.read_csv(os.path.join(data_folder, 'answer_scores.csv'))
df_question_scores = pd.read_csv(os.path.join(data_folder, 'question_scores.csv'))


########## 데이터 전처리를 위한 클래스 #############

class CareerVillageDataPreparation:
    
    
    #꼭 넣을 필요는 없지만,
    #나중에 새로운 함수를 추가하게 될 가능성이 있거나 공백을 유지해야 할 때 사용
    
    def __init__(self):
        pass

    
    # self는 클래스의 인스턴스를 나타내는 변수.
    # 클래스 내부의 메서드(함수)는 인스턴스와 연결되어 실행된다.
    # 해당 인스턴스의 속성(변수)이나 다른 메서드에 접근할 수 있게 해 주는 첫 번째 매개변수

    def _assign_unique_id(self, data, id_col_name):
        """
        Generate unique integer id for users, questions and answers

        Parameters
        ----------
        data: Dataframe
            Pandas Dataframe for Users or Q&A. 
        id_col_name : String 
            New integer id's column name.

        Returns
        -------
        Dataframe
            Updated dataframe containing new id column
        """
        new_dataframe=data.assign(
            int_id_col_name=np.arange(len(data))
            ).reset_index(drop=True)
        return new_dataframe.rename(columns={'int_id_col_name': id_col_name})

    
    #결측치를 제거하기 위한 작업
    
    def _dropna(self, data, column, axis):
        """Drop null values from specific column"""
        return data.dropna(column, axis=axis)

    # SQL처럼 교차하는 키를 가진 데이터 테이블을 상호 연결할 수 있게 함
        
    def _merge_data(self, left_data, left_key, right_data, right_key, how):
    
        """
        This function is used for merging two dataframe.
        
        Parameters
        -----------
        left_data: Dataframe
            Left side dataframe for merge
        left_key: String
            Left Dataframe merge key
        right_data: Dataframe
            Right side dataframe for merge
        right_key: String
            Right Dataframe merge key
        how: String
            Method of merge (inner, left, right, outer)
            
        
        Returns
        --------
        Dataframe
            A new dataframe merging left and right dataframe
        """
        return left_data.merge(
            right_data,
            how=how,
            left_on=left_key,
            right_on=right_key)

    
    # 여러 개의 태그를 '그룹화 하고자 하는 기준'에 맞춰서 일렬로 나열
    
    def _group_tags(self, data, group_by, tag_column):
        """Group multiple tags into single rows sepearated by comma"""
        return data.groupby(
            [group_by])[tag_column].apply(
            ','.join).reset_index()

    
    # 위의 함수를 거쳐 업데이트 된 테이블을 모두 연결
    
    def _merge_cv_datasets(
        self,
        professionals,students,
        questions,answers,
        tags,tag_questions,tag_users, questions_score):
        """
        This function merges all the necessary 
        CareerVillage dataset in defined way. 
        
        Parameters
        ------------
        professionals,students,
        questions,answers,
        tags,tag_questions,
        tag_users,
        questions_score: Dataframe
            Pandas dataframe defined by it's name
        
        
        Returns
        ---------
        questions, professionals: Dataframe
            Updated dataframe after merge
        merge: Dataframe
            A new datframe after merging answers with questions
        """ 
        
    # merge tag_questions with tags name
    # then group all tags for each question into single rows
    
        tag_question = self._merge_data(
            left_data=tag_questions,
            left_key='tag_questions_tag_id',
            right_data=tags,
            right_key='tags_tag_id',
            how='inner')
        tag_question = self._group_tags(
            data=tag_question,
            group_by='tag_questions_question_id',
            tag_column='tags_tag_name')
        
        tag_question = tag_question.rename(
            columns={'tags_tag_name': 'questions_tag_name'})

        
        # merge tag_users with tags name
        # then group all tags for each user into single rows 
        # after that rename the tag column name
        
        tags_pro = self._merge_data(
            left_data=tag_users,
            left_key='tag_users_tag_id',
            right_data=tags,
            right_key='tags_tag_id',
            how='inner')
        tags_pro = self._group_tags(
            data=tags_pro,
            group_by='tag_users_user_id',
            tag_column='tags_tag_name')
        tags_pro = tags_pro.rename(
            columns={'tags_tag_name': 'professionals_tag_name'})
        
        # merge professionals and questions tags with main merge_dataset 
        questions = self._merge_data(
            left_data=questions,
            left_key='questions_id',
            right_data=tag_question,
            right_key='tag_questions_question_id',
            how='left')
        professionals = self._merge_data(
            left_data=professionals,
            left_key='professionals_id',
            right_data=tags_pro,
            right_key='tag_users_user_id',
            how='left')
        
        # merge questions with scores 
        questions = self._merge_data(
            left_data=questions,
            left_key='questions_id',
            right_data=questions_score,
            right_key='id',
            how='left')
        
        # merge questions with students
        questions = self._merge_data(
            left_data=questions,
            left_key='questions_author_id',
            right_data=students,
            right_key='students_id',
            how='left')
        
        # merge answers with questions
        # then merge professionals and questions score with that
        merge = self._merge_data(
            left_data=answers,
            left_key='answers_question_id',
            right_data=questions,
            right_key='questions_id',
            how='inner')
        
        merge = self._merge_data(
            left_data=merge,
            left_key='answers_author_id',
            right_data=professionals,
            right_key='professionals_id',
            how='inner')
        
        return questions, professionals, merge

    
    def _drop_duplicates_tags(self, data, col_name):
        # drop duplicates tags from each row
        return (
            data[col_name].str.split(
                ',').apply(set).str.join(','))



     ########################
     # Merge professionals previous answered
     # questions tags into professionals tags
     ########################
      
    def _merge_pro_pre_ans_tags(self, professionals, merge):
         
        # select professionals answered questions tags
        # and stored as a dataframe
    
        professionals_prev_ans_tags = (
            merge[['professionals_id', 'questions_tag_name']])
        
        # drop null values from that
        professionals_prev_ans_tags = professionals_prev_ans_tags.dropna()
        
        # because professsionals answers multiple questions,
        # we group all of tags of each user into single row
        
        professionals_prev_ans_tags = self._group_tags(
            data=professionals_prev_ans_tags,
            group_by='professionals_id',
            tag_column='questions_tag_name')
        
        # drop duplicates tags from each professionals rows
        professionals_prev_ans_tags['questions_tag_name'] = \
        self._drop_duplicates_tags(
            professionals_prev_ans_tags, 'questions_tag_name')
        
        # finally merge the dataframe with professionals dataframe
       
        professionals = self._merge_data(
            left_data=professionals,
            left_key='professionals_id',
            right_data=professionals_prev_ans_tags,
            right_key='professionals_id',
            how='left')
        
        # join professionals tags and their answered tags 
        # we replace nan values with ""
        professionals['professional_all_tags'] = (
            professionals[['professionals_tag_name',
                           'questions_tag_name']].apply(
                lambda x: ','.join(x.dropna()),
                axis=1))
        return professionals

    
    def prepare(
        self,
        professionals,students,
        questions,answers,
        tags,tag_questions,tag_users, questions_score):
        
        """
        지금까지의 모든 데이터 셋을 토대로 정제 처리하는 과정
        """
        
        # assign unique integer id
        professionals = self._assign_unique_id(
            professionals, 'professionals_id_num')
        students = self._assign_unique_id(
            students, 'students_id_num')
        questions = self._assign_unique_id(
            questions, 'questions_id_num')
        answers = self._assign_unique_id(
            answers, 'answers_id_num')
        
        # just dropna from tags 
        tags = tags.dropna()
        tags['tags_tag_name'] = tags['tags_tag_name'].str.replace(
            '#', '')
        
        
        # merge necessary datasets
        df_questions, df_professionals, df_merge = self._merge_cv_datasets(
            professionals,students,
            questions,answers,
            tags,tag_questions,tag_users,
            questions_score)
        
        #######################
        # Generate some features for calculates weights
        # that will use with interaction matrix
        #######################
        df_merge['num_ans_per_ques'] = df_merge.groupby(
            ['questions_id'])['answers_id'].transform('count')
        
        # merge pro previoius answered question tags with pro tags 
        df_professionals = self._merge_pro_pre_ans_tags(
            df_professionals, df_merge)
        
        # some more pre-processing 
        # handling null values 
        df_questions['score'] = df_questions['score'].fillna(0)
        df_questions['score'] = df_questions['score'].astype(int)
        df_questions['questions_tag_name'] = \
        df_questions['questions_tag_name'].fillna('No Tag')
        
        # remove duplicates tags from each questions 
        df_questions['questions_tag_name'] = \
        df_questions['questions_tag_name'].str.split(
            ',').apply(set).str.join(',')

        # fill nan with 'No Tag' if any 
        df_professionals['professional_all_tags'] = \
        df_professionals['professional_all_tags'].fillna(
            'No Tag')
        # replace "" with "No Tag", because previously we replace nan with ""
        df_professionals['professional_all_tags'] = \
        df_professionals['professional_all_tags'].replace(
            '', 'No Tag')
        
        df_professionals['professionals_location'] = \
        df_professionals['professionals_location'].fillna(
            'No Location')
        
        df_professionals['professionals_industry'] = \
        df_professionals['professionals_industry'].fillna(
            'No Industry')

        # remove duplicates tags from each professionals
        df_professionals['professional_all_tags'] = \
        df_professionals['professional_all_tags'].str.split(
            ',').apply(set).str.join(',')

        # remove some null values from df_merge
        df_merge['num_ans_per_ques']  = \
        df_merge['num_ans_per_ques'].fillna(0)
        
        return df_questions, df_professionals, df_merge



class LightFMDataPrep:
    def __init__(self):
        pass
    def create_features(self, dataframe, features_name, id_col_name):
        """
        Generate features that will be ready for feeding into lightfm

        Parameters
        ----------
        dataframe: Dataframe
            Pandas Dataframe which contains features
        features_name : List
            List of feature columns name avaiable in dataframe
        id_col_name: String
            Column name which contains id of the question or
            answer that the features will map to.
            There are two possible values for this variable.
            1. questions_id_num
            2. professionals_id_num

        Returns
        -------
        Pandas Series
            A pandas series containing process features
            that are ready for feed into lightfm.
            The format of each value
            will be (user_id, ['feature_1', 'feature_2', 'feature_3'])
            Ex. -> (1, ['military', 'army', '5'])
        """

        features = dataframe[features_name].apply(
            lambda x: ','.join(x.map(str)), axis=1)
        features = features.str.split(',')
        features = list(zip(dataframe[id_col_name], features))
        return features



    def generate_feature_list(self, dataframe, features_name):
        """
        Generate features list for mapping 

        Parameters
        ----------
        dataframe: Dataframe
            Pandas Dataframe for Users or Q&A. 
        features_name : List
            List of feature columns name avaiable in dataframe. 

        Returns
        -------
        List of all features for mapping 
        """
        features = dataframe[features_name].apply(
            lambda x: ','.join(x.map(str)), axis=1)
        features = features.str.split(',')
        features = features.apply(pd.Series).stack().reset_index(drop=True)
        return features
    
    def create_data(self, questions, professionals, merge):
        question_feature_list = self.generate_feature_list(
            questions,
            ['questions_tag_name'])

        professional_feature_list = self.generate_feature_list(
            professionals,
            ['professional_all_tags'])
        
        merge['total_weights'] = 1 / (
            merge['num_ans_per_ques'])
        
        # creating features for feeding into lightfm 
        questions['question_features'] = self.create_features(
            questions, ['questions_tag_name'], 
            'questions_id_num')

        professionals['professional_features'] = self.create_features(
            professionals,
            ['professional_all_tags'],
            'professionals_id_num')
        
        return question_feature_list,\
    professional_feature_list,merge,questions,professionals
        
    def fit(self, questions, professionals, merge):
        ########################
        # Dataset building for lightfm
        ########################
        question_feature_list, \
        professional_feature_list,\
        merge,questions,professionals = \
        self.create_data(questions, professionals, merge)
        
        
        # define our dataset variable
        # then we feed unique professionals and questions ids
        # and item and professional feature list
        # this will create lightfm internel mapping
        dataset = Dataset()
        dataset.fit(
            set(professionals['professionals_id_num']), 
            set(questions['questions_id_num']),
            item_features=question_feature_list, 
            user_features=professional_feature_list)


        # now we are building interactions
        # matrix between professionals and quesitons
        # we are passing professional and questions id as a tuple
        # e.g -> pd.Series((pro_id, question_id), (pro_id, questin_id))
        # then we use lightfm build in method for building interactions matrix
        merge['author_question_id_tuple'] = list(zip(
            merge.professionals_id_num,
            merge.questions_id_num,
            merge.total_weights))

        interactions, weights = dataset.build_interactions(
            merge['author_question_id_tuple'])



        # now we are building our questions and
        # professionals features
        # in a way that lightfm understand.
        # we are using lightfm build in method for building
        # questions and professionals features 
        questions_features = dataset.build_item_features(
            questions['question_features'])

        professional_features = dataset.build_user_features(
            professionals['professional_features'])
        
        return interactions,\
    weights,questions_features,professional_features
        
        


# 모델을 훈련시키는 클래스

class TrainLightFM:
    def __init__(self):
        pass
        
    def train_test_split(self, interactions, weights):
        train_interactions, test_interactions = \
        cross_validation.random_train_test_split(
            interactions, 
            random_state=np.random.RandomState(2019))
        
        train_weights, test_weights = \
        cross_validation.random_train_test_split(
            weights, 
            random_state=np.random.RandomState(2019))
        return train_interactions,\
    test_interactions, train_weights, test_weights
    
    def fit(self, interactions, weights,
            questions_features, professional_features,
            cross_validation=False,no_components=150,
            learning_rate=0.05,
            loss='warp',
            random_state=2019,
            verbose=True,
            num_threads=4, epochs=5):
        ################################
        # Model building part
        ################################

        # define lightfm model by specifying hyper-parametre
        # then fit the model with ineteractions matrix,
        # item and user features
        
        model = LightFM(
            no_components,
            learning_rate,
            loss=loss,
            random_state=random_state)
        
        model.fit(
            interactions,
            item_features=questions_features,
            user_features=professional_features, sample_weight=weights,
            epochs=epochs, num_threads=num_threads, verbose=verbose)
        
        return model



class LightFMRecommendations:
    """
    Make prediction given model and professional ids
    """
    def __init__(self, lightfm_model,
                 professionals_features,
                 questions_features,
                 questions,professionals,merge):
        self.model = lightfm_model
        self.professionals_features = professionals_features
        self.questions_features = questions_features
        self.questions = questions
        self.professionals = professionals
        self.merge = merge
        
    def previous_answered_questions(self, professionals_id):
        previous_q_id_num = (
            self.merge.loc[\
                self.merge['professionals_id_num'] == \
                professionals_id]['questions_id_num'])
        
        previous_answered_questions = self.questions.loc[\
            self.questions['questions_id_num'].isin(
            previous_q_id_num)]
        return previous_answered_questions
        
    
    def _filter_question_by_pro(self, professionals_id):
        """Drop questions that professional already answer"""
        previous_answered_questions = \
        self.previous_answered_questions(professionals_id)
        
        discard_qu_id = \
        previous_answered_questions['questions_id_num'].values.tolist()
        
        questions_for_prediction = \
        self.questions.loc[~self.questions['questions_id_num'].isin(discard_qu_id)]
        
        return questions_for_prediction

    # 특정 날짜 기준으로 필터링하고자 할 때
    def _filter_question_by_date(self, questions, start_date, end_date):
        mask = \
        (questions['questions_date_added'] > start_date) & \
        (questions['questions_date_added'] <= end_date)
        
        return questions.loc[mask]
        
    
    def recommend_by_pro_id_general(self,
                                    professional_id,
                                    num_prediction=8):
        questions_for_prediction = self._filter_question_by_pro(professional_id)
        score = self.model.predict(
            professional_id,
            questions_for_prediction['questions_id_num'].values.tolist(), 
            item_features=self.questions_features,
            user_features=self.professionals_features)
        
        questions_for_prediction['recommendation_score'] = score
        questions_for_prediction = questions_for_prediction.sort_values(
            by='recommendation_score', ascending=False)[:num_prediction]
        return questions_for_prediction
    
    def recommend_by_pro_id_frequency_date_range(self,
                                                 professional_id,
                                                 start_date,
                                                 end_date,
                                                 num_prediction=8):
        questions_for_prediction = \
        self._filter_question_by_pro(professional_id)
        
        start_date = datetime.strptime(start_date, '%Y-%m-%d')
        end_date = datetime.strptime(end_date, '%Y-%m-%d')
        
        questions_for_prediction = self._filter_question_by_date(
            questions_for_prediction, start_date, end_date)
        
        score = self.model.predict(
            professional_id,
            questions_for_prediction['questions_id_num'].values.tolist(), 
            item_features=self.questions_features,
            user_features=self.professionals_features)
        
        questions_for_prediction['recommendation_score'] = score
        questions_for_prediction = questions_for_prediction.sort_values(
            by='recommendation_score', ascending=False)[:num_prediction]
        
        return questions_for_prediction



# instiate all class instance
cv_data_prep = CareerVillageDataPreparation()
light_fm_data_prep = LightFMDataPrep()
train_lightfm = TrainLightFM()

# process raw data
df_questions_p, df_professionals_p, df_merge_p = \
cv_data_prep.prepare(
    df_professionals,df_students,
    df_questions,df_answers,
    df_tags,df_tag_questions,df_tag_users,
    df_question_scores)


# prepare data for lightfm 
interactions, weights, \
questions_features, professional_features = \
light_fm_data_prep.fit(
    df_questions_p, df_professionals_p, df_merge_p)


# finally build and trian our model
model = train_lightfm.fit(interactions,
                          weights,
                          questions_features,
                          professional_features)



# define our recommender class
lightfm_recommendations = LightFMRecommendations(
    model,
    professional_features,questions_features,
    df_questions_p, df_professionals_p, df_merge_p)

# let's what our model predict for user id 3
print("Recommendation for professional: " + str(3))
display(lightfm_recommendations.recommend_by_pro_id_general(3)[:8])

