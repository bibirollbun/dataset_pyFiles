import os

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


import warnings
warnings.filterwarnings('ignore')

import plotly.graph_objects as go # 개별 그래프 객체를 직접 생성하여 세밀한 그래프 작성 (선 그래프, 바 그래프 등)
import plotly.figure_factory as ff # 여러 요소를 조합하여 특수한 그래프를 만들 때 사용함 (히트맵, 분포도, 덴드로그램 등)
import plotly.express as px #간단한 코드로 시각화

# iplot을 사용하면 대화형 그래프를 Notebook 내에서 렌더링 할 수 있음
#주피터 노트북에서 Plotly를 사용할 때 필요함. (ex: init_notebook_mode(connected=True))
from plotly.offline import init_notebook_mode, iplot 

from plotly.colors import n_colors # 컬러 스케일 생성시 사용 ('rgb(5,200,100')
from plotly.subplots import make_subplots # 여러 개의 플롯을 한 figure 내아 배치 (#fig = make_subplots(rows=1, cols=2))

from collections import Counter

import json

#IPython은 주피터 노트북 환경에서 디스플레이 기능을 제공할 수 있게 함
import IPython.display # 개체 출력
from IPython.core.display import display, HTML, Javascript # HTML, 자바스크립트 요소를 직접 노트북에 렌더링 할 수 있게 함.
from IPython.display import display, clear_output

!pip install ipywidgets
import ipywidgets as widgets # 대화형 위젯을 만들기 위한 라이브러리. 슬라이더, 버튼, 드롭다운 등 대화형 UI 생성 가능
from ipywidgets import interact, interact_manual # 버튼을 누르면 실행되게 할 수 있음


df = pd.read_csv("../input/kaggle-survey-2022/kaggle_survey_2022_responses.csv")
questions_titles = df.iloc[[0]]

df = df[1:]


questions_titles


def create_scatter_plot(
    x_axis_values,
    y_axis_values, 
    hover_template,
    marker_color, 
    marker_size, 
    title,
    subtitle,
    subtitle_explain):
    """It creates a Scatter Plot."""
    
    # Define the trace
    trace = go.Scatter(
        x=x_axis_values, 
        y=y_axis_values,
        mode='markers', 
        hovertemplate=hover_template,
        marker=dict(
            color=marker_color,
            size=marker_size,
            showscale=True, # 색상 눈금(color scale)을 나타내는 컬러바로, 마커의 색상 값이 어떻게 매핑되는지 보여주는 역할
            colorbar=dict(title="Percent"),
            opacity=0.7,
            colorscale = 'RdBu_r'
        )
    )

    # Define the layout
    layout = go.Layout(
        width=900, 
        height=950, 
        plot_bgcolor="#fff", 
        paper_bgcolor="#fff", 
        showlegend = False, 
        title = {
            'text' : f"<span style='font-size:30px; font-family:Times New Roman'>{title}</span><br><br><sup>{subtitle}</sup><br><sup>{subtitle_explain}</sup>", 
            'x':0.5,
            'xanchor': 'center'
        }, 
        font = {"color" : '#7b6b59'},
        margin = dict(t=170),
    )
    
    fig = go.Figure(data = [trace], layout = layout)
    
    fig.update_xaxes(
        showline=False,
        linewidth=1,
        linecolor='#c9c4c3',
        gridcolor='#c9c4c3',
        tickfont=dict(size=14, family='Verdana', color='#7b6b59'), 
        title="",
        title_font=dict(size=14, family='Verdana', color='#f57369'),
        showgrid=False, 
        tickangle=325
    )
    fig.update_yaxes(
        showline=False,
        linewidth=1,
        linecolor='#000',
        gridcolor='#fff',
        tickfont=dict(size=14, family='Verdana', color='#a43725'), 
        title="",
        title_font=dict(size=14, family='Verdana', color='#f57369'),
        showgrid=False
    )

    fig.show()


def get_bar_plot_trace(x_values, y_values, display_text, top_n, rest_n, hovertext, orientation="h"):
    """It creates the trace for a bar plot."""
    
    trace = go.Bar(
        y = y_values,
        x = x_values,
        name = "",
        orientation = orientation,
        marker = dict(color = ["#E6b6a4"]*rest_n +  ["#a43725"]*top_n),
        text = display_text,
        texttemplate =  "<b style='color: #fff'>%{text}% </b>",
        textposition = ["outside"]*rest_n + ["inside"]*top_n,
        hovertext=hovertext
    )
    
    return trace


def create_single_bar_plot(x_values, y_values, display_text, top_n, rest_n, hovertext, title, subtitle="", orientation="h"):
    """It creates single bar plots."""

    trace = get_bar_plot_trace(x_values, y_values, display_text, top_n, rest_n, hovertext, orientation)

    large_title_format = f"<span style='font-size:30px; font-family:Times New Roman'>{title}</span>"
    
    layout = dict(
        title = large_title_format,
        font = dict(color = '#7b6b59'),
        margin = dict(t=120),
        yaxis={'categoryorder':'array','categoryarray': x_values},
        xaxis=dict(side="top", zerolinecolor = "#4d4d4d", zerolinewidth = 0.5, gridcolor="#e7e7e7", tickformat=",.1%"),
        width = 800,
        height= 700,
        plot_bgcolor = "white"
    )

    fig = go.Figure(data = trace, layout = layout)
    fig.show()
    
    
def create_box_plot(df, x_column_name, y_column_name, title):
    """It creates bar plots."""

    fig = px.box(
        df, 
        x=x_column_name, 
        y=y_column_name,
        title=f"<span style='font-size:30px; color:#7b6b59; font-family:Times New Roman'>{title}</span>")

    layout = go.Layout(
        xaxis= {"title": ""},
        yaxis= {"title": "Compensation in USD"},
        font = dict(color = 'black'),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=800,
        width=1050
    )

    fig.update_layout(layout)
    fig.update_yaxes(showline=True, linewidth=1, gridcolor='lightgrey')
    fig.update_traces(marker_color='#b39a74')

    fig.show() 

def create_heatmap(z, x, y, annotation_text, color_scale, title, subtitle="", xlabel="", ylabel=""):
    """It creates a heatmap."""

    fig = ff.create_annotated_heatmap(z, x=x, y=y, annotation_text=annotation_text,  colorscale=color_scale)

    large_title_format = f"<span style='font-size:30px; font-family:Times New Roman'>{title}</span>"
    small_title_format = f"<span style='font-size:14px; font-family:Helvetica'>{subtitle}</b></span>"


    layout = dict(
        title = large_title_format + "<br>" + small_title_format,
        font = dict(color = '#7b6b59'),
         xaxis= {"title": xlabel},
        yaxis= {"title": ylabel},

    )

    fig['layout'].update(layout)
    fig["layout"]["xaxis"].update(side="bottom")
    fig.show()


map_ml_adoption = {
    "No (we do not use ML methods)": "Not Started" , 
    "We are exploring ML methods (and may one day put a model into production)": "Exploration Stage",
    "We use ML methods for generating insights (but do not put working models into production)": "Generating Insights", 
    "We recently started using ML methods (i.e., models in production for less than 2 years)": "Models in Production",
    "We have well established ML methods (i.e., models in production for more than 2 years)": "Models in Production",
    "I do not know": "Not Known",
    np.nan: "Not Known"
}

ml_adoption_color_discrete_map={
    "Models in Production":"#a43725", 
    "Generating Insights": "#c07156", 
    "Exploration Stage":"#E6b6a4",
    "Not Started": "#e0d5bd",
    "Not Known": "#beb29e"
}

map_ml_usage = {
    "No (we do not use ML methods)": "0. Not Started<br><sup>(No ML)</sup>" , 
    "We are exploring ML methods (and may one day put a model into production)": "1. Exploration<br><sup>Only Exploring ML</sup>",
    "We use ML methods for generating insights (but do not put working models into production)": "2. Beginner Stage<br><sup>Use ML only for Insights</sup>", 
    "We recently started using ML methods (i.e., models in production for less than 2 years)": "3. Intermediate Stage<br><sup>Recently Started Using ML</sup>",
    "We have well established ML methods (i.e., models in production for more than 2 years)": "4. Advance Stage<br><sup>Well Established ML</sup>",
    "I do not know": "Not Known",
    np.nan: "Not Known"   
}

map_company_size = {
    "0-49 employees": "1. 0-49 employees" , 
    "50-249 employees": "2. 50-249 employees",
    "250-999 employees": "3. 250-999 employees",
    "1000-9,999 employees": "4. 1000-9,999 employees",
    "10,000 or more employees": "5. 10,000 or more employees",
    np.nan: np.nan   
}

map_programming_experience = {
    "I have never written code": "1. 0 years",
    "< 1 years": "2. < 1 years",
    "1-3 years": "3. 1-3 years",
    "3-5 years": "4. 3-5 years",
    "5-10 years": "5. 5-10 years",
    "10-20 years": "6. 10-20 years",
    "20+ years": "7. 20+ years",
    np.nan: np.nan
}

map_ml_experience = {
    "I do not use machine learning methods": "1. 0 years",
    "Under 1 year": "2. < 1 years",
    "1-2 years": "3. 1-2 years",
    "2-3 years": "4. 2-3 years",
    "3-4 years": "5. 3-4 years",
    "4-5 years": "6. 4-5 years",
    "5-10 years": "7. 5-10 years",
    "10-20 years": "8. 10-20 years",
    "20+ years": "9. 20+ years",
    np.nan: np.nan
}

map_data_team_size = {
    "0": "1. 0",
    "1-2": "2. 1-2",
    "3-4": "3. 3-4",
    "5-9": "4. 5-9",
    "10-14": "5. 10-14",
    "15-19": "6. 15-19",
    "20+": "7. 20+",
    np.nan: np.nan
}


countries_df = px.data.gapminder().query("year == 2007")
countries_df["country"] = countries_df["country"].str.strip()

map_country_continent = {
    "United States of America": "Americas", 
    "United Kingdom of Great Britain and Northern Ireland": "Europe",
    "South Korea": "Asia", 
    "Russia": "Europe",
    "Viet Nam": "Asia",
    "Hong Kong (S.A.R.)": "Asia",
    "Ukraine": "Europe",
    "United Arab Emirates": "Asia",
    "Iran, Islamic Republic of...": "Asia",
    
}

def fix_map_country_continent(map_countries: dict, country:str, continent:str):
    """It maps a country to its continent"""
    if country in map_countries:
        return map_countries[country]
    
    return continent


def usage_of_a_product_or_service(question_title: str, row: pd.Series, columns_list: list) -> str:
    """
    객관식 답변이 포함된 질문 제목을 인풋으로 받고,
    응답자가 답변 중 하나 이상을 선택했는지 그 여부를 확인함
    예를 들어, 응답자가 클라우드 컴퓨팅 플랫폼을 사용하는지 확인하려면
    31번 문항에서 클라우드 컴퓨팅 플랫폼 선택지 Q31_1, Q31_2 등을 선택했는지 확인
    """
    
    for col in columns_list:  # ✅ 들여쓰기 수정 (for문이 함수 내부에 맞춰 정렬됨)
        if col.startswith(question_title):
            if not pd.isnull(row[col]) and row[col].strip().lower() != "none":
                return "Yes"
    
    # 만약 답변에 대한 모든 칼럼이 비어 있다면, 그 유저는 선택하지 않은 것
    return "No"
    

def categorize_education(education:str) -> str:
    """Assigns more general categories to education levels."""
    if education in [
        "No formal education past high school", 
        "Some college/university study without earning a bachelor’s degree"
    ]:
        return "Lower than Bachelor"
    
    if education == "Bachelor’s degree":
        return "Bachelor"
    
    if education == "Master’s degree":
        return "Master"
    
    if education in ["Doctoral degree", "Professional doctorate"]:
        return "Higher than Master"
    
    return "Other"



def extract_and_count_all_the_multiple_choice_answers(question, df):

    #주어진 질문에 대한 다중 선택 응답 수와 비율을 계산함 - ( 예: 19번 )
    # 설문 응답 데이터 - df에서 주어진 질문과 관련된 모든 칼럼을찾고,
    # 각 응답별 개수를 계산하여 비율까지 정렬된 df 반환
    
    # e.g List of choices for Question, e.g. Q19 (computer vision methods)
    choices_list = [choice for choice in df.columns if choice.startswith(question)]
    dfs_list = []
    for col in choices_list:
        dfs_list.append(df.groupby([col]).agg({"Q2" : "count"}).reset_index().rename(columns={col: question, "Q2": "counts"}))

    agg_df = pd.concat(dfs_list)
    agg_df["relative_percent"] = agg_df.apply(lambda x : (x["counts"] / df.shape[0]), axis = 1)
    agg_df = agg_df.sort_values(by=["relative_percent"], ascending=True)
    
    return agg_df

def assign_label(service:str):
    
    if "google" in service.lower():
        return "Google"
    
    if "aws" in service.lower() or "amazon" in service.lower():
        return "Amazon"
    
    if "azure" in service.lower() or "microsoft" in service.lower():
        return "Microsoft"
    
    if "ibm" in service.lower():
        return "IBM"

    return "Other"


# 주어진 질문에 대한 다중 선택 응답 개수를 계산하는 것
# 설문 응답에서 특정 질문과 관련된 모든 칼럼을 찾아 응답한 개수를 세어 반환함
def extract_the_number_of_responses(question_title: str, row: pd.Series, columns_list: list) -> str:
    
    num_responses = 0 
    for col in columns_list:
        if col.startswith(question_title): # 해당 칼럼이 질문과 관련 있는 경우
            if not pd.isnull(row[col]): # 값이 NaN 아닌 경우 (응답한 경우)
                num_responses = num_responses + 1  # 응답 개수 +1

    return num_responses

def wrap_df_text(df):
    return display(HTML(df.style.background_gradient(axis=0,  cmap='YlOrBr', subset=["Average number of selected choices"]).to_html().replace("\\n","<br>")))


# 현재 학생이 아닌 응답자 수(*Q5* 질문에 *아니오*라고 답함)
# 현재 고용되어 있는 응답자 수 (*Q23* 질문에 "현재 고용되어 있지 않다"고 답하지 않음)
# 현재 어떤 산업에 종사하고 있는지 (또는 은퇴한 경우 가장 최근의 고용주) - *Q24* 질문에 답변

scope_df = df[
    (df["Q5"] == "No") & 
    (df["Q24"].notnull()) &
    (df["Q23"] != "Currently not employed")
]

# 산업에서 머신러닝 도입 상태를 보다 일반적인 카테고리로 할당
scope_df["ML_adoption_class"] = scope_df["Q27"].apply(lambda x : map_ml_adoption[x])

# 머신러닝 도입 상태를 정렬할 수 있도록 숫자를 추가하여 표현 변경
scope_df["ML_adoption"] = scope_df["Q27"].apply(lambda x : map_ml_usage[x])

# 회사 규모를 정렬할 수 있도록 숫자를 추가하여 표현 변경
scope_df["Q25"] = scope_df["Q25"].apply(lambda x : map_company_size[x])

# 응답자가 클라우드 컴퓨팅 플랫폼을 사용했는지 확인
scope_df["Cloud_usage"] = scope_df.apply(lambda row: usage_of_a_product_or_service("Q31", row, list(scope_df.columns)), axis=1)
scope_df["NLP_methods_usage"] = scope_df.apply(lambda row: usage_of_a_product_or_service("Q20", row, list(scope_df.columns)), axis=1)
scope_df["CV_methods_usage"] = scope_df.apply(lambda row: usage_of_a_product_or_service("Q19", row, list(scope_df.columns)), axis=1)
scope_df["GPU_usage"] = scope_df.apply(lambda row: usage_of_a_product_or_service("Q42", row, list(scope_df.columns)), axis=1)

# 프로그래밍 경험을 정렬할 수 있도록 숫자를 추가하여 표현 변경
scope_df["Q11"] = scope_df["Q11"].apply(lambda x : map_programming_experience[x])

# 머신러닝 경험을 정렬할 수 있도록 숫자를 추가하여 표현 변경
scope_df["Q16"] = scope_df["Q16"].apply(lambda x : map_ml_experience[x])

# 데이터 팀 규모를 정렬할 수 있도록 숫자를 추가하여 표현 변경
scope_df["Q26"] = scope_df["Q26"].apply(lambda x : map_data_team_size[x])

# 산업별 응답자 수 집계
industry_totals = scope_df["Q24"].value_counts().to_dict()



df


scope_df


# 데이터 프레임의 전체 행 개수
df.shape[0]


mpl.rcParams.update(mpl.rcParamsDefault)

fig1 = plt.figure(figsize=(5,2),facecolor='white')

ax1 = fig1.add_subplot(1,1,1)
font = 'monospace'

ax1.text(0.9, 0.8, "Key figures",color='#7b6b59',fontsize=26, fontweight='bold', fontfamily=font, ha='center')

# 천 단위 콤마로 숫자를 구분해 주는 포맷: "{:,d}"
# df의 전체 행 개수 (총 응답자 수)
ax1.text(0, 0.4, "{:,d}".format(df.shape[0]), color='#e60000', fontsize=24, fontweight='bold', fontfamily=font, ha='center')
ax1.text(0, 0.001, "# of respondents \nin the survey",color='#757575',fontsize=15, fontweight='light', fontfamily=font,ha='center')

# scope_df에 해당되는 응답자 수
ax1.text(0.6, 0.4, "{}".format(scope_df.shape[0]), color='#e60000', fontsize=24, fontweight='bold', fontfamily=font, ha='center')
ax1.text(0.6, 0.001, "# of professionals",color='#757575',fontsize=15, fontweight='light', fontfamily=font,ha='center')


ax1.text(1.5, 0.4, "{}".format(round((scope_df.shape[0]/df.shape[0] )*100, 2))+"%", color='#e60000', fontsize=24, fontweight='bold', fontfamily=font, ha='center')
ax1.text(1.5, 0.001, "of the respondents are in the analysis \nscope",color='#757575',fontsize=15, fontweight='light', fontfamily=font, ha='center')


# X,Y축 눈금 레이블(값) 숨기기
# X,Y축 눈금(Tick) 길이를 0으로 설정 (눈금 자체를 없앰)

ax1.set_yticklabels('')
ax1.tick_params(axis='y',length=0)
ax1.tick_params(axis='x',length=0)
ax1.set_xticklabels('')

for direction in ['top','right','left','bottom']:
    ax1.spines[direction].set_visible(False)

# Matplotlib에서 서브플롯 간격과 여백을 조정하는 함수
# 그래프의 배치 및 간격을 조정하여 보기 좋게 만들기 위해 사용!
fig1.subplots_adjust(top=0.9, bottom=0.2, left=0, hspace=1)

fig1.patch.set_linewidth(4)
fig1.patch.set_edgecolor('#E6b6a4')
fig1.patch.set_facecolor('white')
ax1.set_facecolor('white')
    
plt.show()


# Collect all the multiple-choice questions

#다중선택 질문을 저장할 빈 딕셔너리 생성
multiple_choice_questions = {}

# 이미 확인한 질문을 저장할 리스트
seen_columns = []

# 칼럼명을 "_" 기준으로 나눠 질문 번호만 추출함
# Q6_1, Q6_2는 모두 Q6로 반환 / 데이터 셋의 질문 타이틀에 '_'이 포함되어 있으면 다중답변을 허용하는 질문이라는 것
# multiple_choice_questions에는 2개 이상인 경우 저장되기 때문에, seen_question에 저장된 후 한 번 더 나와야 하는 것

for col in df.columns:
    question = col.split("_")[0]
    if question in seen_columns: # 이전 반복에서 확인한 질문인지 체크
        if question not in multiple_choice_questions: # 다중답변 질문이 아니라면 체크 / 이 질문에 몇 개의 선택지가 있는지 확인하기 위함
            multiple_choice_questions[question] = 2 # 기존에 이미 한 번 등장했다면 선택지 개수를 2로 설정 
        else: # Q6_3이 등장한다면 Q3의 선택지 개수를 3으로 증가시키는 식
           multiple_choice_questions[question] = multiple_choice_questions[question] + 1
    else: # 처음 보는 질문이라면 seen_columns 리스트에 추가 (질문이 처음 등장할 경우 선택지 개수를 세지 않고 존재 여부만 기록함)
        seen_columns.append(question)

#
for col in list(multiple_choice_questions.keys()):
    scope_df[f"{col}_number_of_responses"] = scope_df.apply(
        lambda x : extract_the_number_of_responses(col,x, df.columns), axis = 1)


# 각 응답자가 다중 선택 질문에서 평균적으로 몇 개의 선택지를 골랐는지 계산
# (professional)
respondents_mean_responses = scope_df[[f"{col}_number_of_responses" for col in list(multiple_choice_questions.keys())]].mean(axis = 1).reset_index().rename(columns={0: "Mean number of responses"})
respondents_mean_responses.iloc[:,1].describe()


scope_df.index


# 평균 응답 개수가 3개 초과인 응답자 행을 필터링
# 인덱스 목록을 시리즈가 아닌 리스트로 변환 why? >>
# filter(item = )...에 들어가는 값은 인덱스 혹은 칼럼의 리스트여야 함
# axis = 0이면 인덱스 (어떠한 기준을 충족하는 행만 가져옴) axis = 1 이면 칼럼을 필터링 (특정 열만 가져옴)
outliers = scope_df.filter(items = respondents_mean_responses[respondents_mean_responses["Mean number of responses"] > 3]["index"].to_list(), axis=0)

# 머신러닝 경험(Q16)에 따라 이상치 응답자 수를 그룹화
outliers = outliers.groupby(
    ["Q16"]
).agg(
    {"Q2" : "count"} # Q16 기준으로 그룹화된 데이터를 "Q2" 칼럼에 대해 count로 집계한다는 것
).reset_index().rename(
    columns={"Q2": "Nbr of respondents", "Q16": "Years of Machine Learning Experience"}
).sort_values(by=["Years of Machine Learning Experience"])
# 각 머신러닝 경험 그룹별 응답자 수를 센다는 것임.

# 그룹별 응답 비율 계산
outliers["%"] = outliers.apply(lambda x : x["Nbr of respondents"] / outliers["Nbr of respondents"].sum(), axis = 1)
outliers["%"] = np.round(outliers["%"]* 100, 2)

# 보기 불편한 000 제거
outliers["%"] = outliers["%"].astype(str).str.rstrip('0').str.rstrip('.')

#pandas의 DataFrame.style 함수를 사용하여 데이터 프레임을 시각화
outliers.style.background_gradient(axis=0,  cmap='YlOrBr', subset=["%"])


# Q11 기준 그룹: 경력 기준

outliers = scope_df.filter(items=respondents_mean_responses[respondents_mean_responses["Mean number of responses"] > 3]["index"].to_list(), axis=0)
outliers = outliers.groupby(
    ["Q11"]
).agg(
    {"Q2" : "count"}
).reset_index().rename(
    columns={"Q2": "Nbr of respondents", "Q11": "Years of Coding Experience"}
).sort_values(by=["Years of Coding Experience"])
outliers["%"] = outliers.apply(lambda x : x["Nbr of respondents"] / outliers["Nbr of respondents"].sum(), axis = 1)
outliers["%"] = np.round(outliers["%"]* 100, 2)

outliers["%"] = outliers["%"].astype(str).str.rstrip('0').str.rstrip('.')

outliers.style.background_gradient(axis=0,  cmap='YlOrBr', subset=["%"])



# 다중 선택 질문의 평균 응답 개수 분석

outlier_analysis = []

for col in list(multiple_choice_questions.keys()):
    mean_responses = round(scope_df[f"{col}_number_of_responses"].mean())
    outlier_analysis.append([
        col,
        multiple_choice_questions[col], #해당 질문의 선택지 개수
        mean_responses, # 평균 선택 개수
    ])

# 이후 outlier_analysis 리스트에는 ["질문 번호", 선택지 개수, 평균 선택 수]에 대한 리스트가 질문별로 생성됨
outlier_analysis


average_responses = pd.DataFrame(outlier_analysis, columns = ["Question", "Nbr of available Choices", "Average number of selected choices"])
average_responses["Question Title"] = questions_titles[[f"{col}_1" for col in list(multiple_choice_questions.keys())]].loc[0].to_list()

#예: "Which ML models do you use? (Select all that apply)" → "Which ML models do you use?"
average_responses["Question Title"] = average_responses["Question Title"].apply(lambda x : x.split("(Select")[0].strip())

#Updates the DataFrame in place
scope_df.drop([f"{col}_number_of_responses" for col in list(multiple_choice_questions.keys())], axis = 1, inplace=True)

average_responses["Question Title"] = average_responses['Question Title'].str.wrap(80)
average_responses = average_responses[["Question", "Question Title", "Nbr of available Choices", "Average number of selected choices"]]

wrap_df_text(average_responses)


# 2022년 기업들의 머신러닝 도입 현황 (산업별 AI/ML 활용도 파악)

# 라벨의 회전 각도와 정렬을 정의
# numpy.rad2deg 함수는 각도의 단위를 라디안(radian)에서 도(degree) 단위로 변환
# angle: 각 라벨이 위치할 각도 - radian
# offset: 전체 차트의 회전 오프셋

# 라벨이 차트 왼쪽 (0~pi: right 정렬 & 회전 + 180도)
# 오른쪽: left 정렬
def get_label_rotation(angle, offset):
    rotation = np.rad2deg(angle + offset)
    if angle <= np.pi: #pi = 180
        alignment = "right"
        rotation = rotation + 180
    else: 
        alignment = "left"
    return rotation, alignment


# 각 막대의 끝 부분에 라벨을 추가하는 역할
#막대 그래프 끝에서 6만큼 떨어진 위치에 산업명, 비율을 라벨로 표시
def add_labels(angles, values, labels, offset, ax):
    
    padding = 6
    
    for angle, value, label, in zip(angles, values, labels):
        angle = angle
        
        rotation, alignment = get_label_rotation(angle, offset)
        y = value + padding
        # And finally add the text
        ax.text(
            x=angle, 
            y=y, 
            s=label, 
            ha=alignment, 
            va="center", 
            rotation=rotation, 
            rotation_mode="anchor",

        ) 

# 차트에 사용할 x축(각도), y축(값), 라벨 데이터를 정의
ANGLES = np.linspace(0, 2 * np.pi, scope_df["Q24"].nunique(), endpoint=False)
VALUES = np.round(scope_df["Q24"].value_counts(normalize=True).values, 2)*400
LABELS = scope_df["Q24"].value_counts().index


#막대 너비 및 차트 기본 설정
WIDTH = 2 * np.pi / len(VALUES)

#차트 생성 및 스타일링
OFFSET = np.pi / 2

fig, ax = plt.subplots(figsize=(10, 5), subplot_kw={"projection": "polar"})

ax.set_theta_offset(OFFSET)  # ✅ 차트 시작 위치를 90도로 설정
ax.set_ylim(-100, 100)  # ✅ y축 범위 설정
ax.set_frame_on(False)  # ✅ 차트 테두리 제거
ax.xaxis.grid(False)  # ✅ x축 그리드 제거
ax.yaxis.grid(False)  # ✅ y축 그리드 제거
ax.set_xticks([])  # ✅ x축 눈금 제거
ax.set_yticks([])  # ✅ y축 눈금 제거

ax.bar(
    ANGLES, VALUES, width=WIDTH, linewidth=2,
    color="#a43725", edgecolor="white")

#라벨 텍스트 추가 "산업명:X%" 형태로 표시
final_labels = []
test = list(np.round(scope_df["Q24"].value_counts(normalize=True).values*100,2))
count = 0 
for label in LABELS:
    final_labels.append(f"{label}: {test[count]}%")
    count = count+ 1
np.array(final_labels)

# Add labels
add_labels(ANGLES, VALUES, final_labels, OFFSET, ax)
plt.show()


# Bar 차트로 표기하는 경우

# ✅ 기존 데이터 사용 (산업별 AI 도입률)
# scope_df["Q24"]에는 응답자의 산업 데이터가 저장됨
# value_counts(normalize=True) * 100 → 각 산업이 차지하는 비율(%)을 계산
# value_counts에서 노멀라이저를 쓰는 이유는 각 값의 비율을 계산할 수 있게 되기 때문
industry_counts = scope_df["Q24"].value_counts(normalize=True) * 100  
industries = industry_counts.index  # 산업 이름
values = industry_counts.values  # 각 산업의 비율 값

# ✅ 데이터 정렬 (낮은 비율부터 정렬하여 barh() 적용)
df_adoption = pd.DataFrame({"Industry": industries, "Percentage": values})
df_adoption = df_adoption.sort_values(by="Percentage", ascending=True)  # 낮은 값부터 정렬 (위에서 아래로 증가하도록)

# ✅ 차트 크기 설정 & 수평 막대 그래프 그리기
fig, ax = plt.subplots(figsize=(12, 7))  # 그래프 크기 지정
ax.barh(df_adoption["Industry"], df_adoption["Percentage"], color="#a43725", edgecolor="black", alpha=0.8)

# ✅ 라벨 추가 (각 막대 끝에 "XX.XX%" 형식으로 도입률 표시)
# enumerate()를 사용하여 df_adoption["Percentage"]의 각 값과 해당 인덱스(i)를 가져옴
#  percentage → 해당 산업의 AI 도입률 (막대 길이, x 좌표) / i → 막대의 y 좌표를 의미
for i, (percentage) in enumerate(df_adoption["Percentage"]):
    ax.text(percentage + 0.5, i, f"{round(percentage, 2)}%",  # ✅ 텍스트(퍼센트 값) 추가 / 소수점 2자리까지 기기
            va='center',  # ✅ 수직 정렬 (막대의 중앙에 위치)
            fontsize=10)  # ✅ 글씨 크기와 볼드 설정

# ✅ 차트 스타일링
ax.set_xlabel("Percentage of AI/ML Adoption (%)", fontsize=14, labelpad=20)  # ✅ x축 라벨 설정
ax.set_ylabel("Industry", fontsize=14, labelpad=20)  # ✅ y축 라벨 설정
ax.set_title("2022 AI/ML Adoption by Industry", fontsize=16, fontweight='bold',pad=10)  # ✅ 제목 설정

plt.xlim(0, max(df_adoption["Percentage"])+5)  # ✅ x축 범위 설정 (최대값보다 5% 크게 설정하여 여백 추가)
plt.grid(axis='x', linestyle="--", alpha=0.4)  # ✅ x축에 점선 스타일의 그리드 추가 (가독성 향상)

# ✅ 차트 표시
plt.show()




# ✅ ML 도입 현황 데이터 집계
ml_adoption = scope_df.groupby(
    ["ML_adoption_class", "ML_adoption"]  # ✅ ML 도입 카테고리(큰 범주)와 세부 항목(세부 범주) 기준으로 그룹화
).agg(
    {"Q2": "count"}  # ✅ Q2(응답자 수)를 count하여 산업별 ML 도입률을 계산
).reset_index().rename(
    columns={"Q2": "counts"}  # ✅ Q2를 "counts"로 컬럼명 변경 (더 직관적인 표현)
)

fig = go.Figure(data=[go.Pie(
    labels = scope_df["ML_adoption_class"].value_counts().index.to_list(),  # ✅ ML 도입 클래스 (라벨)
    values = list(scope_df["ML_adoption_class"].value_counts().values),  # ✅ 해당 클래스의 응답 수 (값)
    hole=.4,  # ✅ 도넛 차트 형태로 만들기 (0.4 크기의 구멍 추가)
    textinfo="label+percent",  # ✅ 파이 조각 내부에 라벨과 퍼센트 표시
    textposition="outside",  # ✅ 텍스트 방향을 방사형으로 조정 (더 읽기 쉽게)
    outsidetextfont=dict(size=14)
)])

# ✅ 원형 차트의 색상 설정 (ML 도입 상태별로 색상 지정)
fig.update_traces(marker=dict(colors=['#a43725', '#e0d5bd', '#beb29e', '#E6b6a4', '#c07156']))

# ✅ 차트 제목 설정 (HTML을 사용해 스타일 적용)
large_title_format = "<span style='font-size:30px; font-family:Times New Roman'>The State of the ML Adoption in Industry in 2022</span>"
small_title_format = "<span style='font-size:20px; font-family:Helvetica'></b></span>"

# ✅ 차트의 레이아웃 설정
layout = dict(
    title = large_title_format + "<br>" + small_title_format,  # ✅ 제목과 부제목 설정
    font = dict(color = '#7b6b59'),  # ✅ 기본 글씨 색상 지정
    showlegend = False,  # ✅ 범례 제거 (파이에 직접 라벨이 표시되므로 불필요)
    margin = dict(t=80, pad=0),  # ✅ 위쪽 여백 설정 (제목과의 간격 조정)
)

# ✅ 차트 레이아웃 업데이트 적용
fig["layout"].update(layout)

# ✅ 첫 번째 플롯 표시
fig.show()



# ✅ Sunburst 차트 생성 (비중 추가)
fig = px.sunburst(
    ml_adoption,  # ✅ ML 도입 데이터 사용
    path=["ML_adoption_class", "ML_adoption"],  # ✅ 계층 구조: (대분류) ML 도입 카테고리 → (소분류) ML 도입 상태
    values="counts",  # ✅ 응답자 수를 값으로 사용
    color="ML_adoption_class",  # ✅ ML 도입 카테고리를 기준으로 색상 지정
    color_discrete_map=ml_adoption_color_discrete_map  # ✅ 미리 정의된 색상 매핑 적용
)

# ✅ 비중(%) 표시 추가 (각 부모 항목 대비 비율 표시)
fig.update_traces(
    textinfo="label+percent parent",  # ✅ 라벨 + 부모 항목 대비 비율(%) 표시
    texttemplate="%{label}<br>%{percentParent:.1%}"  # ✅ 줄바꿈 포함하여 비율 표시
)

# ✅ 두 번째 차트의 레이아웃 설정
layout = dict(
    showlegend = False,  # ✅ 범례 숨기기 (Sunburst 차트는 계층적 구조로 이미 정보 제공)
    margin = dict(t=80, pad=0, l=0),  # ✅ 위쪽과 왼쪽 여백 조정
)

# ✅ 차트 레이아웃 업데이트 적용
fig["layout"].update(layout)

# ✅ 두 번째 플롯 표시
fig.show()



import pandas as pd
import plotly.express as px

# 1. ml_adoption 데이터는 이미 ML 도입 현황(대분류와 세부 항목별 응답 수)를 집계한 DataFrame

# 2. 전체 응답 수 계산
total_counts = ml_adoption["counts"].sum()

# 3. ml_adoption 데이터를 복사하여 바 차트용 DataFrame 생성
df_bar = ml_adoption.copy()

# 4. 각 상위 그룹(ML_adoption_class) 내 총 응답 수 계산
df_bar["group_total"] = df_bar.groupby("ML_adoption_class")["counts"].transform("sum")

# 5. 각 상위 그룹의 전체 비중(%) 계산 : 해당 그룹의 응답 수가 전체에서 차지하는 비율
df_bar["overall_percent"] = df_bar["group_total"] / total_counts * 100

# 6. 각 하위 항목(ML_adoption) 내 비중(%) 계산 : 해당 항목이 그룹 내에서 차지하는 비율
df_bar["sub_percentage"] = df_bar["counts"] / df_bar["group_total"] * 100

# 7. 각 하위 항목의 막대 길이 계산 : 상위 그룹의 전체 비율에 하위 항목 비중을 곱한 값
# 이렇게 하면 한 상위 그룹의 모든 하위 항목 막대의 길이 합은 해당 그룹의 overall_percent와 동일함.
df_bar["scaled_sub"] = df_bar["counts"] / df_bar["group_total"] * df_bar["overall_percent"]

# 8. 누적(스택) 수평 바 차트 생성: 
fig = px.bar(
    df_bar,
    x="scaled_sub",                     # x축: 각 하위 항목의 길이 (계산된 비중)
    y="ML_adoption_class",              # y축: 상위 항목 (ML 도입 클래스)
    color="ML_adoption",                # 색상: 하위 항목 (세부 항목)
    text=df_bar["sub_percentage"].apply(lambda x: f"{x:.1f}%"),  # 하위 항목의 비중(%) 텍스트 표시
    orientation="h",                    # 가로 막대 그래프
    color_discrete_map=ml_adoption_color_discrete_map  # 미리 정의된 색상 매핑
)

# 9. 누적 바 차트 스타일 조정
fig.update_traces(
    textposition="inside",  # 텍스트를 막대 내부에 배치
    marker=dict(line=dict(width=0.5, color='black'))  # 막대 테두리 설정
)

# 10. 각 상위 항목(ML 도입 클래스)의 전체 비중(%)을 별도의 어노테이션으로 추가
annotations = []
# 상위 항목은 중복 없이 추출
for cat in df_bar["ML_adoption_class"].unique():
    # 해당 그룹의 전체 비중은 모든 행에 동일하므로 첫 번째 값 사용
    overall_pct = df_bar[df_bar["ML_adoption_class"] == cat]["overall_percent"].iloc[0]
    annotations.append(dict(
        x=overall_pct + 2,         # x: 막대 끝 오른쪽에 여백 추가하여 배치 (여기서는 전체 비중 값 + 2)
        y=cat,                     # y: 상위 항목 이름 (자동으로 해당 바의 중앙에 위치)
        text=f"<b>{overall_pct:.1f}%</b>",  # 텍스트: 상위 그룹 전체 비율(소수점 한 자리)
        showarrow=False,           # 화살표 제거
        font=dict(size=14, color="black")
    ))

max_overall = df_bar["overall_percent"].max()

# 11. 차트 레이아웃 업데이트 (제목, 축, 어노테이션 등 설정)
fig.update_layout(
    title="ML Adoption by Industry (Hierarchical Percentage Breakdown)",
    xaxis_title="Overall Percentage (%)",
    yaxis_title="ML Adoption Class",
    barmode="stack",  # 누적(스택) 바 차트
    legend_title="ML Adoption Type",
    margin=dict(t=50, l=50, r=50, b=50),
    xaxis_range=[0, max_overall + 5],
    annotations=annotations  # 상위 항목 전체 비중 어노테이션 추가
)

# 12. 차트 표시
fig.show()


# 각 산업(Q24)의 등장 횟수를 계산하여 딕셔너리로 저장
# 예: {"Tech": 500, "Finance": 300, ...}
ai_adoption_totals = scope_df["Q24"].value_counts().to_dict()

# 각 산업(Q24)과 ML 도입 단계(ML_adoption)별 응답자 수를 집계
# scope_df를 그룹화하여 Q2(응답자 수)를 count하고, reset_index()로 DataFrame 형태로 변환한 후,
# 컬럼 "Q2"의 이름을 "counts"로 변경함.
ai_adoption = scope_df.groupby(
    ["Q24", "ML_adoption"]  # ML 도입의 대분류와 소분류 기준으로 그룹화
).agg(
    {"Q2": "count"}         # 각 그룹의 응답자 수를 세기
).reset_index().rename(
    columns={"Q2": "counts"}  # 컬럼명을 "counts"로 변경 (더 직관적으로 표현)
)

# 각 행에 대해, 해당 ML 도입 단계의 응답자 수를 전체 해당 산업의 응답자 수로 나누어
# 상대 비율(relative_percent)을 계산하여 새로운 컬럼에 저장.
# lambda 함수는 각 행(x)에서 x["counts"] 값을 해당 산업의 총 응답 수(ai_adoption_totals[x["Q24"]])로 나눔.
ai_adoption["relative_percent"] = ai_adoption.apply(
    lambda x: x["counts"] / ai_adoption_totals[x["Q24"]],
    axis=1)

# "Not Known"과 같이 의미 없는 ML 도입 단계는 분석에서 제외함.
ai_adoption = ai_adoption[ai_adoption["ML_adoption"] != "Not Known"]

# hover_template: 마우스 오버 시 표시될 텍스트 형식을 지정.
# %{x} → 산업, %{y} → ML 도입 단계, %{marker.size:,} → 마커의 크기를 천 단위 콤마와 함께 표시.

hover_template = (
    "Industry: %{x}<br>"
    "AI Adoption Stage: %{y}<br>"
    "Percentage: %{marker.size:,}<extra></extra>")

# create_scatter_plot 함수 호출:
# 이 함수는 산점도를 생성하여 ML 도입 현황을 시각화하는 사용자 정의 함수(또는 미리 정의된 함수)
# 인자?
#  - x축: ai_adoption["Q24"] (산업 데이터)
#  - y축: ai_adoption["ML_adoption"].apply(lambda x: x.split(".")[-1])
#           → ML 도입 단계 문자열에서 마지막 부분(예: "Advanced", "Beginner" 등)만 추출하여 사용.
#  - hover_template: 마우스 오버 시 표시될 텍스트 형식.
#  - 마커 크기와 색상: ai_adoption['relative_percent']*100 (응답 비율을 백분율로 변환하여 사용)
#  - 제목, 데이터 설명, 그리고 인코딩(마커 크기, 색상이 응답 비율을 나타냄)에 관한 설명을 전달

create_scatter_plot(
    ai_adoption["Q24"],  # x축: 각 산업 정보
    ai_adoption["ML_adoption"].apply(lambda x: x.split(".")[-1]),  # y축: ML 도입 단계의 마지막 부분만 추출 (예: " Adivanced Stage")
    hover_template,  # 마우스 오버 시 표시될 텍스트 템플릿
    ai_adoption["relative_percent"] * 100,  # 마커 크기를 결정할 값 (백분율)
    ai_adoption["relative_percent"] * 100,  # 마커 색상을 결정할 값 (백분율)
    "The State of Machine Learning Adoption by Industry",  # 차트 제목
    "Questions Data: Industry (Q24) and ML Adoption State (Q27)",  # 데이터 출처 및 설명
    (
        "Size,Color: Percentage of Respondents - <br>"
        "The number of respondents of the related sector that chose the relevant adoption stage of their company <br>"
        "divided by the total number of respondents working in that sector."
    )  # 마커의 크기와 색상이 응답 비율을 나타냄
)



# 그룹화하여 각 기업 규모(Q25)와 ML 도입 단계(ML_adoption_class)별 응답자 수를 계산
# Q2 컬럼의 응답자 수를 count하여 "counts"라는 새로운 컬럼명으로 저장
adoption_per_company_size = scope_df.groupby(
    ["Q25", "ML_adoption_class"]  # ML 도입 단계의 대분류(예: "Models in Production", "Exploration Stage", 등)와 기업 규모(Q25)로 그룹화
).agg({
    "Q2": "count"  # 각 그룹 내 응답자 수(count)를 계산
}).reset_index().rename(columns={
    "Q2": "counts"  # 결과 컬럼 Q2를 "counts"로 이름 변경 (더 직관적)
})
# adoption_per_company_size를 확인하면, 각 기업 규모 및 ML 도입 단계별 응답자 수가 나옴
# 아래 코드는 Plotly를 사용해 Polar 차트를 생성하여, 
# 각 ML 도입 단계별 응답자 수를 기업 규모(Q25) 기준으로 시각화함

# 새로운 Figure 객체 생성
fig = go.Figure()

# 1. "Models in Production" 단계에 대한 데이터를 추출하여 Scatterpolar trace 추가
fig.add_trace(go.Scatterpolar(
    # r: 해당 그룹의 응답자 수 (counts 값을 Q25 기준으로 정렬한 후 리스트 변환)
    r = adoption_per_company_size[adoption_per_company_size["ML_adoption_class"] == "Models in Production"]
        .sort_values(by=["Q25"])["counts"].to_list(),
    # theta: 기업 규모(Q25)의 값을 각도(θ)로 사용 (정렬 후 리스트 변환)
    theta=adoption_per_company_size[adoption_per_company_size["ML_adoption_class"] == "Models in Production"]
        .sort_values(by=["Q25"])["Q25"].to_list(),
    fill='toself',  # (포인트를 연결하여 내부를 채움)
    name='Models in Production'  # 범례에 표시될 이름
))

# 2. "Exploration Stage" 단계에 대한 데이터를 추출하여 Scatterpolar trace 추가
fig.add_trace(go.Scatterpolar(
    r=adoption_per_company_size[adoption_per_company_size["ML_adoption_class"] == "Exploration Stage"]
        .sort_values(by=["Q25"])["counts"].to_list(),
    theta=adoption_per_company_size[adoption_per_company_size["ML_adoption_class"] == "Exploration Stage"]
        .sort_values(by=["Q25"])["Q25"].to_list(),
    fill='toself',
    name='Exploration Stage'
))

# 3. "Generating Insights" 단계에 대한 데이터를 추출하여 Scatterpolar trace 추가
fig.add_trace(go.Scatterpolar(
    r=adoption_per_company_size[adoption_per_company_size["ML_adoption_class"] == "Generating Insights"]
        .sort_values(by=["Q25"])["counts"].to_list(),
    theta=adoption_per_company_size[adoption_per_company_size["ML_adoption_class"] == "Generating Insights"]
        .sort_values(by=["Q25"])["Q25"].to_list(),
    fill='toself',
    name='Generating Insights'
))

# 4. "Not Started" 단계에 대한 데이터를 추출하여 Scatterpolar trace 추가
fig.add_trace(go.Scatterpolar(
    r=adoption_per_company_size[adoption_per_company_size["ML_adoption_class"] == "Not Started"]
        .sort_values(by=["Q25"])["counts"].to_list(),
    theta=adoption_per_company_size[adoption_per_company_size["ML_adoption_class"] == "Not Started"]
        .sort_values(by=["Q25"])["Q25"].to_list(),
    fill='toself',
    name='Not Started'
))

# HTML 태그를 사용하여 차트 제목의 서식을 지정
large_title_format = "<span style='font-size:30px; font-family:Times New Roman'>Productionization of ML models by Company's size</span>"
small_title_format = "<span style='font-size:14px; font-family:Helvetica'>소제목</b></span>"

# 차트 레이아웃 설정: 제목, 글꼴 색상, 범례 표시 여부, 여백 등을 설정합니다.
layout = dict(
    title = large_title_format + "<br>" + small_title_format,  # 제목과 부제목을 결합
    font = dict(color = '#7b6b59'),  # 기본 글꼴 색상 설정
    showlegend = True,  # 범례 표시
    margin = dict(t=80, pad=6),  # 상단 여백 및 내부 패딩 설정
)

# 레이아웃 업데이트를 통해 위의 설정을 차트에 적용합니다.
fig["layout"].update(layout)

# 최종적으로 차트를 표시합니다.
fig.show()



cloud_usage = scope_df.groupby(
    ["Cloud_usage"]
).agg({
    "Q2" : "count"
}).reset_index().rename(columns={
    "Q2": "Nbr of respondents",
    "Cloud_usage": "Usage of Cloud Computing Platforms"
})
cloud_usage["%"] = np.round((cloud_usage["Nbr of respondents"] / scope_df.shape[0]) * 100, 2)
cloud_usage["%"] = cloud_usage["%"].astype(str).str.rstrip('0').str.rstrip('.')

cloud_usage.style.background_gradient(axis=0,  cmap='Blues')

#background_gradient() 함수는 수치 데이터를 기반으로 색상 그라디언트를 계산
# % 칼럼을 문자열로 변환했기 때문에 그라디언트가 적용되지 않음


# usage_per_adoption 데이터프레임에는 각 ML 도입 단계와 Cloud 사용 여부별 응답자 수가 포함됨
# 그룹화: ML 도입 단계(ML_adoption)와 Cloud 사용 여부(Cloud_usage)별로 응답자 수를 계산
# Q2 컬럼(예: 응답자 수)을 count하여 각 그룹의 개수를 "counts"라는 컬럼으로 저장

usage_per_adoption = scope_df.groupby(
    ["ML_adoption", "Cloud_usage"]  # ML 도입 단계와 클라우드 사용 여부를 기준으로 그룹화
).agg({
    "Q2": "count"  # 각 그룹 내 응답자 수(count)를 계산
}).reset_index().rename(columns={
    "Q2": "counts"  # 결과 컬럼 Q2의 이름을 "counts"로 변경 (더 직관적 표현)
})


# Figure 객체 생성 (Plotly 그래프를 위한 빈 도화지)

fig = go.Figure()

# 첫 번째 trace 추가: Cloud Usage가 "Yes"인 경우에 대한 극좌표(Polar) 산점도
fig.add_trace(go.Scatterpolar(
    # r: "Yes"인 그룹의 응답자 수 (counts)를 ML 도입 단계 기준으로 정렬 후 리스트로 변환
    r = usage_per_adoption[usage_per_adoption["Cloud_usage"] == "Yes"]
            .sort_values(by=["ML_adoption"])["counts"].to_list(),
    # theta: "Yes"인 그룹의 ML 도입 단계 값을 정렬 후 리스트로 변환
    theta = usage_per_adoption[usage_per_adoption["Cloud_usage"] == "Yes"]
            .sort_values(by=["ML_adoption"])["ML_adoption"].to_list(),
    fill = 'toself',  # 해당 trace 내부 영역을 채움 (폐쇄 영역 채우기)
    name = 'Cloud Usage: Yes'  # 범례에 표시될 이름
))

# 두 번째 trace 추가: Cloud Usage가 "No"인 경우에 대한 극좌표 산점도
fig.add_trace(go.Scatterpolar(
    # r: "No"인 그룹의 응답자 수 (counts)를 ML 도입 단계 기준으로 정렬 후 리스트로 변환
    r = usage_per_adoption[usage_per_adoption["Cloud_usage"] == "No"]
            .sort_values(by=["ML_adoption"])["counts"].to_list(),
    # theta: "No"인 그룹의 ML 도입 단계 값을 정렬 후 리스트로 변환
    theta = usage_per_adoption[usage_per_adoption["Cloud_usage"] == "No"]
            .sort_values(by=["ML_adoption"])["ML_adoption"].to_list(),
    fill = 'toself',  # 해당 trace 내부 영역을 채움
    name = "Cloud Usage: No"  # 범례에 표시될 이름
))

# 제목 서식을 HTML 태그를 사용해 지정 (큰 제목과 작은 부제목)
large_title_format = "<span style='font-size:30px; font-family:Times New Roman'>Cloud Usage by ML Adoption</span>"
small_title_format = "<span style='font-size:14px; font-family:Helvetica'></b></span>"

# 차트 레이아웃 설정: 제목, 글꼴, 범례, 여백 등을 지정
layout = dict(
    title = large_title_format + "<br>" + small_title_format,  # 제목과 부제목 결합
    font = dict(color = '#7b6b59'),  # 기본 글씨 색상 설정
    showlegend = True,  # 범례 표시
    margin = dict(t=80, pad=6),  # 상단 여백 및 내부 패딩 설정
)

# 설정한 레이아웃을 Figure 객체에 업데이트
fig["layout"].update(layout)

# 최종 차트 표시
fig.show()



# 1. df 컬럼 중 "Q31"으로 시작하는 컬럼명을 리스트로 추출함.
#    - 이 코드는 df에 "Q31"으로 시작하는 모든 컬럼명을 찾아 cloud_computing_platforms 리스트에 저장함.
cloud_computing_platforms = [col for col in scope_df.columns if col.startswith("Q31")]

# 2. 빈 리스트 dfs_list를 초기화함.
#    - 이후 각 "Q31" 컬럼별 그룹화 결과를 저장할 예정임.
dfs_list = []

# 3. cloud_computing_platforms 리스트에 있는 각 컬럼에 대해 아래의 작업을 수행함.
for col in cloud_computing_platforms:
    # 3-1. scope_df를 사용하여 그룹화를 수행함.
    #      - 그룹화 기준은 "Q24" (산업)와 현재 컬럼(col)임.
    #      - agg({"Q2": "count"})로 각 그룹별 응답자 수를 셈.
    # 3-2. reset_index()를 호출하여 그룹화 결과를 DataFrame 형태로 변환함.
    # 3-3. rename() 함수를 사용하여 현재 그룹화 기준 컬럼 이름을 "Q31"로,
    #      그리고 응답자 수 컬럼 "Q2"를 "counts"로 변경함.
    dfs_list.append(
        scope_df.groupby(["Q24", col]).agg({"Q2": "count"})
        .reset_index()
        .rename(columns={col: "Q31", "Q2": "counts"})
    )

# 4. dfs_list에 저장된 모든 DataFrame을 pd.concat()을 이용해 하나의 DataFrame으로 결합함.
#    - 이 결과는 여러 "Q31" 컬럼에 대한 그룹화 결과가 모두 합쳐진 DataFrame임.
cloud_computing_platforms_df = pd.concat(dfs_list)

# 5. 각 행에 대해, 해당 그룹("Q24" 산업)의 전체 응답자 수를 industry_totals 딕셔너리에서 가져와,
#    현재 그룹의 응답자 수("counts")와의 비율을 계산하여 "relative_percent"라는 새로운 컬럼에 저장함.
cloud_computing_platforms_df["relative_percent"] = cloud_computing_platforms_df.apply(
    lambda x: x["counts"] / industry_totals[x["Q24"]],
    axis=1
)

# 6. "Q31" 컬럼에서 값이 "None" 또는 "Other"인 행들을 필터링하여 제거함.
#    - 이는 의미 없는 값들을 분석 대상에서 제외하기 위함임.
cloud_computing_platforms_df = cloud_computing_platforms_df[~cloud_computing_platforms_df["Q31"].isin(["None", "Other"])]

# 7. create_scatter_plot 함수를 호출하여 산점도를 생성함.
#    - x축 값: cloud_computing_platforms_df["Q31"]의 각 값에서 "/"를 기준으로 분리하여 첫번째 부분만 사용함.
#    - y축 값: cloud_computing_platforms_df["Q24"] (산업 정보)
#    - hover_template: 마우스 오버 시 표시될 텍스트 템플릿으로, 산업, 클라우드 플랫폼, 퍼센트(마커 크기)를 표시함.
#    - marker_color와 marker_size: 각각 "relative_percent" 값을 100배한 값(백분율)을 사용함.
#    - title, subtitle, subtitle_explain: 차트 제목 및 부제목, 설명을 설정함.
create_scatter_plot(
    cloud_computing_platforms_df["Q31"].apply(lambda x: x.split("/")[0]),  # x축: "/" 앞의 텍스트 사용
    cloud_computing_platforms_df["Q24"],                                  # y축: 산업 정보
    "Cloud Computing Platform: %{x}<br>" +
        "Industry: %{y}<br>" +
        "Percentage: %{marker.size:,}" +
        "<extra></extra>",                                              # hover 템플릿 설정
    cloud_computing_platforms_df['relative_percent'] * 100,               # 마커 색상 (백분율)
    cloud_computing_platforms_df['relative_percent'] * 100,               # 마커 크기 (백분율)
    "Cloud Computing In Different Industries",                          # 차트 제목
    "Questions Data: Industry (Q24) and Cloud Computing Platform (31)",   # 부제목(데이터 출처 및 설명)
    "Size,Color: Percentage of Respondents - <br>The number of respondents of the related sector that chose the relevant Cloud Computing Platform<br>divided by the total number of respondents working in that sector."  
    # 설명: 각 마커의 크기와 색상이 해당 산업에서 클라우드 컴퓨팅 플랫폼을 선택한 응답자의 비율을 나타냄.
)



# Q31로 시작하는 컬럼들을 추출함.
# 이 컬럼들은 클라우드 컴퓨팅 관련 질문들을 나타냄.
cloud_computing_questions = [col for col in scope_df.columns if col.startswith("Q31")]

# Q4(국가)에 대한 전체 응답자 수를 계산하여 딕셔너리 형태로 저장함.
# 예: {"USA": 500, "Germany": 300, ...}
country_totals = scope_df["Q4"].value_counts().to_dict()

# 각 클라우드 컴퓨팅 관련 질문에 대해 그룹화된 결과를 저장할 빈 리스트를 초기화함.
dfs_list = []

# cloud_computing_questions 리스트에 있는 각 컬럼에 대해 반복함.
for col in cloud_computing_questions:
    # scope_df를 그룹화함. 그룹 기준은 Q4(국가)와 현재 컬럼(col)임.
    # 각 그룹에서 Q2(응답자 수)를 count하여 집계한 후, 인덱스를 재설정하고,
    # 현재 컬럼(col)의 이름을 "Q31"으로, Q2 컬럼명을 "counts"로 변경함.
    dfs_list.append(
        scope_df.groupby(["Q4", col]).agg({"Q2": "count"})
        .reset_index()
        .rename(columns={col: "Q31", "Q2": "counts"})
    )

# dfs_list에 저장된 DataFrame들을 모두 결합하여 하나의 DataFrame으로 만듦.
# 만약 dfs_list가 빈 리스트이면 에러가 발생함.
cloud_computing_platforms = pd.concat(dfs_list)

# 각 행에 대해, 해당 그룹(국가)의 응답자 수 대비 현재 그룹의 응답자 수 비율(%)을 계산하여 "relative_percent" 컬럼에 저장함.
# lambda 함수는 각 행(x)의 counts를 country_totals에서 해당 국가(x["Q4"])의 총 응답자 수로 나눔.
cloud_computing_platforms["relative_percent"] = cloud_computing_platforms.apply(
    lambda x: x["counts"] / country_totals[x["Q4"]],
    axis=1
)

# "Q31" 컬럼에서 "None" 또는 "Other" 값이 포함된 행들을 제거함.
cloud_computing_platforms = cloud_computing_platforms[~cloud_computing_platforms["Q31"].isin(["None", "Other"])]

# 각 국가(Q4)별로 가장 많이 사용된 클라우드 컴퓨팅 플랫폼만 남기기 위해,
# 우선 Q4와 counts 기준으로 오름차순 정렬함.
cloud_computing_platforms = cloud_computing_platforms.sort_values(by=["Q4", "counts"], ascending=True)

# 국가별로 중복된 행들 중 마지막 행(가장 높은 counts 값을 가진 행)을 남기고 나머지를 제거함.
cloud_computing_platforms = cloud_computing_platforms.drop_duplicates(subset=["Q4"], keep='last')

# "Q31" 컬럼의 문자열에서 불필요한 공백을 제거함.
cloud_computing_platforms["Q31"] = cloud_computing_platforms["Q31"].str.strip()

# Plotly Express의 choropleth 함수를 사용하여 국가별로 가장 인기있는 클라우드 컴퓨팅 플랫폼을 지도에 표시함.
fig = px.choropleth(
    locations = cloud_computing_platforms['Q4'],  # 국가 이름을 사용하여 위치 지정함.
    locationmode = "country names",              # 국가 이름 모드를 사용함.
    color = cloud_computing_platforms["Q31"],      # 클라우드 플랫폼(Q31)을 색상 기준으로 사용함.
    color_discrete_map = {                         # 각 클라우드 플랫폼별로 지정된 색상 매핑을 적용함.
        'Google Cloud Platform (GCP)' : '#a43725',
        'Amazon Web Services (AWS)': '#cfbd9b',
        'Microsoft Azure': '#edc860',
        'Alibaba Cloud': '#E6b6a4'
    }
)

# 큰 제목 서식을 HTML 태그로 지정함.
large_title_format = "<span style='font-size:30px; font-family:Times New Roman'>Most Popular Cloud Computing Platform by Country</span>"
# 작은 제목(부제목) 서식을 HTML 태그로 지정함.
small_title_format = "<span style='font-size:14px; font-family:Helvetica'></b></span>"

# 차트 레이아웃을 업데이트하여 전체 크기, 범례, 제목, 폰트 색상, 드래그 모드, 여백 등을 설정함.
fig.update_layout(
    width = 750,  # 전체 차트 너비 설정함.
    legend = dict(
        orientation = "h",  # 범례를 수평으로 배치함.
        yanchor = "bottom",
        y = 0.9,
        xanchor = "right",
        x = 1,
        title = '',       # 범례 제목 제거함.
    ),
    title = large_title_format + "<br>" + small_title_format,  # 제목과 부제목을 결합함.
    font = dict(color = '#7b6b59'),  # 기본 글씨 색상 설정함.
    dragmode = False,  # 드래그 모드를 비활성화함.
    margin = dict(
        l = 10,  # 왼쪽 여백
        r = 10,  # 오른쪽 여백
        b = 10,  # 아래 여백
        t = 50,  # 위쪽 여백
        pad = 0  # 내부 패딩
    )
)

# 최종적으로 choropleth 차트를 표시함.
fig.show()



# px.Constant은 Plotly Express에서 제공하는 헬퍼 함수
# treemap 등 계층적 차트를 생성할 때, 고정된(상수) 값을 갖는 루트 노드를 생성하기 위해 사용
# 예를 들어, path 인자에서 px.Constant("AI Tech Stack")을 사용하면
# 모든 데이터의 최상위 레벨에 "AI Tech Stack"이라는 고정된 값을 부여하여
# 트리맵의 루트 노드를 하나로 설정 가능


# 분석할 제품 질문들을 리스트에 저장함.
product_questions = ["Q14", "Q31", "Q33", "Q34", "Q35", "Q36", "Q37", "Q38", "Q41"]

# 결과를 저장할 빈 리스트 초기화함.
dfs_list = []

# 각 제품 질문별로 반복
# extract_and_count_all_the_multiple_choice_answers 함수를 호출하여
# 해당 제품 질문에 대한 다중 선택 응답 데이터를 집계함.
    # 집계 결과에서 제품 질문 컬럼명을 "Service"로 변경함.

for product in product_questions:
    result = extract_and_count_all_the_multiple_choice_answers(product, scope_df)
    result = result.rename(columns={product: "Service"})
    dfs_list.append(result)

# 리스트에 저장된 모든 DataFrame을 하나로 결합함.
products_df = pd.concat(dfs_list)

# "Service" 컬럼의 값을 기반으로 assign_label 함수(사용자 정의)를 적용하여 "Label" 컬럼 생성함.
products_df["Label"] = products_df["Service"].apply(lambda x: assign_label(x))

# "Label" 컬럼 값이 "Amazon", "Microsoft", "Google", "IBM"인 행들만 남김.
products_df = products_df[products_df["Label"].isin(["Amazon", "Microsoft", "Google", "IBM"])]

# Plotly Express의 treemap 차트를 생성함.
# - 최상위 노드는 "AI Tech Stack"로 고정함.
# - 그 다음 노드로 "Label" (예: Amazon 등)을 사용하고,
# - 마지막 하위 노드로 "Service" (제품/기술 이름)을 사용함.
# - 값은 "counts", 색상은 "relative_percent"를 기준으로 지정함.
# - color_continuous_scale으로 'RdBu' 색상 스케일을 적용함.

fig = px.treemap(
    products_df,
    path=[px.Constant("AI Tech Stack"), "Label", "Service"],
    values="counts",
    color="relative_percent",
    color_continuous_scale="RdBu",
    height=650
)

# 트리맵의 루트 노드 색상을 lightgrey로 설정함.
fig.update_traces(root_color="lightgrey")

# 차트 레이아웃의 여백을 설정함.
fig.update_layout(margin=dict(t=50, l=25, r=25, b=25))

# 최종 treemap 차트를 표시함.
fig.show()



### 바 차트에 상위 3개 항목을 제외하면 값이 표시되지 않는 이유?
### 상위 항목을 제외하면 outer에 표시되게끔 설정되어 있는데, 배경도 흰색이고 글씨도 흰색이라 그렇슴


# Q12: 프로그래밍 언어 관련 질문에 대한 다중 선택 응답 집계 결과를 추출함.
programming_languages = extract_and_count_all_the_multiple_choice_answers("Q12", scope_df)

# "None"이나 "Other" 값을 제외함.
programming_languages = programming_languages[~programming_languages["Q12"].isin(["None", "Other"])]

# Q13: 통합 개발 환경(IDE) 관련 질문에 대한 다중 선택 응답 집계 결과를 추출함.
ides = extract_and_count_all_the_multiple_choice_answers("Q13", scope_df)

# "None"이나 "Other" 값을 제외함.
ides = ides[~ides["Q13"].isin(["None", "Other"])]

# 두 개의 차트 트레이스를 저장할 딕셔너리를 초기화함.
traces = dict()

# 프로그래밍 언어에 대한 바 차트 트레이스를 생성함.
# get_bar_plot_trace 함수는 주어진 값들을 바 차트의 트레이스로 변환함.

trace_languages = get_bar_plot_trace(
    programming_languages["relative_percent"].to_list(),   # 각 프로그래밍 언어의 상대 비율 값 리스트
    programming_languages["Q12"].to_list(),                # 프로그래밍 언어 이름 리스트 (Q12 컬럼)
    np.round((programming_languages["relative_percent"] * 100), decimals=2),  # 백분율 값(소수점 2자리) 리스트
    3,                                                     
    programming_languages.shape[0] - 3,                    
    programming_languages["counts"].to_list()              # 각 프로그래밍 언어의 응답자 수 리스트
)

# IDE에 대한 바 차트 트레이스를 생성함.
trace_ides = get_bar_plot_trace(
    ides["relative_percent"].to_list(),                    # 각 IDE의 상대 비율 값 리스트
    ides["Q13"].apply(lambda x: x.split("(")[0]),           # IDE 이름에서 괄호 앞 부분만 추출하여 리스트로 사용함
    np.round((ides["relative_percent"] * 100), decimals=2),  # 백분율 값(소수점 2자리) 리스트
    3,                                                     # 바 차트 시작 인덱스
    ides.shape[0] - 3,                                     # 바 차트 끝 인덱스
    ides["counts"].to_list()                               # 각 IDE의 응답자 수 리스트
)

# 두 개의 바 차트를 하나의 Figure에 나란히 배치하기 위해 서브플롯을 생성함.
fig = make_subplots(
    rows=1,                   # 한 행에 배치함.
    cols=2,                   # 두 개의 열로 나눔.
    shared_yaxes=False,       # y축은 공유하지 않음.
    shared_xaxes=True,        # x축은 공유함.
    horizontal_spacing=0.20,  # 두 차트 사이의 가로 간격을 20%로 설정함.
    vertical_spacing=0.10     # 세로 간격은 10%로 설정함.
)

# 생성된 트레이스를 traces 딕셔너리에 저장함.
traces["Programming_Languages"] = trace_languages
traces["ides"] = trace_ides

# 왼쪽 서브플롯(1행 1열)에 프로그래밍 언어 관련 트레이스를 추가함.
fig.append_trace(traces["Programming_Languages"], 1, 1)

# 오른쪽 서브플롯(1행 2열)에 IDE 관련 트레이스를 추가함.
fig.append_trace(traces["ides"], 1, 2)

# 큰 제목 서식을 HTML 태그로 지정함.
large_title_format = "<span style='font-size:30px; font-family:Times New Roman'>Top programming languages for Data Science & ML in 2022</span>"

# 부제목 서식을 HTML 태그로 지정함.
small_title_format = ("<span style='font-size:14px; font-family:Helvetica'>"
                      "Python Is Essential for Data Analysis and Data Science.</br>"
                      "The length of the bars denotes the <b>percentage of professionals</b> that use the relevant language.</br>"
                      "The counts are also visible by hover."
                      "</span>")

# 차트 레이아웃 설정함.
layout = dict(
    title = large_title_format + "<br>" + small_title_format,  # 제목과 부제목을 결합함.
    font = dict(color = '#7b6b59'),   # 기본 글씨 색상 설정함.
    showlegend = False,               # 범례는 표시하지 않음.
    margin = dict(t=160, pad=6),        # 위쪽 여백을 160pt, 내부 패딩 6pt로 설정함.
    plot_bgcolor = '#fff',            # 플롯 배경색을 흰색으로 설정함.
    bargap = 0.10                     # 막대 간 간격을 10%로 설정함.
)

# 설정한 레이아웃을 Figure에 업데이트함.
fig['layout'].update(layout)

# 최종적으로 차트를 표시함.
fig.show()



# Q15: 데이터 시각화 라이브러리 관련 질문의 다중 선택 응답 집계 결과를 추출함.
data_viz_libs = extract_and_count_all_the_multiple_choice_answers("Q15", scope_df)
# 상대 비율(relative_percent)을 백분율로 변환하고 소수점 2자리로 반올림함.
data_viz_libs["relative_percent"] = round(data_viz_libs["relative_percent"] * 100, 2)
# 컬럼명을 더 직관적으로 변경함.
data_viz_libs = data_viz_libs.rename(
    columns={
        "Q15": "Data Visualization Libraries", 
        "counts": "# of respondents", 
        "relative_percent": "% of respondents"
    }
)
# % of respondents 값을 기준으로 내림차순 정렬한 후 인덱스를 재설정함.
data_viz_libs = data_viz_libs.sort_values(by=["% of respondents"], ascending=False).reset_index(drop=True)


# Q17: ML Frameworks 관련 질문의 다중 선택 응답 집계 결과를 추출함.
ml_frameworks = extract_and_count_all_the_multiple_choice_answers("Q17", scope_df)
# 상대 비율을 백분율로 변환하고 소수점 2자리로 반올림함.
ml_frameworks["relative_percent"] = round(ml_frameworks["relative_percent"] * 100, 2)
# 컬럼명을 변경하여 ML Frameworks 관련 정보를 저장함.
ml_frameworks = ml_frameworks.rename(
    columns={
        "Q17": "ML Frameworks", 
        "counts": "# of respondents", 
        "relative_percent": "% of respondents"
    }
)
# % of respondents 값을 기준으로 내림차순 정렬하고 인덱스를 재설정함.
ml_frameworks = ml_frameworks.sort_values(by=["% of respondents"], ascending=False).reset_index(drop=True)


# 색상 팔레트를 생성함.
# n_colors() 함수를 사용하여 'rgb(230, 182, 164)'에서 'rgb(164, 55, 37)'까지 15가지 색상(rgb 형식) 배열을 생성함.
colors = n_colors('rgb(230, 182, 164)', 'rgb(164, 55, 37)', 15, colortype='rgb')
# a 리스트는 색상 배열에서 인덱스를 재배열하는 용도로 사용됨.
a = [14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]


# 서브플롯을 이용하여 두 개의 테이블을 하나의 Figure에 나란히 배치함.
fig = make_subplots(
    rows=1, 
    cols=2,
    # 각 서브플롯은 표(table) 유형임.
    specs=[[{"type": "table"}, {"type": "table"}]],
    vertical_spacing=0.03  # 서브플롯 간의 수직 간격 설정
)

# 첫 번째 테이블: 데이터 시각화 라이브러리 관련 결과 표시
fig.add_trace(
    go.Table(
        header=dict(
            values=["Data Visualization Libraries", "% of respondents"],
            line_color='white', 
            fill_color='white',
            align='center', 
            font=dict(color='black', size=12)
        ),
        cells=dict(
            # 테이블의 각 셀 값: 첫 번째 열은 라이브러리 이름, 두 번째 열은 백분율 값
            values=[data_viz_libs["Data Visualization Libraries"], data_viz_libs["% of respondents"]],
            # fill_color를 설정하여 각 셀 배경색에 색상 팔레트를 적용함.
            fill_color=[np.array(colors)[a]],
            align='center', 
            font=dict(color='white', size=13, family='Arial Rounded MT Bold')
        )
    ),
    row=1, col=1
)

# 두 번째 테이블: ML Frameworks 관련 결과 표시
fig.add_trace(
    go.Table(
        header=dict(
            values=["ML Frameworks", "% of respondents"],
            line_color='white', 
            fill_color='white',
            align='center', 
            font=dict(color='black', size=12)
        ),
        cells=dict(
            values=[ml_frameworks["ML Frameworks"], ml_frameworks["% of respondents"]],
            fill_color=[np.array(colors)[a]],
            align='center', 
            font=dict(color='white', size=13, family='Arial Rounded MT Bold')
        )
    ),
    row=1, col=2
)

# 큰 제목 서식을 HTML로 지정함.
large_title_format = "<span style='font-size:30px; font-family:Times New Roman'>Top Data Visualization Libraries and ML Frameworks</span>"
# 부제목 서식을 HTML로 지정함.
small_title_format = "<span style='font-size:14px; font-family:Helvetica'></b></span>"

# 차트 레이아웃 설정: 제목, 글꼴 색상, 범례(없음), 여백, 배경색, 그리고 막대 간격을 설정함.
layout = dict(
    title = large_title_format + "<br>" + small_title_format,  # 제목과 부제목을 결합함.
    font = dict(color = '#7b6b59'),
    showlegend = False,
    margin = dict(t=160, pad=6),
    plot_bgcolor = '#fff',
    bargap = 0.10  # 막대 간 간격 10%
)

# 레이아웃 설정을 Figure 객체에 업데이트함.
fig['layout'].update(layout)

# 최종적으로 두 테이블이 포함된 Figure를 표시함.
fig.show()



dfs_list = []

for col in [column for column in scope_df.columns if column.startswith("Q18")]:
    dfs_list.append(scope_df.groupby([col]).agg({"Q2" : "count"}).reset_index().rename(columns={"Q2": "counts", col: "ML Algorithms"}))

ml_algorithms = pd.concat(dfs_list)
ml_algorithms["relative_percent"] = ml_algorithms.apply(lambda x : x["counts"] / scope_df.shape[0], axis = 1)


ml_algorithms = ml_algorithms.sort_values(by=["relative_percent"], ascending=True)
ml_algorithms = ml_algorithms[~ml_algorithms["ML Algorithms"].isin(["None", "Other"])]

create_single_bar_plot(
    x_values=ml_algorithms["relative_percent"].to_list(), 
    y_values=ml_algorithms["ML Algorithms"].to_list(),
    display_text=np.round((ml_algorithms["relative_percent"] *100), decimals = 2),
    top_n=3,
    rest_n=ml_algorithms.shape[0]-3, 
    hovertext = ml_algorithms["counts"].to_list(),
    title="Top 12 Machine Learning Algorithms",
    subtitle="", 
    orientation="h"
)


ml_algorithms = [col for col in scope_df.columns if col.startswith("Q18")]

dfs_list = []
for col in ml_algorithms:
    dfs_list.append(
        scope_df.groupby(["Q24", col]).agg({"Q2" : "count"}).reset_index().rename(columns={col: "Q18", "Q2": "counts"})
    )

ml_algorithms_df = pd.concat(dfs_list)
ml_algorithms_df["relative_percent"] = ml_algorithms_df.apply(lambda x : x["counts"] / industry_totals[x["Q24"]], axis = 1)
ml_algorithms_df = ml_algorithms_df[~ml_algorithms_df["Q18"].isin(["None", "Other"])]


create_scatter_plot(
    ml_algorithms_df["Q18"].apply(lambda x : x.split("(")[0]),
    ml_algorithms_df["Q24"], 
     "ML Algorithm: %{x}<br>" +
        "Industry: %{y}<br>" +
        "Percentage: %{marker.size:,}" +
        "<extra></extra>",
    ml_algorithms_df['relative_percent']*100, 
    ml_algorithms_df['relative_percent']*100, 
    "Commonly Used Machine Learning Algorithms in Different Industries",
    "Questions Data: Industry (Q24) and ML Algorithm (Q18)",
    "Size,Color: Percentage of Respondents - <br>The number of respondents of the related sector that chose the relevant ML Algorithm<br>divided by the total number of respondents working in that sector."
)


# CV / NLP 관련 분석 (19번, 20번, 21번)

map_cv_methods = {
    "Vision transformer networks (ViT, DeiT, BiT, BEiT, Swin, etc)": "Vision transformer<br>networks" , 
    "Generative Networks (GAN, VAE, etc)": "Generative Networks",
    "General purpose image/video tools (PIL, cv2, skimage, etc)": "General purpose<br><sup>image/video tools</sup>", 
    "Object detection methods (YOLOv6, RetinaNet, etc)": "Object detection<br>methods",
    "Image classification and other general purpose networks (VGG, Inception, ResNet, ResNeXt, NASNet, EfficientNet, etc)": "Image classification Nets",
    "Image segmentation methods (U-Net, Mask R-CNN, etc)": "Image segmentation<br>methods"
}

map_nlp_methods = {
    "Contextualized embeddings (ELMo, CoVe)": "Contextualized<br>embeddings" , 
    "Encoder-decoder models (seq2seq, vanilla transformers)": "Encoder-decoder models",
    "Word embeddings/vectors (GLoVe, fastText, word2vec)": "Word embeddings<br><sup>GLoVe, fastText, word2vec</sup>", 
    "Transformer language models (GPT-3, BERT, XLnet, etc)": "Transformer <br>language models",

}

computer_vision_methods =  extract_and_count_all_the_multiple_choice_answers("Q19", scope_df)
computer_vision_methods = computer_vision_methods[~computer_vision_methods["Q19"].isin(["None", "Other"])]
computer_vision_methods["Q19"] = computer_vision_methods["Q19"].apply(lambda x : map_cv_methods[x])

nlp_methods = extract_and_count_all_the_multiple_choice_answers("Q20", scope_df)
nlp_methods = nlp_methods[~nlp_methods["Q20"].isin(["None", "Other"])]
nlp_methods["Q20"] = nlp_methods["Q20"].apply(lambda x : map_nlp_methods[x])

pre_trained_models =  extract_and_count_all_the_multiple_choice_answers("Q21", scope_df)
pre_trained_models["Q21"] = np.where(pre_trained_models["Q21"] == "No, I do not download pre-trained model weights on a regular basis", "No, I do not download <br>pre-trained model weights", pre_trained_models["Q21"])


traces = dict()

    
# Creating the bar chart
trace_nlp = get_bar_plot_trace(
    nlp_methods["relative_percent"].to_list(),
    nlp_methods["Q20"].to_list(),
    np.round((nlp_methods["relative_percent"] *100), decimals = 2),
    2, 
    nlp_methods.shape[0]-2, 
    nlp_methods["counts"].to_list()
) 


trace_cv = get_bar_plot_trace(
    computer_vision_methods["relative_percent"].to_list(),
    computer_vision_methods["Q19"].to_list(),
    np.round((computer_vision_methods["relative_percent"] *100), decimals = 2),
    2, 
    computer_vision_methods.shape[0]-2, 
    computer_vision_methods["counts"].to_list()
) 


trace_models = get_bar_plot_trace(
    pre_trained_models["Q21"].apply(lambda x : x.split("(")[0]).to_list(),
    pre_trained_models["relative_percent"].to_list(),
    np.round((pre_trained_models["relative_percent"] *100), decimals = 2),
    3, 
    pre_trained_models.shape[0]-3, 
    pre_trained_models["counts"].to_list(),
    orientation = "v"
) 

traces["NLP_methods"] = trace_nlp
traces["CV_methods"] = trace_cv


fig = make_subplots(
    rows=1, 
    cols=2 ,
    shared_yaxes=False, 
    shared_xaxes=True, 
    horizontal_spacing = 0.15, 
    subplot_titles=("Most common Computer Vision methods", "Most common NLP methods", "Do you download Pre-Trained Models for Transfer Learning?"))


fig.append_trace(traces["CV_methods"],1,1)
fig.append_trace(traces["NLP_methods"],1,2)            

large_title_format = "<span style='font-size:30px; font-family:Times New Roman'>How Transfer Learning is being used today</span>"
small_title_format = "<span style='font-size:14px; font-family:Helvetica'>The length of the bars denotes the <b>percentage of professionals in the field that use the specified model</b>.</span>"


layout = dict(
    title = large_title_format + "<br>" + small_title_format + "<br><br>",
    showlegend = False,
    font = dict(color = '#7b6b59'),
    margin = dict(t=150),
    plot_bgcolor='#fff',
    bargap = 0.10,

)


fig['layout'].update(layout)

fig.show()

large_title_format = "<span style='font-size:22px; font-family:Times New Roman'>Do you download pre-trained model weights from any <br>of the public available services? </span>"


fig = go.Figure(trace_models)
layout = dict(
    title = large_title_format + "<br>",
    showlegend = False,
    font = dict(color = '#7b6b59'),
    margin = dict(t=40),
    plot_bgcolor='#fff',
    bargap = 0.10,

)


fig['layout'].update(layout)

fig.show()


nlp_usage = scope_df.groupby(
    ["NLP_methods_usage"]
).agg({
    "Q2" : "count"
}).reset_index().rename(columns={
    "Q2": "Nbr of respondents",
    "NLP_methods_usage": "Use of NLP Methods and Pre-trained Models"
})
nlp_usage["%"] = np.round((nlp_usage["Nbr of respondents"] / scope_df.shape[0]) * 100, 2)
nlp_usage.style.background_gradient(axis=0,  cmap='Blues')


# Get the counts of occurrences of each job role
roles_totals = scope_df["Q23"].value_counts().to_dict()

nlp_usage = scope_df[scope_df["NLP_methods_usage"] == "Yes"].groupby(["Q23"]).agg({"Q2" : "count"}).reset_index().rename(columns={"Q2": "Nbr of respondents", "Q23" : "Role"})

nlp_usage["%"] = nlp_usage.apply(lambda x : x["Nbr of respondents"] / roles_totals[x["Role"]], axis = 1)
nlp_usage["%"]  = np.round(nlp_usage["%"] * 100, 2)
nlp_usage = nlp_usage.sort_values(by=["%"], ascending=False).reset_index(drop=True)

nlp_usage.style.background_gradient(axis=0,  cmap='Oranges')


cv_usage = scope_df.groupby(
    ["CV_methods_usage"]
).agg({
    "Q2" : "count"
}).reset_index().rename(columns={
    "Q2": "Nbr of respondents",
    "CV_methods_usage": "Use of CV Methods and Pre-trained Models"
})
cv_usage["%"] = np.round((cv_usage["Nbr of respondents"] / scope_df.shape[0]) * 100, 2)
cv_usage.style.background_gradient(axis=0,  cmap='Blues')


# Get the counts of occurrences of each job role
roles_totals = scope_df["Q23"].value_counts().to_dict()

cv_usage = scope_df[scope_df["CV_methods_usage"] == "Yes"].groupby(["Q23"]).agg({"Q2" : "count"}).reset_index().rename(columns={"Q2": "Nbr of respondents", "Q23" : "Role"})

cv_usage["%"] = cv_usage.apply(lambda x : x["Nbr of respondents"] / roles_totals[x["Role"]], axis = 1)
cv_usage["%"]  = np.round(cv_usage["%"] * 100, 2)
cv_usage = cv_usage.sort_values(by=["%"], ascending=False).reset_index(drop=True)

cv_usage.style.background_gradient(axis=0,  cmap='Oranges')


hardware_usage = scope_df.groupby(
    ["GPU_usage"]
).agg({
    "Q2" : "count"
}).reset_index().rename(columns={
    "Q2": "Nbr of respondents",
    "GPU_usage": "Specialized Hardware Usage"
})
hardware_usage["%"] = np.round((hardware_usage["Nbr of respondents"] / scope_df.shape[0]) * 100, 2)
hardware_usage.style.background_gradient(axis=0,  cmap='Blues')


usage_per_adoption = scope_df.groupby(
    ["ML_adoption", "GPU_usage"]
).agg({
    "Q2" : "count"
}).reset_index().rename(columns={
    "Q2": "counts"
})
usage_per_adoption

fig = go.Figure()

fig.add_trace(go.Scatterpolar(
      r=usage_per_adoption[usage_per_adoption["GPU_usage"] == "Yes"].sort_values(by=["ML_adoption"])["counts"].to_list(),
      theta=usage_per_adoption[usage_per_adoption["GPU_usage"] == "Yes"].sort_values(by=["ML_adoption"])["ML_adoption"].to_list(),
      fill='toself',
      name='Specialized hardware usage: Yes'
))

large_title_format = "<span style='font-size:30px; font-family:Times New Roman'>Specialized hardware usage for ML models training by ML adoption stage</span>"
small_title_format = "<span style='font-size:14px; font-family:Helvetica'></b></span>"


layout = dict(
    title = large_title_format + "<br>" + small_title_format,
    font = dict(color = '#7b6b59'),
    showlegend = True,
    margin = dict(t=80,pad=6),

)
fig["layout"].update(layout)

fig.show()


dfs_list = []

for col in [column for column in scope_df.columns if column.startswith("Q42")]:
    dfs_list.append(scope_df.groupby([col]).agg({"Q2" : "count"}).reset_index().rename(columns={"Q2": "counts", col: "Hardware"}))

hardware = pd.concat(dfs_list)
hardware["relative_percent"] = hardware.apply(lambda x : x["counts"] / scope_df.shape[0], axis = 1)


hardware = hardware.sort_values(by=["relative_percent"], ascending=True)
hardware = hardware[~hardware["Hardware"].isin(["None", "Other"])]

create_single_bar_plot(
    x_values=hardware["relative_percent"].to_list(), 
    y_values=hardware["Hardware"].to_list(),
    display_text=np.round((hardware["relative_percent"] *100), decimals = 2),
    top_n=2,
    rest_n=hardware.shape[0]-2, 
    hovertext = hardware["counts"].to_list(),
    title="Commonly Used Types of Specialized Hardware",
    subtitle="", 
    orientation="h"
)


roles_totals = scope_df["Q23"].value_counts().to_dict()

gpu_usage = scope_df[scope_df["GPU_usage"] == "Yes"].groupby(["Q23"]).agg({"Q2" : "count"}).reset_index().rename(columns={"Q2": "Nbr of respondents", "Q23" : "Role"})

gpu_usage["%"] = gpu_usage.apply(lambda x : x["Nbr of respondents"] / roles_totals[x["Role"]], axis = 1)
gpu_usage["%"]  = np.round(gpu_usage["%"] * 100, 2)
gpu_usage = gpu_usage.sort_values(by=["%"], ascending=False).reset_index(drop=True)

gpu_usage.style.background_gradient(axis=0,  cmap='Oranges')


data_science_roles = scope_df.groupby(["Q23"]).agg({"Q2" : "count"}).reset_index().rename(columns={"Q2": "counts"})

data_science_roles["relative_percent"] = data_science_roles.apply(lambda x : (x["counts"] / scope_df.shape[0]), axis = 1)
data_science_roles = data_science_roles.sort_values(by=["relative_percent"], ascending=True)
data_science_roles = data_science_roles[~data_science_roles["Q23"].isin(["None", "Other"])]


create_single_bar_plot(
    x_values=data_science_roles["relative_percent"].to_list(),
    y_values=data_science_roles["Q23"].to_list(),
    display_text=np.round((data_science_roles["relative_percent"] *100), decimals = 2),
    top_n=2,
    rest_n=data_science_roles.shape[0]-2, 
    hovertext = data_science_roles["counts"].to_list(),
    title="Top AI Jobs in the Market",
    subtitle="", 
    orientation="h"
)


roles_df = scope_df.groupby(["Q24", "Q23"]).agg({"Q2" : "count"}).reset_index().rename(columns={"Q2": "counts"})
roles_df["relative_percent"] = roles_df.apply(lambda x : x["counts"] / industry_totals[x["Q24"]], axis = 1)

create_scatter_plot(
    roles_df["Q23"].apply(lambda x : x.split("(")[0]),
    roles_df["Q24"], 
     "Role: %{x}<br>" +
        "Industry: %{y}<br>" +
        "Percentage: %{marker.size:,}" +
        "<extra></extra>",
    roles_df['relative_percent']*100, 
    roles_df['relative_percent']*100, 
    "What Industries are Hiring the Most AI Technology Specialists?",
    "Questions Data: Industry (Q24) and Job Role (Q23)",
    "Size,Color: Percentage of Respondents - <br>The number of respondents with the relevant job position in the related sector<br>divided by the total number of respondents working in that sector."
)


dfs_list = []

# 'scope_df' 데이터프레임에서 관심있는 직군만 추출
# "Machine Learning/ MLops Engineer", "Data Scientist" 또는 "Data Analyst"가 포함된 행만 선택
ml_scope_df = scope_df[
    (scope_df["Q23"].isin(["Machine Learning/ MLops Engineer", "Data Scientist"])) |
    (scope_df["Q23"].str.contains("Data Analyst"))
]

# 'scope_df'에서 이름이 "Q28"으로 시작하는 모든 컬럼에 대해 반복 작업을 수행
for col in [column for column in scope_df.columns if column.startswith("Q28")]:
    dfs_list.append(
        ml_scope_df.groupby([col])              # 해당 컬럼 기준으로 그룹화
                   .agg({"Q2" : "count"})         # 각 그룹별로 'Q2'의 개수를 셈
                   .reset_index()                 # 인덱스를 다시 설정
                   .rename(columns={"Q2": "counts", col: "ML Activities"})  # 컬럼명 변경
    )
    
# 그룹화한 데이터프레임들을 하나로 병합
ml_activities = pd.concat(dfs_list)

# 각 활동이 전체에서 차지하는 비율을 계산
ml_activities["relative_percent"] = ml_activities.apply(lambda x: x["counts"] / ml_activities["counts"].sum(), axis=1)

# 상대 비율을 기준으로 내림차순 정렬
ml_activities = ml_activities.sort_values(by=["relative_percent"], ascending=False)

# "None"이나 "Other"가 포함된 행들은 제거
ml_activities = ml_activities[
    ~((ml_activities["ML Activities"].str.contains("None")) | 
      (ml_activities["ML Activities"].str.contains("Other")))
]

map_ml_activities = {
    "Analyze and understand data to influence product or business decisions": "Analyze and understand data<br>to influence product or business decisions",
    "Build prototypes to explore applying machine learning to new areas": "Build prototypes to explore <br>applying machine learning to new areas",
    "Build and/or run the data infrastructure that my business uses for storing, analyzing, and operationalizing data": "Build and/or run the data infrastructure",
    "Experimentation and iteration to improve existing ML models": "Experimentation and iteration<br>to improve existing ML models",
    "Build and/or run a machine learning service that operationally improves my product or workflows": "Build and/or run<br>a machine learning service",
    "Do research that advances the state of the art of machine learning": "Do research that advances the<br>state of the art of machine learning"
}

ml_activities["ML Activities"] = ml_activities["ML Activities"].apply(lambda x: map_ml_activities[x])

# Plotly를 사용하여 funnel area 차트를 생성

fig = go.Figure(go.Funnelarea(
    values = ml_activities["counts"].to_list(),   # 각 활동별 카운트 값 리스트
    text = ml_activities["ML Activities"].to_list(), # 포맷팅된 활동명 리스트
    marker = {"colors": ["#a43725","#c07156", "#E6b6a4", "#edc860", "#e5b01c", "#cfbd9b", "#a43725"]},
    textfont = {"family": "Times New Roman", "size": 22, "color": "black"},
    opacity = 0.65  # 약간의 투명도로 부드러운 느낌을 더합니다.
))


large_title_format = "<span style='font-size:30px; font-family:Times New Roman'>A Day in the Life of a Data Scientist / Analyst or ML Engineer</span>"

layout = dict(
    title = large_title_format,
    font = dict(color = '#7b6b59'),
    margin = dict(t=170),
    width = 800,
    height = 700,
    plot_bgcolor = "white"
)

# 설정한 레이아웃을 차트에 적용
fig.update_layout(layout)
fig.update_traces(showlegend=False)

fig.show()



# 관심 직무 목록 설정
jobs_in_scope = [
    "Data Scientist",
    "Data Analyst (Business, Marketing, Financial, Quantitative, etc)",
    "Research Scientist",
    "Machine Learning/ MLops Engineer"
]

# df의 Q28로 시작하는 열들을 ML 활동 관련 질문으로 선택
activities = [col for col in scope_df.columns if col.startswith("Q28")]

# Q23 열의 직무별 응답자 수를 계산 (앞뒤 공백 제거 후 딕셔너리 생성)
job_roles = scope_df["Q23"].str.strip().value_counts().to_dict()

# 각 직무와 활동별로 데이터를 저장할 빈 리스트 생성
dfs_list = []

# 관심 있는 각 직무와 각 활동별로 응답자 수를 집계
for role in jobs_in_scope:
    for col in activities:
        roles_df = scope_df[scope_df["Q23"].str.strip() == role] \
            .groupby(["Q23", col]) \
            .agg({"Q2": "count"}) \
            .reset_index() \
            .rename(columns={"Q2": "counts", col: "ML Activities"})
        dfs_list.append(roles_df)

# 개별 데이터프레임들을 하나로 합침
results = pd.concat(dfs_list)

# Q23 열의 불필요한 공백 제거
results["Q23"] = results["Q23"].str.strip()

# 직무별 총 응답자 수를 기준으로 각 활동의 상대 비율 계산
results["relative_percent"] = results.apply(lambda x: x["counts"] / job_roles[x["Q23"]], axis=1)

# ML 활동에서 "None" 또는 "Other"가 포함된 행 제거
results = results[
    ~((results["ML Activities"].str.contains("None")) | 
      (results["ML Activities"].str.contains("Other")))
]

# ML 활동명을 포맷팅하기 위한 매핑 딕셔너리 (번호 및 줄바꿈 추가)
map_ml_activities = {
    "Analyze and understand data to influence product or business decisions": "1. Analyze and understand data<br><sup>to influence product or business decisions</sup>",
    "Build prototypes to explore applying machine learning to new areas": "2. Build prototypes to explore <br><sup>applying machine learning to new areas</sup>",
    "Build and/or run the data infrastructure that my business uses for storing, analyzing, and operationalizing data": "3. Build and/or run the data infrastructure</sup>",
    "Experimentation and iteration to improve existing ML models": "4. Experimentation and iteration<br><sup>to improve existing ML models</sup>",
    "Build and/or run a machine learning service that operationally improves my product or workflows": "5. Build and/or run a machine learning service",
    "Do research that advances the state of the art of machine learning": "6. Do research that advances <br><sup>the state of the art of machine learning</sup>"
}

# ML 활동명을 매핑 딕셔너리로 변경
results["ML Activities"] = results["ML Activities"].apply(lambda x: map_ml_activities[x])

# ML 활동명을 기준으로 내림차순 정렬
results = results.sort_values(by=["ML Activities"], ascending=False)

# 산점도 차트를 생성하는 사용자 정의 함수 호출
create_scatter_plot(
    # x축: 직무명에서 괄호 앞부분만 추출하여 리스트 생성
    results["Q23"].apply(lambda x: x.split("(")[0]).to_list(), 
    # y축: ML 활동명에서 괄호 앞부분만 추출하여 리스트 생성
    results["ML Activities"].apply(lambda x: x.split("(")[0]),
    # 툴팁에 표시할 텍스트 포맷 설정
    "Role: %{x}<br>" +
        "ML Activity: %{y}<br>" +
        "Percentage: %{marker.size:,}" +
        "<extra></extra>",
    # 마커 크기: 상대 비율(백분율) 값
    results['relative_percent'] * 100, 
    # 마커 색상: 상대 비율(백분율) 값
    results['relative_percent'] * 100, 
    # 차트 제목
    "Tasks among ML and Data Science Roles",
    # 차트 부제목
    "Questions Data: ML Activity (Q28) and Job Role (Q23)",
    # 차트 설명: 마커의 크기와 색상이 의미하는 내용
    "Size,Color: Percentage of Respondents - <br>The number of respondents with the relevant job position doing the respective ML activity<br>divided by the total number of respondents with the same job position."
)



# 새로운 노드 색상 팔레트 (10개 색상)
color_node_new = [
    "#003f5c",  # 어두운 파랑
    "#58508d",  # 보라
    "#bc5090",  # 핑크
    "#ff6361",  # 붉은 계열
    "#ffa600",  # 노랑
    "#2f4b7c",  # 청자색
    "#a05195",  # 보라톤
    "#d45087",  # 분홍톤
    "#f95d6a",  # 오렌지
    "#ff7c43"   # 주황
]

# 새로운 링크 색상 팔레트 (24개 색상, 투명도 0.5 적용)
color_link_new = [
    "rgba(0,63,92,0.5)", "rgba(88,80,141,0.5)", "rgba(0,63,92,0.5)", "rgba(88,80,141,0.5)",
    "rgba(88,80,141,0.5)", "rgba(88,80,141,0.5)",
    "rgba(188,80,144,0.5)", "rgba(188,80,144,0.5)", "rgba(255,99,97,0.5)", "rgba(255,99,97,0.5)",
    "rgba(255,99,97,0.5)", "rgba(255,99,97,0.5)",
    "rgba(255,166,0,0.5)", "rgba(255,166,0,0.5)", "rgba(255,166,0,0.5)", "rgba(255,166,0,0.5)",
    "rgba(255,166,0,0.5)", "rgba(255,166,0,0.5)",
    "rgba(47,75,124,0.5)", "rgba(47,75,124,0.5)", "rgba(47,75,124,0.5)", "rgba(47,75,124,0.5)",
    "rgba(47,75,124,0.5)", "rgba(47,75,124,0.5)"
]

fig = go.Figure(data=[go.Sankey(
    node=dict(
        pad=10,
        thickness=20,
        line=dict(color="black", width=0.3),
        label=label,
        color=color_node_new,
    ),
    link=dict(
        source=source,
        target=target,
        value=value,
        color=color_link_new
    ),
    arrangement='snap'
)])

layout = dict(
    font=dict(color='#7b6b59'),
    height=700
)

fig.update_layout(layout)
fig.show()



# 관심 직무 목록 (노드 인덱스 0~3에 해당)
jobs_in_scope = [
    "Data Scientist",
    "Data Analyst (Business, Marketing, Financial, Quantitative, etc)",
    "Research Scientist",
    "Machine Learning/ MLops Engineer"
]

# ML 채택 단계 (노드 인덱스 4~7에 해당)
models_in_scope = [
    "Models in Production",
    "Not Started",
    "Exploration Stage",
    "Generating Insights"
]

# 업무(task) 관련 질문 열 (나중에 job-model에서 task로의 연결에 사용)
tasks_in_scope = [
    "Q28_1",
    "Q28_2",
    "Q28_3",
    "Q28_4",
    "Q28_5",
    "Q28_6",
]

# Sankey 다이어그램에 사용될 전체 노드 레이블 (인덱스 0~13)
# 0~3: 직무, 4~7: 모델 단계, 8~13: 각 업무(task)
label = [
    "Data Scientist",              # 0
    "Data Analyst",                # 1
    "Research Scientist",          # 2
    "Machine Learning Engineer",   # 3
    "Models in Production",        # 4
    "Not Started",                 # 5
    "Exploration Stage",           # 6
    "Generating Insights",         # 7
    'Analyze and Understand Data', # 8
    'Build and run data infrastructure', # 9
    'Create ML to explore new areas',      # 10
    'Build and run ML',                    # 11
    'Improve ML Models',                   # 12 
    'Research to advance the state of ML'  # 13
]

# source와 target 리스트는 노드 간 연결(링크)을 정의함
# 예: 0 -> 4는 "Data Scientist"에서 "Models in Production"으로의 연결을 의미
source = [0, 0, 0, 0, 4,4,4,4,4,4, 5,5,5,5,5,5, 6,6,6,6,6,6, 7,7,7,7,7,7,  
         1, 1, 1, 1, 4,4,4,4,4,4, 5,5,5,5,5,5, 6,6,6,6,6,6, 7,7,7,7,7,7,
         2, 2, 2, 2, 4,4,4,4,4,4, 5,5,5,5,5,5, 6,6,6,6,6,6, 7,7,7,7,7,7,
         3, 3, 3, 3, 4,4,4,4,4,4, 5,5,5,5,5,5, 6,6,6,6,6,6, 7,7,7,7,7,7]

target = [4,5,6,7,  8,9,10,11,12,13, 8,9,10,11,12,13, 8,9,10,11,12,13, 8,9,10,11,12,13, 
          4, 5, 6, 7,  8,9,10,11,12,13, 8,9,10,11,12,13, 8,9,10,11,12,13, 8,9,10,11,12,13, 
          4, 5, 6, 7,  8,9,10,11,12,13, 8,9,10,11,12,13, 8,9,10,11,12,13, 8,9,10,11,12,13, 
          4, 5, 6, 7,  8,9,10,11,12,13, 8,9,10,11,12,13, 8,9,10,11,12,13, 8,9,10,11,12,13 ]

# value 리스트는 각 링크의 값을 지정
# 첫 번째 for문: 직무별로 모델 단계(ML 채택 상태)에 해당하는 응답 수(job -> model 연결)
# 두 번째 for문: 직무와 모델 단계별로 각 업무(task)에 해당하는 응답 수(job-model -> task 연결)

value = []

for job in jobs_in_scope:
    # 직무별 모델 단계 응답 수 (job -> model)
    for model in models_in_scope:
        value.append(
            scope_df[
                (scope_df["Q23"] == job) &
                (scope_df["ML_adoption_class"] == model)
            ].shape[0])
    # 직무별, 모델 단계별 업무(task) 응답 수 (job-model -> task)
    for model in models_in_scope:
        for col in tasks_in_scope:
            value.append(
               scope_df[
                (scope_df["Q23"] == job) &
                (scope_df["ML_adoption_class"] == model)][col].count())

# 노드 색상 설정: 
# 앞 4개는 직무, 다음 4개는 모델 단계, 마지막 6개는 업무(task)
color_node = ["#CC5600", "#9D4800",  "#91281A", "#DA9300"] + ["#c07156"]*4 + ["#325C6E"]*6 

# 링크 색상 설정 (현재 여러 색상 지정 후 주석처리되어 있음)
color_link = ["#DDCECC"]*4 + ["#89CFF0"]*24 + ["#DA9300"]*4 + ["pink"]*24 + ["#FAC898"] * 4 + ["pink"]*24 + ["#F8EED9"] * 4 + ["pink"]*24 

# Sankey 다이어그램 생성
fig = go.Figure(data=[go.Sankey(
    node = dict(
      pad = 15,                      # 노드 간 간격 조절
      thickness = 20,                # 노드 두께 설정
      line = dict(color = "black", width = 0.5),  # 노드 테두리 색상과 두께 설정
      label = label,                 # 노드 레이블 지정
      color = color_node,            # 각 노드의 색상 지정
    ),
    link = dict(
      source = source,               # 연결의 시작 노드 인덱스
      target = target,               # 연결의 목표 노드 인덱스
      value = value,                 # 각 연결의 값(강도)
      color = color_link          # 연결 색상 (필요시 주석 해제)
  ))])

# 제목 포맷 문자열 (현재 레이아웃에 사용되지는 않음)
large_title_format = "<span style='font-size:30px; font-family:Times New Roman'>Tasks among ML and Data Science Roles</span>"

# 레이아웃 설정 (폰트 색상 등)
layout = dict(
    font = dict(color = '#7b6b59'),
)

# 레이아웃 업데이트 후 Sankey 다이어그램 출력
fig.update_layout(layout)
fig.show()


