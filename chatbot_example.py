# %%
import openai
import streamlit as st
import os
import pandas as pd
import numpy as np
from numpy import dot
from numpy.linalg import norm
import ast
from openai.embeddings_utils import get_embedding
from streamlit_chat import message

# from dotenv import load_dotenv
# load_dotenv()

# openai.api_key = ''

folder_path = './data'
file_name = 'embedding.csv'
file_path = os.path.join(folder_path, file_name)
eda_customer_center_path = './data/eda-customer-center.csv'
eda_wiki_path = './data/eda-wiki.csv'

if os.path.isfile(file_path):
    print(f"{file_path} 파일이 존재합니다.")
    df = pd.read_csv(file_path)
    df['embedding'] = df['embedding'].apply(ast.literal_eval)

else:
    df_customer_center = pd.read_csv(eda_customer_center_path)
    df_wiki = pd.read_csv(eda_wiki_path)

    # 병합된 데이터프레임 생성
    df = pd.concat([df_customer_center, df_wiki], ignore_index=True)

    # 데이터프레임의 text 열에 대해서 embedding을 추출
    df['embedding'] = df.apply(lambda row: get_embedding(
        row.text,
        engine="text-embedding-ada-002"
    ), axis=1)
    df.to_csv(file_path, index=False, encoding='utf-8-sig')


# %%
def cos_sim(A, B):
  return dot(A, B)/(norm(A)*norm(B))

def return_answer_candidate(df, query):
    query_embedding = get_embedding(
        query,
        engine="text-embedding-ada-002"
    )
    df["similarity"] = df.embedding.apply(lambda x: cos_sim(np.array(x), np.array(query_embedding)))
    top_three_doc = df.sort_values("similarity", ascending=False).head(3)
    return top_three_doc

def create_prompt(df, query):
    result = return_answer_candidate(df, query)
    system_role = f"""You are an artificial intelligence language model named "저스트 스캔" that specializes in summarizing \
    and answering documents about just scan. You were developed by a developer named Lee Hae-young.
    You need to take a given document and return a very detailed summary of the document in the query language.
    Here are the document: 
            doc 1 :""" + str(result.iloc[0]['text']) + """
            doc 2 :""" + str(result.iloc[1]['text']) + """
            doc 3 :""" + str(result.iloc[2]['text']) + """
    You must return in Korean. Return a accurate answer based on the document.
    """
    user_content = f"""User question: "{str(query)}". """

    messages = [
        {"role": "system", "content": system_role},
        {"role": "user", "content": user_content}
    ]

    return messages

# # 대화 히스토리를 최적화
# def manage_chat_history(messages, max_tokens=4096):
#     # 메시지의 총 토큰 수 계산
#     total_tokens = sum(len(tokenize_message(msg)) for msg in messages)

#     # 토큰 수가 최대 허용치를 초과하는 경우 필요 없는 메시지 제거
#     while total_tokens > max_tokens and messages:
#         removed_tokens = len(tokenize_message(messages.pop(0)))
#         total_tokens -= removed_tokens

#     return messages

def tokenize_message(message):
    # 메시지를 토큰화하여 토큰 수 반환
    # 실제로 사용하는 토큰화 방법에 따라 수정이 필요합니다.
    return message['content'].split()


def generate_response(messages):
    try:
        result = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0.4,
            max_tokens=500)
        response_content = result['choices'][0]['message']['content']
        return response_content
    except Exception as e:
        # st.error(f"오류 발생: {e}")
        return "챗봇 응답 생성에 실패했습니다. 고객 센터에 문의해주세요."


# st.image('images/ask_me_chatbot.png')

if 'generated' not in st.session_state:
    st.session_state['generated'] = []

if 'past' not in st.session_state:
    st.session_state['past'] = []

with st.form('form', clear_on_submit=True):
    user_input = st.text_input('저스트 스캔에 대해 물어보세요!', '', key='input')
    submitted = st.form_submit_button('Send')

if submitted and user_input:
    with st.spinner("답변을 작성 중이에요. 조금만 기다려주세요 :)"):
        # 프롬프트 생성 후 프롬프트를 기반으로 챗봇의 답변을 반환
        prompt = create_prompt(df, user_input)
        chatbot_response = generate_response(prompt)
        st.session_state['past'].append(user_input)
        st.session_state["generated"].append(chatbot_response)

if st.session_state['generated']:
    for i in reversed(range(len(st.session_state['generated']))):
        message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
        message(st.session_state["generated"][i], key=str(i))
# %%
