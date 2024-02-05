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

folder_path = './data'
file_name = 'embedding.csv'
file_path = os.path.join(folder_path, file_name)
gloud_path = './data/eda-gloud-customer-center.csv'

if os.path.isfile(file_path):
    print(f"{file_path} 파일이 존재합니다.")
    df = pd.read_csv(file_path)
    df['embedding'] = df['embedding'].apply(ast.literal_eval)

else:
    df = pd.read_csv(gloud_path)

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
    and answering documents about just scan
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

def generate_response(messages):
    result = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0.4,
        max_tokens=500)
    return result['choices'][0]['message']['content']

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