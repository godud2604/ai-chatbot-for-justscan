# %%
import pandas as pd
from bs4 import BeautifulSoup

# CSV 파일 읽기
df = pd.read_csv('./data/customer-center.csv')

# %%

df = pd.DataFrame(df)
df = df[['title', 'body']]

# HTML 태그 제거 함수
def remove_html_tags(text):
    soup = BeautifulSoup(text, 'html.parser')
    return soup.get_text()

# 'body' 컬럼의 HTML 태그 제거
df['body'] = df['body'].apply(remove_html_tags)

# %%
# title과 body를 하나의 문장으로 정리
df['text'] = df.apply(lambda row: f"{row['title'][:-1] if row['title'].endswith('?') else row['title']}. {row['body']}", axis=1)

# %%
# title과 body를 하나의 문장으로 정리
def combine_text(row):
    title = row['title']
    body = row['body']

    # title에 물음표가 있을 경우, 유연하게 문장 이어주기
    if title.endswith('?'):
        combined_text = f"{body}"
    else:
        combined_text = f"{title}. {body}"

    return combined_text

df['text'] = df.apply(combine_text, axis=1)

# 결과를 새로운 CSV 파일로 저장
df[['text']].to_csv('./data/eda-customer-center.csv', index=False)

# %%
