# %%
import pandas as pd

# CSV 파일 읽기
df = pd.read_csv('./data/wiki.csv')

# %%

df = pd.DataFrame(df)
df = df[['Title', '뜻', '사용예시']]

# %%
# text 컬럼 생성
df['text'] = df.apply(lambda row: f"{row['Title']}의 뜻은 {row['뜻']}이다." +
                                    f" {row['Title']}의 사용예시는 {row['사용예시']} 입니다."
                     if pd.notna(row['사용예시']) else f"{row['Title']}의 뜻은 {row['뜻']}이다.",
                     axis=1)

# 사용예시가 NaN인 경우 해당 부분 제거
df['text'] = df['text'].apply(lambda x: x.replace('(사용예시) ', ''))

# %%
# 결과를 새로운 CSV 파일로 저장
df['text'].to_csv('data/eda-wiki.csv', index=False)
# %%
