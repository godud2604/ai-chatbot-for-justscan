# %%
import requests
from bs4 import BeautifulSoup
import re
import pandas as pd

# %%
title_array = []

def crawl_page_title(url):
    # 페이지 요청
    response = requests.get(url)
    
    if response.status_code != 200:
        print('페이지 없음')
    else:
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # id가 pro-gallery-margin-container-pro-blog인 요소 찾기
        gallery_container = soup.find(id='pro-gallery-margin-container-pro-blog')
        
        if gallery_container:
            # class가 gallery-item-container인 모든 요소 찾기
            gallery_items = gallery_container.find_all(class_='gallery-item-container')
            
            # 각 요소의 aria-label 값 출력
            for item in gallery_items:
                aria_label = item.get('aria-label')
                title_array.append(aria_label)

# %% gloud 홈페이지 기사 제목 크롤링
for page_num in range(10):
    if page_num == 0:
        crawl_page_title('https://www.gloud.io/media')       
    else:   
        crawl_page_title('https://www.gloud.io/media/page/' + str(page_num + 1))


# %% title 가공
def transform_title(title):
    # 영어 대문자를 소문자로 변경
    title = title.lower()

    # 특수문자를 띄어쓰기로 변경
    title = re.sub(r"[^가-힣a-zA-Z0-9\s]", " ", title)

    # 중복된 띄어쓰기 제거
    title = re.sub(r"\s+", " ", title)

    # 띄어쓰기를 '-'로 변경
    title = title.replace(" ", "-")

    title = title.strip()

    # result가 맨 앞이 "-"일 경우 제거
    if title.startswith("-"):
        title = title[1:]

    # result가 맨 뒤가 "-"일 경우 제거
    if title.endswith("-"):
        title = title[:1]

    return title

# %% title 가공 결과값 array로 저장
transform_title_array = []

for idx, title in enumerate(title_array):
    transform_title_array.append(transform_title(title))

print(transform_title_array)

# %% 기사 본문 내용 크롤링 함수
def crawl_page_content(url):
    # 페이지 요청
    response = requests.get(url)
    
    if response.status_code != 200:
        print('페이지 없음')
    else:
        # 1. data-breakout 속성이 normal인 것 모두 찾기
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')

        # 2. span의 class가 UYHF3인 것 찾기
        elements_with_class_uyhf3 = soup.find_all('span', class_='UYHF3')

        # 3. 2번에서 찾은 자식들 중 text가 있는 span의 text 가져오기
        texts = []
        for element in elements_with_class_uyhf3:

            # 직계 자식인 span 엘리먼트들만 가져오기
            span_elements_with_text = element.find_all('span', text=True, recursive=False)
            texts.extend([span.text for span in span_elements_with_text])

        # 한 문장으로 출력
        return ' '.join(texts)
    
# %% 기사 본문 내용 크롤링 결과값 array로 저장
contents_array = []

for idx, title in enumerate(transform_title_array):
    contents_array.append(crawl_page_content('https://www.gloud.io/post/' + title))

# %% 데이터프레임 생성
df = pd.DataFrame(contents_array, columns=["text"])

# %% 값이 빈 문자열인 행 제거
df = df.replace("", pd.NA)
df = df.dropna()
# %% 
# CSV 파일로 저장
df.to_csv('data/crawling-gloud-article.csv', index=False)
# %%
