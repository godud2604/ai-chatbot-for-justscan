# %%
import openai
import streamlit as st
import os
import pandas as pd
import numpy as np
from numpy import dot
from numpy.linalg import norm
from sklearn.manifold import TSNE
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import CountVectorizer
import ast
import torch
import torch.nn as nn
import torch.optim as optim
from konlpy.tag import Kkma
from openai.embeddings_utils import get_embedding
from streamlit_chat import message

folder_path = './data'
file_name = 'embedding.csv'
file_path = os.path.join(folder_path, file_name)
eda_customer_center_path = './data/eda-customer-center.csv'
eda_wiki_path = './data/eda-wiki.csv'


df_customer_center = pd.read_csv(eda_customer_center_path)
df_wiki = pd.read_csv(eda_wiki_path)

# 병합된 데이터프레임 생성
df = pd.concat([df_customer_center, df_wiki], ignore_index=True)

# %%
# CountVectorizer를 사용하여 단어로 토큰화
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['text'])
# toarray()를 사용하여 희소 행렬을 밀집 행렬로 변환
one_hot_encoded = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())

# 기존 데이터프레임과 합치기
df_encoded = pd.concat([df, one_hot_encoded], axis=1)

# %%
class Encoder(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.encoder_block = nn.Sequential(
            nn.Linear(input_size, 250),
            nn.ReLU(),
            nn.BatchNorm1d(250),
            nn.Linear(250, 500),
            nn.ReLU(),
            nn.BatchNorm1d(500),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.BatchNorm1d(500),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.BatchNorm1d(500),
            nn.Linear(500, 1536),
        )
        # self.weight_initialize()  # Initialize weights

    def weight_initialize(self):
        for layer in self.encoder_block:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor):
        return self.encoder_block(x)

class Decoder(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.decoder_block = nn.Sequential(
            nn.Linear(1536, 500),
            nn.ReLU(),
            nn.BatchNorm1d(500),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.BatchNorm1d(500),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.BatchNorm1d(500),
            nn.Linear(500, 250),
            nn.ReLU(),
            nn.BatchNorm1d(250),
            nn.Linear(250, input_size),
        )
        # self.weight_initialize()  # Initialize weights

    def weight_initialize(self):
        for layer in self.decoder_block:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor):
        return self.decoder_block(x)


# %%
encoder = Encoder(input_size=5539)
decoder = Decoder(input_size=5539)

optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=1e-3)
criterion = nn.MSELoss()

# %%
from torch.utils.data import DataLoader, TensorDataset
data = torch.tensor(df_encoded.iloc[:, 1:].to_numpy(), dtype=torch.float32)
dataset = TensorDataset(data)
dataloader = DataLoader(dataset, batch_size=16)

encoder.train()
decoder.train()

for epoch in range(1000):
    for batch in dataloader:
        optimizer.zero_grad()
        encoded_vec = encoder(batch[0])
        reconst_vec = decoder(encoded_vec)

        loss = criterion(reconst_vec, batch[0])
        loss.backward()
        optimizer.step()
        
    print(f'Epoch | {epoch + 1}/{1000}, Loss: {loss.item()}')

# %%
data = torch.tensor(df_encoded.iloc[:, 1:].to_numpy(), dtype=torch.float32)
dataset = TensorDataset(data)
dataloader = DataLoader(dataset, batch_size=1)

encoder.eval()
decoder.eval()
count = 0
df['embedding'] = ''
with torch.no_grad():
    for batch in dataloader:
        encoded_vec = encoder(batch[0])
        df.at[count, 'embedding'] = encoded_vec.detach().numpy().reshape(-1,).tolist()
        count += 1

# %%
df.to_csv('./embedding_test.csv', index=False, encoding='utf-8-sig')

# %%
