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
words = []
for idx, row in df.iterrows():
    words.append(row.values[0].split())

# %%
stop_words = ["'", '[안내]', '.', '!', '?', '/', ',', '[업데이트]']

remove_stopword_words = []
for word in words:
    for stem in word:
        if stem not in stop_words:
            remove_stopword_words.append(word)

remove_stopword_words
# %%
class Encoder(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.encoder_block = nn.Sequential(
            nn.Linear(input_size, 250),
            nn.CELU(),
            nn.Linear(250, 250),
            nn.CELU(),
            nn.Linear(250, 2),
        )
    
    def forward(self, x: torch.Tensor):
        return self.encoder_block(x)
    
class Decoder(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.decoder_block = nn.Sequential(
            nn.Linear(2, 250),
            nn.CELU(),
            nn.Linear(250, 250),
            nn.CELU(),
            nn.Linear(250, input_size),
        )

    def forward(self, x: torch.Tensor):
        return self.decoder_block(x)

# %%
encoder = Encoder(input_size=1)
decoder = Decoder(input_size=1)

optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=1e-3)
criterion = nn.MSELoss()

# %%
from torch.utils.data import DataLoader, TensorDataset
data = torch.tensor(df['w2v'].values * 1e-10, dtype=torch.float32)
dataset = TensorDataset(data)
dataloader = DataLoader(dataset, batch_size=8)

for epoch in range(100):
    for batch in dataloader:
        break
# %%
