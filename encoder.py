import numpy as np
from transformers import BertJapaneseTokenizer, BertModel

import torch

# BERTの日本語モデル
MODEL_NAME = 'izumi-lab/bert-small-japanese'

#トークナイザとモデルのロード
tokenizer = BertJapaneseTokenizer.from_pretrained(MODEL_NAME)
bertModel = BertModel.from_pretrained(MODEL_NAME)

def encode(text):
    encoding = tokenizer(
        text,
        return_tensors = 'pt'
        )
    attention_mask = encoding['attention_mask']

    #文章ベクトルを計算
    with torch.no_grad():
        output = bertModel(**encoding)
        last_hidden_state = output.last_hidden_state
        averaged_hidden_state =(last_hidden_state*attention_mask.unsqueeze(-1)).sum(1)/attention_mask.sum(1,keepdim=True) 
        
    sentence_vectors = []
    #文章ベクトルとラベルを追加
    sentence_vectors.append(averaged_hidden_state[0].cpu().numpy())

    #ベクトルとラベルをnumpy.ndarrayにする
    sentence_vectors = np.vstack(sentence_vectors)
    return sentence_vectors


musics = [
'おとぼけダンス', '大混乱', 'Funny_Funny', '全力で逃げる時のBGM', 'トッカータとフーガ〜ギャグVer〜', 'シラけムードは少し気まずい',
'修羅場_怒り心頭', 'おばけとかぼちゃのスープ', 'いちごホイップ', 'eye-catch', '夏の霧', '昼下がり気分',
'Happy_birthday', 'はっぴいばあすでいつーゆー', 'yonhonnorecorder', 'happytime', '夏休みの探検', 'Recollections', 'パステルハウス'
]

from tensorflow.keras.models import load_model

model = load_model('model.h5')
def getMusic(sentence):
    predicted = model(np.array(encode(sentence)))
    index = np.argmax(predicted[0])
    return musics[index]
