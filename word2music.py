from transformers import BertJapaneseTokenizer, BertModel
import torch

class SentenceBertJapanese:
    def __init__(self, model_name_or_path, device=None):
        self.tokenizer = BertJapaneseTokenizer.from_pretrained(model_name_or_path)
        self.model = BertModel.from_pretrained(model_name_or_path)
        self.model.eval()


    def _mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    @torch.no_grad()
    def encode(self, sentences, batch_size=8):
        all_embeddings = []
        iterator = range(0, len(sentences), batch_size)
        for batch_idx in iterator:
            batch = sentences[batch_idx:batch_idx + batch_size]

            encoded_input = self.tokenizer.batch_encode_plus(batch, padding="longest", 
                                           truncation=True, return_tensors="pt")
            model_output = self.model(**encoded_input)
            sentence_embeddings = self._mean_pooling(model_output, encoded_input["attention_mask"]).to('cpu')

            all_embeddings.extend(sentence_embeddings)

        # return torch.stack(all_embeddings).numpy()
        return torch.stack(all_embeddings)


bertModel = SentenceBertJapanese("bertModel")

from tensorflow.keras.models import load_model
import numpy as np

musics = [
'おとぼけダンス', '大混乱', 'Funny_Funny', '全力で逃げる時のBGM', 'トッカータとフーガ〜ギャグVer〜', 'シラけムードは少し気まずい',
'修羅場_怒り心頭', 'おばけとかぼちゃのスープ', 'いちごホイップ', 'eye-catch', '夏の霧', '昼下がり気分',
'Happy_birthday', 'はっぴいばあすでいつーゆー', 'yonhonnorecorder', 'happytime', '夏休みの探検', 'Recollections', 'パステルハウス'
]

model = load_model('model.h5')

def getMusic(sentence):
    predicted = model(np.array(bertModel.encode([sentence], batch_size=8)))
    index = np.argmax(predicted[0])
    return musics[index]