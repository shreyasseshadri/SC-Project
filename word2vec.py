# import tensorflow as tf
import numpy as np
import itertools
from collections import namedtuple


import pandas as pd
train_pos=pd.read_csv('./data/imdb/train/pos.csv')
train_neg=pd.read_csv('./data/imdb/train/neg.csv')
test_pos=pd.read_csv('./data/imdb/test/pos.csv')
test_neg=pd.read_csv('./data/imdb/test/neg.csv')
print('imported data.....')

def clear_text(text):
    return text.replace('"', '').replace("'", "").replace("<br \>","")

# train_pos['text']=train_pos['text'].apply(clear_text)
# print(train_pos[train_pos.id==0]['text'])
# for word in train_pos[train_pos.id==0]['text']:
#     print(word,end=' ')


# li=train_pos[train_pos.id==0]['text'].apply(lambda x:x.split('.'))
# for word in li:
#     print(word,end=' ')
# # print(train_pos['t'])

train_pos_sent=train_pos['text'].apply(clear_text).apply(lambda x:x.split('.')).tolist()
train_neg_sent=train_neg['text'].apply(clear_text).apply(lambda x:x.split('.')).tolist()
test_pos_sent=test_pos['text'].apply(clear_text).apply(lambda x:x.split('.')).tolist()
test_neg_sent=test_neg['text'].apply(clear_text).apply(lambda x:x.split('.')).tolist()
print('text preprocessing done......')

# print(train_pos_sent[0])
# print(train_pos_sent[1])
# for word in train_pos_sent[0]:
#     print(word,end=' ')

def drop_less(vec):
    for sent in vec:
        # print(sent)
        for word in sent:
            if len(word)<2:
                del word
    return list(itertools.chain.from_iterable(vec))
# print(train_pos_sent[0])
train_pos_sent=drop_less(train_pos_sent)
train_neg_sent=drop_less(train_neg_sent)
test_pos_sent=drop_less(test_pos_sent)
test_neg_sent=drop_less(test_neg_sent)

corpus=train_pos_sent+train_neg_sent+test_pos_sent+test_neg_sent
print(corpus[0])
corpus_lenght=len(corpus)
for i,text in enumerate(corpus):
    corpus[corpus_lenght-i-1]=corpus[corpus_lenght-i-1].split()
    if len(corpus[corpus_lenght-i-1])<2:
        del corpus[corpus_lenght-i-1]

print('Total number of sentences : ',len(corpus),'!!! how will i run in my computer!')

print(corpus[:3])
from gensim.models import Word2Vec
# define training data
# train model
# print(corpus[:3])
model = Word2Vec(corpus, min_count=1)
# summarize the loaded model
print(model)
# # summarize vocabulary
words = list(model.wv.vocab)
# print(words)
# # access vector for one word
# print(model['Ya'])
# save model
print("model saved")
model.save('model.bin')
# load model
# new_model = Word2Vec.load('model.bin')
# print(new_model)
