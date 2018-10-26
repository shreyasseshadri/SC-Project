import tensorflow as tf
import numpy as np
import itertools
from collections import namedtuple

class word2vec:
    
    def __init__(self,corpus):
        self.corpus=None
        self.vectors=None
        words=[]
        sentences=[]
        for sent in corpus:
            sentences.append(sent.split(" "))
            for i in sent.split(" "):
                if i!="." or i!="\n" or i!="-":
                    words.append(i)
        words=set(words)
        self.word2int = {}
        int2word = {}
        self.vocab_size = len(words) # gives the total number of unique words
        for i,word in enumerate(words):
            self.word2int[word] = i
            int2word[i] = word
        
        data = []
        WINDOW_SIZE = 3
        for sentence in sentences:
            for word_index, word in enumerate(sentence):
                for nb_word in sentence[max(word_index - WINDOW_SIZE, 0) : min(word_index + WINDOW_SIZE, len(sentence)) + 1] : 
                    if nb_word != word:
                        data.append([word, nb_word])

        x_train = [] 
        y_train = [] 
        for data_word in data:
            x_train.append(self.to_one_hot(self.word2int[ data_word[0] ], self.vocab_size))
            y_train.append(self.to_one_hot(self.word2int[ data_word[1] ], self.vocab_size))
        X_train = np.asarray(x_train)
        y_train = np.asarray(y_train)
        
        x = tf.placeholder(tf.float32, shape=(None, self.vocab_size))
        y_label = tf.placeholder(tf.float32, shape=(None, self.vocab_size))

        EMBEDDING_DIM = 5 # Same as number of features
        W1 = tf.Variable(tf.random_normal([self.vocab_size, EMBEDDING_DIM]))
        b1 = tf.Variable(tf.random_normal([EMBEDDING_DIM])) 
        hidden_representation = tf.add(tf.matmul(x,W1), b1)
        W2 = tf.Variable(tf.random_normal([EMBEDDING_DIM, self.vocab_size]))
        b2 = tf.Variable(tf.random_normal([self.vocab_size]))
        prediction = tf.nn.softmax(tf.add( tf.matmul(hidden_representation, W2), b2))

        sess = tf.Session()

        cross_entropy_loss = tf.reduce_mean(-tf.reduce_sum(y_label * tf.log(prediction), reduction_indices=[1]))
        train_step = tf.train.AdamOptimizer(0.1).minimize(cross_entropy_loss)
        init = tf.global_variables_initializer()

        n_iters = 2
        with tf.Session() as sess:
            sess.run(init)    
            for _ in range(n_iters):
                sess.run(train_step, feed_dict={x: X_train, y_label: y_train})
                print('loss is : ', sess.run(cross_entropy_loss, feed_dict={x: x_train, y_label: y_train}))
            self.vectors = sess.run(W1 + b1)

    def to_one_hot(self,data_point_index, vocab_size):
        temp = np.zeros(vocab_size)
        temp[data_point_index] = 1
        return temp

    def get_vector(self,word):
        assert type(word)==type("h")
        return self.vectors[self.word2int[word]]

    def euclidean_dist(self,vec1, vec2):
        return np.sqrt(np.sum((vec1-vec2)**2))

    def find_closest(self,word, vectors):
        min_dist = 10000 
        min_index = -1
        query_vector = self.get_vector(word)
        for index, vector in enumerate(vectors):
            if self.euclidean_dist(vector, query_vector) < min_dist and not np.array_equal(vector, query_vector):
                min_dist = self.euclidean_dist(vector, query_vector)
                min_index = index
        return min_index
    

# corpus=[]
# with open('./data/imdb/imdb.vocab') as f:
#     lines=f.readlines()
#     for line in lines:
#         corpus.append(line.split('\n')[0])

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
