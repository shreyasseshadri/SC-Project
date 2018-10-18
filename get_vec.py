import gensim


model = gensim.models.KeyedVectors.load_word2vec_format('./data/GoogleNews-vectors-negative300.bin', binary=True,limit=500000)  

corpus=[]
with open('./data/imdb/imdb.vocab') as f:
    lines=f.readlines()
    # print(lines[0].split('\n'))
    # print(lines[1].split('\n'))
    for line in lines:
        corpus.append(line.split('\n')[0])

common_words=[word for word in corpus if word in model.vocab]
print(len(corpus),len(common_words))
print(common_words[:10])
print(model[common_words[:10]])
# common_words=[word for ]

