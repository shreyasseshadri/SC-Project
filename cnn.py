import pandas as pd
train_pos=pd.read_csv('./data/imdb/train/pos.csv')
train_neg=pd.read_csv('./data/imdb/train/neg.csv')
test_pos=pd.read_csv('./data/imdb/test/pos.csv')
test_neg=pd.read_csv('./data/imdb/test/neg.csv')
print('imported data.....')

def clear_text(text):
    return text.replace('"', '').replace("'", "").replace("<br />","").replace(",","").replace("(","").replace(")","").replace("-","")

# train_pos['text']=train_pos['text'].apply(clear_text)
# print(train_pos[train_pos.id==0]['text'])
# for word in train_pos[train_pos.id==0]['text']:
#     print(word,end=' ')


# li=train_pos[train_pos.id==0]['text'].apply(lambda x:x.split('.'))
# for word in li:
#     print(word,end=' ')
# # print(train_pos['t'])

train_pos_sent=train_pos['text'].apply(clear_text).tolist()
train_neg_sent=train_neg['text'].apply(clear_text).tolist()
test_pos_sent=test_pos['text'].apply(clear_text).tolist()
test_neg_sent=test_neg['text'].apply(clear_text).tolist()


corpus=train_pos_sent+train_neg_sent+test_pos_sent+test_neg_sent
