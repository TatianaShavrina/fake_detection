
# coding: utf-8

# In[7]:

import gensim
import os
import collections
import smart_open
import random
import re
import pandas as pd
from sklearn.cross_validation import train_test_split


# 
# path = r'/media/mi_air/0F0B7DDE62EEA81E/Documents/vk_crawler/5000goldtexts_with_joined_feats.csv'
# data = pd.read_csv(path, sep="\t")

# X = data.text
# y = data.label_num
# print(X.shape)
# print(y.shape)
# x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# In[ ]:

# Set file names for train and test data
"""test_data_dir = '{}'.format(os.sep).join([gensim.__path__[0], 'test', 'test_data'])
lee_train_file = test_data_dir + os.sep + 'lee_background.cor'
lee_test_file = test_data_dir + os.sep + 'lee.cor'"""

X = []
y = []
data2 = open(r'/media/mi_air/0F0B7DDE62EEA81E/Documents/vk_crawler/5000goldtexts_bare.txt', "r")
for line in data2:
    label, text = line.split("\t")[2].strip("\n"),line.split("\t")[1]
    X.append(text)
    y.append(label)
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[ ]:

def read_corpus(fname, tokens_only=False):
    with smart_open.smart_open(fname, encoding="iso-8859-1") as f:
        for i, line in enumerate(f):
            if tokens_only:
                yield gensim.utils.simple_preprocess(line)
            else:
                # For training data, add tags
                yield gensim.models.doc2vec.TaggedDocument(gensim.utils.simple_preprocess(line), [i])


# In[ ]:

train_corpus = list(read_corpus(x_train))
test_corpus = list(read_corpus(x_test, tokens_only=True))


# In[10]:

train_corpus = list(x_train)
test_corpus = list(x_test)



# In[ ]:




# In[11]:

model = gensim.models.doc2vec.Doc2Vec(size=50, min_count=2, iter=55)

model.build_vocab(train_corpus)



# In[ ]:

get_ipython().magic('time model.train(train_corpus, total_examples=model.corpus_count, epochs=model.iter)')



# In[ ]:

model.infer_vector(['только', 'ты', 'можешь', 'предотвратить', 'переобучение'])


# In[ ]:

ranks = []
second_ranks = []
for doc_id in range(len(train_corpus)):
    inferred_vector = model.infer_vector(train_corpus[doc_id].words)
    sims = model.docvecs.most_similar([inferred_vector], topn=len(model.docvecs))
    rank = [docid for docid, sim in sims].index(doc_id)
    ranks.append(rank)
    
    second_ranks.append(sims[1])


# In[ ]:

collections.Counter(ranks)  # Results vary due to random seeding and very small corpus


# In[ ]:

print('Document ({}): «{}»\n'.format(doc_id, ' '.join(train_corpus[doc_id].words)))
print(u'SIMILAR/DISSIMILAR DOCS PER MODEL %s:\n' % model)
for label, index in [('MOST', 0), ('MEDIAN', len(sims)//2), ('LEAST', len(sims) - 1)]:
    print(u'%s %s: «%s»\n' % (label, sims[index], ' '.join(train_corpus[sims[index][0]].words)))


# In[ ]:

# Pick a random document from the test corpus and infer a vector from the model
doc_id = random.randint(0, len(test_corpus))
inferred_vector = model.infer_vector(test_corpus[doc_id])
sims = model.docvecs.most_similar([inferred_vector], topn=len(model.docvecs))

# Compare and print the most/median/least similar documents from the train corpus
print('Test Document ({}): «{}»\n'.format(doc_id, ' '.join(test_corpus[doc_id])))
print(u'SIMILAR/DISSIMILAR DOCS PER MODEL %s:\n' % model)
for label, index in [('MOST', 0), ('MEDIAN', len(sims)//2), ('LEAST', len(sims) - 1)]:
    print(u'%s %s: «%s»\n' % (label, sims[index], ' '.join(train_corpus[sims[index][0]].words)))

