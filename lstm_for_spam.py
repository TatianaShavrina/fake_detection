
# coding: utf-8

# In[50]:

# LSTM with dropout for sequence classification 
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
from keras.preprocessing import sequence, text
from keras.layers.embeddings import Embedding
from keras.optimizers import RMSprop, Adam


# In[9]:

import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
get_ipython().magic('pylab inline')


# In[21]:

# fix random seed for reproducibility
numpy.random.seed(7)

#fetching sms spam dataset
path = r'/media/mi_air/0F0B7DDE62EEA81E/Documents/vk_crawler/5000goldtexts_with_joined_feats.csv'
data = pd.read_csv(path, sep="\t")

#binarizing
data['label_num'] = data.allclass.map({'notspam':0, 'spam':1})
data.head()



# In[22]:

X = data.text
y = data.label_num
print(X.shape)
print(y.shape)


# In[51]:






###################################
tk = text.Tokenizer(num_words=200, lower=True)
tk.fit_on_texts(X)

x = tk.texts_to_sequences(X)

print (len(tk.word_counts))

###################################
max_len = 80
print ("max_len ", max_len)
print('Pad sequences (samples x time)')

x = sequence.pad_sequences(x, maxlen=max_len)



max_features = 300
model = Sequential()
print('Build model...')

model.add(Embedding(max_features, 128, input_length=max_len, dropout=0.2))
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1))
model.add(Activation('hard_sigmoid'))

optimizer = RMSprop(lr=0.01)
optimizer2 = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

model.compile(loss='binary_crossentropy', optimizer="adam", metrics=['accuracy'])

model.fit(x, y=y, batch_size=128, epochs=10, verbose=1, validation_split=0.2, shuffle=True)


# In[52]:

json_string = model.to_json()
with open(r"/media/mi_air/0F0B7DDE62EEA81E/Documents/vk_crawler/models/model_lstm10_hard_sigmoid_adam_batch64.json", "w") as text_file:
    text_file.write(json_string)
model.save_weights(r"/media/mi_air/0F0B7DDE62EEA81E/Documents/vk_crawler/models/model_weights_lstm10_hard_sigmoid_adam_batch64.hdf5")
model.save_weights(r"/media/mi_air/0F0B7DDE62EEA81E/Documents/vk_crawler/models/model_weights_lstm10_hard_sigmoid_adam_batch64.h5")
print("saved model to disk")


# #на сигмоиде, батч 128, 10 эпох - loss: 0.4983 - val_acc: 0.8120
# #на софтмаксе совсем плохо! elu неплохо, быстро растет
# #relu хорошо начинает, резко прыгает
# #hard_sigmoid пока лучше всего loss: 0.4452 - acc: 0.8369 - val_loss: 0.6239 - val_acc: 0.8180
# #linear - ужас!
# #на сигмоиде, но с векторов длины 300 уже лучше: loss: 0.2795 - acc: 0.8631 - val_loss: 0.5342 - val_acc: 0.8020
# 
# 
# #ошибка большая, явно нужно больше эпох и батч другой
# 
# #на батче 64
# #loss: 0.2555 - acc: 0.8704 - val_loss: 0.5702 - val_acc: 0.7780
# 
# #уменьш батч, поменять кол-во фич в векторе - сейчас 200
# #посмотреть dence и dropout!
