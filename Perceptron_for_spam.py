
# coding: utf-8

# In[1]:

from __future__ import print_function

import numpy as np
import keras
from keras.datasets import reuters
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence, text
from keras.optimizers import RMSprop, Adam
import pandas as pd
import re

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
get_ipython().magic('pylab inline')


# In[2]:

from sklearn.cross_validation import train_test_split
path = r'/media/mi_air/0F0B7DDE62EEA81E/Documents/vk_crawler/5000goldtexts_with_joined_feats.csv'
data = pd.read_csv(path, sep="\t")


# In[3]:

X = data.text
y = data.label_num
print(X.shape)
print(y.shape)
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[4]:


tk = Tokenizer(num_words=1000, lower=True)
tk.fit_on_texts(X)

x = tk.texts_to_sequences(X)
x2 = X

print (len(tk.word_counts))

max_len = 80
print ("max_len ", max_len)
print('Pad sequences (samples x time)')

x = sequence.pad_sequences(x, maxlen=max_len)



#max_features = 300
model = Sequential()


# In[5]:

'''Trains and evaluate a simple MLP
on the Reuters newswire topic classification task.
'''
#idx = len(x_train)
#x_train, y_train = np.array(xs[:idx]), np.array(labels[:idx])
#x_test, y_test = np.array(xs[idx:]), np.array(labels[idx:])

max_words = num_words = 1000
batch_size = 16
epochs = 100

print('Loading data...')
#(x_train, y_train), (x_test, y_test) = load_data(num_words=max_words)
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')




# In[6]:

num_classes = 2
print(num_classes, 'classes')

print('Vectorizing sequence data...')

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(X)
x_train = tokenizer.texts_to_sequences(x_train)
x_test = tokenizer.texts_to_sequences(x_test)
x2 = tokenizer.texts_to_sequences(x2)


# In[7]:

x_train = tokenizer.sequences_to_matrix(x_train, mode='tfidf')
x_test = tokenizer.sequences_to_matrix(x_test, mode='tfidf')
x2 = tokenizer.sequences_to_matrix(x2, mode='tfidf')
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)


# In[8]:

print('Convert class vector to binary class matrix '
      '(for use with categorical_crossentropy)')
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
y = keras.utils.to_categorical(y, num_classes)
print('y_train shape:', y_train.shape)
print('y_test shape:', y_test.shape)

print('Building model...')
model = Sequential()
model.add(Dense(256, input_shape=(max_words,)))
#model.add(Activation('tanh'))
model.add(Activation('elu'))
model.add(Dropout(0.5))
model.add(Dense(64))
model.add(Activation('sigmoid'))
model.add(Dropout(0.5))
#model.add(Dense(16))
#model.add(Activation('linear'))
#model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

optimizer = RMSprop(lr=0.01,decay=0.0002)
optimizer2 = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0002)

model.compile(loss='categorical_crossentropy',
              optimizer=optimizer2,
              metrics=['accuracy'])

history = model.fit(x2, y=y,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_split=0.2)
#score = model.evaluate(x_test, y_test,
#                       batch_size=batch_size, verbose=1)
print('Test score:', score[0])
print('Test accuracy:', score[1])


# In[ ]:

json_string = model.to_json()
with open(r"/media/mi_air/0F0B7DDE62EEA81E/Documents/vk_crawler/models/model_simplenn2_100_adam_batch16.json", "w") as text_file:
    text_file.write(json_string)
model.save_weights(r"/media/mi_air/0F0B7DDE62EEA81E/Documents/vk_crawler/models/model_weights_simplenn2_100_adam_batch16.hdf5")
model.save_weights(r"/media/mi_air/0F0B7DDE62EEA81E/Documents/vk_crawler/models/model_weights_simplenn2_100_adam_batch16.h5")
print("saved model to disk")


# In[ ]:



