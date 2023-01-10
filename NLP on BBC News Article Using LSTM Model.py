#%%
# Import packages

import pandas as pd
import re
import numpy as np
import datetime
import os
import pickle
import json

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Dense, LSTM, Dropout, Embedding
from tensorflow.keras import Sequential
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, confusion_matrix

# %%
# Data loading
url = 'https://raw.githubusercontent.com/susanli2016/PyCon-Canada-2019-NLP-Tutorial/master/bbc-text.csv'
df = pd.read_csv(url)

# %%
# Data inspection
print(df.info())
print(df.head())

# Checking for duplicated data
print(df.duplicated().sum())
# looking for what to remove from the text
print(df['text'][150])

# %%
# Data cleaning
"""
Things to remove
1) numbers
2) punctuations
3) remove special character
4) Make all lower case
"""

for index, data in enumerate(df['text']):
    df['text'][index] = re.sub('[^a-zA-Z]', ' ', df['text'][index]).lower()

print(df['text'][500])

# %%
# Feature selection

text = df['text']
category = df['category']

# %%
# Data preprocessing

# Tokenizer

num_words = 5000
oov_token = '<oov>'

# instantiate Tokenizer()
tokenizer = Tokenizer(num_words=num_words, oov_token=oov_token)

# fitting tokenizer to text
tokenizer.fit_on_texts(text)
word_index = tokenizer.word_index
print(dict(list(word_index.items())[0:10]))

# to transform the text using tokenizer --> mms.transform
text = tokenizer.texts_to_sequences(text)

#%%
# padding
padded_text = pad_sequences(text, maxlen=200, padding='post', truncating='post')

#%%
# One hot encoder
ohe = OneHotEncoder(sparse=False)
category = ohe.fit_transform(category[::,None])

#%%
# Train test split
x_train, x_test, y_train, y_test = train_test_split(padded_text, category, shuffle=True, test_size=0.2, random_state=64)

# %%
# Model development

embedding_layer = 64

model = Sequential()
model.add(Embedding(num_words, embedding_layer))
model.add(LSTM(embedding_layer))
model.add(Dropout(0.35))
model.add(Dense(5, activation='softmax'))

model.summary()

plot_model(model, show_shapes=True)

# %%
# Model compilation
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

#%%
# tensorboard callbacks and earlystopping
logs_path = os.path.join(os.getcwd(), 'multi_label_NLP_project_logs', datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
es = EarlyStopping(patience=5, verbose=1, restore_best_weights=True)
tb = TensorBoard(log_dir=logs_path)

# %%
# Model training
history = model.fit(x_train, y_train, validation_data=(x_test, y_test), batch_size=128, epochs=100, callbacks=[tb, es])

# %%
# Model prediction
y_predicted = model.predict(x_test)

# %%
# Model Analysis

y_predicted = np.argmax(y_predicted, axis=1)
y_test = np.argmax(y_test, axis=1)

#%%
# confusion matrix and classification report of model prediction
print(classification_report(y_test, y_predicted))
cm = confusion_matrix(y_test, y_predicted)
ConfusionMatrixDisplay(cm)

# %%
disp = ConfusionMatrixDisplay(cm)
disp.plot()

# %%
# Model saving
model.save('multi_label_NLP_project_model.h5')

#to save one hot encoder model
with open('multi_label_NLP_project_ohe.pkl', 'wb') as f:
    pickle.dump(ohe, f)

# to save tokenizer
token_json = tokenizer.to_json()
with open('multi_label_NLP_project_tokenizer.json', 'w') as f:
    json.dump(token_json, f)