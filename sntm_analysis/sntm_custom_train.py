import numpy as np
import pandas as pd
import re
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Bidirectional
from keras.callbacks import EarlyStopping, ModelCheckpoint

stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = text.lower()
    tokens = re.findall(r'\b\w+\b|[^\w\s]', text)
    words = [stemmer.stem(word) for word in tokens if word not in stop_words]
    preprocessed_text = ' '.join(words)
    return preprocessed_text

df = pd.read_csv('data/dataset.csv', sep=';')

df = df.dropna(subset=['text', 'sentiment'])
X = df['text'].values
y = df['sentiment'].values

# encode labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=27)

max_words = 10000
max_len = 100

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(X_train)

X_train_processed = [preprocess_text(text) for text in X_train]
X_test_processed = [preprocess_text(text) for text in X_test]

X_train_seq = tokenizer.texts_to_sequences(X_train_processed)
X_test_seq = tokenizer.texts_to_sequences(X_test_processed)

X_train_pad = pad_sequences(X_train_seq, maxlen=max_len)
X_test_pad = pad_sequences(X_test_seq, maxlen=max_len)

# --------------
# defining model
# --------------

embedding_dim = 100
vocab_size = min(len(tokenizer.word_index) + 1, max_words)

embedding_index = {}
with open('glove/glove.6B.100d.txt', 'r', encoding='utf-8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embedding_index[word] = coefs

embedding_matrix = np.zeros((vocab_size, embedding_dim))
for word, i in tokenizer.word_index.items():
    if i >= max_words:
        continue
    embedding_vector = embedding_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_len, weights=[embedding_matrix], trainable=False))
model.add(Bidirectional(LSTM(units=128, dropout=0.2, recurrent_dropout=0.2)))
model.add(Dense(units=3, activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

callbacks = [
    EarlyStopping(monitor='val_loss', patience=3),
    ModelCheckpoint('models/best_model.h5', save_best_only=True, save_weights_only=False, monitor='val_accuracy')
]

# --------------
# training model
# --------------

epochs = 20
batch_size = 32
history = model.fit(X_train_pad, y_train, epochs=epochs, batch_size=batch_size, 
                    validation_data=(X_test_pad, y_test), callbacks=callbacks)

loss, accuracy = model.evaluate(X_test_pad, y_test)
print(f'Loss: {loss:.4f}')
print(f'Accuracy: {accuracy:.4f}')

# Predict probabilities for each class
y_pred_probs = model.predict(X_test_pad)

# Convert probabilities to class labels
y_pred = y_pred_probs.argmax(axis=-1)

# Print classification report
print('Classification Report:')
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

model.save('models/sentiment_model_nn.h5')
with open('models/tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)
