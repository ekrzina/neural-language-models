import random
import re
import os
import time

from nltk.corpus import gutenberg
from nltk.tokenize import sent_tokenize, word_tokenize

import numpy as np
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, GRU, Embedding
from keras.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.model_selection import train_test_split

OUTPUT_DIR = "model_out"
MODEL_NAME = "sentgen_v2_weights.h5"

# Get random sentence for check
def get_random_sentence(text):
    sentences = sent_tokenize(text)
    random_sentence = random.choice(sentences)
    return random_sentence

# Gets random word for sentence generation
def get_random_word(text):
    sentence = get_random_sentence(text)
    words = word_tokenize(sentence)
    random_word = random.choice(words)
    return random_word

# Makes text lowercase, removes punctuation and saves only long words
def clean_text(text):
    new_text = text.lower()
    new_text = re.sub(r"'s\b", "", new_text)
    new_text = re.sub("[^a-zA-Z]", " ", new_text)

    # Remove short words
    long_words = [i for i in new_text.split() if len(i) >= 3]
    return " ".join(long_words).strip()

# Creates sequences for predicting words
def create_sequence(text):
    print("Creating model sequences...")
    length = 30
    sequences = []
    for i in range(length, len(text)):
        seq = text[i - length:i + 1]
        sequences.append(seq)
    print("Sequences created.")
    return sequences

# Encodes sequences
def encode_sequences(seq, mapping):
    sequences = []
    for line in seq:
        encoded_seq = [mapping[char] for char in line]
        sequences.append(encoded_seq)
    return sequences

# Creates a training and validation set for the model
def create_train_val_sets(vocabulary_size, sequences):
    sequences = np.array(sequences)
    x, y = sequences[:, :-1], sequences[:, -1]
    y = to_categorical(y, num_classes=vocabulary_size)

    # Create train and val sets
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.1, random_state=74)
    print(f"Training shape: {x_train.shape}\nValidation shape: {x_val.shape}")
    return x_train, x_val, y_train, y_val

# Trains the dataset
def start_training(mapping, sequences):
    vocabulary_size = len(mapping)
    x_train, x_val, y_train, y_val = create_train_val_sets(vocabulary_size, sequences)

    model = Sequential()
    model.add(Embedding(vocabulary_size, 50, input_length=30))
    model.add(GRU(150, recurrent_dropout=0.1, dropout=0.1))
    model.add(Dense(vocabulary_size, activation='softmax'))
    print(model.summary())

    model.compile(loss='categorical_crossentropy', metrics=['acc'], optimizer='adam')
    
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10),
        ModelCheckpoint(filepath=os.path.join(OUTPUT_DIR, MODEL_NAME), save_best_only=True)
    ]

    start_time = time.time()
    model.fit(x_train, y_train, epochs=200, verbose=2, validation_data=(x_val, y_val), callbacks=callbacks)
    end_time = time.time()

    epoch_duration = (end_time - start_time) / 500
    print(f"Average time per epoch: {epoch_duration:.2f} seconds")

    print("Model finished training.")

# Saves the trained model
def save_model(model):
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    model_path = os.path.join(OUTPUT_DIR, MODEL_NAME)
    model.save_weights(model_path)
    print(f"Model weights saved to {model_path}.")

# Generates new text from created model - model mapping, length of the sequence, random seed and number of characters
def generate_text_from_model(model, mapping, seq_length, seed_text, n_chars):
    in_text = seed_text

    for _ in range(n_chars):
        # Encode characters as integers
        encoded = [mapping[char] for char in in_text]
        # Truncate sequences to fixed length
        encoded = pad_sequences([encoded], maxlen=seq_length, truncating='pre')
        # Predict character
        yhat = np.argmax(model.predict(encoded, verbose=0), axis=-1)
        # Reverse map integer to character
        out_char = ''
        for char, index in mapping.items():
            if index == yhat:
                out_char = char
                break
        # Append to input
        in_text += out_char
    return in_text

# Get texts for model
modeltexts = (
    gutenberg.raw('austen-emma.txt') + '\n' +
    gutenberg.raw('shakespeare-hamlet.txt') + '\n' +
    gutenberg.raw('carroll-alice.txt')
)

preprocessed = clean_text(modeltexts)
sequen = create_sequence(preprocessed)

# Encode characters
chars = sorted(list(set(preprocessed)))
mapping = dict((c, i) for i, c in enumerate(chars))
sequen = encode_sequences(sequen, mapping)

# Start training model
model = start_training(mapping, sequen)

# See what it can do
generated_text = generate_text_from_model(model, mapping, 30, get_random_word(), 100)
print(generated_text)

# Save model
save_model(model)
