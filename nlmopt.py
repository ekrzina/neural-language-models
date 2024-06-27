import random
import re
import os
import time

from nltk.corpus import gutenberg
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

import numpy as np
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, GRU, Embedding
from keras.callbacks import EarlyStopping, ModelCheckpoint
from nltk.translate.bleu_score import sentence_bleu

from sklearn.model_selection import train_test_split

# Constants
OUTPUT_DIR = "model_out_fin"
MODEL_NAME = "sentgen_v1_weights.h5"
EMBEDDING_DIM = 100     # glove embeddings dimension
BATCH_SIZE = 128
TEXT_FILE = "sample.txt"

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

# Function to load GloVe embeddings
def load_glove_embeddings(embedding_dim):
    glove_dir = 'glove'
    print("Loading GloVe embeddings...")
    embedding_file = os.path.join(glove_dir, f'glove.6B.{embedding_dim}d.txt')
    embeddings_index = {}
    with open(embedding_file, encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    print(f"Total {len(embeddings_index)} word vectors loaded.")
    return embeddings_index

# Function to create embedding matrix
def create_embedding_matrix(word_index, embeddings_index, embedding_dim):
    vocabulary_size = len(word_index) + 1
    embedding_matrix = np.zeros((vocabulary_size, embedding_dim))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return embedding_matrix

# Function to preprocess text
def preprocess_text(text):
    # Lowercase and remove punctuation
    new_text = text.lower()
    new_text = re.sub(r"'s\b", "", new_text)
    new_text = re.sub("[^a-zA-Z]", " ", new_text)

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    long_words = [word for word in new_text.split() if len(word) >= 3 and word not in stop_words]
    
    return " ".join(long_words).strip()

# Function to create sequences from text
def create_sequences(text, seq_length):
    print("Creating model sequences...")
    sequences = []
    for i in range(seq_length, len(text)):
        seq = text[i - seq_length:i + 1]
        sequences.append(seq)
    print(f"Total sequences: {len(sequences)}")
    return sequences

# Function to encode sequences
def encode_sequences(sequences, mapping):
    sequences_encoded = []
    for seq in sequences:
        encoded_seq = [mapping[char] for char in seq]
        sequences_encoded.append(encoded_seq)
    return np.array(sequences_encoded)

# Function to create and compile model with GloVe embedding
def create_model(vocabulary_size, embedding_dim, embedding_matrix, seq_length):
    model = Sequential()
    model.add(Embedding(vocabulary_size, embedding_dim, weights=[embedding_matrix], input_length=seq_length, trainable=False))
    model.add(GRU(150, recurrent_dropout=0.1, dropout=0.1))
    model.add(Dense(vocabulary_size, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# Function to save model weights
def save_model_weights(model):
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    model_path = os.path.join(OUTPUT_DIR, MODEL_NAME)
    model.save_weights(model_path)
    print(f"Model weights saved to {model_path}.")

# Function to generate text from model
def generate_text(model, mapping, seq_length, seed_text, n_chars):
    in_text = seed_text
    for _ in range(n_chars):
        encoded = [mapping[char] for char in in_text]
        encoded = pad_sequences([encoded], maxlen=seq_length, truncating='pre')
        yhat = np.argmax(model.predict(encoded), axis=-1)
        out_char = ''
        for char, index in mapping.items():
            if index == yhat:
                out_char = char
                break
        in_text += out_char
    return in_text

# Function to evaluate BLEU score for generated text
def evaluate_bleu(reference_text, generated_text):
    reference_tokens = reference_text.split()
    generated_tokens = generated_text.split()
    return sentence_bleu([reference_tokens], generated_tokens)

# Imports text data from local source
def import_data(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        text = file.read()
    return text

# Main function to train and use the model
def main():
    # Load and preprocess text data
    modeltexts = (
        gutenberg.raw('austen-emma.txt') + '\n' +
        gutenberg.raw('shakespeare-hamlet.txt') + '\n' +
        gutenberg.raw('carroll-alice.txt')
    )
    #modeltexts = import_data(TEXT_FILE)
    preprocessed_text = preprocess_text(modeltexts)

    # Create sequences and mappings
    chars = sorted(list(set(preprocessed_text)))
    mapping = dict((c, i) for i, c in enumerate(chars))
    sequences = create_sequences(preprocessed_text, seq_length=30)
    sequences_encoded = encode_sequences(sequences, mapping)

    # Create GloVe embeddings
    embeddings_index = load_glove_embeddings(EMBEDDING_DIM)
    embedding_matrix = create_embedding_matrix(mapping, embeddings_index, EMBEDDING_DIM)

    # Create and compile model
    vocabulary_size = len(mapping) + 1
    model = create_model(vocabulary_size, EMBEDDING_DIM, embedding_matrix, seq_length=30)
    print(model.summary())

    # Split data into train and validation sets
    x, y = sequences_encoded[:, :-1], sequences_encoded[:, -1]
    y = to_categorical(y, num_classes=vocabulary_size)
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.1, random_state=74)

    # Train model and evaluate BLEU score
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10),
        ModelCheckpoint(filepath=os.path.join(OUTPUT_DIR, MODEL_NAME), save_best_only=True)
    ]
    start_time = time.time()
    model.fit(x_train, y_train, epochs=200, batch_size=BATCH_SIZE, verbose=2, validation_data=(x_val, y_val), callbacks=callbacks)
    end_time = time.time()
    epoch_duration = (end_time - start_time) / 200
    print(f"Average time per epoch: {epoch_duration:.2f} seconds")

    # Generate text
    seed_text = get_random_word(modeltexts)
    generated_text = generate_text(model, mapping, 30, seed_text, 100)
    print("Generated Text:")
    print(generated_text)

    # Evaluate BLEU score
    reference_text = get_random_sentence(modeltexts)
    bleu_score = evaluate_bleu(reference_text, generated_text)
    print(f"BLEU score: {bleu_score}")

    # Save model weights
    save_model_weights(model)

if __name__ == "__main__":
    main()
