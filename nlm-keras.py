import tensorflow as tf
from keras.layers import StringLookup, Embedding, GRU, Dense
import numpy as np
import os
import time
import random
from nltk.corpus import gutenberg
from nltk import sent_tokenize, word_tokenize

SEQ_LEN = 100
BATCH_SIZE = 64
BUFFER_SIZE = 10000
EMBEDDING_DIM = 256
RNN_UNITS = 1024
EPOCHS = 20

CHECKPOINT_DIR = "sentgen_checkpoints"

class OneStep(tf.keras.Model):
    def __init__(self, model, chars_from_ids, ids_from_chars, temperature=1.0):
        super().__init__()
        self.temperature = temperature
        self.model = model
        self.chars_from_ids = chars_from_ids
        self.ids_from_chars = ids_from_chars

        # Create a mask to prevent "[UNK]" from being generated.
        skip_ids = self.ids_from_chars(['[UNK]'])[:, None]
        sparse_mask = tf.SparseTensor(
            # Put a -inf at each bad index.
            values=[-float('inf')]*len(skip_ids),
            indices=skip_ids,
            # Match the shape to the vocabulary
            dense_shape=[len(ids_from_chars.get_vocabulary())])
        self.prediction_mask = tf.sparse.to_dense(sparse_mask)

    @tf.function
    def generate_one_step(self, inputs, states=None):
        input_chars = tf.strings.unicode_split(inputs, 'UTF-8')
        input_ids = self.ids_from_chars(input_chars).to_tensor()

        # predicted_logits.shape is [batch, char, next_char_logits]
        predicted_logits, states = self.model(inputs=input_ids, states=states,
                                              return_state=True)
        # Only use the last prediction
        predicted_logits = predicted_logits[:, -1, :]
        predicted_logits = predicted_logits/self.temperature
        # Apply the prediction mask: prevent "[UNK]" from being generated.
        predicted_logits = predicted_logits + self.prediction_mask

        # Sample the output logits to generate token IDs.
        predicted_ids = tf.random.categorical(predicted_logits, num_samples=1)
        predicted_ids = tf.squeeze(predicted_ids, axis=-1)

        # Convert from token ids to characters
        predicted_chars = self.chars_from_ids(predicted_ids)

        # Return the characters and model state.
        return predicted_chars, states

class LMModel(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, rnn_units):
        super().__init__(self)
        self.embedding = Embedding(vocab_size, embedding_dim)
        self.gru = GRU(rnn_units, return_sequences=True, return_state=True)
        self.dense = Dense(vocab_size)
    
    def call(self, inputs, states=None, return_state=False, training=False):
        x = inputs
        x = self.embedding(x, training=training)
        if states is None:
            states = self.gru.get_initial_state(x)
        x, states = self.gru(x, initial_state=states, training=training)
        x = self.dense(x, training=training)

        if return_state:
            return x, states
        else:
            return x

# Get random sentence for check
def get_random_sentence(text):
    sentences = sent_tokenize(text)
    return random.choice(sentences)

# Gets random word for sentence generation
def get_random_word(text):
    sentence = get_random_sentence(text)
    words = word_tokenize(sentence)
    return random.choice(words)

# Gets raw text files to process
def get_raw_text_files():
    file_ids = gutenberg.fileids()
    combined_text = ""
    for file_id in file_ids:
        if file_id != 'bible-kjv.txt':
            combined_text += gutenberg.raw(file_id) + '\n'
    return combined_text

# Define the function to create the StringLookup layer for character to ID conversion
def create_ids_from_chars(vocab):
    return StringLookup(vocabulary=list(vocab), mask_token=None)

# Define the function to create the inverse lookup for ID to character conversion
def create_chars_from_ids(ids_from_chars):
    return StringLookup(vocabulary=ids_from_chars.get_vocabulary(), invert=True, mask_token=None)

# Convert text to IDs
def text_from_ids(ids, chars_from_ids):
    return tf.strings.reduce_join(chars_from_ids(ids), axis=-1)

# Function to define input - label pairs (current char - next char)
def split_input_target(sequence):
    input_text = sequence[:-1]
    target_text = sequence[1:]
    return input_text, target_text

def create_checkpoint_dir():
    if not os.path.exists(CHECKPOINT_DIR):
        os.makedirs(CHECKPOINT_DIR)

def train_model(vocab_size, dataset, chars_from_ids):
    model = LMModel(vocab_size=vocab_size,
                    embedding_dim=EMBEDDING_DIM,
                    rnn_units=RNN_UNITS)
    
    for input_example_batch, target_example_batch in dataset.take(1):
        example_batch_predictions = model(input_example_batch)
        print(example_batch_predictions.shape, "# (batch_size, sequence_length, vocab_size)")

    model.summary()

    loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
    example_batch_mean_loss = loss(target_example_batch, example_batch_predictions)
    print("Prediction shape: ", example_batch_predictions.shape, " # (batch_size, sequence_length, vocab_size)")
    print("Mean loss:        ", example_batch_mean_loss)
    tf.exp(example_batch_mean_loss).numpy()
    model.compile(optimizer='adam', loss=loss)

    create_checkpoint_dir()
    checkpoint_prefix = os.path.join(CHECKPOINT_DIR, "ckpt_{epoch}")
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_prefix,
        save_weights_only=True)
    history = model.fit(dataset, epochs=EPOCHS, callbacks=[checkpoint_callback])

    return model

def generate_text(model, start_string, chars_from_ids, ids_from_chars):
    one_step_model = OneStep(model, chars_from_ids, ids_from_chars)
    states = None
    next_char = tf.constant([start_string])
    result = [next_char]

    for n in range(1000):  # Generate 1000 characters
        next_char, states = one_step_model.generate_one_step(next_char, states=states)
        result.append(next_char)

    result = tf.strings.join(result)
    return result[0].numpy().decode('utf-8')

# Load the latest checkpoint
def load_latest_checkpoint(vocab_size):
    model = LMModel(vocab_size=vocab_size,
                    embedding_dim=EMBEDDING_DIM,
                    rnn_units=RNN_UNITS)
    checkpoint_dir = CHECKPOINT_DIR
    latest = tf.train.latest_checkpoint(checkpoint_dir)
    model.load_weights(latest)
    return model

def main():
    texts = get_raw_text_files()
    vocab = sorted(set(texts))
    
    ids_from_chars = create_ids_from_chars(vocab)
    chars_from_ids = create_chars_from_ids(ids_from_chars)
    
    all_ids = ids_from_chars(tf.strings.unicode_split(texts, 'UTF-8'))
    ids_dataset = tf.data.Dataset.from_tensor_slices(all_ids)

    sequences = ids_dataset.batch(SEQ_LEN + 1, drop_remainder=True)
    dataset = sequences.map(split_input_target)
    dataset = (
        dataset
        .shuffle(BUFFER_SIZE)
        .batch(BATCH_SIZE, drop_remainder=True)
        .prefetch(tf.data.experimental.AUTOTUNE))

    vocab_size = len(ids_from_chars.get_vocabulary())
    
    train_model(vocab_size, dataset, chars_from_ids)

    # After training, generate text
    model = load_latest_checkpoint(vocab_size)
    start_string = "Once upon a time"
    generated_text = generate_text(model, start_string, chars_from_ids, ids_from_chars)
    print(generated_text)

if __name__ == "__main__":
    main()
