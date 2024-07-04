import tensorflow as tf
from keras.layers import StringLookup, Embedding, GRU, Dense
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from nltk.corpus import gutenberg
from nltk import sent_tokenize, word_tokenize
import os
import random

SEQ_LEN = 100
BATCH_SIZE = 64
BUFFER_SIZE = 10000
EMBEDDING_DIM = 256
RNN_UNITS = 1024
EPOCHS = 20

CHECKPOINT_DIR = "sentgen_checkpoints_v2"
MODELS_PATH = "model_out"
MODEL_SAVE_PATH = os.path.join(MODELS_PATH, "sentgen_model_v2")
# checkpoint / model
LOAD_METHOD = "model"

class OneStep(tf.keras.Model):
    def __init__(self, model, chars_from_ids, ids_from_chars, temperature=1.0):
        super().__init__()
        # temperature scales logits before computing categorical distribution over token IDs
        self.temperature = temperature
        self.model = model
        self.chars_from_ids = chars_from_ids
        self.ids_from_chars = ids_from_chars

        # retrieves unknown token and reshapes to column vector
        skip_ids = self.ids_from_chars(['[UNK]'])[:, None]
        # sparse_mask ensures that the model doesn't predict UNK token
        sparse_mask = tf.SparseTensor(
            values=[-float('inf')]*len(skip_ids),
            indices=skip_ids,
            dense_shape=[len(ids_from_chars.get_vocabulary())])
        self.prediction_mask = tf.sparse.to_dense(sparse_mask)

    # converts generate_one_step to TF graph for faster execution
    '''
    Sets up input IDs and predicts output. Output logits are scaled by temperature and prediction mask is applied.
    Removes added dimension for a flat token and returns predicted characters.
    '''
    @tf.function
    def generate_one_step(self, inputs, states=None):
        # splits input text to individual characters
        input_chars = tf.strings.unicode_split(inputs, 'UTF-8')
        input_ids = self.ids_from_chars(input_chars).to_tensor()

        # prediction of output
        predicted_logits, states = self.model(inputs=input_ids, states=states, return_state=True)
        predicted_logits = predicted_logits[:, -1, :]
        predicted_logits = predicted_logits/self.temperature
        predicted_logits = predicted_logits + self.prediction_mask
        predicted_ids = tf.random.categorical(predicted_logits, num_samples=1)
        predicted_ids = tf.squeeze(predicted_ids, axis=-1)

        predicted_chars = self.chars_from_ids(predicted_ids)
        return predicted_chars, states

class LMModel(tf.keras.Model):
    '''
    Function sets up embedding layers for map integers (word tokens) into a dense vector sized embedding_dim,
    sets up a gated recurrent unit GRU (RNN) and a fully-connected dense layer 
    '''
    def __init__(self, vocab_size, embedding_dim, rnn_units):
        super().__init__(self)
        self.embedding = Embedding(vocab_size, embedding_dim)
        self.gru = GRU(rnn_units, return_sequences=True, return_state=True)
        self.dense = Dense(vocab_size)
    
    # on call, set up inputs and states
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

# gets random sentence (for e.g. start string)
def get_random_sentence(text):
    sentences = sent_tokenize(text)
    return random.choice(sentences)

# gets random word (for e.g. start string)
def get_random_word(text):
    sentence = get_random_sentence(text)
    words = word_tokenize(sentence)
    return random.choice(words)

# gathers all gutenberg texts and adds them to a single file
'''
Since the model will behave like given files, include only files that you want model to learn from.
E.g. given old English texts (The Bible, Shakespeare etc.), model will most likely generate old English sentences. For modern text, include more modern files.
'''
def get_raw_text_files():
    file_ids = gutenberg.fileids()
    combined_text = ""
    for file_id in file_ids:
        if file_id != 'bible-kjv.txt':
            combined_text += gutenberg.raw(file_id) + '\n'
    return combined_text

# creates identifier from characters
def create_ids_from_chars(vocab):
    return StringLookup(vocabulary=list(vocab), mask_token=None)

# returns characters from identifiers
def create_chars_from_ids(ids_from_chars):
    return StringLookup(vocabulary=ids_from_chars.get_vocabulary(), invert=True, mask_token=None)

# converts sequences of character identifiers into text and concatenates them
def text_from_ids(ids, chars_from_ids):
    return tf.strings.reduce_join(chars_from_ids(ids), axis=-1)

# splits data to text - prediction
'''
The data is split in two parts: the input text, and the target prediction. Target prediction is set to
the next string in the sequence that should be predicted. Sequence is encoded and uses identifiers.
'''
def split_input_target(sequence):
    input_text = sequence[:-1]
    target_text = sequence[1:]
    return input_text, target_text

# creates a directory for checkpoints from specified path
def create_checkpoint_dir():
    if not os.path.exists(CHECKPOINT_DIR):
        os.makedirs(CHECKPOINT_DIR)

# loads latest checkpoint so as to not reset training
def load_latest(vocab_size):
    if LOAD_METHOD == "checkpoint":
        model = LMModel(vocab_size=vocab_size, embedding_dim=EMBEDDING_DIM, rnn_units=RNN_UNITS)
        latest_checkpoint = tf.train.latest_checkpoint(CHECKPOINT_DIR)
        if latest_checkpoint:
            try:
                status = model.load_weights(latest_checkpoint)
                status.expect_partial()
                print(f"Loaded model from checkpoint {latest_checkpoint}")
                return model
            except Exception as e:
                print(f"Error loading weights from checkpoint: {e}")
                return None
        else:
            print("No checkpoint found. Training a new model.")
            return None
    elif LOAD_METHOD == "model":
        if os.path.exists(MODEL_SAVE_PATH):
            model = tf.keras.models.load_model(MODEL_SAVE_PATH)
            print(f"Loaded model from saved file {MODEL_SAVE_PATH}")
            return model
        else:
            print(f"No model file found at {MODEL_SAVE_PATH}. Training a new model.")
            return None
    else:
        print("No loading method was chosen. Training a new model.")
        return None

# saves model to specified path
def save_model(model):
    if not os.path.exists(MODEL_SAVE_PATH):
        os.makedirs(MODEL_SAVE_PATH)
    save_path = os.path.join(MODELS_PATH, MODEL_SAVE_PATH)
    model.save(save_path)
    print(f"Model saved to {MODEL_SAVE_PATH}")

# function for training language model
def train_model(vocab_size, dataset):
    model = LMModel(vocab_size=vocab_size, embedding_dim=EMBEDDING_DIM, rnn_units=RNN_UNITS)
    # sets up model config; displays losses and accuracy while training
    model.compile(optimizer='adam', loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
    checkpoint_prefix = os.path.join(CHECKPOINT_DIR, "ckpt_{epoch}")
    #added checkpoint, tensorflow early stopping callback
    checkpoint_callback = ModelCheckpoint(filepath=checkpoint_prefix, 
                                          save_weights_only=True, 
                                          save_best_only=True, 
                                          monitor='loss', 
                                          mode='min')
    
    early_stopping_callback = EarlyStopping(monitor='val_loss', 
                                            patience=3, 
                                            verbose=1, 
                                            mode='min', 
                                            restore_best_weights=True)
    
    tensorboard_callback = TensorBoard(log_dir=MODELS_PATH, 
                                       histogram_freq=1)
    # making validation dataset
    split = int(len(dataset) * 0.9)
    train_dataset = dataset.take(split)
    val_dataset = dataset.skip(split)

    # start training  
    model.fit(train_dataset, 
              epochs=EPOCHS, 
              callbacks=[checkpoint_callback, early_stopping_callback, tensorboard_callback], 
              validation_data=val_dataset)
    
    # evaluation
    loss, accuracy = model.evaluate(dataset)
    print(f'Final Validation Loss: {loss:.4f}, Accuracy: {accuracy:.4f}')
    save_model(model)
    
    return model

# generates new text from specified start string and range
def generate_text(model, start_string, chars_from_ids, ids_from_chars, text_range):
    one_step_model = OneStep(model, chars_from_ids, ids_from_chars)
    states = None
    # character is converted to tensorflow constant
    next_char = tf.constant([start_string])
    result = [next_char]
    for _ in range(text_range):
        # generates next step of character
        next_char, states = one_step_model.generate_one_step(next_char, states=states)
        result.append(next_char)
    result = tf.strings.join(result)
    # result is decoded before beign returned
    return result[0].numpy().decode('utf-8')

def main():
    texts = get_raw_text_files()
    vocab = sorted(set(texts))
    
    ids_from_chars = create_ids_from_chars(vocab)
    chars_from_ids = create_chars_from_ids(ids_from_chars)
    
    all_ids = ids_from_chars(tf.strings.unicode_split(texts, 'UTF-8'))
    ids_dataset = tf.data.Dataset.from_tensor_slices(all_ids)
    sequences = ids_dataset.batch(SEQ_LEN + 1, drop_remainder=True)
    dataset = sequences.map(split_input_target)
    dataset = (dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE))
    vocab_size = len(ids_from_chars.get_vocabulary())

    # if model is none, creates new model for training; else loads checkpoints / model
    model = load_latest(vocab_size)
    if model is None:
        create_checkpoint_dir()
        model = train_model(vocab_size, dataset)
    
    # set / generate the starting string and generate sequence
    start_string = "Once upon a time"
    print(generate_text(model, start_string, chars_from_ids, ids_from_chars, 500))

if __name__ == "__main__":
    main()
