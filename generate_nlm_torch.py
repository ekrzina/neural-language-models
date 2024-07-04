import torch
import torch.nn as nn
import numpy as np
import os
import random
from nltk.corpus import gutenberg
from nltk.tokenize import word_tokenize
import re
from nltk.corpus import stopwords

# replace X with model
MODEL_PATH = "model_out/sentgen_vX_weights.pt"
EMBEDDING_DIM = 200
SEQ_LENGTH = 30
HIDDEN_DIM = 150

# function to load GloVe embeddings
def load_glove_embeddings(embedding_dim):
    glove_dir = 'glove'
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

# function to preprocess text
def preprocess_text(text):
    new_text = text.lower()
    new_text = re.sub(r"'s\b", "", new_text)
    new_text = re.sub("[^a-zA-Z]", " ", new_text)
    stop_words = set(stopwords.words('english'))
    long_words = [word for word in new_text.split() if len(word) >= 3 and word not in stop_words]
    return " ".join(long_words).strip()

# define the text generation model in PyTorch
class TextGenerator(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, embedding_matrix=None):
        super(TextGenerator, self).__init__()
        if embedding_matrix is not None:
            self.embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze=False)
        else:
            self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        embedded = self.embedding(x)
        output, _ = self.gru(embedded)
        output = self.fc(output[:, -1, :])
        return output

# function to create sequences from text
def create_sequences(text, seq_length):
    sequences = []
    for i in range(seq_length, len(text)):
        seq = text[i - seq_length:i + 1]
        sequences.append(seq)
    return sequences

# function to encode sequences
def encode_sequences(sequences, mapping):
    return [[mapping[char] for char in seq] for seq in sequences]

# function to create embedding matrix
def create_embedding_matrix(word_index, embeddings_index, embedding_dim):
    vocabulary_size = len(word_index) + 1
    embedding_matrix = np.zeros((vocabulary_size, embedding_dim))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return torch.tensor(embedding_matrix, dtype=torch.float32)

# function to generate text from the trained model
def generate_text(model, mapping, seq_length, seed_text, n_chars, top_k=5):
    model.eval()
    inverse_mapping = {v: k for k, v in mapping.items()}
    with torch.no_grad():
        in_text = seed_text
        generated_text = in_text
        for _ in range(n_chars):
            encoded = [mapping[char] for char in in_text if char in mapping]
            encoded = torch.LongTensor([encoded[-seq_length:]])
            outputs = model(encoded)
            #top k predictions
            _, top_indices = torch.topk(outputs, top_k)
            top_indices = top_indices.squeeze().tolist()
            yhat_item = random.choice(top_indices)
            if yhat_item in inverse_mapping:
                out_char = inverse_mapping[yhat_item]
                # avoid words that repeat
                if out_char in generated_text.split()[-3:]:
                    continue
                generated_text += ' ' + out_char
                in_text += out_char
                in_text = in_text[-seq_length:]
            else:
                print(f"Warning: {yhat_item} not in mapping")
                break
    return generated_text
def main():

    modeltext = gutenberg.raw('austen-emma.txt')
    preprocessed_text = preprocess_text(modeltext)
    chars = sorted(list(set(preprocessed_text)))
    mapping = {char: i for i, char in enumerate(chars)}
    embeddings_index = load_glove_embeddings(EMBEDDING_DIM)
    embedding_matrix = create_embedding_matrix(mapping, embeddings_index, EMBEDDING_DIM)
    vocab_size = len(mapping)
    model = TextGenerator(vocab_size, EMBEDDING_DIM, HIDDEN_DIM, embedding_matrix)
    model.load_state_dict(torch.load(MODEL_PATH))
    
    # generate sentence
    seed_text = word_tokenize(modeltext)[random.randint(0, len(word_tokenize(modeltext))-1)]
    generated_text = generate_text(model, mapping, SEQ_LENGTH, seed_text, 100)
    
    print("Generated Text:")
    print(generated_text)

if __name__ == "__main__":
    main()
