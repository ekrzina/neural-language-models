import random
import re
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from nltk.corpus import gutenberg, stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.translate.bleu_score import sentence_bleu
import numpy as np

# already set up for next version - always
OUTPUT_DIR = "model_out"
MODEL_NAME = "sentgen_v7_weights.pt"
LOG_NAME = "sentgen_v7"
EMBEDDING_DIM = 200     # GloVe embeddings dimension
BATCH_SIZE = 128
SEQ_LENGTH = 30
# 4 - 0.0025
LEARNING_RATE = 0.004
NUM_EPOCHS = 100
HIDDEN_DIM = 150

# Get random sentence for check
def get_random_sentence(text):
    sentences = sent_tokenize(text)
    return random.choice(sentences)

# Gets random word for sentence generation
def get_random_word(text):
    sentence = get_random_sentence(text)
    words = word_tokenize(sentence)
    return random.choice(words)

# Function to load GloVe embeddings
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

# Function to preprocess text
def preprocess_text(text):
    new_text = text.lower()
    new_text = re.sub(r"'s\b", "", new_text)
    new_text = re.sub("[^a-zA-Z]", " ", new_text)
    stop_words = set(stopwords.words('english'))
    long_words = [word for word in new_text.split() if len(word) >= 3 and word not in stop_words]
    return " ".join(long_words).strip()

# Function to create sequences from text
def create_sequences(text, seq_length):
    sequences = []
    for i in range(seq_length, len(text)):
        seq = text[i - seq_length:i + 1]
        sequences.append(seq)
    return sequences

# Function to encode sequences
def encode_sequences(sequences, mapping):
    return [[mapping[char] for char in seq] for seq in sequences]

# Define the text generation model in PyTorch
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

# Dataset class for PyTorch DataLoader
class TextDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return torch.LongTensor(self.sequences[idx][:-1]), torch.LongTensor([self.sequences[idx][-1]])

# Function for training and evaluating the model
def train_model(model, dataloader, criterion, optimizer, num_epochs, patience):
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    writer = SummaryWriter(log_dir=LOG_NAME)
    best_val_loss = float('inf')
    counter = 0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        for inputs, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels.squeeze(1))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == labels.squeeze(1)).sum().item()
            total_samples += labels.size(0)

        epoch_loss = running_loss / len(dataloader)
        epoch_accuracy = correct_predictions / total_samples

        writer.add_scalar('Loss/train', epoch_loss, epoch)
        writer.add_scalar('Accuracy/train', epoch_accuracy, epoch)
        writer.add_scalar('Learning rate', LEARNING_RATE, epoch)
        writer.add_scalar('Batch size', BATCH_SIZE, epoch)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in dataloader:
                outputs = model(inputs)
                loss = criterion(outputs, labels.squeeze(1))
                val_loss += loss.item()

        val_loss /= len(dataloader)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, MODEL_NAME))
        else:
            counter += 1
            if counter >= patience:
                break

    writer.close()

# Function to generate text from the trained model
def generate_text(model, mapping, seq_length, seed_text, n_chars):
    model.eval()
    inverse_mapping = {v: k for k, v in mapping.items()}  # Create inverse mapping
    with torch.no_grad():
        in_text = seed_text
        for _ in range(n_chars):
            encoded = [mapping[char] for char in in_text if char in mapping]
            encoded = torch.LongTensor([encoded[-seq_length:]])
            outputs = model(encoded)
            yhat = torch.argmax(outputs, dim=-1)
            yhat_item = yhat.item()
            if yhat_item in inverse_mapping:
                out_char = inverse_mapping[yhat_item]
            else:
                print(f"Warning: {yhat_item} not in mapping")
                break  # Or handle appropriately
            in_text += out_char
            print(f"Generated so far: {in_text}")  # Debugging info
    return in_text

# Function to evaluate BLEU score for generated text
def evaluate_bleu(reference_text, generated_text):
    reference_tokens = reference_text.split()
    generated_tokens = generated_text.split()
    return sentence_bleu([reference_tokens], generated_tokens)

def create_embedding_matrix(word_index, embeddings_index, embedding_dim):
    vocabulary_size = len(word_index) + 1
    embedding_matrix = np.zeros((vocabulary_size, embedding_dim))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return torch.tensor(embedding_matrix, dtype=torch.float32)

# Main function
def main():
    modeltext = gutenberg.raw('austen-emma.txt')
    preprocessed_text = preprocess_text(modeltext)
    chars = sorted(list(set(preprocessed_text)))
    mapping = {char: i for i, char in enumerate(chars)}

    sequences = create_sequences(preprocessed_text, seq_length=SEQ_LENGTH)
    sequences_encoded = encode_sequences(sequences, mapping)
    embeddings_index = load_glove_embeddings(EMBEDDING_DIM)
    embedding_matrix = create_embedding_matrix(mapping, embeddings_index, EMBEDDING_DIM)
    vocab_size = len(mapping)

    model = TextGenerator(vocab_size, EMBEDDING_DIM, HIDDEN_DIM, embedding_matrix)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    dataset = TextDataset(sequences_encoded)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    train_model(model, dataloader, criterion, optimizer, NUM_EPOCHS, patience=10)
    seed_text = get_random_word(modeltext)
    generated_text = generate_text(model, mapping, SEQ_LENGTH, seed_text, 100)
    
    print("Generated Text:")
    print(generated_text)

    reference_text = get_random_sentence(modeltext)
    bleu_score = evaluate_bleu(reference_text, generated_text)
    
    print(f"BLEU score: {bleu_score}")
    
    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, MODEL_NAME))
    print(f"Model weights saved to {os.path.join(OUTPUT_DIR, MODEL_NAME)}.")

if __name__ == "__main__":
    main()