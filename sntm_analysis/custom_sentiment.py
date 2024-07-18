import tensorflow as tf
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
import numpy as np
import pickle

with open('models/tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)
model = load_model('models/sentiment_model_nn.h5')

def predict_sentiment(text):
    text_sequence = tokenizer.texts_to_sequences([text])
    text_sequence_padded = pad_sequences(text_sequence, maxlen=100)
    
    prediction = model.predict(text_sequence_padded)
    
    labels = ['negative', 'neutral', 'positive']
    predicted_label = labels[np.argmax(prediction)]
    
    return predicted_label

if __name__ == "__main__":
    while True:
        text = input("Insert text to analyze (or 'exit' to quit): ")
        
        if text.lower() == 'exit':
            break
        
        sentiment = predict_sentiment(text)
        print(f"Predicted sentiment: {sentiment}")
