import random
import re
from nltk.corpus import gutenberg
from nltk.tokenize import sent_tokenize

# import nltk
#nltk.download('gutenberg')
#nltk.download('punkt')

file_ids = gutenberg.fileids()

combined_text = ""
for file_id in file_ids:
    combined_text += gutenberg.raw(file_id) + '\n'

def clean_text(text):
    cleaned_text = re.sub(r'[^\w\s.,!?]', '', text)
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
    return cleaned_text.strip()

sentences = sent_tokenize(combined_text)
random_sentence = random.choice(sentences)
cleaned_sentence = clean_text(random_sentence)

print(cleaned_sentence)