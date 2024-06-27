# neural-language-models
The repository contains files for neural language modelling. Uses NLTK and Keras for model training / validation.

## Installation

Before running any application, get the necessary Python libraries listed in `requirements.txt`. Fetch all necessary NLTK files (e.g. gutenberg, from which text files are read). This is done with the following code which can be added into existing files or can be written in a separate file :

```
import nltk

nltk.download('punkt')
nltk.download('gutenberg')

print("Downloads complete.")
```

The `nlmopt.py` file uses GloVe for added embeddings. The files are downloaded using the following command lines:

```
mkdir glove
cd glove
wget http://nlp.stanford.edu/data/glove.6B.zip
unzip glove.6B.zip
```

## File Description

Currently, the repository contains three separate Python applications:
- nlm.py - NLM from scratch
- nlmopt.py - optimized NLM; includes GloVe embeddings - *recommended*
- get_gutenberg_sentence.py - selects random sentence from given input Gutenberg texts

The present language models all use the following Gutenberg texts:
- Alice in Wonderland - Lewis Carroll
- Emma - Jane Austen
- Hamlet - Shakespeare

To see all available text files, or add all possible files, write the following lines of code:

```
file_ids = gutenberg.fileids()

combined_text = ""
for file_id in file_ids:
    combined_text += gutenberg.raw(file_id) + '\n'
```

In the `nlmopt.py` file, a function for importing your own text data has been added. To use your own textual dataset, uncomment the `modeltexts = import_data(TEXT_FILE)` line and change the `TEXT_FILE` parameter. You can comment out `gutenberg.raw` files now, or you may add the imported text by concatting it to the given Gutenberg raws by adding the following code snippet after import:

```
modeltexts = modeltexts_gutenberg + '\n' + modeltexts_file
```

The application uses Keras to create a model, train it, validate it and save it. One can change training parameters for varied results.
