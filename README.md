# neural-language-models
The repository contains files for neural language modelling. Uses NLTK and Keras / Pytorch for model training / validation and generating new text. Uses [TensorBoard](https://www.tensorflow.org/tensorboard) for visualizing training. 

## Installation

Before running any application, get the necessary Python libraries listed in `requirements.txt`. If only `nlm-keras.py` model is used, torch isn't needed. If only `nlm-torch.py` is used, keras download isn't needed. Fetch all necessary NLTK files (e.g. gutenberg, from which text files are read). This is done with the following code which can be added into existing files or can be written in a separate file :

```
import nltk

nltk.download('punkt')
nltk.download('gutenberg')
nltk.download('stopwords')

print("Downloads complete.")
```

The `nlm-torch.py` file uses [GloVe](https://nlp.stanford.edu/projects/glove/) for added embeddings. The files are downloaded using the following command lines:

```
mkdir glove
cd glove
wget http://nlp.stanford.edu/data/glove.6B.zip
unzip glove.6B.zip
```

## File Description

Currently, the repository contains three separate Python applications:
- `nlm-keras.py` - Keras and NLM; includes GloVe embeddings; *recommended*
- `nlm-torch.py` - Torch and NLM; includes GloVe embeddings
- `get_gutenberg_sentence.py` - selects random sentence from given input Gutenberg texts
- `generate_nlm_torch.py` - generates torch text from imported model

To see all available text files for picking specific texts, write the following lines of code:

```
file_ids = gutenberg.fileids()
print(file_ids)
```

The `nlm-keras.py` application uses Keras to create a model, train it, validate it and save it, while `nlm-torch.py` uses Pytorch. One can change training parameters for varied results.
