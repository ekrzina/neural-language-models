# neural-language-models
The repository contains files for neural language modelling. Uses NLTK and Keras / Pytorch for model training / validation and generating new text. Uses [TensorBoard](https://www.tensorflow.org/tensorboard) for visualizing training. 

## Installation

Before running any application, get the necessary Python libraries listed in `requirements.txt`. Fetch all necessary NLTK files (e.g. gutenberg, from which text files are read). This is done with the following code which can be added into existing files or can be written in a separate file:

```
import nltk

nltk.download('punkt')
nltk.download('gutenberg')
nltk.download('stopwords')

print("Downloads complete.")
```

The `nlm-torch.py` file and the sentiment-analysis file for custom data training use [GloVe](https://nlp.stanford.edu/projects/glove/) for added embeddings and transfer learning. The files are downloaded using the following command lines:

```
mkdir glove
cd glove
wget http://nlp.stanford.edu/data/glove.6B.zip
unzip glove.6B.zip
```

Make sure to use this command in the same directory as the Python script you want to run, or change the pathway to the GloVe files in the scripts where it is required.

## File Description

### Creating a Neural Language Model
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

### Sentiment Analysis
Python code for sentiment analysis has been implemented. The directory `sntm_analysis` contains the following Python scripts:
- `sntm.py` - simple sentiment analysis that uses pre-trained polarity scores for positive, negative and neutral sentiments
- `custom_sentiment.py` - sentiment analysis that uses a custom model for sentiment analysis
- `sntm_custom_train.py` - Keras and NLM; trains custom model for sentiment analysis
- `dataset_analyzer.py` - auxiliary file for analyzing dataset; shows dataset sentiment count and WordCloud for positive and negative sentiment

Example `dataset_analyzer` output:

![Figure_1. Sentiment Count](https://github.com/user-attachments/assets/18ffb794-11b1-4ac4-9ee4-78fbfdf4ccc0)

![Figure_2. Word Cloud for positive and negative sentiment](https://github.com/user-attachments/assets/5e795e33-81dc-4402-b2c5-41c26e5569f7)

The current implementation predicts the following sentiment:
- positive
- negative
- neutral

Any number of sentiments can be analyzed this way given a correctly-structured and balanced dataset. The dataset for the application has not been included due to its size, but sample sentences have been provided.
