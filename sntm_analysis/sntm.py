# Simple program that uses vader to predict sentiment

from nltk.sentiment.vader import SentimentIntensityAnalyzer
#import nltk
#nltk.download('vader_lexicon')

def analyze_sentiment(sid, text):
    scores = sid.polarity_scores(text)
    return scores

if __name__ == "__main__":
    sid = SentimentIntensityAnalyzer()
    while 1:
        txt = input("Insert text to analyze: ")
        print(analyze_sentiment(sid, txt))
