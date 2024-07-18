import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from wordcloud import WordCloud
from pandas.api.types import is_categorical_dtype
import numpy as np

# Read data and clean sentiment column
df = pd.read_csv("data/dataset.csv", sep=";")
df['sentiment'] = df['sentiment'].str.strip().str.lower()

# Count sentiments
sentiment_count = df['sentiment'].value_counts()

# Plot histogram for sentiment count
plt.figure(figsize=(8,6))
sns.histplot(data=df, x='sentiment', discrete=True, shrink=0.8)
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.title('Sentiment Analysis Count Visualizer')
plt.show()

# Generate word clouds
positive_texts = ' '.join(df[df['sentiment'] == 'positive']['text'])
negative_texts = ' '.join(df[df['sentiment'] == 'negative']['text'])

positive_wordcloud = WordCloud(width=800, height=400, background_color='white').generate(positive_texts)
negative_wordcloud = WordCloud(width=800, height=400, background_color='white').generate(negative_texts)

# Plot word clouds
fig, axes = plt.subplots(2, 1, figsize=(8, 12))

axes[0].imshow(positive_wordcloud, interpolation='bilinear')
axes[0].axis('off')
axes[0].set_title('Word Cloud for Positive Sentiment')

axes[1].imshow(negative_wordcloud, interpolation='bilinear')
axes[1].axis('off')
axes[1].set_title('Word Cloud for Negative Sentiment')

plt.tight_layout()
plt.show()