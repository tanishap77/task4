import pandas as pd
from textblob import TextBlob
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# Load dataset
data = pd.read_csv("social sentiments.csv")

# Preview
print("🔹 First 5 Comments:")
print(data.head())

# Function to compute sentiment polarity and label
def get_sentiment(text):
    analysis = TextBlob(str(text))
    polarity = analysis.sentiment.polarity
    if polarity > 0:
        return 'Positive'
    elif polarity < 0:
        return 'Negative'
    else:
        return 'Neutral'

# Apply sentiment analysis
data["Sentiment"] = data["Comment Text"].apply(get_sentiment)

# ✅ Sentiment Distribution
sns.countplot(x="Sentiment", data=data, palette="Set2")
plt.title("Sentiment Distribution")
plt.show()

# 📊 Word Cloud for Positive Comments
positive_text = " ".join(data[data['Sentiment']=='Positive']["Comment Text"].dropna())
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(positive_text)
plt.figure(figsize=(10,5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.title("Word Cloud of Positive Comments")
plt.show()

# 🔎 Sentiment per Topic (if available)
if 'Topic' in data.columns:
    plt.figure(figsize=(10,5))
    sns.countplot(x="Topic", hue="Sentiment", data=data)
    plt.title("Sentiment by Topic")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Optional: Export updated data with sentiment
data.to_csv("sentiment_results.csv", index=False)
print("✅ Sentiment results saved as sentiment_results.csv")
