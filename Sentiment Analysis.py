#!/usr/bin/env python
# coding: utf-8

# #### Import necessary packages

# In[1]:


import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from textblob import TextBlob
from wordcloud import WordCloud, STOPWORDS
import nltk
from nltk.corpus import stopwords
import warnings
warnings.filterwarnings("ignore")
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[ ]:





# #### Load the dataset

# In[56]:


Df = pd.read_csv("mcdonalds.csv", encoding="ISO-8859-1")
# remove unnecessary columns
df = Df[["city", "review"]]
df.head()


# #### The structure of the dataset

# In[57]:


df.info


# #### Dimension of the dataset

# In[58]:


df.shape


# #### Data visualization

# In[59]:


# define a function to perform sentiment analysis
def get_sentiment(review):
    blob = TextBlob(review)
    if blob.sentiment.polarity > 0:
        return 'positive'
    elif blob.sentiment.polarity < 0:
        return 'negative'
    else:
        return 'neutral'

df.loc[:, 'sentiment'] = df['review'].apply(get_sentiment)

print(df.head(10))


# #### Bar chart of sentiment distribution

# In[60]:


plt.figure(figsize=(8,6))
sns.barplot(x=df["sentiment"].value_counts().index, y=df["sentiment"].value_counts().values)
plt.xlabel("Sentiment")
plt.ylabel("Count")
plt.savefig("sentiment_distribution.png")
plt.show()


# #### Stacked bar chart

# In[61]:


city_sentiment_counts = df.groupby(["city", "sentiment"]).size().unstack()
plt.figure(figsize=(10,10))
city_sentiment_counts.plot(kind="bar", stacked=True)
plt.xlabel("City")
plt.ylabel("Count")
plt.legend(title="Sentiment", loc="upper right")
plt.savefig("Stacked.png")
plt.show();


# #### Word cloud for negative, positive, and neutral sentiments

# In[84]:


stopwords = set(STOPWORDS)
stopwords_file = open("stopwords_en.txt", "r")
stopwords_list = [line.rstrip('\n') for line in stopwords_file]
stopwords.update(stopwords_list)


df['review'] = df['review'].apply(lambda x: ' '.join([word for word in x.split() if word.lower() not in stopwords]))

def get_sentiment_score(sentiment):
    if sentiment == 'positive':
        return 1
    elif sentiment == 'negative':
        return -1
    else:
        return 0

df['sentiment_score'] = df['sentiment'].apply(get_sentiment_score)

text = ' '.join(df['review'])
wordcloud = WordCloud(width=800, height=800, background_color='white', stopwords=STOPWORDS).generate(text)

print("")
print('')
plt.figure(figsize=(8, 8), facecolor=None)
plt.imshow(wordcloud)
plt.axis('off')
plt.tight_layout(pad=0)
plt.savefig("wordcloud.png")
plt.show()


# #### Sentiment Score

# In[85]:


sentiment_score = df['sentiment_score'].sum()
print('Sentiment Score:', sentiment_score)
print("")


# #### Word cloud for negative sentiments`

# In[66]:


# select the positive reviews
negative_reviews = df[df['sentiment'] == 'negative']
# join the reviews into a single string
text = ' '.join(positive_reviews['review'])
wordcloud = WordCloud(width=800, height=800, background_color='white', stopwords=STOPWORDS).generate(text)
print("")
print('')
plt.figure(figsize=(8, 8), facecolor=None)
plt.imshow(wordcloud)
plt.axis('off')
plt.tight_layout(pad=0)
plt.savefig("wordnegative.png")
plt.show()


# ## Data Analysis

# #### Frequency-inverse document frequency

# In[96]:


from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
stop_words = set(stopwords.words('english'))
ps = PorterStemmer()
def preprocess_text(text):
    text = text.lower()
    words = nltk.word_tokenize(text)
    words = [w for w in words if w not in stop_words]
    words = [ps.stem(w) for w in words]
    return ' '.join(words)

df['processed_review'] = df['review'].apply(preprocess_text)

tfidf = TfidfVectorizer()
tfidf_scores = tfidf.fit_transform(df['processed_review'])
feature_names = tfidf.get_feature_names()
df_tfidf = pd.DataFrame(tfidf_scores.toarray(), columns=feature_names)

top_words = df_tfidf.sum().sort_values(ascending=False).head(20)
print(top_words)


# #### Modelling: Logistic Regression, Random Forest, Support Vector Machine,  and Decision Tree Models

# In[111]:


import nltk
from nltk.corpus import stopwords
stopwords = set(stopwords.words('english'))
df['review'] = df['review'].apply(lambda x: ' '.join([word.lower() for word in word_tokenize(x) if word.lower() not in stopwords]))
df['sentiment_score'] = df['sentiment'].apply(lambda x: 1 if x == 'positive' else -1 if x == 'negative' else 0)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['review'], df['sentiment_score'], test_size=0.2, random_state=42)

# Convert the text data into a numerical representation
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)


clf = LogisticRegression()

clf.fit(X_train_vec, y_train)


y_pred = clf.predict(X_test_vec)
clf_accuracy = round(accuracy_score(y_test, y_pred),4)


from sklearn.ensemble import RandomForestClassifier


rfc = RandomForestClassifier()
rfc.fit(X_train_vec, y_train)


y_pred = rfc.predict(X_test_vec)
rfc_accuracy = round(accuracy_score(y_test, y_pred), 4)


from sklearn.tree import DecisionTreeClassifier


dtc = DecisionTreeClassifier()
dtc.fit(X_train_vec, y_train)

y_pred = dtc.predict(X_test_vec)
dtc_accuracy = round(accuracy_score(y_test, y_pred), 4)


from sklearn.svm import SVC


svm = SVC()
svm.fit(X_train_vec, y_train)


y_pred = svm.predict(X_test_vec)
svm_accuracy = round(accuracy_score(y_test, y_pred), 4)



models = [ 'Random Forest', 'Logistic Regression', 'SVM','Decision Tree']
accuracies = [rfc_accuracy, clf_accuracy, svm_accuracy,dtc_accuracy]

df_models = pd.DataFrame({'Model': models, 'Accuracy': accuracies})
print(df_models)

