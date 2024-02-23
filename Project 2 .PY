#!/usr/bin/env python
# coding: utf-8

# In[1]:


import string
from collections import Counter
import warnings

import nltk
from nltk.corpus import stopwords

# Download required resources
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('vader_lexicon')
nltk.download('averaged_perceptron_tagger')
nltk.download('movie_reviews')
nltk.download('punkt')
nltk.download('conll2000')
nltk.download('brown')


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    confusion_matrix,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC

# Suppress warnings
warnings.filterwarnings("ignore")


# In[2]:


df = pd.read_csv("financial_sentiment_data.csv")
df_final = df.copy()
df.head(10)


# In[3]:


df_final.head()


# In[4]:


df.shape


# In[5]:


df.describe()


# In[6]:


df.info()


# In[7]:


# Drop rows with NaN values in the 'Sentence' column
df = df.dropna(subset=['Sentence'])
df


# In[8]:


duplicated_sentence =df["Sentence"].duplicated().sum()
duplicated_sentence


# In[9]:


df = df.drop_duplicates(subset="Sentence")


# In[10]:


df.shape


# #EDA

# In[11]:


df['word_counts'] = df['Sentence'].apply(lambda x: len(str(x).strip().split(' ')))
df[['Sentence','word_counts']]


# In[12]:


sns.histplot(data=df, x='word_counts', hue='Sentiment', kde=True)
plt.title('Distribution of Word Counts by Sentiments')
plt.show()


# In[13]:


sns.countplot(data = df, x = 'Sentiment', palette = 'deep')


# **Word Cloud**

# In[14]:


for sentiment in df['Sentiment'].unique():
    subset = df[df['Sentiment'] == sentiment]
    text = ' '.join(subset['Sentence'])
    wordcloud = WordCloud(width=800, height=400, random_state=42, background_color='white').generate(text)

    plt.figure(figsize=(10, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f'Word Cloud for {sentiment} Sentiments')
    plt.show()


# **Stop Words**

# In[15]:


stop = stopwords.words('english')
df['stop_words'] = df['Sentence'].apply(lambda x : len([x for x in x.split() if x in stop]))

df[['Sentence','stop_words']].head(10)


# **The Most Common Words in the Sentences for each Sentiment Class**

# In[16]:


sentiment_words = {}

for sentiment in df['Sentiment'].unique():
    subset = df[df['Sentiment'] == sentiment]
    text = ' '.join(subset['Sentence'])


    text = text.translate(str.maketrans("", "", string.punctuation))
    words = text.split()


    words = [word for word in words if word.lower() not in stop]

    word_freq = Counter(words)
    sentiment_words[sentiment] = word_freq.most_common(10)

for sentiment, words in sentiment_words.items():
    print(f'Most common words in {sentiment} sentiment (without stopwords and punctuation): {words}')


# Plotting those above Most Common Words

# In[17]:


# Plotting
fig, axes = plt.subplots(nrows=len(sentiment_words), figsize=(10, 6 * len(sentiment_words)))

for idx, (sentiment, words) in enumerate(sentiment_words.items()):
    ax = axes[idx]
    ax.bar([word[0] for word in words], [count[1] for count in words])
    ax.set_title(f'Most common words in {sentiment} sentiment')
    ax.set_xlabel('Words')
    ax.set_ylabel('Frequency')

plt.tight_layout()
plt.show()


# UNIQUE WORD COUNTS

# 
# *This can help identify the diversity of vocabulary used in different sentiments.

# In[18]:


import pandas as pd

# Tokenize the sentences into words
df['tokenized_words'] = df['Sentence'].apply(lambda x: x.split())

# Calculate the number of unique words in each sentence
df['unique_word_count'] = df['tokenized_words'].apply(lambda x: len(set(x)))

# Display the DataFrame with the new column
print(df[['Sentence', 'unique_word_count']])


# In[19]:


#HISTOGRAM FOR VISUAL REPRESENTATION OF ABOVE CODE

# Set up the figure and axes
plt.figure(figsize=(8, 4))

# Plot a histogram of unique word counts
sns.histplot(df['unique_word_count'], bins=20, kde=True, color='skyblue')

# Set labels and title
plt.xlabel('Number of Unique Words')
plt.ylabel('Frequency')
plt.title('Distribution of Unique Word Counts in Sentences')

# Show the plot
plt.show()


# N GRAMS ANALYSIS

# In[20]:


import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import ngrams
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Function for text preprocessing
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation and special characters
    text = ''.join([char for char in text if char not in string.punctuation and char.isalnum() or char.isspace()])
    
    # Tokenize the text
    tokens = word_tokenize(text)
    
    # Remove stop words
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    
    return tokens

# Tokenize and preprocess the sentences into words
df['tokenized_words'] = df['Sentence'].apply(preprocess_text)

# Extract unigrams 
df['unigrams'] = df['tokenized_words'].apply(lambda x: list(ngrams(x, 1)))

# Flatten the list of unigrams
all_unigrams = [unigram for sentence_unigrams in df['unigrams'] for unigram in sentence_unigrams]

# Convert the results to a DataFrame for visualization
unigrams_df = pd.DataFrame(all_unigrams, columns=['Word1'])

# Display the most common unigrams
common_unigrams = unigrams_df['Word1'].value_counts().reset_index().rename(columns={'index': 'Word1', 'Word1': 'Frequency'})
print(common_unigrams.head(10))



# In[22]:


plt.figure(figsize=(8, 4))
sns.barplot(x='Word1', y='Frequency', data=common_unigrams.head(10), palette='viridis')
plt.xlabel('Word')
plt.ylabel('Frequency')
plt.show()


# TOP 10 BIGRAMS(N-GRAMS ANALYSIS)

# In[23]:


# Function for text preprocessing
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation and special characters
    text = ''.join([char for char in text if char not in string.punctuation and char.isalnum() or char.isspace()])
    
    # Tokenize the text
    tokens = word_tokenize(text)
    
    # Remove stop words
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    
    return tokens

# Tokenize and preprocess the sentences into words
df['tokenized_words'] = df['Sentence'].apply(preprocess_text)

# Extract bigrams 
df['bigrams'] = df['tokenized_words'].apply(lambda x: list(ngrams(x, 2)))

# Flatten the list of bigrams
all_bigrams = [bigram for sentence_bigrams in df['bigrams'] for bigram in sentence_bigrams]

# Convert the results to a DataFrame for visualization
bigrams_df = pd.DataFrame(all_bigrams, columns=['Word1', 'Word2'])

# Combine the bigrams into a single column for counting
bigrams_df['Word1 Word2'] = bigrams_df['Word1'] + ' ' + bigrams_df['Word2']

# Display the most common bigrams
common_bigrams = bigrams_df['Word1 Word2'].value_counts().reset_index().rename(columns={'index': 'Word1 Word2', 'Word1 Word2': 'Frequency'})
print(common_bigrams.head(10))


# In[25]:


plt.figure(figsize=(6, 3))
sns.barplot(x='Word1 Word2', y='Frequency', data=common_bigrams.head(10), palette='viridis')
plt.xlabel('Bigram')
plt.ylabel('Frequency')
plt.title('Top 10 Most Common Bigrams')
plt.xticks(rotation=45, ha='right')
plt.show()


# TOP 10 TRIGRAMS

# In[26]:


from nltk.util import ngrams

# Function for text preprocessing
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation and special characters
    text = ''.join([char for char in text if char not in string.punctuation and char.isalnum() or char.isspace()])
    
    # Tokenize the text
    tokens = word_tokenize(text)
    
    # Remove stop words
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    
    return tokens

# Tokenize and preprocess the sentences into words
df['tokenized_words'] = df['Sentence'].apply(preprocess_text)

# Extract trigrams
df['trigrams'] = df['tokenized_words'].apply(lambda x: list(ngrams(x, 3)))

# Flatten the list of trigrams
all_trigrams = [trigram for sentence_trigrams in df['trigrams'] for trigram in sentence_trigrams]

# Convert the results to a DataFrame for visualization
trigrams_df = pd.DataFrame(all_trigrams, columns=['Word1', 'Word2', 'Word3'])

# Combine the trigrams into a single column for counting
trigrams_df['Word1 Word2 Word3'] = trigrams_df['Word1'] + ' ' + trigrams_df['Word2'] + ' ' + trigrams_df['Word3']

# Display the most common trigrams
common_trigrams = trigrams_df['Word1 Word2 Word3'].value_counts().reset_index().rename(columns={'index': 'Word1 Word2 Word3', 'Word1 Word2 Word3': 'Frequency'})
print(common_trigrams.head(10))


# In[29]:


plt.figure(figsize=(6, 3))
sns.barplot(x='Word1 Word2 Word3', y='Frequency', data=common_trigrams.head(10), palette='viridis')
plt.xlabel('Trigram')
plt.ylabel('Frequency')
plt.title('Top 10 Most Common Trigrams')
plt.xticks(rotation=45, ha='right')
plt.show()


# In[30]:


# Creating column "processed_reviewText"

df_final['processed_Sentence'] = df_final['Sentence'].copy()

df_final.head()


# In[31]:


df_final['processed_Sentence'] = df_final['processed_Sentence'].str.lower()
df_final.head()


# 4. Data Pre-processing
# 
# Data pre-processing was done to remove the noise present in data for easier processing of the data by the algorithm.
# 
# Pre-processing steps followed :
# i. Conversion of the characters into lower case
# ii. Removal of Punctuation
# iii. Removal of Stopwords
# iv. Removal of frequently occuring words
# v. Removal of rare words
# vi. Stemming
# vii. Lemmatization
# viii. Removal of emojis
# i. Conversion of the characters into Lowercase
# Lower casing is a common text pre-processing technique. The idea of conversion of the characters into lowercase is to make all the characters into the same case so that the words like 'review', 'Review', 'REVIEW' are treated in the same way.
# 
# This conversion also helps in feature extraction techniques which will come in the later stage of pre-processing. In techniques like TF-IDF, it helps to combine the same words together thereby reducing the duplication and get correct counts or tfidf values.

# In[32]:


df_final[['processed_Sentence']]


# ii. Removal of Punctuations, URL, HTML TAGS
# 
# Our second step in preprocessing is to remove the punctuations from the text data which is a text standardization process that will help to treat 'hurray' and 'hurray!' in the same way.
# 
# For example, the string.punctuation in python contains the following punctuation symbols
# 
# !"#$%&'()*+,-./:;<=>?@[\]^_{|}~`
# 
# Removal of URL
# 
# There also exists some possibility that users will post the link of other products which they find useful while writing a review. Links containing URL will pose an obstacle while processing. The URL was also removed in this step.
# 
# e.g. : 'https?://\S+|www.\S+', '',
# 
# Removal of HTML Tags
# 
# One another common preprocessing technique that will come handy in multiple places is removal of html tags. This is especially useful, if we scrap the data from different websites. We might end up having html strings as part of our text.
# 
# e.g. : '<.*?>+', ''

# In[33]:


import re

def review_cleaning(text):
    '''Make text lowercase, remove text in square brackets,remove links,remove punctuation
    and remove words containing numbers.'''
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    ##df['text'] = df['text'].apply(lambda x: re.sub(r'\$', '', x))
    text = re.sub('\$', '', text)
    text = re.sub('\,', '', text)
    text = re.sub('\.', '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text


# In[34]:


df_final['processed_Sentence']=df_final['processed_Sentence'].apply(lambda x:review_cleaning(x))
df_final[['processed_Sentence']]


# iii. Removal of Stopwords
# 
# Stopwords are commonly occuring words in a language like 'the', 'a' and so on.As they don't add any valuable information for downstream analysis, they can be removed from the texts. In cases like Part of Speech tagging, we should not remove them as they provide very valuable information about the POS.
# 
# These stopword lists are already compiled for different languages and we can safely use them. For example, the stopword list for english language from the nltk package can be seen below.

# In[35]:


from nltk.corpus import stopwords
", ".join(stopwords.words('english'))


# In[36]:


STOPWORDS = set(stopwords.words('english'))
def remove_stopwords(text):
    """custom function to remove the stopwords"""
    return " ".join([word for word in str(text).split() if word not in STOPWORDS])

df_final["processed_Sentence"] = df_final["processed_Sentence"].apply(lambda text: remove_stopwords(text))
df_final[['processed_Sentence']]


# vii. Lemmatization
# 
# Lemmatization is similar to stemming in reducing inflected words to their word stem but differs in the way that it makes sure the root word (also called as lemma) retains in the text.
# 
# We have used the WordNetLemmatizer in nltk for the process.

# In[37]:


from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
def lemmatize_words(text):
    return " ".join([lemmatizer.lemmatize(word) for word in text.split()])

df_final["processed_Sentence_lemmatized"] = df_final["processed_Sentence"].apply(lambda text: lemmatize_words(text))
df_final[['processed_Sentence_lemmatized']]


# TextBlob is a Python library for processing textual data. It provides a simple API for common natural language processing (NLP) tasks, including part-of-speech tagging, noun phrase extraction, sentiment analysis, classification, translation, and more.
# 
# The sentiment analysis in TextBlob returns a sentiment polarity and subjectivity score for each sentence.

# TEXTBLOB APPLICATION

# In[38]:


pip install textblob


# In[39]:


from textblob import TextBlob


# #applying textblob to my dataset
# 1) converting each sentence to a textblob object.
# 2) use the 'sentiment' property to get the sentiment polarity and subjectivity.

# In[40]:


df_final["predicted_sentiment"] = df_final["processed_Sentence_lemmatized"].apply(lambda text: TextBlob(text).sentiment.polarity)

# Classify the sentiment as 'positive', 'neutral', or 'negative'
df_final["predicted_sentiment_label"] = df_final["predicted_sentiment"].apply(lambda x: 'positive' if x > 0 else ('neutral' if x == 0 else 'negative'))



# This code calculates the sentiment polarity for each lemmatized sentence using TextBlob. The sentiment is then classified into 'positive', 'neutral', or 'negative' based on the polarity score.

# In[41]:


#displaying the result

print(df_final[['processed_Sentence', 'processed_Sentence_lemmatized', 'predicted_sentiment', 'predicted_sentiment_label']])


# In[42]:


import matplotlib.pyplot as plt

# Count the occurrences of each sentiment label
sentiment_counts = df_final['predicted_sentiment_label'].value_counts()

# Create a bar chart
plt.figure(figsize=(8, 5))
sentiment_counts.plot(kind='bar', color=['green', 'gray', 'red'])
plt.title('Sentiment Distribution')
plt.xlabel('Sentiment Label')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.show()


# In[43]:


# Creating a pie chart
plt.figure(figsize=(8, 8))
plt.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', colors=['green', 'gray', 'red'])
plt.title('Sentiment Distribution')
plt.show()


# MODEL BUILDING USING TEXT BLOB TO FIND THE ACCURACY

# Define Features (Input) and Target (Output)
# 
# Input Feature (X):
# 
# processed_Sentence_lemmatized: This is our lemmatized and processed text. we'll use this as the input feature.
# 
# Target Variable (y):
# 
# predicted_sentiment_label: This is the sentiment label you want to predict. This will be your target variable.

# LABEL ENCODING FOR TARGET VARIABLE: THIS WILL CONVERT THE CATEGORICAL LABELS INTO NUMERICAL VALUES

# In[44]:


from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
df_final['encoded_sentiment'] = label_encoder.fit_transform(df_final['predicted_sentiment_label'])


# NOW WE HAVE A NEW COLUMN NAMED 'ENCODED_SENTIMENT' THAT WE CAN USE AS OUR TARGET VARIABLE FOR THE CLASSIFICATION MODELS.
# 
# "SPLITTING THE DATA INTO TRAIN AND TEST SETS"

# In[45]:


from sklearn.model_selection import train_test_split

X = df_final['processed_Sentence_lemmatized']
y = df_final['encoded_sentiment']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[46]:


#FEATURE EXTRACTION
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vectorizer = TfidfVectorizer()
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)


# BUILD AND TRAIN A CLASSIFICATION MODEL

# In[47]:


from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train_tfidf, y_train)

y_pred = model.predict(X_test_tfidf)

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# In[48]:


from sklearn.naive_bayes import MultinomialNB

# Create and train the model
nb_model = MultinomialNB()
nb_model.fit(X_train_tfidf, y_train)

# Make predictions
y_pred_nb = nb_model.predict(X_test_tfidf)

# Evaluate the model
accuracy_nb = accuracy_score(y_test, y_pred_nb)
print(f"Naive Bayes Accuracy: {accuracy_nb:.2f}")


# In[49]:


from sklearn.ensemble import RandomForestClassifier

# Create and train the model
rf_model = RandomForestClassifier()
rf_model.fit(X_train_tfidf, y_train)

# Make predictions
y_pred_rf = rf_model.predict(X_test_tfidf)

# Evaluate the model
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print(f"Random Forest Accuracy: {accuracy_rf:.2f}")


# In[50]:


from sklearn.svm import SVC

# Create and train the model
svm_model = SVC()
svm_model.fit(X_train_tfidf, y_train)

# Make predictions
y_pred_svm = svm_model.predict(X_test_tfidf)

# Evaluate the model
accuracy_svm = accuracy_score(y_test, y_pred_svm)
print(f"SVM Accuracy: {accuracy_svm:.2f}")


# MODEL BUILDING USING WORD EMBEDDING TECHNIQUES

# In[51]:


# Step 1: Train-Test Split
X = df_final['processed_Sentence_lemmatized']
y = df_final['Sentiment']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[52]:


# Step 2: Label Encoding for Target Variable
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)


# In[53]:


# Step 3: TF-IDF Vectorization
tfidf_vectorizer = TfidfVectorizer()
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)


# In[54]:


# Step 4: Bag of Words Vectorization (CountVectorizer)
count_vectorizer = CountVectorizer()
X_train_bow = count_vectorizer.fit_transform(X_train)
X_test_bow = count_vectorizer.transform(X_test)


# In[55]:


# Step 5: Build and Evaluate Models
def evaluate_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))


# In[56]:


# Logistic Regression with TF-IDF
logistic_regression_tfidf = LogisticRegression()
evaluate_model(logistic_regression_tfidf, X_train_tfidf, X_test_tfidf, y_train_encoded, y_test_encoded)

# Naive Bayes with TF-IDF
naive_bayes_tfidf = MultinomialNB()
evaluate_model(naive_bayes_tfidf, X_train_tfidf, X_test_tfidf, y_train_encoded, y_test_encoded)

# Random Forest with TF-IDF
random_forest_tfidf = RandomForestClassifier()
evaluate_model(random_forest_tfidf, X_train_tfidf, X_test_tfidf, y_train_encoded, y_test_encoded)


# In[57]:


# Logistic Regression with Bag of Words
logistic_regression_bow = LogisticRegression()
evaluate_model(logistic_regression_bow, X_train_bow, X_test_bow, y_train_encoded, y_test_encoded)

# Naive Bayes with Bag of Words
naive_bayes_bow = MultinomialNB()
evaluate_model(naive_bayes_bow, X_train_bow, X_test_bow, y_train_encoded, y_test_encoded)

# Random Forest with Bag of Words
random_forest_bow = RandomForestClassifier()
evaluate_model(random_forest_bow, X_train_bow, X_test_bow, y_train_encoded, y_test_encoded)


# In[ ]:


#TF-IDF

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer


# Step 1: Train-Test Split
X = df_final['processed_Sentence_lemmatized']
y = df_final['Sentiment']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 2: Label Encoding for Target Variable
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# Step 3: TF-IDF Vectorization (or you can use CountVectorizer if you have a single feature)
tfidf_vectorizer = TfidfVectorizer()
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Step 4: Build and Evaluate Decision Tree Model
decision_tree = DecisionTreeClassifier(max_depth=30, random_state=42)
decision_tree.fit(X_train_tfidf, y_train_encoded)

# Evaluate the model
y_pred = decision_tree.predict(X_test_tfidf)

accuracy = accuracy_score(y_test_encoded, y_pred)
print(f"Accuracy: {accuracy:.2f}")

print("\nClassification Report:")
print(classification_report(y_test_encoded, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test_encoded, y_pred))




# In[ ]:


from sklearn.model_selection import GridSearchCV

# Define the parameter grid
param_grid = {
    'max_depth': [5, 10, 15, 20, 25, 30],  
    'random_state': [5, 10, 15, 20, 25, 30,42] 
}

# Create the Decision Tree model
decision_tree = DecisionTreeClassifier()

# Create the GridSearchCV object
grid_search = GridSearchCV(decision_tree, param_grid, cv=5, scoring='accuracy')

# Fit the model to the data
grid_search.fit(X_train_tfidf, y_train_encoded)

# Get the best hyperparameters
best_params = grid_search.best_params_
print("Best Hyperparameters:", best_params)

# Get the best model
best_decision_tree = grid_search.best_estimator_

# Evaluate the best model
y_pred_best = best_decision_tree.predict(X_test_tfidf)

accuracy_best = accuracy_score(y_test_encoded, y_pred_best)
print(f"Best Model Accuracy: {accuracy_best:.2f}")

print("\nClassification Report (Best Model):")
print(classification_report(y_test_encoded, y_pred_best))

print("\nConfusion Matrix (Best Model):")
print(confusion_matrix(y_test_encoded, y_pred_best))


# In[ ]:


#AFTER GRID SEARCH CV

# Step 4: Build and Evaluate Decision Tree Model
decision_tree = DecisionTreeClassifier(max_depth=25, random_state=5)
decision_tree.fit(X_train_tfidf, y_train_encoded)

# Evaluate the model
y_pred = decision_tree.predict(X_test_tfidf)

accuracy = accuracy_score(y_test_encoded, y_pred)
print(f"Accuracy: {accuracy:.2f}")

print("\nClassification Report:")
print(classification_report(y_test_encoded, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test_encoded, y_pred))




# In[ ]:


#BOW

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer

# Step 1: Train-Test Split
X = df_final['processed_Sentence_lemmatized']
y = df_final['Sentiment']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 2: Label Encoding for Target Variable
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# Step 3: Bag of Words Vectorization (CountVectorizer)
count_vectorizer = CountVectorizer()
X_train_bow = count_vectorizer.fit_transform(X_train)
X_test_bow = count_vectorizer.transform(X_test)

# Step 4: Build and Evaluate Decision Tree Model
decision_tree_bow = DecisionTreeClassifier(max_depth=30, random_state=42)
decision_tree_bow.fit(X_train_bow, y_train_encoded)

# Evaluate the model
y_pred_bow = decision_tree_bow.predict(X_test_bow)

accuracy_bow = accuracy_score(y_test_encoded, y_pred_bow)
print(f"Accuracy (Bag of Words): {accuracy_bow:.2f}")

print("\nClassification Report (Bag of Words):")
print(classification_report(y_test_encoded, y_pred_bow))

print("\nConfusion Matrix (Bag of Words):")
print(confusion_matrix(y_test_encoded, y_pred_bow))


# In[ ]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer

# Step 1: Train-Test Split
X = df_final['processed_Sentence_lemmatized']
y = df_final['Sentiment']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 2: Label Encoding for Target Variable
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# Step 3: Bag of Words Vectorization (CountVectorizer)
count_vectorizer = CountVectorizer()
X_train_bow = count_vectorizer.fit_transform(X_train)
X_test_bow = count_vectorizer.transform(X_test)

# Step 4: Hyperparameter Tuning with Grid Search
param_grid = {
   'max_depth': [5, 10, 15, 20, 25, 30],  
    'random_state': [5, 10, 15, 20, 25, 30,42],
}

# Create the Decision Tree model
decision_tree = DecisionTreeClassifier()

# Create the GridSearchCV object
grid_search = GridSearchCV(decision_tree, param_grid, cv=5, scoring='accuracy')

# Fit the model to the data
grid_search.fit(X_train_bow, y_train_encoded)

# Get the best hyperparameters
best_params = grid_search.best_params_
print("Best Hyperparameters:", best_params)

# Get the best model
best_decision_tree_bow = grid_search.best_estimator_

# Evaluate the best model
y_pred_best_bow = best_decision_tree_bow.predict(X_test_bow)

accuracy_best_bow = accuracy_score(y_test_encoded, y_pred_best_bow)
print(f"Best Model Accuracy (Bag of Words): {accuracy_best_bow:.2f}")

print("\nClassification Report (Best Model - Bag of Words):")
print(classification_report(y_test_encoded, y_pred_best_bow))

print("\nConfusion Matrix (Best Model - Bag of Words):")
print(confusion_matrix(y_test_encoded, y_pred_best_bow))


# In[ ]:


##ADA boosting for BOW

from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer

# Step 1: Train-Test Split
X = df_final['processed_Sentence_lemmatized']
y = df_final['Sentiment']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 2: Label Encoding for Target Variable
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# Step 3: Bag of Words Vectorization (CountVectorizer)
count_vectorizer = CountVectorizer()
X_train_bow = count_vectorizer.fit_transform(X_train)
X_test_bow = count_vectorizer.transform(X_test)

# Step 4: Build and Evaluate AdaBoost Model
adaboost = AdaBoostClassifier(n_estimators=50, random_state=42)
adaboost.fit(X_train_bow, y_train_encoded)

# Evaluate the model
y_pred_adaboost = adaboost.predict(X_test_bow)

accuracy_adaboost = accuracy_score(y_test_encoded, y_pred_adaboost)
print(f"AdaBoost Accuracy: {accuracy_adaboost:.2f}")

print("\nClassification Report (AdaBoost):")
print(classification_report(y_test_encoded, y_pred_adaboost))

print("\nConfusion Matrix (AdaBoost):")
print(confusion_matrix(y_test_encoded, y_pred_adaboost))


# In[ ]:


##gradient boosting for BOW

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer

# Step 1: Train-Test Split
X = df_final['processed_Sentence_lemmatized']
y = df_final['Sentiment']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 2: Label Encoding for Target Variable
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# Step 3: Bag of Words Vectorization (CountVectorizer)
count_vectorizer = CountVectorizer()
X_train_bow = count_vectorizer.fit_transform(X_train)
X_test_bow = count_vectorizer.transform(X_test)

# Step 4: Build and Evaluate Gradient Boosting Model
gradient_boosting = GradientBoostingClassifier(n_estimators=400, random_state=42)
gradient_boosting.fit(X_train_bow, y_train_encoded)

# Evaluate the model
y_pred_gb = gradient_boosting.predict(X_test_bow)

accuracy_gb = accuracy_score(y_test_encoded, y_pred_gb)
print(f"Gradient Boosting Accuracy: {accuracy_gb:.2f}")

print("\nClassification Report (Gradient Boosting):")
print(classification_report(y_test_encoded, y_pred_gb))

print("\nConfusion Matrix (Gradient Boosting):")
print(confusion_matrix(y_test_encoded, y_pred_gb))


# In[ ]:


pip install xgboost


# In[ ]:


#XG gradient boosting for bOW

import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer

# Step 1: Train-Test Split
X = df_final['processed_Sentence_lemmatized']
y = df_final['Sentiment']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 2: Label Encoding for Target Variable
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# Step 3: Bag of Words Vectorization (CountVectorizer)
count_vectorizer = CountVectorizer()
X_train_bow = count_vectorizer.fit_transform(X_train)
X_test_bow = count_vectorizer.transform(X_test)

# Step 4: Build and Evaluate XGBoost Model
xgb_model = xgb.XGBClassifier(n_estimators=300, learning_rate=0.1,random_state=42)
xgb_model.fit(X_train_bow, y_train_encoded)

# Evaluate the model
y_pred_xgb = xgb_model.predict(X_test_bow)

accuracy_xgb = accuracy_score(y_test_encoded, y_pred_xgb)
print(f"XGBoost Accuracy: {accuracy_xgb:.2f}")

print("\nClassification Report (XGBoost):")
print(classification_report(y_test_encoded, y_pred_xgb))

print("\nConfusion Matrix (XGBoost):")
print(confusion_matrix(y_test_encoded, y_pred_xgb))


# In[ ]:


#XGgradient boosting for TF-IDF

import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer

# Step 1: Train-Test Split
X = df_final['processed_Sentence_lemmatized']
y = df_final['Sentiment']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 2: Label Encoding for Target Variable
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# Step 3: TF-IDF Vectorization (TF-IDFVectorizer)
tfidf_vectorizer = TfidfVectorizer()
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Step 4: Build and Evaluate XGBoost Model
xgb_model = xgb.XGBClassifier(n_estimators=100, random_state=42)
xgb_model.fit(X_train_tfidf, y_train_encoded)

# Evaluate the model
y_pred_xgb = xgb_model.predict(X_test_tfidf)

accuracy_xgb = accuracy_score(y_test_encoded, y_pred_xgb)
print(f"XGBoost Accuracy: {accuracy_xgb:.2f}")

print("\nClassification Report (XGBoost):")
print(classification_report(y_test_encoded, y_pred_xgb))

print("\nConfusion Matrix (XGBoost):")
print(confusion_matrix(y_test_encoded, y_pred_xgb))


# In[ ]:


#gradient boosting for TF-IDF

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer

# Step 1: Train-Test Split
X = df_final['processed_Sentence_lemmatized']
y = df_final['Sentiment']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 2: Label Encoding for Target Variable
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# Step 3: Bag of Words Vectorization (CountVectorizer)
tfidf_vectorizer = TfidfVectorizer()
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Step 4: Build and Evaluate Gradient Boosting Model
gradient_boosting = GradientBoostingClassifier(n_estimators=100, random_state=42)
gradient_boosting.fit(X_train_tfidf, y_train_encoded)

# Evaluate the model
y_pred_gb = gradient_boosting.predict(X_test_tfidf)

accuracy_gb = accuracy_score(y_test_encoded, y_pred_gb)
print(f"Gradient Boosting Accuracy: {accuracy_gb:.2f}")

print("\nClassification Report (Gradient Boosting):")
print(classification_report(y_test_encoded, y_pred_gb))

print("\nConfusion Matrix (Gradient Boosting):")
print(confusion_matrix(y_test_encoded, y_pred_gb))


# In[ ]:


#ADABOOST FOR TF-IDF

from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer

# Step 1: Train-Test Split
X = df_final['processed_Sentence_lemmatized']
y = df_final['Sentiment']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 2: Label Encoding for Target Variable
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# Step 3: Bag of Words Vectorization (CountVectorizer)
tfidf_vectorizer = TfidfVectorizer()
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Step 4: Build and Evaluate AdaBoost Model
adaboost = AdaBoostClassifier(n_estimators=50, random_state=42)
adaboost.fit(X_train_tfidf, y_train_encoded)

# Evaluate the model
y_pred_adaboost = adaboost.predict(X_test_tfidf)

accuracy_adaboost = accuracy_score(y_test_encoded, y_pred_adaboost)
print(f"AdaBoost Accuracy: {accuracy_adaboost:.2f}")

print("\nClassification Report (AdaBoost):")
print(classification_report(y_test_encoded, y_pred_adaboost))

print("\nConfusion Matrix (AdaBoost):")
print(confusion_matrix(y_test_encoded, y_pred_adaboost))


# In[ ]:


pip install imbalanced-learn


# In[ ]:


#main code

from collections import Counter
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import make_pipeline

seed = 66

# Train-Test Split
X = df_final['processed_Sentence_lemmatized']
y = df_final['Sentiment']
print(sorted(Counter(y).items()))

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.33,
    random_state=seed
)

# Label Encoding for Target Variable
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# TF-IDF Vectorization
tfidf_vectorizer = TfidfVectorizer()
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Using SMOTE in a pipeline with RandomForestClassifier
smote_pipeline = make_pipeline(TfidfVectorizer(), SMOTE(random_state=seed), RandomForestClassifier(random_state=seed))

print(sorted(Counter(y_train).items()))

# Fit the model and evaluate
smote_pipeline.fit(X_train, y_train)
accuracy = smote_pipeline.score(X_test, y_test)
print(f"Accuracy: {accuracy:.2f}")


# In[ ]:


from sklearn.model_selection import GridSearchCV

# Define the parameter grid to search
param_grid = {
    'randomforestclassifier__n_estimators': [50, 100, 200],
    'randomforestclassifier__max_depth': [None, 10, 20],
    'randomforestclassifier__min_samples_split': [2, 5, 10],
    'randomforestclassifier__min_samples_leaf': [1, 2, 4]
}

# Create the pipeline with TF-IDF, SMOTE, and RandomForestClassifier
smote_pipeline = make_pipeline(TfidfVectorizer(), SMOTE(random_state=seed), RandomForestClassifier(random_state=seed))

# Wrap the pipeline in GridSearchCV
grid_search = GridSearchCV(smote_pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1)

# Fit the grid search to the data
grid_search.fit(X_train, y_train)

# Print the best parameters
print("Best Parameters:", grid_search.best_params_)

# Evaluate the model with the best parameters
best_model = grid_search.best_estimator_
accuracy = best_model.score(X_test, y_test)
print(f"Accuracy with Best Parameters: {accuracy:.2f}")


# In[ ]:


from collections import Counter
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import make_pipeline

seed = 66

# Train-Test Split
X = df_final['processed_Sentence_lemmatized']
y = df_final['Sentiment']
print(sorted(Counter(y).items()))

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.33,
    random_state=seed
)

# Label Encoding for Target Variable
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# TF-IDF Vectorization
tfidf_vectorizer = TfidfVectorizer()
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Using SMOTE in a pipeline with Logistic Regression
smote_pipeline = make_pipeline(TfidfVectorizer(), SMOTE(random_state=seed), LogisticRegression(random_state=seed))

print(sorted(Counter(y_train).items()))

# Fit the model and evaluate
smote_pipeline.fit(X_train, y_train)
accuracy = smote_pipeline.score(X_test, y_test)
print(f"Accuracy: {accuracy:.2f}")


# In[ ]:


from collections import Counter
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

seed = 66

# Train-Test Split
X = df_final['processed_Sentence_lemmatized']
y = df_final['Sentiment']
print(sorted(Counter(y).items()))

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.33,
    random_state=seed
)

# Label Encoding for Target Variable
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# TF-IDF Vectorization
tfidf_vectorizer = TfidfVectorizer()
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Feature Scaling (Optional but can help with convergence)
scaler = StandardScaler(with_mean=False)  # Sparse matrix, so don't center the data
X_train_tfidf_scaled = scaler.fit_transform(X_train_tfidf)
X_test_tfidf_scaled = scaler.transform(X_test_tfidf)

# Using SMOTE in a pipeline with Logistic Regression
smote_pipeline = make_pipeline(TfidfVectorizer(), SMOTE(random_state=seed), LogisticRegression(C=1.0, penalty='l2', solver='liblinear', random_state=seed))

print(sorted(Counter(y_train).items()))

# Fit the model and evaluate
smote_pipeline.fit(X_train, y_train)
y_pred = smote_pipeline.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# In[ ]:


from collections import Counter
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from imblearn.combine import SMOTETomek, SMOTEENN
from imblearn.pipeline import make_pipeline
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

seed = 66

# Load your dataset (assuming df_final is your DataFrame)

# Train-Test Split
X = df_final['processed_Sentence_lemmatized']
y = df_final['Sentiment']

# Counter before any resampling
print("Counter before resampling:", sorted(Counter(y).items()))

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.33,
    random_state=seed,
    stratify=y  # Ensure stratified split for imbalanced classes
)

# Label Encoding for Target Variable
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# Using SMOTETomek in a pipeline with Logistic Regression
smotetomek_pipeline = make_pipeline(TfidfVectorizer(), SMOTETomek(random_state=seed), LogisticRegression(C=1.0, penalty='l2', solver='liblinear', random_state=seed))

# Using SMOTEENN in a pipeline with Logistic Regression
smoteenn_pipeline = make_pipeline(TfidfVectorizer(), SMOTEENN(random_state=seed), LogisticRegression(C=1.0, penalty='l2', solver='liblinear', random_state=seed))

# Counter before resampling
print("Counter before SMOTETomek:", sorted(Counter(y_train_encoded).items()))

# Stratified K-Fold Cross-Validation for SMOTETomek
stratified_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

cv_scores_smt = cross_val_score(smotetomek_pipeline, X_train, y_train_encoded, cv=stratified_kfold, scoring='accuracy')
print("Cross-Validation Scores (SMOTETomek):", cv_scores_smt)
print("Mean Accuracy (SMOTETomek):", cv_scores_smt.mean())

# Fit the SMOTETomek model and evaluate on the test set
smotetomek_pipeline.fit(X_train, y_train_encoded)
y_pred_smt = smotetomek_pipeline.predict(X_test)

# Counter after applying SMOTETomek
print("Counter after SMOTETomek:", sorted(Counter(y_pred_smt).items()))

accuracy_smt = accuracy_score(y_test_encoded, y_pred_smt)
print(f"\nAccuracy on Test Set (SMOTETomek): {accuracy_smt:.2f}")

print("\nClassification Report (SMOTETomek):")
print(classification_report(y_test_encoded, y_pred_smt))

print("\nConfusion Matrix (SMOTETomek):")
print(confusion_matrix(y_test_encoded, y_pred_smt))

# Counter before resampling
print("Counter before SMOTEENN:", sorted(Counter(y_train_encoded).items()))

# Stratified K-Fold Cross-Validation for SMOTEENN
cv_scores_senn = cross_val_score(smoteenn_pipeline, X_train, y_train_encoded, cv=stratified_kfold, scoring='accuracy')
print("\nCross-Validation Scores (SMOTEENN):", cv_scores_senn)
print("Mean Accuracy (SMOTEENN):", cv_scores_senn.mean())

# Fit the SMOTEENN model and evaluate on the test set
smoteenn_pipeline.fit(X_train, y_train_encoded)
y_pred_senn = smoteenn_pipeline.predict(X_test)

# Counter after applying SMOTEENN
print("Counter after SMOTEENN:", sorted(Counter(y_pred_senn).items()))

accuracy_senn = accuracy_score(y_test_encoded, y_pred_senn)
print(f"\nAccuracy on Test Set (SMOTEENN): {accuracy_senn:.2f}")

print("\nClassification Report (SMOTEENN):")
print(classification_report(y_test_encoded, y_pred_senn))

print("\nConfusion Matrix (SMOTEENN):")
print(confusion_matrix(y_test_encoded, y_pred_senn))


# In[ ]:


#Pickle File
import pickle
pickle_out = open('smote_pipeline_model.pkl', 'wb')
pickle.dump(smote_pipeline,pickle_out)
pickle_out.close()

