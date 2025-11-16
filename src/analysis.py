# analysis.py
# Auto-generated from the original notebook.
# Contains the full linear analysis pipeline used in the project.

# ---- Cell ----
import pandas as pd
import re
import nltk
import random
import spacy
import missingno as msno
from gensim import corpora
from gensim.models import LdaModel
import gensim
import seaborn as sns
import matplotlib.pyplot as plt

from wordcloud import WordCloud
from collections import Counter
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import bigrams, trigrams
from nltk.probability import FreqDist
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer

import warnings
warnings.filterwarnings('ignore')

# ---- Cell ----
# Read the Excel file and store it in a DataFrame. Upload the Dataset File to the Collab if you get error
df = pd.read_excel('/content/drive/MyDrive/DSPM/Final Project/Diabetes Continuous Glucose Monitoring – Data Export.xlsx')

# ---- Cell ----
# Display the first few rows of the dataset
df.head(2)

# ---- Cell ----
df.shape

# ---- Cell ----
df.columns

# ---- Cell ----
# Extract 5 random samples
sample_df = df.sample(n=5)

sample_post = []
for i, row in sample_df.iterrows():
    post = f"Post {i+1}: {row['Sound Bite Text']}"
    sample_post.append(post)
print('\n\n'.join(sample_post))

# ---- Cell ----
# Drop duplicates based solely on 'Sound Bite Text'
df_cleaned = df.drop_duplicates(subset='Sound Bite Text')

print(f"Original Dataset Shape: {df.shape}")
print(f"Dataset Shape after Dropping Unnecessary Column: {df_cleaned.shape}")

# ---- Cell ----
# Visualize missing data
msno.bar(df)
plt.title("Missing Data Visualization", fontsize=16)
plt.show()

# ---- Cell ----
# Display the information of the dataset
df.info()

# ---- Cell ----
# List of columns to drop due to missing or irrelevant data
columns_to_drop = [
    'Ratings and Scores', 'Author Location - Other', 'Tags', 'Post Shares',
    'Post Views', 'Post Dislikes', 'Product Name', 'Product Hierarchy',
    '@Mention Media Tags', 'LexisNexis Source Publisher', 'LexisNexis Source Category',
    'LexisNexis Source Genre', 'LexisNexis Source Quality', 'LexisNexis Company - High',
    'LexisNexis Company - Any', 'LexisNexis Person - High', 'LexisNexis Person - Any',
    'LexisNexis Institution - High', 'LexisNexis Institution - Any', 'LexisNexis Subject Group 1',
    'LexisNexis Subject 1', 'LexisNexis Subject Group 2', 'LexisNexis Subject 2',
    'LexisNexis Other Subjects', 'Is Paid'
]

# Drop these columns from the dataset
df_cleaned = df_cleaned.drop(columns=columns_to_drop)

# Display the shape of the dataset after dropping irrelevant columns
print(f"Original Dataset Shape: {df.shape}")
print(f"Dataset Shape after Dropping Irrelevant Columns: {df_cleaned.shape}")

# ---- Cell ----
# Define a function to handle combined text based on various conditions
def combine_text_and_context(row):
    # Handle Original Posts from Twitter with Quoted Post
    if row['Source Type'] == 'Twitter' and row['Post Type'] == 'Original' and pd.notnull(row['Quoted Post']):
        return f"{row['Sound Bite Text']} {row['Quoted Post']}"

    # Combine Title and Sound Bite Text for forums and blogs
    elif row['Source Type'] in ['Forum', 'Blog'] and pd.notnull(row['Title']):
        return f"{row['Title']} {row['Sound Bite Text']}"

    # Use only Sound Bite Text for other cases
    else:
        return row['Sound Bite Text']

# Apply the function to create a new 'Combined Text' column
df_cleaned['Combined Text'] = df_cleaned.apply(combine_text_and_context, axis=1)

# ---- Cell ----
# Display the first few rows to verify the changes
df_cleaned[['Source Type', 'Post Type', 'Quoted Post', 'Title', 'Sound Bite Text', 'Combined Text']].head()

# ---- Cell ----
# Convert to lowercase
df_cleaned['Combined Text'] = df_cleaned['Combined Text'].str.lower()
for i in df_cleaned['Combined Text'].sample(5):
    print("Text: ",i,'\n')

# ---- Cell ----
# Fill missing values in the 'Followers/Daily Unique Visitors/Subscribers' column with zero
df_cleaned['Followers/Daily Unique Visitors/Subscribers'] = df_cleaned['Followers/Daily Unique Visitors/Subscribers'].fillna(0)

# ---- Cell ----
import matplotlib.pyplot as plt
import seaborn as sns

# Boxplot to visualize the distribution of 'Followers/Daily Unique Visitors/Subscribers'
plt.figure(figsize=(10, 6))
sns.boxplot(x=df_cleaned['Followers/Daily Unique Visitors/Subscribers'])
plt.title('Distribution of Followers/Daily Unique Visitors/Subscribers')
plt.xlabel('Followers/Daily Unique Visitors/Subscribers')
plt.show()

# ---- Cell ----
# Explore the distribution of followers
plt.figure(figsize=(10, 6))
sns.histplot(df_cleaned['Followers/Daily Unique Visitors/Subscribers'], bins=30, kde=True)
plt.title("Distribution of Followers/Daily Unique Visitors", fontsize=16)
plt.xlabel("Followers/Daily Unique Visitors")
plt.ylabel("Frequency")
plt.show()

# ---- Cell ----
# Determine threshold (e.g., 90th percentile)
followers_threshold = df_cleaned['Followers/Daily Unique Visitors/Subscribers'].quantile(0.90)
print(f"Threshold for large followers (90th percentile): {followers_threshold}")

# Filter out authors with large followers
df_cleaned['Is Influencer'] = df_cleaned['Followers/Daily Unique Visitors/Subscribers'] >= followers_threshold

# Step 4: Remove influencers from the dataset
df_cleaned = df_cleaned[df_cleaned['Is Influencer'] == False]

# Display the number of rows before and after filtering
print(f"update dataset shape: {df_cleaned.shape}")

# ---- Cell ----
# Explore the distribution of followers after removing influencers
plt.figure(figsize=(10, 6))
sns.histplot(df_cleaned['Followers/Daily Unique Visitors/Subscribers'].dropna(), bins=30, kde=True)
plt.title("Distribution of Followers/Daily Unique Visitors After Influencers Removal", fontsize=16)
plt.xlabel("Followers/Daily Unique Visitors")
plt.ylabel("Frequency")
plt.show()

# ---- Cell ----
# List of professions that indicate non-consumer or professional experts
non_consumer_professions = ['Sales and Marketing', 'Blogging', 'Journalism', 'Creative Arts', 'Executive Management', 'Entrepreneurship', 'Health and Medicine']

# Define a function to check if the profession belongs to non-consumer professions
def is_non_consumer(profession):
    # If any non-consumer profession is found in the 'Professions' column, return True
    if pd.notnull(profession):
        for non_consumer in non_consumer_professions:
            if non_consumer in profession:
                return True
    return False

# Apply the filter to flag posts from non-consumers
df_cleaned['Is Non Consumer'] = df_cleaned['Professions'].apply(is_non_consumer)

# Filter out non-consumer posts
df_cleaned = df_cleaned[df_cleaned['Is Non Consumer'] == False]

# Display the number of rows before and after filtering
print(f"update dataset shape: {df_cleaned.shape}")

# ---- Cell ----
def cleanTxt(text):
    text = re.sub(r'https?://\S+|www\.\S+', '', text)  # Remove URLs
    text = re.sub('\n', ' ', text)  # Remove newline characters
    text = re.sub('\n\n', ' ', text)  # Remove newline characters
    return text

# Apply the clean_text function to the Combined Text column
df_cleaned['Combined Text'] = df_cleaned['Combined Text'].apply(cleanTxt)

# ---- Cell ----
# Function to check for the presence of emojis in the text
def contains_emoji(text):
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002500-\U00002BEF"  # chinese characters
                           u"\U00002702-\U000027B0"
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           u"\U0001F926-\U0001F937"
                           u"\U00010000-\U0010FFFF"
                           u"\u200d"
                           u"\u2640-\u2642"
                           u"\u2600-\u2B55"
                           u"\u23cf"
                           u"\u23e9"
                           u"\u231a"
                           u"\ufe0f"  # dingbats
                           u"\u3030"
                           "]+", flags=re.UNICODE)
    return bool(emoji_pattern.search(text))

# Apply the function to check for emojis in 'Combined Text'
df_with_emojis = df_cleaned[df_cleaned['Combined Text'].apply(contains_emoji)]

# Display the number of rows containing emojis (if any)
if not df_with_emojis.empty:
    print(f"Found {df_with_emojis.shape[0]} rows with emojis.")
    print(df_with_emojis[['Combined Text']].head())
else:
    print("No emojis found in the 'Combined Text' column.")

# ---- Cell ----
import re

# Function to remove all icons (including emojis)
def remove_icons(text):
    icon_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002500-\U00002BEF"  # miscellaneous symbols
                           u"\U0001F900-\U0001F9FF"  # supplemental symbols and pictographs
                           u"\U00002000-\U00002BFF"  # arrows, bullets, stars, math operators
                           u"\U0000FE0F-\U0000FEFF"  # variation selectors
                           u"\U0001F700-\U0001F77F"  # alchemical symbols
                           u"\u200d"                 # zero-width joiner
                           u"\u2640-\u2642"          # gender symbols
                           u"\u2600-\u26FF"          # miscellaneous symbols
                           u"\u23cf"                 # eject symbol
                           u"\u23e9"                 # fast-forward
                           u"\u231a"                 # watch
                           u"\ufe0f"                 # dingbats
                           u"\u3030"                 # wavy dash
                           "]+", flags=re.UNICODE)
    return re.sub(icon_pattern, '', text)

# Apply icon removal to relevant columns in the datasets
df_cleaned['Combined Text Without Icon'] = df_cleaned['Combined Text'].apply(remove_icons)

# ---- Cell ----
# Install emoji library if not installed
!pip install emoji

# ---- Cell ----
# Here, if we want to include emojis in the context as part of our sentiment analysis, we can convert them from symbols into text
import emoji

# Convert emojis to their textual representation
def convert_emoji_to_text(text):
    return emoji.demojize(text)

# Apply to the 'Combined Text' column
df_cleaned['Combined Text'] = df_cleaned['Combined Text'].apply(convert_emoji_to_text)

# ---- Cell ----
# Function to check for the presence of hashtags, mentions, or retweets
def check_social_media_elements(text):
    return any([re.search(r'#', text), re.search(r'@', text), re.search(r'\bRT\b', text)])

# Apply the function to check for social media elements in 'Combined Text'
df_with_social_elements = df_cleaned[df_cleaned['Combined Text'].apply(check_social_media_elements)]

# Display the number of rows containing hashtags, mentions, or retweets (if any)
if not df_with_social_elements.empty:
    print(f"Found {df_with_social_elements.shape[0]} rows with social media elements (hashtags, mentions, RT).")
    print(df_with_social_elements[['Combined Text']].head(3))
else:
    print("No hashtags, mentions, or retweets found in the 'Combined Text' column.")

# ---- Cell ----
# Create a function to clean the tweets
def remove_social_media_elements(text):
    text = re.sub(r'@[A-Za-z0–9]+', '', text) #Remove @mentions replace with blank
    text = re.sub(r'#', '', text) #Remove the ‘#’ symbol, replace with blank
    text = re.sub(r'RT[\s]+', '', text) #Removing RT, replace with blank
    text = re.sub(r':', '', text) # Remove :
    return text

df_cleaned['Combined Text'] = df_cleaned['Combined Text'].apply(remove_social_media_elements)
df_cleaned['Combined Text Without Icon'] = df_cleaned['Combined Text Without Icon'].apply(remove_social_media_elements)

# ---- Cell ----
# Remove punctuation
def remove_punc(text):
    punc_pattern = r'[^\w\s]'
    return re.sub(punc_pattern, '', text)

df_cleaned['Combined Text'] = df_cleaned['Combined Text'].apply(remove_punc)
df_cleaned['Combined Text Without Icon'] = df_cleaned['Combined Text Without Icon'].apply(remove_punc)

# After executing this code, the 'Combined Text' column in the DataFrame
# will have punctuation removed from its text entries.

#Example:
print(remove_punc("#qa#werqe-q/- Hello ####"))

# ---- Cell ----
# Concatenate all the narratives into a single string
all_narratives = ' '.join(df_cleaned['Combined Text'])

# Tokenize the text and count word frequencies
word_counts = Counter(all_narratives.split())

# Convert the word counts into a list of tuples for sorting
sorted_word_counts = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)

# Extract the top N words and their counts
top_words = [word for word, count in sorted_word_counts[:10]]
top_counts = [count for word, count in sorted_word_counts[:10]]

# Create a horizontal bar chart
plt.figure(figsize=(10, 6))
plt.barh(top_words, top_counts, color='skyblue')
plt.xlabel('Word Frequency')
plt.ylabel('Words')
plt.title('Top 10 Most Frequent Words Before Handling Stopwords, Stemming and Lemmatization')
plt.gca().invert_yaxis()  # Invert y-axis to have the highest frequency on top
plt.show()

# ---- Cell ----
word_counts

# ---- Cell ----
# Handling stopwords
nltk.download('punkt')  # For tokenization
nltk.download('stopwords')  # For stopwords
stopwords.words('english')

# ---- Cell ----
# Load stopwords
stop_words = set(stopwords.words('english'))

# Removing Stopwords and Updating the Dataset
def remove_stopwords(text):
    words = word_tokenize(text)
    # Remove stopwords and keep only alphabetic words
    words = [word.lower() for word in words if word.isalpha() and word.lower() not in stop_words]
    return ' '.join(words)

# Apply stopword removal to the 'Combined Text' column and update the dataset
df_cleaned['Combined Text'] = df_cleaned['Combined Text'].apply(remove_stopwords)
df_cleaned['Combined Text Without Icon'] = df_cleaned['Combined Text Without Icon'].apply(remove_stopwords)

# ---- Cell ----
# Concatenate all the narratives into a single string
all_narratives_cleaned = ' '.join(df_cleaned['Combined Text'])

# Tokenize the cleaned text and count word frequencies
word_counts_cleaned = Counter(all_narratives_cleaned.split())

# Convert the word counts into a list of tuples for sorting
sorted_word_counts_cleaned = sorted(word_counts_cleaned.items(), key=lambda x: x[1], reverse=True)

# Extract the top N words and their counts
top_words_cleaned = [word for word, count in sorted_word_counts_cleaned[:10]]
top_counts_cleaned = [count for word, count in sorted_word_counts_cleaned[:10]]

# Print most common words after stopword removal
print("Top 10 most common words after stopword removal:")
print(top_words_cleaned)

# ---- Cell ----
# Concatenate all the narratives into a single string
all_narratives_cleaned_wthicon = ' '.join(df_cleaned['Combined Text Without Icon'])

# Tokenize the cleaned text and count word frequencies
word_counts_cleaned_wthicon = Counter(all_narratives_cleaned_wthicon.split())

# Convert the word counts into a list of tuples for sorting
sorted_word_counts_cleaned_wthicon = sorted(word_counts_cleaned_wthicon.items(), key=lambda x: x[1], reverse=True)

# Extract the top N words and their counts
top_words_cleaned_wthicon = [word for word, count in sorted_word_counts_cleaned_wthicon[:10]]
top_counts_cleaned_wthicon = [count for word, count in sorted_word_counts_cleaned_wthicon[:10]]

# Print most common words after stopword removal
print("Top 10 most common words after stopword removal:")
print(top_words_cleaned_wthicon)

# ---- Cell ----
# Generate bigrams and trigrams from the cleaned words
cleaned_words_list = all_narratives_cleaned.split()
bigram_list = list(bigrams(cleaned_words_list))
trigram_list = list(trigrams(cleaned_words_list))

# Calculate frequencies of bigrams and trigrams
bigram_freq = FreqDist(bigram_list)
trigram_freq = FreqDist(trigram_list)

# Print most common bigrams and trigrams
print("Top 10 most common bigrams:")
print(bigram_freq.most_common(10))

print("\nTop 10 most common trigrams:")
print(trigram_freq.most_common(10))

# ---- Cell ----
# Generate bigrams and trigrams from the cleaned words
cleaned_words_list_wthicon = all_narratives_cleaned_wthicon.split()
bigram_list_wthicon = list(bigrams(cleaned_words_list_wthicon))
trigram_list_wthicon = list(trigrams(cleaned_words_list_wthicon))

# Calculate frequencies of bigrams and trigrams
bigram_freq_wthicon = FreqDist(bigram_list_wthicon)
trigram_freq_wthicon = FreqDist(trigram_list_wthicon)

# Print most common bigrams and trigrams
print("Top 10 most common bigrams:")
print(bigram_freq_wthicon.most_common(10))

print("\nTop 10 most common trigrams:")
print(trigram_freq_wthicon.most_common(10))

# ---- Cell ----
# Define frequent words (top 10 most common words after stopword removal)
frequent_words = ['dexcom', 'cgm', 'im', 'pump', 'get', 'blood', 'like', 'insulin', 'glucose', 'libre']

# Define frequent bigrams (top 10 most common bigrams)
frequent_bigrams = [('blood', 'sugar'), ('freestyle', 'libre'), ('continuous', 'glucose'), ('glucose', 'monitor'), ('pump', 'cgm'),
                    ('insulin', 'pump'), ('blood', 'glucose'), ('blood', 'sugars'), ('glucose', 'monitoring'), ('dexcom', 'cgm')]

# Define frequent trigrams (top 10 most common trigrams)
frequent_trigrams = [('continuous', 'glucose', 'monitor'), ('continuous', 'glucose', 'monitoring'), ('blood', 'sugar', 'levels'),
                     ('using', 'freestyle', 'libre'), ('continuous', 'glucose', 'monitors'), ('glucose', 'monitor', 'cgm'),
                     ('use', 'freestyle', 'libre'), ('closed', 'loop', 'system'), ('freestyle', 'libre', 'sensor'), ('low', 'blood', 'sugar')]

# Remove Frequent Words, Bigrams, and Trigrams
def remove_frequent_phrases(text):
    # Tokenize the text into individual words
    words = text.split()

    # Generate bigrams and trigrams from the tokenized words
    bigram_list = list(bigrams(words))
    trigram_list = list(trigrams(words))

    # Remove frequent words
    words = [word for word in words if word.lower() not in frequent_words]

    # Remove frequent bigrams
    bigram_list = [bigram for bigram in bigram_list if bigram not in frequent_bigrams]

    # Remove frequent trigrams
    trigram_list = [trigram for trigram in trigram_list if trigram not in frequent_trigrams]

    # Join the remaining words back into a string
    preprocessed_text = ' '.join(words)  # Join individual words
    return preprocessed_text

# Apply the function to the 'Combined Text' column to remove frequent phrases
df_cleaned['Combined Text'] = df_cleaned['Combined Text'].apply(remove_frequent_phrases)

# Verify the Results by Rechecking Word Statistics After Removal
# Concatenate all the narratives into a single string after cleaning
all_narratives_cleaned_final = ' '.join(df_cleaned['Combined Text'])

# Tokenize the cleaned text and count word frequencies again
word_counts_cleaned_final = Counter(all_narratives_cleaned_final.split())

# Convert the word counts into a list of tuples for sorting
sorted_word_counts_cleaned_final = sorted(word_counts_cleaned_final.items(), key=lambda x: x[1], reverse=True)

# Extract the top N words and their counts after removal of frequent phrases
top_words_cleaned_final = [word for word, count in sorted_word_counts_cleaned_final[:10]]
top_counts_cleaned_final = [count for word, count in sorted_word_counts_cleaned_final[:10]]

# Create a horizontal bar chart to visualize the most common words
plt.figure(figsize=(10, 6))
plt.barh(top_words_cleaned_final, top_counts_cleaned_final, color='skyblue')
plt.xlabel('Word Frequency')
plt.ylabel('Words')
plt.title('Top 10 Most Frequent Words After Stopword and Bigrams-Trigrams Removal')
plt.gca().invert_yaxis()  # Invert y-axis to have the highest frequency on top
plt.show()

# Print most common words after removing frequent phrases
print("Top 10 most common words after removing frequent words, bigrams, and trigrams:")
print(top_words_cleaned_final)

# ---- Cell ----
# Define frequent words (top 10 most common words after stopword removal)
frequent_words_wthicon = ['dexcom', 'cgm', 'im', 'pump', 'get', 'blood', 'like', 'insulin', 'glucose', 'libre']

# Define frequent bigrams (top 10 most common bigrams)
frequent_bigrams_wthicon = [('blood', 'sugar'), ('freestyle', 'libre'), ('continuous', 'glucose'), ('glucose', 'monitor'), ('pump', 'cgm'),
                    ('insulin', 'pump'), ('blood', 'glucose'), ('blood', 'sugars'), ('glucose', 'monitoring'), ('dexcom', 'cgm')]

# Define frequent trigrams (top 10 most common trigrams)
frequent_trigrams_wthicon = [('continuous', 'glucose', 'monitor'), ('continuous', 'glucose', 'monitoring'), ('blood', 'sugar', 'levels'),
                     ('using', 'freestyle', 'libre'), ('continuous', 'glucose', 'monitors'), ('glucose', 'monitor', 'cgm'),
                     ('use', 'freestyle', 'libre'), ('closed', 'loop', 'system'), ('freestyle', 'libre', 'sensor'), ('low', 'blood', 'sugar')]

# Remove Frequent Words, Bigrams, and Trigrams
def remove_frequent_phrases(text):
    # Tokenize the text into individual words
    words = text.split()

    # Generate bigrams and trigrams from the tokenized words
    bigram_list = list(bigrams(words))
    trigram_list = list(trigrams(words))

    # Remove frequent words
    words = [word for word in words if word.lower() not in frequent_words_wthicon]

    # Remove frequent bigrams
    bigram_list = [bigram for bigram in bigram_list if bigram not in frequent_bigrams_wthicon]

    # Remove frequent trigrams
    trigram_list = [trigram for trigram in trigram_list if trigram not in frequent_trigrams_wthicon]

    # Join the remaining words back into a string
    preprocessed_text = ' '.join(words)  # Join individual words
    return preprocessed_text

# Apply the function to the 'Combined Text Without Icon' column to remove frequent phrases
df_cleaned['Combined Text Without Icon'] = df_cleaned['Combined Text Without Icon'].apply(remove_frequent_phrases)

# Verify the Results by Rechecking Word Statistics After Removal
# Concatenate all the narratives into a single string after cleaning
all_narratives_cleaned_wthicon_final = ' '.join(df_cleaned['Combined Text Without Icon'])

# Tokenize the cleaned text and count word frequencies again
word_counts_cleaned_wthicon_final = Counter(all_narratives_cleaned_wthicon_final.split())

# Convert the word counts into a list of tuples for sorting
sorted_word_counts_cleaned_wthicon_final = sorted(word_counts_cleaned_wthicon_final.items(), key=lambda x: x[1], reverse=True)

# Extract the top N words and their counts after removal of frequent phrases
top_words_cleaned_wthicon_final = [word for word, count in sorted_word_counts_cleaned_wthicon_final[:10]]
top_counts_cleaned_wthicon_final = [count for word, count in sorted_word_counts_cleaned_wthicon_final[:10]]

# Create a horizontal bar chart to visualize the most common words
plt.figure(figsize=(10, 6))
plt.barh(top_words_cleaned_wthicon_final, top_counts_cleaned_wthicon_final, color='skyblue')
plt.xlabel('Word Frequency')
plt.ylabel('Words')
plt.title('Top 10 Most Frequent Words Without icon After Stopword and Bigrams-Trigrams Removal')
plt.gca().invert_yaxis()  # Invert y-axis to have the highest frequency on top
plt.show()

# Print most common words after removing frequent phrases
print("Top 10 most common words without icon after removing frequent words, bigrams, and trigrams:")
print(top_words_cleaned_wthicon_final)

# ---- Cell ----
# Extract 10 random samples
sample_df = df_cleaned.sample(n=5)

sample_posts = []
for i, row in sample_df.iterrows():
    tweet = f"Tweet {i+1}: {row['Combined Text']}"
    sample_posts.append(tweet)
print('\n\n'.join(sample_posts))

# ---- Cell ----
nltk.download('wordnet')

# ---- Cell ----
def preprocess_text(text):
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)

    # Tokenize the text
    words = word_tokenize(text)

    # Perform stemming (using Porter Stemmer)
    stemmer = PorterStemmer()
    stemmed_words = [stemmer.stem(word) for word in words]

    # Perform lemmatization (using WordNet Lemmatizer)
    lemmatizer = WordNetLemmatizer()
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words]

    # Join the processed words back into a sentence
    stemmed_text = ' '.join(stemmed_words)
    lemmatized_text = ' '.join(lemmatized_words)

    return stemmed_words, lemmatized_words

# Apply the preprocessing function to the 'Sound Bite Text' column
df_cleaned['Stemmed Text'], df_cleaned['Lemmatized Text'] = zip(*df_cleaned['Combined Text'].apply(preprocess_text))

# ---- Cell ----
df_cleaned.head()

# ---- Cell ----
cleaned_words_stemmed = [stem for stems_list in df_cleaned['Lemmatized Text'] for stem in stems_list]

# Use Counter to count the frequency of each root
word_freq = Counter(cleaned_words_stemmed)

# Get the most common roots
most_common_roots = word_freq.most_common(10)

# Extract the roots and their frequencies for plotting
roots, frequencies = zip(*most_common_roots)

# Create a bar plot
plt.figure(figsize=(10, 6))
plt.bar(roots, frequencies)
plt.xlabel('Root Words')
plt.ylabel('Frequency')
plt.title('Most Common Roots in Cleaned Text')
plt.xticks(rotation=45)
plt.tight_layout()

# Show the plot
plt.show()

# ---- Cell ----
# Group and aggregate data to count occurrences
source_counts = df_cleaned['Source Type'].value_counts().nlargest(10)

# Create a single subplot for visualizations
fig, ax = plt.subplots(figsize=(12, 6))

# Plot most common sources
sns.barplot(x=source_counts.values, y=source_counts.index, ax=ax)
ax.set_title('Top 10 Most Common Sources')
ax.set_xlabel('Count')
ax.set_ylabel('Source')

# Show the plot
plt.show()

# ---- Cell ----
df_sentiment = df_cleaned.groupby('Sentiment').count()['Post ID'].reset_index().sort_values(by='Post ID',ascending=False)
df_sentiment = df_sentiment.rename(columns={'Post ID': 'Total'})

# ---- Cell ----
import plotly.express as px

# Flatten the list of lemmatized words
cleaned_words_lemmatized = [lemma for lemma_list in df_cleaned['Lemmatized Text'] for lemma in lemma_list]

# Use Counter to count the frequency of each root word (lemmatized)
word_freq_lemmatized = Counter(cleaned_words_lemmatized)

# Get the most common lemmatized words
most_common_lemmatized = word_freq_lemmatized.most_common(20)

# Convert to DataFrame
temp_lemmatized = pd.DataFrame(most_common_lemmatized, columns=['Common_words', 'count'])

# Fancy bar chart for lemmatized words
fig = px.bar(temp_lemmatized, x="count", y="Common_words", title='Most Common Lemmatized Words', orientation='h',
             width=700, height=700, color='Common_words')
fig.show()

# Treemap for lemmatized words
fig = px.treemap(temp_lemmatized, path=['Common_words'], values='count', title='Treemap of Most Common Lemmatized Words')
fig.show()

# ---- Cell ----
df_cleaned.head()

# ---- Cell ----
# Import necessary libraries
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Define a function to generate word clouds for a specific sentiment
def generate_wordcloud(text, title, ax):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(text))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    ax.set_title(title, fontsize=16)

# Filter data by sentiment
df_positive = df_cleaned[df_cleaned['Sentiment'] == 'Positives']
df_negative = df_cleaned[df_cleaned['Sentiment'] == 'Negatives']
df_neutral = df_cleaned[df_cleaned['Sentiment'] == 'Neutrals']

# Extract lemmatized words for each sentiment
positive_words = [lemma for lemma_list in df_positive['Lemmatized Text'] if isinstance(lemma_list, list) for lemma in lemma_list]
negative_words = [lemma for lemma_list in df_negative['Lemmatized Text'] if isinstance(lemma_list, list) for lemma in lemma_list]
neutral_words = [lemma for lemma_list in df_neutral['Lemmatized Text'] if isinstance(lemma_list, list) for lemma in lemma_list]

# Create a single row of 3 subplots (one for each sentiment)
fig, axes = plt.subplots(1, 3, figsize=(20, 8))

# Generate word clouds for each sentiment and display them in the subplots
generate_wordcloud(positive_words, "Positive Sentiment WordCloud", axes[0])
generate_wordcloud(negative_words, "Negative Sentiment WordCloud", axes[1])
generate_wordcloud(neutral_words, "Neutral Sentiment WordCloud", axes[2])

# Display the word clouds
plt.tight_layout()
plt.show()

# ---- Cell ----
df_sentiment

# ---- Cell ----
from plotly import graph_objs as go
import plotly.express as px
import plotly.figure_factory as ff

# Create the funnel chart
fig = go.Figure(go.Funnelarea(
    text = df_sentiment.Sentiment,  # Labels for the funnel areas
    values = df_sentiment.Total,    # Values for each section of the funnel
))

# Set the title of the chart
fig.update_layout(
    title={
        'text': "Funnel-Chart of Sentiment Distribution",
        'y':0.9,  # Title position (higher on the plot)
        'x':0.5,  # Center the title
        'xanchor': 'center',
        'yanchor': 'top'
    }
)

# Show the chart
fig.show()

# ---- Cell ----
#add a column to keep the original texts from sound bite text and title
df_cleaned['Original Combined Text'] = df_cleaned.apply(combine_text_and_context, axis=1)
df_cleaned.head()

# ---- Cell ----
products = ['FreeStyle Libre', 'Dexcom', 'Medtronic', 'Senseonics']

def detect_products(text):
    detected_products = []
    for product in products:
        if product.lower() in text.lower():
            detected_products.append(product)
    return detected_products if detected_products else ['Other']
#detect which product is mentioned in the reviews
df_cleaned['Product Mentions'] = df_cleaned['Original Combined Text'].apply(detect_products)
df_cleaned.head()

# ---- Cell ----
#how many reviews for each product (give us a hints about the popularity of products on internet)
df_cleaned['Product Mentions'].value_counts()

# ---- Cell ----
# visualization
all_product_mentions = [product for sublist in df_cleaned['Product Mentions'] for product in sublist]
product_counts = Counter(all_product_mentions)
labels, counts = zip(*product_counts.items())
plt.figure(figsize=(8, 8))
colors = ['#ff9999','#66b3ff','#99ff99','#ffcc99', '#c2c2f0', '#ffb3e6', '#c4e17f']  # Color palette
explode = [0.1 if count == min(counts) else 0 for count in counts]  # Explode the smallest segments
wedges, texts, autotexts = plt.pie(counts, labels=labels, autopct=lambda p: f'{p:.2f}%', startangle=90,
                                   colors=colors, explode=explode, shadow=True, wedgeprops={'edgecolor': 'black'}, pctdistance=0.85)
#readability
for text in texts:
    text.set_fontsize(12)
for autotext in autotexts:
    autotext.set_fontsize(10)
    autotext.set_color('black')

centre_circle = plt.Circle((0, 0), 0.70, fc='white')
fig = plt.gcf()
fig.gca().add_artist(centre_circle)

plt.axis('equal')
plt.title('Distribution of Reviews by CGM Products', fontsize=16)
plt.show()

# ---- Cell ----
#positive and negative reviews for each product
df_exploded = df_cleaned.explode('Product Mentions')
sentiment_counts = df_exploded.groupby(['Product Mentions', 'Sentiment']).size().unstack(fill_value=0)

sentiment_percentage = sentiment_counts.div(sentiment_counts.sum(axis=1), axis=0) * 100
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()
products = ['Dexcom', 'FreeStyle Libre', 'Medtronic', 'Senseonics']
colors = ['#99ff99', '#66b3ff', '#ff9999', '#ffcc99']
for i, product in enumerate(products):
    sentiment_percentage.loc[product].plot(kind='bar', stacked=True, ax=axes[i], color=colors, edgecolor='black')
    axes[i].set_title(f'Sentiment Distribution for {product}', fontsize=14)
    axes[i].set_ylabel('Percentage (%)')
    axes[i].set_ylim(0, 100)
    axes[i].legend(loc='upper left', bbox_to_anchor=(1.0, 1.0))
plt.tight_layout()
plt.show()

# ---- Cell ----
df_exploded = df_cleaned.explode('Product Mentions')

# ---- Cell ----
df_exploded.columns

# ---- Cell ----
# Group by 'Product Mentions' and concatenate the 'Combined Text' for each product
df_exploaded_merged = df_exploded.groupby('Product Mentions')['Combined Text'].apply(lambda texts: ' '.join(texts)).reset_index()

# ---- Cell ----
from transformers import pipeline

# Load the summarization model from Hugging Face
summarizer = pipeline("summarization")

# Function to summarize text using Hugging Face summarization model
def summarize_text(text, max_length=1024, min_length=30):
    try:
        summary = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
        return summary[0]['summary_text']
    except Exception as e:
        return text  # If there's an issue, return the original text

# Function to process job description
def process_combined_text(description):
    # Summarize the job description using the transformers model
    summary = summarize_text(description)

    # Return only the summary
    return summary

# ---- Cell ----
# Apply the summarization function to the 'Description' column
df_exploaded_merged['Summarized Combined Text'] = df_exploaded_merged['Combined Text'].apply(process_combined_text)

# ---- Cell ----
df_exploaded_merged.head(1)

# ---- Cell ----
!pip install vaderSentiment

# ---- Cell ----
# from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# # Initialize VADER sentiment analyzer
# analyzer = SentimentIntensityAnalyzer()

# # Function to classify the sentiment of the comment
# def classify_sentiment(comment):
#     sentiment_score = analyzer.polarity_scores(comment)
#     compound = sentiment_score['compound']

#     if compound > 0.05:
#         return 'Positive'
#     elif compound < -0.05:
#         return 'Negative'
#     else:
#         return 'Neutral'

# # Apply sentiment analysis to the 'Comment' column
# df_exploaded_merged['Summarized Sentiment'] = df_exploaded_merged['Combined Text'].apply(classify_sentiment)

# ---- Cell ----
#review sources for each products
df_exploded = df_cleaned.explode('Product Mentions')
source_counts = df_exploded.groupby(['Product Mentions', 'Source Type']).size().unstack(fill_value=0)
source_percentage = source_counts.div(source_counts.sum(axis=1), axis=0) * 100
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()
products = ['Dexcom', 'FreeStyle Libre', 'Medtronic', 'Senseonics']
for i, product in enumerate(products):
    source_percentage.loc[product].plot(kind='bar', stacked=True, ax=axes[i], color=['#99ff99', '#66b3ff', '#ff9999', '#ffcc99', '#c2c2f0', '#ffb3e6'])
    axes[i].set_title(f'Source Distribution for {product}', fontsize=14)
    axes[i].set_ylabel('Reviews Percentage')
    axes[i].set_ylim(0, source_percentage.max().max() + 1)
    axes[i].legend(title='Source Types', bbox_to_anchor=(1.0, 1.0))
plt.tight_layout()
plt.show()

# ---- Cell ----
df_cleaned.head()

# ---- Cell ----
# Use LDA mode to do topic analysis
#For Dexcom
df_dex = df_cleaned[df_cleaned['Product Mentions'].apply(lambda x: 'Dexcom' in x)]
dictionary = corpora.Dictionary(df_dex['Lemmatized Text'])
corpus = [dictionary.doc2bow(text) for text in df_dex['Lemmatized Text']]
# Train LDA Model
num_topics = 10
lda_model = LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=15)
topics = lda_model.print_topics(num_words=5)
topics_df = pd.DataFrame(topics, columns=['Topic', 'Words'])
print("Topics:")
print(topics_df)

# ---- Cell ----
# Determine Most Important Topic
def get_most_important_topic(product_corpus):
    topic_distribution = lda_model.get_document_topics(product_corpus)

    topic_counts = {}
    for doc_topics in topic_distribution:
        for topic, prob in doc_topics:
            if topic in topic_counts:
                topic_counts[topic] += prob
            else:
                topic_counts[topic] = prob

    most_important_topic = max(topic_counts, key=topic_counts.get)
    return most_important_topic

top_topic_dexcom = get_most_important_topic(corpus)
print(f"Most Important Topic for Dexcom is: {topics_df.Words[top_topic_dexcom]}")

# ---- Cell ----
#For FreeStyle Libre
df_fre = df_cleaned[df_cleaned['Product Mentions'].apply(lambda x: 'FreeStyle Libre' in x)]
dictionary = corpora.Dictionary(df_fre['Lemmatized Text'])
corpus = [dictionary.doc2bow(text) for text in df_fre['Lemmatized Text']]
# Train LDA Model
num_topics = 10
lda_model = LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=15)
topics = lda_model.print_topics(num_words=5)
topics_df = pd.DataFrame(topics, columns=['Topic', 'Words'])
print("Topics:")
print(topics_df)

# ---- Cell ----
top_topic_FreeStyle = get_most_important_topic(corpus)
print(f"Most Important Topic for FreeStyle Libre is: {topics_df.Words[top_topic_FreeStyle]}")

# ---- Cell ----
#For 'Medtronic'
df_med = df_cleaned[df_cleaned['Product Mentions'].apply(lambda x: 'Medtronic' in x)]
dictionary = corpora.Dictionary(df_med['Lemmatized Text'])
corpus = [dictionary.doc2bow(text) for text in df_med['Lemmatized Text']]
# Train LDA Model
num_topics = 5
lda_model = LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=15)
topics = lda_model.print_topics(num_words=5)
topics_df = pd.DataFrame(topics, columns=['Topic', 'Words'])
print("Topics:")
print(topics_df)

# ---- Cell ----
top_topic_med = get_most_important_topic(corpus)
print(f"Most Important Topic for Medtronic is: {topics_df.Words[top_topic_med]}")

# ---- Cell ----
#For ''Senseonics''
df_sen = df_cleaned[df_cleaned['Product Mentions'].apply(lambda x: 'Senseonics' in x)]
dictionary = corpora.Dictionary(df_sen['Lemmatized Text'])
corpus = [dictionary.doc2bow(text) for text in df_sen['Lemmatized Text']]
# Train LDA Model
num_topics = 5
lda_model = LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=15)
topics = lda_model.print_topics(num_words=5)
topics_df = pd.DataFrame(topics, columns=['Topic', 'Words'])
print("Topics:")
print(topics_df)

# ---- Cell ----
top_topic_sen = get_most_important_topic(corpus)
print(f"Most Important Topic for Senseonics is: {topics_df.Words[top_topic_sen]}")

# ---- Cell ----
#apart from that we could also take a loot at the most common words mentioned for each products
product_list = ['Dexcom', 'FreeStyle Libre', 'Medtronic', 'Senseonics']
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()
for i, product in enumerate(product_list):
    cleaned_words_stemmed = [stem for stems_list in df_cleaned[df_cleaned['Product Mentions'].apply(lambda x: product in x)]['Stemmed Text'] for stem in stems_list]


    # Use Counter to count the frequency of each root
    word_freq = Counter(cleaned_words_stemmed)

    # Get the most common roots
    most_common_roots = word_freq.most_common(10)

    # Extract the roots and their frequencies for plotting
    roots, frequencies = zip(*most_common_roots)

    # Create a bar plot
    axes[i].bar(roots, frequencies)
    axes[i].set_xlabel('Root Words')
    axes[i].set_ylabel('Frequency')
    axes[i].set_title(f'Most Common Roots in Cleaned Text for {product}')
    axes[i].tick_params(axis='x',rotation=45)
plt.tight_layout()
# Show the plot
plt.show()

# ---- Cell ----
stemmer = PorterStemmer()

attributes_map = {
    'performance': ['performance', 'reliable', 'accurate', 'fast', 'efficiency', 'consistency', 'work', 'well','blood','sugar','diabet'],
    'usability': ['easy', 'user','friendly', 'simple', 'intuitive', 'convenient', 'accessible','use'],
    'features': ['sense','system','pump','sensor', 'app', 'integration', 'battery', 'life', 'display', 'alerts', 'connectivity', 'data', 'sharing', 'real'],
    'support': ['customer support', 'service', 'help', 'technical', 'support', 'assistance', 'response'],
    'competitors': ['dexcom','medtronic', 'senseonics', 'tandem', 'g6','freestyle','libre','2','freestyl','libr'],
    'price': ['affordable', 'expensive', 'cost', 'price', 'value',  'money', 'cheap', 'worth'],
    'accuracy': ['accuracy', 'precise', 'reliable', 'readings', 'correctness'],
    'quality': ['durability', 'build','quality', 'materials', 'design', 'sturdiness','year','time'],
    'side effects': ['side', 'effects', 'reaction', 'issue', 'problem', 'complications'],
    'availability': ['availability', 'stock', 'purchase', 'buy', 'order', 'shipping'],
    'compatibility': ['compatible', 'works', 'with', 'integration', 'connect', 'to'],
}
#stemed the word first so that it could better match the original stemmed text column
stemmed_attributes = {key: [stemmer.stem(word) for word in keywords] for key, keywords in attributes_map.items()}

def detect_attributes(stemmed_text_list):
    detected_counts = {key: 0 for key in stemmed_attributes.keys()}
    for key, keywords in stemmed_attributes.items():
        if any(stemmed_word in keywords for stemmed_word in stemmed_text_list):  # Direct match with stemmed word
            detected_counts[key] += 1

    return detected_counts


attribute_counts = df_cleaned['Stemmed Text'].apply(detect_attributes)
attributes_df = pd.DataFrame(attribute_counts.tolist(), index=df_cleaned.index)
df_combined = pd.concat([df_cleaned, attributes_df], axis=1)
df_combined.head()

# ---- Cell ----
# Define the product list and sentiment labels
product_list = ['Dexcom', 'FreeStyle Libre', 'Medtronic', 'Senseonics']
sentiment_labels = ['Positives', 'Negatives']

fig, axes = plt.subplots(2, 4, figsize=(20, 10))  # 2 rows for sentiment, 4 columns for products
axes = axes.flatten()

for i, product in enumerate(product_list):
    for j, sentiment in enumerate(sentiment_labels):
        filtered_data = df_combined[(df_combined['Product Mentions'].apply(lambda x: product in x)) &(df_combined['Sentiment'] == sentiment)]
        # Sum the attribute columns
        attribute_counts = filtered_data[attributes_map.keys()].sum()
        total_reviews = len(filtered_data)
        attribute_counts = (attribute_counts / total_reviews) * 100
        attributes, counts = attribute_counts.index, attribute_counts.values
        axes[i * 2 + j].bar(attributes, counts)
        axes[i * 2 + j].set_xlabel('Attributes')
        axes[i * 2 + j].set_ylabel('Count')
        axes[i * 2 + j].set_title(f'{sentiment} Sentiment for {product}')
        axes[i * 2 + j].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()

# ---- Cell ----
# Focus attributes for this part
focus_attributes = ['price', 'performance', 'accuracy', 'quality']
product_list = ['Dexcom', 'FreeStyle Libre', 'Medtronic', 'Senseonics']
sentiment_labels = ['Positives', 'Negatives']

# Initialize an empty DataFrame to store the results for each attribute and sentiment
results = pd.DataFrame(columns=['Product', 'Attribute', 'Positive Count', 'Negative Count'])

# Loop over each product and focus attribute
for product in product_list:
    for attribute in focus_attributes:
        # Filter rows for the specific product
        product_df = df_combined[df_combined['Product Mentions'].apply(lambda x: product in x)]

        # Calculate positive and negative counts for the attribute
        positive_count = product_df[(product_df['Sentiment'] == 'Positives')][attribute].sum()
        negative_count = product_df[(product_df['Sentiment'] == 'Negatives')][attribute].sum()

        # Create a DataFrame for the current row
        new_row = pd.DataFrame({
            'Product': [product],
            'Attribute': [attribute],
            'Positive Count': [positive_count],
            'Negative Count': [negative_count]
        })

        # Concatenate the new row to the results DataFrame
        results = pd.concat([results, new_row], ignore_index=True)

# Display the results
print(results)

# ---- Cell ----
# Visualize the counts for each product and attribute
fig, axes = plt.subplots(2, 2, figsize=(16, 10))  # 2 rows and 2 columns for products
axes = axes.flatten()

# Plot for each product
for i, product in enumerate(product_list):
    product_data = results[results['Product'] == product]

    # Plot Positive and Negative Counts side by side for each attribute
    bar_width = 0.35
    index = range(len(focus_attributes))

    pos_counts = product_data['Positive Count']
    neg_counts = product_data['Negative Count']
    colors = ['#99ff99', '#ff9999']
    # Plot bars
    axes[i].bar(index, pos_counts, bar_width, label='Positive', color=colors[0])
    axes[i].bar([x + bar_width for x in index], neg_counts, bar_width, label='Negative', color=colors[1])

    # Set chart details
    axes[i].set_xlabel('Attributes')
    axes[i].set_ylabel('Count')
    axes[i].set_title(f'Sentiment for {product}')
    axes[i].set_xticks([x + bar_width / 2 for x in index])
    axes[i].set_xticklabels(focus_attributes)
    axes[i].legend()

plt.tight_layout()
plt.show()

# ---- Cell ----
twitter_reviews = df_cleaned[df_cleaned['Source Type'] == 'Twitter'].sample(5)
for index, row in twitter_reviews.iterrows():
    review = row['Original Combined Text']
    stemmed_text = row['Stemmed Text']
    print(f"Some Sound Bite Text and Quoted Post for Twitter reviews: {review}")
    print(f"Some Stemmed reviews: {stemmed_text}")

# ---- Cell ----
nontwitter_reviews = df_cleaned[df_cleaned['Source Type'] != 'Twitter'].sample(5)
for index, row in nontwitter_reviews.iterrows():
    review = row['Original Combined Text']
    stemmed_text = row['Stemmed Text']
    print(f"Some Sound Bite Text and Quoted Post for Non-Twitter reviews: {review}")
    print(f"Some Stemmed reviews: {stemmed_text}")

# ---- Cell ----
!pip install vaderSentiment

# ---- Cell ----
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

analyzer = SentimentIntensityAnalyzer()

#function to classify sentiment
def get_vader_sentiment(text):
    score = analyzer.polarity_scores(text)
    if score['compound'] >= 0.05:
        return 'Positive'
    elif score['compound'] <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'

df_cleaned['Predicted Sentiment'] = df_cleaned['Combined Text'].apply(get_vader_sentiment)

#Compare the predicted sentiment with existing sentiment
conf_matrix = confusion_matrix(df_cleaned['Sentiment'], df_cleaned['Predicted Sentiment'], labels=['Positive', 'Negative', 'Neutral', 'Mixed'])
print("Confusion Matrix:\n", conf_matrix)

# ---- Cell ----
df_cleaned['Predicted Sentiment Without Icon'] = df_cleaned['Combined Text Without Icon'].apply(get_vader_sentiment)

#Compare the predicted sentiment with existing sentiment
conf_matrix_wthicon = confusion_matrix(df_cleaned['Sentiment'], df_cleaned['Predicted Sentiment Without Icon'], labels=['Positive', 'Negative', 'Neutral', 'Mixed'])
print("Confusion Matrix:\n", conf_matrix_wthicon)

# ---- Cell ----
df_cleaned.head(1)

# ---- Cell ----
#confusion matrix
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Positive', 'Negative', 'Neutral', 'Mixed'], yticklabels=['Positive', 'Negative', 'Neutral', 'Mixed'])
plt.xlabel('Predicted Sentiment')
plt.ylabel('Actual Sentiment')
plt.title('Confusion Matrix for Sentiment Analysis')
plt.show()

#accuracy and classification report
print("Classification Report:\n", classification_report(df_cleaned['Sentiment'], df_cleaned['Predicted Sentiment']))

# Explode the 'Product Mentions' column to convert lists into separate rows
df_exploded = df_cleaned.explode('Product Mentions')

# Check the structure after exploding
print(df_exploded[['Product Mentions', 'Predicted Sentiment']].head())

# Sentiment distribution for each product
product_sentiment = df_exploded.groupby(['Product Mentions', 'Predicted Sentiment']).size().unstack(fill_value=0)

# Plot the sentiment distribution
product_sentiment.plot(kind='bar', stacked=True, figsize=(10, 6))
plt.title('Sentiment Distribution for Each Product')
plt.xlabel('Product')
plt.ylabel('Count')
plt.show()

# ---- Cell ----
#Standardizing the labels in the 'Sentiment' column to match predicted labels
df_cleaned['Standardized Sentiment'] = df_cleaned['Sentiment'].replace({
    'Positives': 'Positive',
    'Negatives': 'Negative',
    'Neutrals': 'Neutral',
    'Mixed': 'Mixed'
})

#Re-compute the sentiment match based on standardized labels
df_cleaned['Sentiment Match'] = df_cleaned['Predicted Sentiment'] == df_cleaned['Standardized Sentiment']

#Calculate the number of matches and mismatches again
num_matches = df_cleaned['Sentiment Match'].sum()
num_mismatches = len(df_cleaned) - num_matches
print(f"Number of Matching Sentiments: {num_matches}")
print(f"Number of Mismatching Sentiments: {num_mismatches}")
match_percentage = (num_matches / len(df_cleaned)) * 100
print(f"Match Percentage: {match_percentage:.2f}%")
print(df_cleaned[['Standardized Sentiment', 'Predicted Sentiment', 'Sentiment Match']].head())

# ---- Cell ----
!pip install textblob flair transformers torch

# ---- Cell ----
from textblob import TextBlob

# Function to classify sentiment using TextBlob
def textblob_sentiment(text):
    analysis = TextBlob(text)
    if analysis.sentiment.polarity > 0.5:
        return 'Positive'
    elif analysis.sentiment.polarity < -0.5:
        return 'Negative'
    else:
        return 'Neutral'

# Apply TextBlob sentiment analysis
df_cleaned['TextBlob Sentiment'] = df_cleaned['Combined Text'].apply(textblob_sentiment)

print(df_cleaned[['Sentiment', 'TextBlob Sentiment']].head())

# ---- Cell ----
from textblob import TextBlob

# the function to classify sentiment using TextBlob
def textblob_sentiment(text):
    analysis = TextBlob(text)
    if analysis.sentiment.polarity > 0.5:
        return 'Positive'
    elif analysis.sentiment.polarity < -0.5:
        return 'Negative'
    else:
        return 'Neutral'

# Applying TextBlob sentiment analysis to the 'Combined Text' column
df_cleaned['TextBlob Sentiment'] = df_cleaned['Combined Text'].apply(textblob_sentiment)

# Standardizing the labels in the 'Sentiment' column to match TextBlob predicted labels
df_cleaned['Standardized Sentiment'] = df_cleaned['Sentiment'].replace({
    'Positives': 'Positive',
    'Negatives': 'Negative',
    'Neutrals': 'Neutral',
    'Mixed': 'Mixed'
})

# Re-computing the sentiment match based on TextBlob predictions
df_cleaned['TextBlob Sentiment Match'] = df_cleaned['TextBlob Sentiment'] == df_cleaned['Standardized Sentiment']

# Calculating the number of matches and mismatches
num_matches = df_cleaned['TextBlob Sentiment Match'].sum()
num_mismatches = len(df_cleaned) - num_matches
print(f"Number of Matching Sentiments (TextBlob): {num_matches}")
print(f"Number of Mismatching Sentiments (TextBlob): {num_mismatches}")
match_percentage = (num_matches / len(df_cleaned)) * 100
print(f"Match Percentage (TextBlob): {match_percentage:.2f}%")
print(df_cleaned[['Standardized Sentiment', 'TextBlob Sentiment', 'TextBlob Sentiment Match']].head())

# ---- Cell ----
import matplotlib.pyplot as plt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob

# Initialize VADER sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# Define VADER sentiment function
def get_vader_sentiment(text):
    score = analyzer.polarity_scores(text)
    if score['compound'] >= 0.5:
        return 'Positive'
    elif score['compound'] <= -0.5:
        return 'Negative'
    else:
        return 'Neutral'

# Define TextBlob sentiment function
def get_textblob_sentiment(text):
    analysis = TextBlob(text)
    if analysis.sentiment.polarity > 0.5:
        return 'Positive'
    elif analysis.sentiment.polarity < -0.5:
        return 'Negative'
    else:
        return 'Neutral'

# Step 1: Apply VADER and TextBlob sentiment analysis to the 'Combined Text' column
df_cleaned['VADER Sentiment'] = df_cleaned['Combined Text'].apply(get_vader_sentiment)
df_cleaned['TextBlob Sentiment'] = df_cleaned['Combined Text'].apply(get_textblob_sentiment)

# Step 2: Explode 'Product Mentions' to get each product in separate rows
df_exploded = df_cleaned.explode('Product Mentions')

# Step 3: Calculate sentiment counts for each product using VADER and TextBlob
vader_sentiment_counts = df_exploded.groupby(['Product Mentions', 'VADER Sentiment']).size().unstack(fill_value=0)
textblob_sentiment_counts = df_exploded.groupby(['Product Mentions', 'TextBlob Sentiment']).size().unstack(fill_value=0)

# Step 4: Calculate sentiment percentage for each product
vader_sentiment_percentage = vader_sentiment_counts.div(vader_sentiment_counts.sum(axis=1), axis=0) * 100
textblob_sentiment_percentage = textblob_sentiment_counts.div(textblob_sentiment_counts.sum(axis=1), axis=0) * 100

# Step 5: Plot sentiment distribution for each product using VADER and TextBlob

# Define product list and colors
products = ['Dexcom', 'FreeStyle Libre', 'Medtronic', 'Senseonics']
colors = ['#99ff99', '#66b3ff', '#ff9999']  # Colors for Positive, Neutral, Negative

# Create subplots for VADER and TextBlob
fig, axes = plt.subplots(2, 2, figsize=(14, 12))  # 2 rows, 2 columns
fig.suptitle("Sentiment Distribution for Each Product Using VADER", fontsize=16)
axes = axes.flatten()

# Plot for VADER Sentiment Analysis
for i, product in enumerate(products):
    if product in vader_sentiment_percentage.index:
        vader_sentiment_percentage.loc[product].plot(kind='bar', stacked=True, ax=axes[i], color=colors, edgecolor='black')
        axes[i].set_title(f'Sentiment Distribution for {product} (VADER)', fontsize=14)
        axes[i].set_ylabel('Percentage (%)')
        axes[i].set_ylim(0, 100)
        axes[i].legend(loc='upper left', bbox_to_anchor=(1.0, 1.0))

plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust the layout
plt.show()

# Create subplots for TextBlob sentiment analysis
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
fig.suptitle("Sentiment Distribution for Each Product Using TextBlob", fontsize=16)
axes = axes.flatten()

# Plot for TextBlob Sentiment Analysis
for i, product in enumerate(products):
    if product in textblob_sentiment_percentage.index:
        textblob_sentiment_percentage.loc[product].plot(kind='bar', stacked=True, ax=axes[i], color=colors, edgecolor='black')
        axes[i].set_title(f'Sentiment Distribution for {product} (TextBlob)', fontsize=14)
        axes[i].set_ylabel('Percentage (%)')
        axes[i].set_ylim(0, 100)
        axes[i].legend(loc='upper left', bbox_to_anchor=(1.0, 1.0))

plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust the layout
plt.show()

# ---- Cell ----
import matplotlib.pyplot as plt
import pandas as pd

# Define the product you want to analyze
product_name = 'Dexcom'  # Change this to analyze a different product

# Step 1: Explode the 'Product Mentions' to handle multiple products in a single post
df_exploded = df_cleaned.explode('Product Mentions')

# Step 2: Filter the dataframe for the selected product
df_product = df_exploded[df_exploded['Product Mentions'] == product_name]

# Step 3: Calculate sentiment counts for each sentiment column (Standardized, VADER, and TextBlob)
sentiment_counts_standard = df_product['Standardized Sentiment'].value_counts()
sentiment_counts_vader = df_product['VADER Sentiment'].value_counts()
sentiment_counts_textblob = df_product['TextBlob Sentiment'].value_counts()

# Step 4: Combine the counts into a single DataFrame for side-by-side comparison
comparison_df = pd.DataFrame({
    'Standardized Sentiment': sentiment_counts_standard,
    'VADER Sentiment': sentiment_counts_vader,
    'TextBlob Sentiment': sentiment_counts_textblob
}).fillna(0)  # Fill missing values with 0

# Step 5: Plot the comparison as a grouped bar chart
comparison_df.plot(kind='bar', figsize=(12, 6), width=0.8, color=['#99ccff', '#ff9999', '#66b3ff'])

# Step 6: Customize the chart
plt.title(f'Side-by-Side Sentiment Comparison for {product_name}', fontsize=16)
plt.xlabel('Sentiment Categories', fontsize=14)
plt.ylabel('Count of Sentiments', fontsize=14)
plt.xticks(rotation=0)
plt.legend(title='Sentiment Source', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# ---- Cell ----
import matplotlib.pyplot as plt
import pandas as pd

# Step 1: Define the products to analyze
products = ['Dexcom', 'FreeStyle Libre', 'Medtronic', 'Senseonics']

# Step 2: Explode the 'Product Mentions' to handle multiple products in a single post
df_exploded = df_cleaned.explode('Product Mentions')

# Step 3: Create a 2x2 subplot structure for the four products
fig, axes = plt.subplots(2, 2, figsize=(16, 12))  # Create a 2x2 grid of plots
fig.suptitle("Side-by-Side Sentiment Comparison for All Products", fontsize=18)
axes = axes.flatten()  # Flatten axes to iterate over them easily

# Step 4: Loop through each product and create the comparison plots
for i, product in enumerate(products):
    # Filter the data for the current product
    df_product = df_exploded[df_exploded['Product Mentions'] == product]

    # Calculate sentiment counts for Standardized, VADER, and TextBlob sentiments
    sentiment_counts_standard = df_product['Standardized Sentiment'].value_counts()
    sentiment_counts_vader = df_product['VADER Sentiment'].value_counts()
    sentiment_counts_textblob = df_product['TextBlob Sentiment'].value_counts()

    # Combine the counts into a single DataFrame for side-by-side comparison
    comparison_df = pd.DataFrame({
        'Standardized Sentiment': sentiment_counts_standard,
        'VADER Sentiment': sentiment_counts_vader,
        'TextBlob Sentiment': sentiment_counts_textblob
    }).fillna(0)  # Fill missing values with 0

    # Plot the comparison as a grouped bar chart on the respective subplot
    comparison_df.plot(kind='bar', ax=axes[i], width=0.8, color=['#99ccff', '#ff9999', '#66b3ff'], edgecolor='black')

    # Customize each subplot
    axes[i].set_title(f'Sentiment Comparison for {product}', fontsize=14)
    axes[i].set_xlabel('Sentiment Categories', fontsize=12)
    axes[i].set_ylabel('Count of Sentiments', fontsize=12)
    axes[i].legend(title='Sentiment Source', fontsize=10)
    axes[i].grid(axis='y', linestyle='--', alpha=0.7)
    axes[i].set_ylim(0, max(comparison_df.max()) + 10)  # Adjust y-limit based on the highest count

# Step 5: Adjust layout and show the plot
plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to fit the suptitle
plt.show()

# ---- Cell ----
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Step 1: Initialize VADER sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# Step 2: Define a function to classify sentiment using VADER
def get_vader_sentiment(text):
    score = analyzer.polarity_scores(text)
    if score['compound'] >= 0.05:
        return 'Positive'
    elif score['compound'] <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'

# Step 3: Check if 'VADER Sentiment' column exists, if not create it
if 'VADER Sentiment' not in df_combined.columns:
    # Apply VADER sentiment analysis to the 'Combined Text' column
    df_combined['VADER Sentiment'] = df_combined['Combined Text'].apply(get_vader_sentiment)

# Step 4: Verify that the 'VADER Sentiment' column is created successfully
print("Columns in the DataFrame:", df_combined.columns)

# Define the product list and sentiment labels for VADER Sentiment
product_list = ['Dexcom', 'FreeStyle Libre', 'Medtronic', 'Senseonics']
vader_sentiment_labels = ['Positive', 'Negative']  # Use VADER sentiment labels

# Create a 2x4 subplot structure: 2 rows (sentiments), 4 columns (products)
fig, axes = plt.subplots(2, 4, figsize=(20, 10))  # 2 rows for sentiments, 4 columns for products
axes = axes.flatten()

# Loop through each product and sentiment category
for i, product in enumerate(product_list):
    for j, sentiment in enumerate(vader_sentiment_labels):
        # Step 5: Filter data for the selected product and VADER sentiment
        filtered_data = df_combined[(df_combined['Product Mentions'].apply(lambda x: product in x)) & (df_combined['VADER Sentiment'] == sentiment)]

        # Sum the attribute columns
        attribute_counts = filtered_data[attributes_map.keys()].sum()
        total_reviews = len(filtered_data)

        # Calculate percentage of reviews mentioning each attribute
        if total_reviews > 0:
            attribute_counts = (attribute_counts / total_reviews) * 100
        else:
            attribute_counts = attribute_counts * 0  # Set to zero if no reviews for that sentiment

        # Prepare data for plotting
        attributes, counts = attribute_counts.index, attribute_counts.values

        # Plot the bar chart on the respective subplot
        axes[i * 2 + j].bar(attributes, counts, color='#66b3ff' if sentiment == 'Positive' else '#ff9999')
        axes[i * 2 + j].set_xlabel('Attributes')
        axes[i * 2 + j].set_ylabel('Percentage (%)')
        axes[i * 2 + j].set_title(f'{sentiment} Sentiment for {product}')
        axes[i * 2 + j].tick_params(axis='x', rotation=45)

# Adjust layout and show the plot
plt.tight_layout()
plt.show()

# ---- Cell ----
# Import necessary libraries
import spacy
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Initialize spaCy and VADER
nlp = spacy.load("en_core_web_sm")
analyzer = SentimentIntensityAnalyzer()

# Define a function to perform aspect-based sentiment analysis
def aspect_based_sentiment(text):
    positive_aspects = []
    negative_aspects = []

    # Parse the text using spaCy
    doc = nlp(text)

    # Perform dependency parsing to identify aspects and associated sentiments
    for token in doc:
        # Check for nouns (potential aspects)
        if token.pos_ in ['NOUN', 'PROPN']:
            # Get the sentiment of the associated adjective or verb (if any)
            sentiment_score = None

            # Check the left and right children for adjectives describing the noun
            for child in token.children:
                if child.dep_ in ['amod', 'acomp'] and child.pos_ == 'ADJ':
                    sentiment_score = analyzer.polarity_scores(child.text)['compound']
                elif child.dep_ in ['advmod'] and child.pos_ == 'ADV':
                    sentiment_score = analyzer.polarity_scores(child.text)['compound']

            # Check the noun's head for verbs indicating sentiment
            if token.head.pos_ in ['VERB'] and token.dep_ in ['nsubj']:
                sentiment_score = analyzer.polarity_scores(token.head.text)['compound']

            # Categorize based on the sentiment score
            if sentiment_score is not None:
                if sentiment_score >= 0.05:
                    positive_aspects.append(token.text)
                elif sentiment_score <= -0.05:
                    negative_aspects.append(token.text)

    return positive_aspects, negative_aspects

# Applying the aspect-based sentiment analysis to each post in the dataframe
df_cleaned['Positive Objects'], df_cleaned['Negative Objects'] = zip(*df_cleaned['Combined Text'].apply(aspect_based_sentiment))
print(df_cleaned[['Combined Text', 'Positive Objects', 'Negative Objects']].head())

# ---- Cell ----
# Load the spaCy model
nlp = spacy.load("en_core_web_sm")
analyzer = SentimentIntensityAnalyzer()

# Define a function to get VADER sentiment
def get_vader_sentiment(text):
    score = analyzer.polarity_scores(text)
    if score['compound'] >= 0.05:
        return 'Positive'
    elif score['compound'] <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'

# Separate original posts and replies/comments
original_posts = df_cleaned[df_cleaned['Post Type'] == 'Original']
replies_comments = df_cleaned[df_cleaned['Post Type'].isin(['Replies and Comments'])]

# Create a mapping from URLs to Original Post Text
post_context_mapping = pd.Series(original_posts['Combined Text'].values, index=original_posts['URL']).to_dict()

# Map original post content to each reply/comment
replies_comments['Original Post Context'] = replies_comments['URL'].map(post_context_mapping)

# Merge the context for comments/replies
replies_comments['Full Context'] = replies_comments['Original Post Context'].fillna('') + " " + replies_comments['Combined Text']

# Perform aspect-based sentiment analysis using combined context for replies
replies_comments['Contextual Sentiment'] = replies_comments['Full Context'].apply(get_vader_sentiment)

# Compare with the original sentiment column to see how it changes
comparison_df = replies_comments[['Combined Text', 'Original Post Context', 'Full Context', 'Sentiment', 'Contextual Sentiment']]
print(comparison_df.head())

# ---- Cell ----
import pandas as pd
pd.set_option('display.max_columns', None)
import seaborn as sns
import matplotlib.pyplot as plt

# ---- Cell ----
raw = df_cleaned.copy()

# ---- Cell ----
bold_start = "\033[1m"
bold_end = "\033[0m"

# Step 1: Separate the data into Female, Male, and Unknown groups
female_df = raw[raw['Author Gender'] == 'Female']
male_df = raw[raw['Author Gender'] == 'Male']
unknown_df = raw[raw['Author Gender'] == 'Unknown']

# Step 2: Define a function to calculate sentiment ratio for each group
def calc_sentiment_ratio(df):
    total = len(df)
    positive_ratio = len(df[df['Sentiment'] == 'Positives']) / total * 100
    neutral_ratio = len(df[df['Sentiment'] == 'Neutrals']) / total * 100
    mixed_ratio  = len(df[df['Sentiment'] == 'Mixed']) / total * 100
    negative_ratio = len(df[df['Sentiment'] == 'Negatives']) / total * 100
    return pd.Series({'Positives': positive_ratio, 'Neutrals': neutral_ratio,'Mixed':mixed_ratio, 'Negatives': negative_ratio})

# Calculate sentiment ratios for Female, Male, and Unknown
female_sentiment_ratio = calc_sentiment_ratio(female_df)
male_sentiment_ratio = calc_sentiment_ratio(male_df)
unknown_sentiment_ratio = calc_sentiment_ratio(unknown_df)

#Count the number of Female, Male, and Unknown
gender_counts = raw['Author Gender'].value_counts()

#Calculate the total number of entries
total_entries = len(raw)

#Create a DataFrame to store the counts and percentages
gender_summary = pd.DataFrame({
    'Count': gender_counts,
    'Percentage': (gender_counts / total_entries) * 100
})


#Pretty print the summary table with formatting
print(f"{bold_start}\nSummary Table:{bold_end}")
print(gender_summary.to_string(formatters={'Count': '{:,.0f}'.format, 'Percentage': '{:.1f}%'.format}))


# Display the results for sentiment ratio

print(f"{bold_start}\nFemale Sentiment Ratio:{bold_end}")
print(female_sentiment_ratio)

print(f"{bold_start}\nMale Sentiment Ratio:{bold_end}")
print(male_sentiment_ratio)

print(f"{bold_start}\nUnknown Sentiment Ratio:{bold_end}")
print(unknown_sentiment_ratio)

# ---- Cell ----
top_10_source_names = raw['Source Name'].value_counts().nlargest(10)

# Show the result
print(top_10_source_names)

# ---- Cell ----
import pandas as pd

# Assuming raw is your original DataFrame and it contains a 'Sentiment' column

# Define a function to label the source as T1 or T2 based on Source Name
def label_type(row):
    # Check if 'Source Name' is a string before applying .lower()
    if isinstance(row['Source Name'], str):
        if 't1' in row['Source Name'].lower() or 'type1' in row['Source Name'].lower():
            return 'T1'
        elif 't2' in row['Source Name'].lower() or 'type2' in row['Source Name'].lower():
            return 'T2'
        else:
            return 'Unknown'
    else:
        return 'Unknown'  # Return 'Unknown' if 'Source Name' is not a string

# Step 1: Apply the function to the entire raw DataFrame to add Type_Label
raw['Type_Label'] = raw.apply(label_type, axis=1)

# Step 2: Divide the dataset into T1, T2 and others.
raw_T1 = raw[raw['Type_Label'] == 'T1']
raw_T2 = raw[raw['Type_Label'] == 'T2']
raw_unknown = raw[raw['Type_Label'] == 'Unknown']

# Step 3: Calculate the ratio of each sentiment.
T1_ratio = calc_sentiment_ratio(raw_T1)
T2_ratio = calc_sentiment_ratio(raw_T2)
unknown_ratio = calc_sentiment_ratio(raw_unknown)

#Count the number of Female, Male, and Unknown
Type_counts = raw['Type_Label'].value_counts()

#Calculate the total number of entries
total_entries = len(raw)

#Create a DataFrame to store the counts and percentages
Type_summary = pd.DataFrame({
    'Count': Type_counts,
    'Percentage': (Type_counts / total_entries) * 100
})



#Pretty print the summary table with formatting
print(f"{bold_start}\nSummary Table:{bold_end}")
print(Type_summary.to_string(formatters={'Count': '{:,.0f}'.format, 'Percentage': '{:.1f}%'.format}))
# Display the sentiment ratio for T1 and T2

# Display the results
print(f"{bold_start}\nType 1 Sentiment Ratio:{bold_end}")
print(T1_ratio)

print(f"{bold_start}\nType 2 Sentiment Ratio:{bold_end}")
print(T2_ratio)

print(f"{bold_start}\nUnknown Sentiment Ratio:{bold_end}")
print(unknown_ratio)

# ---- Cell ----
# Get unique values from the 'Professions' column in the raw DataFrame
unique_professions = raw['Professions'].unique()

# Display the list of unique professions
print(unique_professions)

# ---- Cell ----
# Step 1: Get unique values from the 'Professions' column
unique_professions = raw['Professions'].dropna().unique()

# Step 2: Define a function to calculate sentiment ratio for each group (already defined)
# def calc_sentiment_ratio(df):
#     total = len(df)
#     positive_ratio = len(df[df['Sentiment'] == 'Positives']) / total * 100
#     neutral_ratio = len(df[df['Sentiment'] == 'Neutrals']) / total * 100
#     mixed_ratio  = len(df[df['Sentiment'] == 'Mixed']) / total * 100
#     negative_ratio = len(df[df['Sentiment'] == 'Negatives']) / total * 100
#     return pd.Series({'Positives': positive_ratio, 'Neutrals': neutral_ratio, 'Mixed': mixed_ratio, 'Negatives': negative_ratio})

# Step 3: Loop through each unique profession and calculate statistics
profession_summary = {}

for profession in unique_professions:
    # Filter rows for each profession
    profession_df = raw[raw['Professions'] == profession]

    # Calculate the sentiment ratio for this profession
    sentiment_ratio = calc_sentiment_ratio(profession_df)

    # Store the sentiment ratio in a dictionary
    profession_summary[profession] = sentiment_ratio

# Convert the dictionary to a DataFrame for easy viewing
profession_sentiment_df = pd.DataFrame(profession_summary).T

# Pretty print the summary table for profession sentiment ratio
print(f"{bold_start}\nProfession Sentiment Ratios:{bold_end}")
print(profession_sentiment_df)

# Step 4: Count the number of entries per profession
profession_counts = raw['Professions'].value_counts()

# Step 5: Calculate the total number of entries
total_entries = len(raw)

# Step 6: Create a DataFrame to store the counts and percentages for professions
profession_summary_df = pd.DataFrame({
    'Count': profession_counts,
    'Percentage': (profession_counts / total_entries) * 100
})

# Pretty print the summary table for profession count and percentage
print(f"{bold_start}\nSummary Table for Professions:{bold_end}")
print(profession_summary_df.to_string(formatters={'Count': '{:,.0f}'.format, 'Percentage': '{:.1f}%'.format}))

# Step 7: Filter rows where 'Professions' or 'Interests' contains 'health' or 'Health'
filtered_df_health = raw[(raw['Professions'].str.contains('health', case=False, na=False)) |
                         (raw['Interests'].str.contains('health', case=False, na=False))]

# Step 8: Group by 'Sentiment' and calculate the ratio for health-related professions
health_sentiment_ratio = calc_sentiment_ratio(filtered_df_health)

# Display the sentiment ratio for health-related professions
print(f"{bold_start}\nHealth-Related Profession Sentiment Ratio:{bold_end}")
print(health_sentiment_ratio)

# ---- Cell ----
!pip install textstat
import pandas as pd
import nltk
from textstat import flesch_reading_ease, gunning_fog
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Assuming 'raw' is your original DataFrame and 'Combined Text' is the relevant column
# Step 1: Tokenize the text and calculate readability scores
def calculate_readability(text):
    # Calculate different readability scores
    flesch_score = flesch_reading_ease(text)
    fog_score = gunning_fog(text)
    return pd.Series([flesch_score, fog_score])

# Apply the readability functions to each Combined Text
raw[['Flesch Reading Ease', 'Gunning Fog Index']] = raw['Combined Text'].apply(calculate_readability)

# Step 2: Feature extraction for vocabulary richness (optional)
# Example: Calculate average word length, total words, etc.
def vocab_richness(text):
    words = nltk.word_tokenize(text)
    avg_word_len = sum(len(word) for word in words) / len(words) if words else 0
    return avg_word_len

# Apply vocabulary richness calculation
raw['Avg Word Length'] = raw['Combined Text'].apply(vocab_richness)

# Step 3: Define a labeling function manually (Assuming you have labeled data for supervised learning)
# If no labels are available, this step can be used for manual labeling of training data
def label_adult_child(row):
    # Logic for manual labeling or heuristic-based labeling
    # For example, you might use heuristic rules to classify based on readability and vocab richness
    if row['Flesch Reading Ease'] < 60 or row['Avg Word Length'] > 4.5:
        return 'Adult'
    else:
        return 'Child'

# Apply labeling function
raw['Age_Label'] = raw.apply(label_adult_child, axis=1)

# Step 4: Train-test split
X = raw[['Flesch Reading Ease', 'Gunning Fog Index', 'Avg Word Length']]  # Feature matrix
y = raw['Age_Label']  # Target labels

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Train a classifier (e.g., Logistic Regression)
clf = LogisticRegression()
clf.fit(X_train, y_train)

# Step 6: Predict and evaluate
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))

# ---- Cell ----
# Count occurrences of each label
label_counts = raw['Age_Label'].value_counts()

#Calculate the total number of entries
total_entries = len(raw)

#Create a DataFrame to store the counts and percentages
Age_summary = pd.DataFrame({
    'Count': label_counts,
    'Percentage': (label_counts / total_entries) * 100
})



#Pretty print the summary table with formatting
print(f"{bold_start}\nSummary Table:{bold_end}")
print(Age_summary.to_string(formatters={'Count': '{:,.0f}'.format, 'Percentage': '{:.1f}%'.format}))

# Step 1: Separate the data into Adult and Child groups
adult_df = raw[raw['Age_Label'] == 'Adult']
child_df = raw[raw['Age_Label'] == 'Child']


# Calculate sentiment ratios for Adult and Child
adult_sentiment_ratio = calc_sentiment_ratio(adult_df)
child_sentiment_ratio = calc_sentiment_ratio(child_df)

# Display the results
print("\nAdult Sentiment Ratio:")
print(adult_sentiment_ratio)

print("\nChild Sentiment Ratio:")
print(child_sentiment_ratio)

# ---- Cell ----
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score

# Step 1: Extract relevant features from the DataFrame
# Make sure to have the relevant columns
data = raw[['Author Gender', 'Type_Label', 'Professions', 'Age_Label', 'Sentiment']]

# Manual encoding for Author Gender
data['Author Gender'] = data['Author Gender'].map({'Male': 2, 'Female': 1, 'Unknown': 0})

# Manual encoding for Type Label
data['Type_Label'] = data['Type_Label'].map({'T1': 1, 'T2': 2, 'Unknown': 0})

# Manual encoding for Professions (assuming Health is part of this column)
data['Professions'] = data['Professions'].apply(lambda x: 1 if pd.notna(x) and 'Health' in x else 0)

# Manual encoding for Age Label
data['Age_Label'] = data['Age_Label'].map({'Adult': 1, 'Child': 0})

# Sentiment encoding (already in 1, 0, -1 format)
data['Sentiment'] = data['Sentiment'].map({'Positives': 1, 'Neutrals': 0,'Negatives': -1, 'Mixed': 0})


# Step 2: Convert categorical data into numeric labels using LabelEncoder
label_encoders = {}
for column in ['Author Gender', 'Type_Label', 'Professions', 'Age_Label']:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column].astype(str))  # Convert to string in case of NaNs
    label_encoders[column] = le  # Save the encoder for potential inverse_transform use

# Step 3: Standardize the data to have mean 0 and variance 1
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data[['Author Gender', 'Type_Label', 'Professions', 'Age_Label']])

# ---- Cell ----
pca = PCA(n_components=2)  # Reduce to 2 dimensions for visualization
pca_result = pca.fit_transform(scaled_data)

# ---- Cell ----
# Check explained variance ratio for each component
explained_variance_ratio = pca.explained_variance_ratio_
explained_variance_ratio

# ---- Cell ----
#Step 4Detemine the number of clusters.
wcss = []

for i in range(1,20):
    kmeans_pca = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans_pca.fit(pca_result)
    wcss.append(kmeans_pca.inertia_)


plt.figure(figsize=(12,6))
plt.plot(range(1,20), wcss, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')

# ---- Cell ----
# Step 4: Apply K-Means with the optimal number of clusters (K) based on the previous plot
# Let's assume from the Elbow Method that K=3 is optimal
optimal_k = 4  # You can change this based on the previous plot results
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
clusters = kmeans.fit_predict(scaled_data)

# Step 6: Add cluster labels back to the original DataFrame
data['Cluster'] = clusters

# Step 7: Visualize the clusters with PCA (for 2D visualization)
pca = PCA(n_components=2)
pca_result = pca.fit_transform(scaled_data)

plt.figure(figsize=(10, 6))
plt.scatter(pca_result[:, 0], pca_result[:, 1], c=data['Cluster'], cmap='viridis', alpha=0.6)
plt.colorbar(label='Cluster')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title('K-Means Clustering Visualization with PCA')
plt.show()

# Step 8: Group by Cluster and Sentiment to analyze the sentiment distribution in each cluster
sentiment_distribution = data.groupby(['Cluster', 'Sentiment']).size().unstack().fillna(0)
print(sentiment_distribution)

# Step 9: Plot the sentiment distribution within each cluster
sentiment_distribution.plot(kind='bar', stacked=True, figsize=(10, 6))
plt.title('Sentiment Distribution per Cluster')
plt.xlabel('Cluster')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.show()

# ---- Cell ----
!pip install openai
import os
from openai import OpenAI
import openai

# ---- Cell ----
client = OpenAI(
    api_key="",
)

# ---- Cell ----
# function for calling the model and the prompt

def get_completion(prompt, model="gpt-3.5-turbo"):
    messages = [{"role": "user", "content": prompt}]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,
    )
    return(response.choices[0].message.content)

# ---- Cell ----
import openai
import pandas as pd


# Define a function to process the OpenAI API request for each row
def analyze_sentiment_and_extract(dataframe):
    prompt_template = """Please extract the number of customers with gender and age (adult or child) and the summary of the result with the distribution of positives, negatives, neutrals, mixed in each segmentation for the following:

    Sound Bite Text: {sound_bite_text}
    Title: {title}
    Sentiment: {sentiment}
    """

    # Iterate through the dataframe rows and create API requests
    for index, row in dataframe.iterrows():
        prompt = prompt_template.format(
            sound_bite_text=row['Sound Bite Text'],
            title=row['Title'],
            sentiment=row['Sentiment']
        )

        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            max_tokens=150
        )

        # Print the response for each entry
        print(f"Response for row {index}:\n", response['choices'][0]['text'], "\n")

# Use the function with the 'raw' dataframe
analyze_sentiment_and_extract(raw)

