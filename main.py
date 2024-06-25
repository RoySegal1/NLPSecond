import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from bs4 import BeautifulSoup
import requests
from collections import Counter
# from gensim.models import Word2Vec
# Function to print word statistics
def print_word_statistics(words, title):
    word_counts = Counter(words)
    total_words = len(words)
    unique_words = len(word_counts)
    most_common_words = word_counts.most_common(5)

    print(f"{title} Statistics:")
    print(f"Total words: {total_words}")
    print(f"Unique words: {unique_words}")
    print(f"Most common words: {most_common_words}")
    print("\n")

# Function to tokenize text
def tokenize(text):
    return word_tokenize(text)

# Function to lemmatize tokens
def lemmatize(tokens):
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(token) for token in tokens]

# Function to remove stopwords and punctuation, and convert to lowercase
def preprocess_text(tokens):
    stopwords = set(nltk.corpus.stopwords.words('english'))
    return [token.lower() for token in tokens if token.lower() not in stopwords and token.isalpha()]

# URLs of Wikipedia pages
url_turing = 'https://en.wikipedia.org/wiki/Alan_Turing'
url_einstein = 'https://en.wikipedia.org/wiki/Albert_Einstein'

# Function to scrape text from Wikipedia pages
def scrape_wikipedia(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    paragraphs = soup.find_all('p')
    data = [paragraph.text for paragraph in paragraphs]
    return ' '.join(data)

# Scrape text from Wikipedia pages
data_turing = scrape_wikipedia(url_turing)
data_einstein = scrape_wikipedia(url_einstein)

# Tokenize, preprocess, and lemmatize text
tokens_turing = tokenize(data_turing)
tokens_einstein = tokenize(data_einstein)

filtered_tokens_turing = preprocess_text(tokens_turing)
filtered_tokens_einstein = preprocess_text(tokens_einstein)

lemmas_turing = lemmatize(filtered_tokens_turing)
lemmas_einstein = lemmatize(filtered_tokens_einstein)

# Calculate Bag-of-Words (FreqDist) for Alan Turing
bow_turing = Counter(lemmas_turing)
print("Bag-of-Words (FreqDist) for Alan Turing:\n", bow_turing)
print("\n")

# Calculate Bag-of-Words (FreqDist) for Albert Einstein
bow_einstein = Counter(lemmas_einstein)
print("Bag-of-Words (FreqDist) for Albert Einstein:\n", bow_einstein)
print("\n")

# TF-IDF Vectorization
# Combine the preprocessed tokens into a list of documents
tokenized_documents = [' '.join(filtered_tokens_turing), ' '.join(filtered_tokens_einstein)]

# Initialize the TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer()

# Fit and transform the documents into TF-IDF matrix
tfidf_matrix = tfidf_vectorizer.fit_transform(tokenized_documents)

# Print TF-IDF matrix
print("TF-IDF Matrix:\n", tfidf_matrix.toarray())

# Retrieve feature names (terms)
feature_names = tfidf_vectorizer.get_feature_names_out()


# model = Word2Vec(
#     sentences=filtered_tokens_einstein,      # The corpus to train the model on
#     vector_size=500,       # The size of the word vectors to be learned
#     window=5,              # The size of the window of words to be considered
#     min_count=5,           # The minimum frequency required for a word to be included in the vocabulary
#     sg=0,                  # 0 for CBOW, 1 for skip-gram
#     negative=5,            # The number of negative samples to use for negative sampling
#     ns_exponent=0.75,      # The exponent used to shape the negative sampling distribution
#     alpha=0.03,            # The initial learning rate
#     min_alpha=0.0007,      # The minimum learning rate to which the learning rate will be linearly reduced
#     epochs=30,             # The number of epochs (iterations) over the corpus
#     workers=4,             # The number of worker threads to use for training the model
#     seed=42,               # The seed for the random number generator
#     max_vocab_size=None    # The maximum vocabulary size (None means no limit)
# )
#
# # Get the vector representation of a word
# vector = model.wv['albert']
#
# # Find the most similar words to a given word
# similar_words = model.wv.most_similar('man')
#
# # Print the vector and similar words
# print("Vector for 'man':", vector)
# print("Most similar words to 'man':", similar_words)
# # Print feature names and corresponding TF-IDF scores
# for doc_idx, doc in enumerate(tokenized_documents):
#     print(f"Document {doc_idx + 1}:")
#     for term_idx, term in enumerate(feature_names):
#         tfidf_score = tfidf_matrix[doc_idx, term_idx]
#         if tfidf_score > 0:
#             print(f"{term}: {tfidf_score}")
#     print("\n")
#
# # # Normalize TF-IDF matrix (optional)
# # from sklearn.preprocessing import normalize
# # tfidf_matrix_normalized = normalize(tfidf_matrix, norm='l2')

