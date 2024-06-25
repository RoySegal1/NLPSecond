import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from bs4 import BeautifulSoup
import requests
from collections import Counter
from nltk import CFG
from nltk.parse.chart import ChartParser
from nltk.grammar import Nonterminal


def cyk_parse(sentence, grammar):
    words = sentence.split()
    n = len(words)
    table = [[set() for _ in range(n)] for _ in range(n)]

    # Fill the table with lexical entries
    for j in range(n):
        for prod in grammar.productions(rhs=(words[j],)):
            table[j][j].add(prod.lhs())

    # Fill the table with productions
    for length in range(2, n + 1):  # Length of the span
        for i in range(n - length + 1):  # Start of the span
            j = i + length - 1  # End of the span
            for k in range(i, j):  # Split point of the span
                for prod in grammar.productions():
                    if len(prod.rhs()) == 2:
                        B, C = prod.rhs()
                        if B in table[i][k] and C in table[k + 1][j]:
                            table[i][j].add(prod.lhs())

    return table


def print_cyk_table(table, words):
    n = len(table)
    print("CYK Parse Table:")
    print(" " * 8 + " ".join(f"{word:7}" for word in words))
    for i, row in enumerate(table):
        row_str = f"{words[i]:<7}" if i < len(words) else " " * 7
        for j, cell in enumerate(row):
            if j >= i:
                cell_str = "{" + ", ".join(str(nt) for nt in cell) + "}"
                row_str += f"{cell_str:<8}"
            else:
                row_str += " " * 8
        print(row_str)
    print("\n" + "-" * 50 + "\n")


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


# Define the grammar in CNF
grammar = CFG.fromstring("""
    S -> NP VP
    VP -> V NP | V PP | V
    NP -> Det N | N
    PP -> P NP
    V -> 'eats' | 'barks' | 'fly' | 'play' | 'rises'
    Det -> 'the' | 'a'
    N -> 'cat' | 'dog' | 'birds' | 'children' | 'park' | 'sun' | 'fish'
    P -> 'in'
    Adv -> 'loudly' | 'high'
""")

# Initialize sentences
sentences = [
    "the cat eats fish",
    "a dog barks loudly",
    "birds fly high",
    "children play in the park",
    "the sun rises"
]

# Create a parser
parser = ChartParser(grammar)

# Parse each sentence and print the parse trees
for sentence in sentences:
    print(f"Parsing sentence: '{sentence}'")
    words = sentence.split()
    cyk_table = cyk_parse(sentence, grammar)
    print_cyk_table(cyk_table, words)
    parses = list(parser.parse(words))
    if parses:
        for tree in parses:
            tree.pretty_print()
    else:
        print("No valid parse found.")
    print("\n" + "-"*50 + "\n")
