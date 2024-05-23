import pandas as pd
import spacy
import nltk
from nltk.stem import WordNetLemmatizer
from itertools import chain
from nltk.corpus import stopwords
from bertopic import BERTopic
from umap import UMAP
from hdbscan import HDBSCAN
from gensim import corpora
from gensim.models.coherencemodel import CoherenceModel

# Read Book of Isaiah From Archive
print("Reading Data...")
data = pd.read_csv('Conflict Scenarios Research.csv')
# create documents list
documents = data["Describe a past experience you've had that involved conflict with a family member, friend, or significant other. Be as detailed as you like."].tolist()
print("Number of Documents: " + str(len(documents)))

print("Initializing main class...")

class info:
    def __init__(self, documents, lowercasing=False, lemmatization=False):
        self.lowercase = lowercasing
        self.lemmatization = lemmatization
        
        # Load the English language model
        self.nlp = spacy.load("en_core_web_sm")

        # Initialize the WordNet lemmatizer
        if lemmatization:
            self.lemmatizer = WordNetLemmatizer()

        # declare stopwords for global usage
        self.others = ["wa", "art", "one", "nt", "lot", "-", ".", ",", "?", "!", "'s", "n't", "'re", "'m", "'ve"]

        dict = []
        self.tokenized_docs = []
        for item in documents:
            doc = self.nlp(item)
            tokens = [token.text for token in doc] # process token into list
            if lowercasing:
                lowercase_list = [word.lower() for word in tokens] # lowercase all words to improve search
                tokens = lowercase_list
            if lemmatization:
                lemmatized_words = [self.lemmatizer.lemmatize(word) for word in tokens] # lemmatize words to enhace retrieval comprehension
                tokens = lemmatized_words
            self.tokenized_docs.append(tokens) 
            temp = set(tokens) # reduce overhead by performing initial set
            for term in temp:
                dict.append(term)

        dictionary = set(dict)

        self.inverted_index = []
        for term in dictionary:
            temp = [term]
            for doc in self.tokenized_docs:
                for word in doc:
                    if term == word:
                        temp.append(self.tokenized_docs.index(doc))
                        break  # break operation to prevent duplicate postings
            self.inverted_index.append(temp)

    def query(self, phrase):
        # Tokenize input
        tokenize = self.nlp(phrase)
        tokens = [token.text for token in tokenize] # process token into list
        if self.lowercase:
            lowercased = [word.lower() for word in tokens] # lowercase all words to improve search
            tokens = lowercased
        if self.lemmatization:
            lemma = [self.lemmatizer.lemmatize(word) for word in tokens] # final lemmatization of query to match postings
            tokens = lemma
            
    
        # retrieve postings for each token
        retrieve = set(tokens)
        postings = []
        for word in retrieve:
            for post in self.inverted_index:
                if word == post[0]:
                    postings.append(post)
    
        # check for intersection
        combine = list(chain.from_iterable(postings))
        exact_matches = []
        exact_count = len(retrieve)
        for item in combine:
            n = combine.count(item)
            if n == exact_count:
                exact_matches.append(item)
                
        if len(exact_matches) > 0:
            matches = set(exact_matches)
        else:
            matches = set(combine)
            
        # retrieve relevant documents
        results = []
        for item in matches:
            if isinstance(item, int):
                content = documents[item]
                results.append(content)
                   
        return results

    def text_cleaning(self):
        stop_words = list(set(stopwords.words('english')))
        
        for item in self.others:
            stop_words.append(item)
        filter = []
        for token_text in self.tokenized_docs:
            output = ""
            for word in token_text:
                if word not in stop_words:
                    output = output + " " + word
            filter.append(output)
            
        return filter

    def get_tokenized_corpus(self):
        stop_words = list(set(stopwords.words('english')))
        
        for item in self.others:
            stop_words.append(item)
        filter = []
        for token_text in self.tokenized_docs:
            temp = []
            for word in token_text:
                if word not in stop_words:
                    temp.append(word)
            filter.append(temp)
            
        return filter

# generate search
print("Generating Search...")
information = info(documents, lowercasing=True, lemmatization=True)


# get cleaned corpus for training with BERTopic
print("Cleaning Text...")
get_text = information.text_cleaning()


# Set up UMAP with a fixed random state
print("Setting up BERTopic model...")
umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine', random_state=42)

#Set up HDBSCAN
hdbscan_model = HDBSCAN(min_cluster_size=10, metric='euclidean', cluster_selection_method='eom')


#------------------------------------------------------------------------------------------------------------
# Calculate Coherence - GRID SEARCH
topics_per_cluster = range(2, 25, 1)

# get tokenized corpus for coherence calculation
print("Fetching tokenized corpus for Gensim calculation...")
tokenized_corpus = information.get_tokenized_corpus()

# create dictionary of unique words from the tokenized corpus
print("Creating dictionary for Gensim calculation...")
dict_ = corpora.Dictionary(tokenized_corpus)

# The doc_term_matrix contain tuple entries with the token_id for each word in the corpus, along with its frequency of
# occurence.
print("Creating document term matrix for Gensim calculation...")
doc_term_matrix = [dict_.doc2bow(i) for i in tokenized_corpus]

best_cm = -1
high_score = []
scores = []
scoresheet = open("scores.txt", 'w', encoding='utf-8')

# Generate BERTopic Model
for tpc in topics_per_cluster:
    print("Calculating Scores (topics per cluster = " + str(tpc) + ")")
    topic_model = BERTopic(top_n_words=tpc, min_topic_size=30, umap_model=umap_model, hdbscan_model=hdbscan_model)
    topics, probs = topic_model.fit_transform(get_text)

    # Get topics as a dictionary
    topic_dict = topic_model.get_topics()
    topic_list = list(topic_dict.values())
    # Convert topics dictionary to a list of lists
    raw_topics = []
    for item in topic_list:
        temp = []
        for topics in item:
            temp.append(topics[0])
        raw_topics.append(temp)

    raw_topics.pop(0) # remove low prob words

    #calculate coherence and obtain score
    cm = CoherenceModel(topics=raw_topics, texts=tokenized_corpus, corpus=doc_term_matrix, dictionary=dict_, coherence='c_npmi')
    coherence = cm.get_coherence()
    temp = str(coherence) + "," + str(tpc) + "\n"

    scoresheet.write(temp)

    if coherence > best_cm:
        best_cm = coherence
        high_score = [best_cm, tpc]

scoresheet.close()
print("High Score:")
print(high_score)