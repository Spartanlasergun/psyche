import pandas as pd
import spacy
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from bertopic import BERTopic
from umap import UMAP
from hdbscan import HDBSCAN
from gensim import corpora
from gensim.models.coherencemodel import CoherenceModel
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from bertopic.vectorizers import ClassTfidfTransformer

# preprocessing function
def preprocessing(documents, stop):
    # Load the English language model
    nlp = spacy.load("en_core_web_sm")

    lemmatizer = WordNetLemmatizer()

    tokenized_docs = []
    for item in documents:
        doc = nlp(item)
        tokens = [token.text for token in doc] # process token into list
        lowercase_list = [word.lower() for word in tokens] # lowercase all words
        # lemmatize words to prevent overlap between extremely similar words (e.g. "friend" and "friends")
        lemmatized_words = [lemmatizer.lemmatize(word) for word in lowercase_list]
        tokenized_docs.append(lemmatized_words) 

    stop_words = list(set(stopwords.words('english')))

    for item in stop:
        stop_words.append(item)
    
    processed = []
    corpus_tokens = []
    for token_text in tokenized_docs:
        output = ""
        temp = []
        for word in token_text:
            if word not in stop_words:
                output = output + " " + word
                temp.append(word)
        processed.append(output)
        corpus_tokens.append(temp)
            
    return processed, corpus_tokens

# Read Data From Archive
print("Reading Data...")
data = pd.read_csv('Conflict Scenarios Research.csv')
# create documents list
documents = data["Contrary to the previous question, describe a past experience you've had that did not involve conflict with a family member, friend or significant other. Be as detailed as you like."].tolist()

print("Number of Documents: " + str(len(documents)))

stop = ["wa", "art", "one", "nt", "lot", "-", ".", ",", "?", "!", "'s", "n't", "'re", "'m", "'ve", " ",
        "get", "conflict", "really", "went", "time", "dunno", "yeah"]

get_text, tokenized_corpus = preprocessing(documents, stop)


#----------------------------------------------------------------------------------------------
print("Initializing Model...")
# set transformer model for use with BERTopic
sentence_model = SentenceTransformer("all-mpnet-base-v2")

# Setup CountVectorizer
vectorizer_model = CountVectorizer(ngram_range=(1, 3), # considers word groupings in n-gram range (in this case, 1 to 3)
                                   stop_words="english") # additional stop word removal

# Setup c-TF-IDF model
ctfidf_model = ClassTfidfTransformer(bm25_weighting=True, # weighting that works better with small datasets
                                     reduce_frequent_words=True)

# create dictionary of unique words from the tokenized corpus
dict_ = corpora.Dictionary(tokenized_corpus)

# The doc_term_matrix contain tuple entries with the token_id for each word in the corpus, along with its frequency of
# occurence.
doc_term_matrix = [dict_.doc2bow(i) for i in tokenized_corpus]


# Set up UMAP with a fixed random state
umap_model = UMAP(n_neighbors=5, 
                  n_components=4, 
                  min_dist=0.0, 
                  metric='cosine', 
                  random_state=42)

# Setup clustering algorithm
hdbscan_model = HDBSCAN(min_cluster_size=10, 
                        metric='euclidean', 
                        cluster_selection_method='eom',
                        prediction_data=True)

#cluster_model = KMeans(n_clusters=cs) # K-means model is typically less effective that HDBSCAN

# Initialize BERTopic model
topic_model = BERTopic(top_n_words=2, 
                       min_topic_size=30, # note: min_topic_size is not used when the HDBSCAN algorithm is specified
                       umap_model=umap_model, 
                       hdbscan_model=hdbscan_model,
                       vectorizer_model=vectorizer_model,
                       ctfidf_model=ctfidf_model,
                       embedding_model=sentence_model)

# Generate Topics
print("Generating Topics...")
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

print(raw_topics)