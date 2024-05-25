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

# Read Data From Archive
print("Reading Data...")
data = pd.read_csv('Conflict Scenarios Research.csv')
# create documents list
documents = data["Describe a past experience you've had that involved conflict with a family member, friend, or significant other. Be as detailed as you like."].tolist()
print("Number of Documents: " + str(len(documents)))

# preprocessing function
def preprocessing(documents):
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
    others = ["wa", "art", "one", "nt", "lot", "-", ".", ",", "?", "!", "'s", "n't", "'re", "'m", "'ve", " ",
              "get", "conflict", "really"]
    for item in others:
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


# get cleaned raw and tokenized data for training with BERTopic and coherence calculation
print("Preprocessing Text...")
get_text, tokenized_corpus = preprocessing(documents)


#------------------------------------------------------------------------------------------------------------
# Calculate Coherence - GRID SEARCH
topics_per_cluster = range(2, 6, 1)
cluster_size = range(10, 31, 1)

# Set up UMAP with a fixed random state
print("Setting up BERTopic model...")
umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine', random_state=42)

# set transformer model for use with BERTopic
sentence_model = SentenceTransformer("all-mpnet-base-v2")
# Pre-calculate embeddings
print("    Pre-Caculating Embeddings...")
embeddings = sentence_model.encode(get_text, show_progress_bar=True)

# create dictionary of unique words from the tokenized corpus
print("    Creating dictionary for Gensim calculation...")
dict_ = corpora.Dictionary(tokenized_corpus)

# The doc_term_matrix contain tuple entries with the token_id for each word in the corpus, along with its frequency of
# occurence.
print("    Creating document term matrix for Gensim calculation...")
doc_term_matrix = [dict_.doc2bow(i) for i in tokenized_corpus]

best_cm = -1
high_score = []
scores = []
scoresheet = open("scores.txt", 'w', encoding='utf-8')

# Conduct Grid Search
for tpc in topics_per_cluster:
    print("------------------------------------------------------------------------------------------")
    for cs in cluster_size:
        print("Calculating Scores:\ntopics per cluster = " + str(tpc) + "\n" + "min cluster size = " + str(cs))
        
        # Setup clustering algorithm
        hdbscan_model = HDBSCAN(min_cluster_size=cs, 
                                metric='euclidean', 
                                cluster_selection_method='eom',
                                prediction_data=True)

        # Setup CountVectorizer
        vectorizer_model = CountVectorizer(ngram_range=(1, 3), # considers word groupings in n-gram range (in this case, 1 to 3)
                                           stop_words="english") # additional stop word removal

        # Setup c-TF-IDF model
        ctfidf_model = ClassTfidfTransformer(bm25_weighting=True, # weighting that works better with small datasets
                                             reduce_frequent_words=True)

        # Initialize BERTopic model
        topic_model = BERTopic(top_n_words=tpc, 
                               min_topic_size=30, # note: min_topic_size is not used when the HDBSCAN algorithm is specified
                               umap_model=umap_model, 
                               hdbscan_model=hdbscan_model,
                               vectorizer_model=vectorizer_model,
                               ctfidf_model=ctfidf_model)

        # Generate Topics
        topics, probs = topic_model.fit_transform(get_text, embeddings)

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
        temp = str(coherence) + "," + str(tpc) + "," + str(cs) + "\n"

        scoresheet.write(temp)

        if coherence > best_cm:
            best_cm = coherence
            high_score = [best_cm, tpc]

scoresheet.close()
print("High Score:")
print(high_score)