import pandas as pd
import spacy
import nltk
import os
import multiprocessing
import pymongo
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

class grid_search:
	def __init__(self, documents=[None], add_stopwords=[None], ngram_range=(1, 3), bm25_weighting=False,
				 show_progress_bar=True, reduce_frequent_words=True, transformer_model="all-mpnet-base-v2",
				 tpc=[None], cs=[None], nb=[None], comp=[None], umap_metric=[None], hdb_metric=[None],
				 worker_count=10):
		if (documents[0] != None) and (tpc[0] != None) and (nb != None) and (cs != None) and (comp != None) and (umap_metric[0] != None) and (hdb_metric != None):
			self.documents = documents
			self.add_stopwords = add_stopwords

			# text cleaning and preprocessing
			print("Cleaning text and preprocessing...")
			self.get_text, self.tokenized_corpus = self.preprocessing()

			# set transformer model for use with BERTopic
			sentence_model = SentenceTransformer(transformer_model)
			# Pre-calculate embeddings
			print("Initializing...")
			self.embeddings = sentence_model.encode(self.get_text, show_progress_bar=show_progress_bar)

			self.vectorizer_model = CountVectorizer(ngram_range=ngram_range, # considers word groupings in n-gram range (in this case, 1 to 3)
	            									stop_words="english") # additional stop word removal

			self.ctfidf_model = ClassTfidfTransformer(bm25_weighting=bm25_weighting, # weighting that works better with small datasets
	                                            	  reduce_frequent_words=reduce_frequent_words)

			# create dictionary of unique words from the tokenized corpus
			self.dict_ = corpora.Dictionary(self.tokenized_corpus)

			# The doc_term_matrix contain tuple entries with the token_id for each word in the corpus, 
			# along with its frequency of occurence.
			self.doc_term_matrix = [self.dict_.doc2bow(i) for i in self.tokenized_corpus]

			# connect to mongodb to manage flow of data
			self.uri = "mongodb+srv://cluster0.cd4m7jc.mongodb.net/?authSource=%24external&authMechanism=MONGODB-X509&retryWrites=true&w=majority&appName=Cluster0"
			self.client = pymongo.MongoClient(uri,
			                             	  tls=True,
			                             	  tlsCertificateKeyFile='Spartanlasergun-certificate.pem',
			                             	  server_api=pymongo.server_api.ServerApi(version="1", strict=True, deprecation_errors=True))

			self.db = client['watchtower']
			self.collection = db['coherence_parameters']

			# generate grid
			grid = []
			for a in tpc:
				for b in cs:
					for c in nb:
						for d in comp:
							for e in umap_metric:
								for f in hdb_metric:
									temp = [a, b, c, d, e, f]
									grid.append(temp)

			
			# Creating batches from grid
			batch_size = int(len(grid) / worker_count)
			batches = [grid[i:i + batch_size] for i in range(0, len(grid), batch_size)] # creates batches equivalent to the number of workers

			# calculate coherence scores
			print("Calculating Coherence Scores...")
			processes = []
			count = 0
			for batch in batches:
				score = multiprocessing.Process(target=self.coherence_calc, args=(batch,))
				processes.append(score)
				score.start()
				count = count + 1
				print("Starting Batch: " + str(count))

			# Wait for all processes to complete
			for p in processes:
				p.join()

			print("Grid Search Complete")
		else:
			print("Initialization Error: Incorrect Parameter")



	# preprocessing function
	def preprocessing(self):
	    # Load the English language model
	    nlp = spacy.load("en_core_web_sm")

	    lemmatizer = WordNetLemmatizer()

	    tokenized_docs = []
	    for item in self.documents:
	        doc = nlp(item)
	        tokens = [token.text for token in doc] # process token into list
	        lowercase_list = [word.lower() for word in tokens] # lowercase all words
	        # lemmatize words to prevent overlap between extremely similar words (e.g. "friend" and "friends")
	        lemmatized_words = [lemmatizer.lemmatize(word) for word in lowercase_list]
	        tokenized_docs.append(lemmatized_words) 

	    stop_words = list(set(stopwords.words('english')))
	    
	    for item in self.add_stopwords:
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

	def coherence_calc(self, parameters):
		try:
			for parameter in parameters:
				# Set up UMAP with a fixed random state
				umap_model = UMAP(n_neighbors=parameter[2], 
				                  n_components=parameter[3], 
				                  min_dist=0.0, 
				                  metric=parameter[4], 
				                  random_state=42)

				# Setup clustering algorithm
				hdbscan_model = HDBSCAN(min_cluster_size=parameter[1], 
				                        metric=parameter[5],
				                        cluster_selection_method='eom',
				                        prediction_data=True)


				# Initialize BERTopic model
				topic_model = BERTopic(top_n_words=parameter[0], 
				                       min_topic_size=30, # note: min_topic_size is not used when the HDBSCAN algorithm is specified
				                       umap_model=umap_model, 
				                       hdbscan_model=hdbscan_model,
				                       vectorizer_model=self.vectorizer_model,
				                       ctfidf_model=self.ctfidf_model)

				# Generate Topics
				topics, probs = topic_model.fit_transform(self.get_text, self.embeddings)

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
				cm = CoherenceModel(topics=raw_topics, texts=self.tokenized_corpus, corpus=self.doc_term_matrix, dictionary=self.dict_, coherence='c_npmi')
				coherence = cm.get_coherence()

				self.collection.insert_one({"coherence" : coherence,
										    "topics_per_cluster" : parameter[0],
										    "min_cluster_size" : parameter[1],
										    "n_neighbors" : parameter[2],
										    "n_components" : parameter[3],
										    "umap_metric" : parameter[4],
										    "hdb_metric" : parameter[5]})
		except:
			print("coherence calc error")


if __name__ == "__main__":
	# Read Data From Archive
	data = pd.read_csv('Conflict Scenarios Research.csv')
	# create documents list
	conflict = data["Describe a past experience you've had that involved conflict with a family member, friend, or significant other. Be as detailed as you like."].tolist()

	ud_stopwords = ["wa", "art", "one", "nt", "lot", "-", ".", ",", "?", "!", "'s", "n't", "'re", "'m", "'ve", " ",
	             "get", "conflict", "really"]

	check = grid_search(documents=conflict, 
	                    ngram_range=(1, 3),
	                    add_stopwords=ud_stopwords,
	                    bm25_weighting=True,
	                    show_progress_bar=True,
	                    reduce_frequent_words=True,
	                    tpc= range(2, 21, 1),
	                    cs= [10],
	                    nb= [13],
	                    comp= [3],
	                    umap_metric=['cosine'],
	                    hdb_metric=['euclidean'],
	                    worker_count=2)
