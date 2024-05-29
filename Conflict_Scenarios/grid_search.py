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

class grid_search:
	def __init__(self, documents=[None], stopwords=[None], ngram_range=(1, 3), bm25_weighting=False,
				 show_progress_bar=True, reduce_frequent_words=True, transformer_model="all-mpnet-base-v2",
				 tpc=[None], cs=[None], nb=[None], comp=[None], umap_metric=[None], hdb_metric=[None]):
		if (documents[0] != None) and (tpc[0] != None) and (nb != None) and (cs != None) and (comp != None) and (umap_metric[0] != None) and (hdb_metric != None):
			self.documents = documents
			self.stopwords = stopwords

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
			self.doc_term_matrix = [dict_.doc2bow(i) for i in self.tokenized_corpus]

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

			# calculate coherence scores
			print("Calculating Coherence Scores...")
			scores = []
			for parameters in grid:
				score = coherence_calc(tpc=parameters[0],
							   		   cs=parameters[1],
							   		   nb=parameters[2],
							   		   comp=parameters[3],
							   		   umap_met=parameters[4],
							   		   hdb_met=parameters[5])
				temp = parameters
				temp.append(score)
				scores.append(temp)
			print(scores)
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
	    
	    for item in self.stopwords:
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

	def coherence_calc(self, tpc, cs, nb, comp, umap_met, hdb_met):
		# Set up UMAP with a fixed random state
		umap_model = UMAP(n_neighbors=nb, 
		                  n_components=comp, 
		                  min_dist=0.0, 
		                  metric=umap_met, 
		                  random_state=42)

		# Setup clustering algorithm
		hdbscan_model = HDBSCAN(min_cluster_size=cs, 
		                        metric=hdb_met,
		                        cluster_selection_method='eom',
		                        prediction_data=True)


		# Initialize BERTopic model
		topic_model = BERTopic(top_n_words=tpc, 
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

		return coherence