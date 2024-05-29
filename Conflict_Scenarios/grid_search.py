import pandas as pd
import spacy
import nltk
import psutil
import os
import multiprocessing
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
				 tpc=[None], cs=[None], nb=[None], comp=[None], umap_metric=[None], hdb_metric=[None],
				 worker_count=10):
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
			self.doc_term_matrix = [self.dict_.doc2bow(i) for i in self.tokenized_corpus]

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
			batches = [grid[i:i + thread_count] for i in range(0, len(grid), thread_count)]

			# calculate coherence scores
			print("Calculating Coherence Scores...")
			self.scores = []
			processes = []
			for batch in batches:
				for parameters in batch:
					score = multiprocessing.Process(target=self.coherence_calc, args=(parameters[0],
																		   		      parameters[1],
																		   		      parameters[2],
																		   		      parameters[3],
																		   		      parameters[4],
																		   		      parameters[5]))
					processes.append(score)
					score.start()

				# Wait for all processes to complete
				for p in processes:
					p.join()

				# Print CPU and memory usage after each cycle
				self.print_usage()

			scoresheet = open("grid_scores.txt", 'w', encoding='utf8')
			for item in self.scores:
				temp = str(item[0]) + "," + str(item[1]) + "," + str(item[2]) + "," + str(item[3]) + "," + str(item[4]) + "," + str(item[5]) + "," + str(item[6]) + "\n"
				scoresheet.write(temp)
			scoresheet.close()
				
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
		try:
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

			scoresheet = [coherence, tpc, cs, nb, comp, umap_met, hdb_met]
			self.scores.append(scoresheet)
		except:
			pass

	# Function to print CPU and memory usage
	def print_usage(self):
	    process = psutil.Process(os.getpid())
	    memory_info = process.memory_info()
	    cpu_percent = process.cpu_percent(interval=0.0)
	    print(f'CPU usage: {cpu_percent}%')
	    print(f'Memory usage: RSS={memory_info.rss / (1024 ** 2)} MB, VMS={memory_info.vms / (1024 ** 2)} MB')
