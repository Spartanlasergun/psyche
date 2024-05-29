import pandas as pd
from grid_search import grid_search

# Read Data From Archive
data = pd.read_csv('Conflict Scenarios Research.csv')
# create documents list
conflict = data["Describe a past experience you've had that involved conflict with a family member, friend, or significant other. Be as detailed as you like."].tolist()

stopwords = ["wa", "art", "one", "nt", "lot", "-", ".", ",", "?", "!", "'s", "n't", "'re", "'m", "'ve", " ",
             "get", "conflict", "really"]

check = grid_search(documents=conflict, 
                    ngram_range=(1, 3),
                    stopwords=stopwords,
                    bm25_weighting=True,
                    show_progress_bar=True,
                    reduce_frequent_words=True,
                    tpc= range(2, 6, 1),
                    cs= range(10, 21, 1),
                    nb= range(5, 26, 1),
                    comp= range(2, 11, 1),
                    umap_metric=['cosine'],
                    hdb_metric=['euclidean'],
                    thread_count=20)

