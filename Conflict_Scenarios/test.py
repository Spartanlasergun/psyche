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
                    tpc= [2, 3],
                    cs= [13],
                    nb= [10],
                    comp= [3],
                    umap_metric=['cosine'],
                    hdb_metric=['euclidean'],
                    worker_count=2)

