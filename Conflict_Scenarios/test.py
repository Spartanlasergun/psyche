import pandas as pd
import spacy
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

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
        tokenized_docs.append(lowercase_list) 

    stop_words = list(set(stopwords.words('english')))
    others = ["wa", "art", "one", "nt", "lot", "-", ".", ",", "?", "!", "'s", "n't", "'re", "'m", "'ve", " "]
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
            
    return processed, tokenized_docs

# get cleaned raw and tokenized data for training with BERTopic and coherence calculation
print("Preprocessing Text...")
get_text, tokenized_corpus = preprocessing(documents)

print(get_text[0])
print(tokenized_corpus[0])