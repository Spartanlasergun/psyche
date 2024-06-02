import pymongo
from collections import Counter
import matplotlib.pyplot as plt

# connect to mongodb to retrieve data and clear old coherence scores
uri = "mongodb+srv://cluster0.cd4m7jc.mongodb.net/?authSource=%24external&authMechanism=MONGODB-X509&retryWrites=true&w=majority&appName=Cluster0"
client = pymongo.MongoClient(uri,
                             tls=True,
                             tlsCertificateKeyFile='Spartanlasergun-certificate.pem',
                             server_api=pymongo.server_api.ServerApi(version="1", strict=True, deprecation_errors=True))
db = client['watchtower']
collection = db['stopwords']

print("fetching stopwords...")
documents = collection.find()

all_documents = list(documents)

# Extract the 'stopword' field from each document
data_list = [doc['stopword'] for doc in all_documents]

length = len(set(data_list))
print(f"Number of topic outliers: {length}")

# Count the frequency of each string in the 'data' field
counter = Counter(data_list)

# Extract the data for plotting
labels = list(counter.keys())
counts = list(counter.values())

# Create the bar chart
plt.figure(figsize=(10, 8))
plt.grid(True)
plt.bar(labels, counts, color='skyblue')
plt.xticks(rotation=90)

# Add title and labels
plt.title('Frequency Of Occurence For Each Outlier')
plt.xlabel('Frequency')
plt.ylabel('Topic Outliers')
# Reduce the font size of the y-axis labels
plt.yticks(fontsize=7)

# Display the plot
plt.show()
