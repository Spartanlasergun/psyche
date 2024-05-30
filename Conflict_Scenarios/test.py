import pymongo

uri = "mongodb+srv://cluster0.cd4m7jc.mongodb.net/?authSource=%24external&authMechanism=MONGODB-X509&retryWrites=true&w=majority&appName=Cluster0"
client = pymongo.MongoClient(uri,
                             tls=True,
                             tlsCertificateKeyFile='Spartanlasergun-certificate.pem',
                             server_api=pymongo.server_api.ServerApi(version="1", strict=True, deprecation_errors=True))


db = client['watchtower']
collection = db['coherence_parameters']
doc_count = collection.count_documents({})
collection.insert_one({"test_entry" : "This is some test data"})

print("Done")
