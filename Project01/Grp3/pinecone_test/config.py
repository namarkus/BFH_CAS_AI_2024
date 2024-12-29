# ...

# bibliotheken laden
import os
import sys

theLoglevels = ['MAIN','TEST','CLASS'] # ['MAIN','CLASS','QUERIES','TEST']
theRootpath = sys.path[0]
theDBUser='hans'
theDBPath = os.path.join(theRootpath,f"db_user_{theDBUser}.db")
thePineconeKey = 'pcsk_1Mzry_NNT38EXwhAhfTCtT8LfgFuUUguXjwkcAEayh9nfAzopKCtgHJMP6jj7ypTQjRdB'
thePineconeIndex = 'projekt01'
thePineconeNSCorpus = 'Corpus'
thePineconeModell = "multilingual-e5-large"
theAnbieterListe=["ALLE"]
theProduktListe=["ALLE"]

# OPENAI_API_KEY bestimmen
try:
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if openai_api_key is None:
        print("ERROR @ CONFIG.PY > OPENAI_API_KEY not found")
except:
    raise Exception("ERROR @ CONFIG.PY > OPENAI_API_KEY not found")

# PINECONE_API_KEY bestimmen
try:
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    if pinecone_api_key is None:
        print("ERROR @ CONFIG.PY > PINECONE_API_KEY not found")
except:
   raise Exception("ERROR @ CONFIG.PY > PINECONE_API_KEY not found")

thePineconeKey=pinecone_api_key

# for name, value in os.environ.items():
#     print("{0}: {1}".format(name, value))