###########################################################################
# Demo für die Behandlung der Embeddings
###########################################################################

###########################################################################
# Vorbereiten > Module ladan
###########################################################################

# import externe module
import random
import json

# import eigene module
import config as myC
import timestamp
myTimestamp = timestamp.timestamp()
import db
myDB = db.db()
import pc
myPC = pc.pc()

###########################################################################
# Vorbereiten > Singeltons konfigurieren und starten
###########################################################################

random.seed(42)
theDebugTag = "MAIN.PY"

myTimestamp.start()

myDB.start()
myDB.config(myC.theDBPath)
# myDB.leeren()

# myPC.deleteIndex(myC.thePineconeIndex)
myPC.start()
myPC.config(myC.thePineconeIndex)


###########################################################################
# Vorbereiten > Data
###########################################################################

myEmbeddingDataset=""

###########################################################################
# SCHRITT 1 > Embeddings > Typ=Corpus > daten vorbereiten
###########################################################################

# corpus daten > texte bestimmen
myCorpus = [
    {"text": "Apple is a popular and wellknown fruit known for its sweetness and crisp texture."},
    {"text": "The tech company Apple is known for its innovative products like the iPhone."},
    {"text": "Many people enjoy eating apples as a healthy snack."},
    {"text": "Apple Inc. has revolutionized the tech industry with its sleek designs and user-friendly interfaces."},
    {"text": "An apple a day keeps the doctor away, as the saying goes."},
    {"text": "Apple Computer Company was founded on April 1, 1976, by Steve Jobs, Steve Wozniak, and Ronald Wayne as a partnership."}
]

# corpus daten > raw data ergänzen mit ID und
for i in myCorpus:
    theText=i["text"]
    i["id"]=f"{myTimestamp.generateTS()}"
    theAnbieter=random.choice(myC.theAnbieterListe)
    theProdukt=random.choice(myC.theProduktListe)
    i["metadata_anbieter"] = f"{theAnbieter}"
    i["metadata_produkt"] = f"{theProdukt}"
    i["metadata"] = {"anbieter":f"{theAnbieter}","produkt":f"{theProdukt}","text":f"{theText}"}

# corpus daten > check
if 'MAIN' in myC.theLoglevels:
    print(f"\n{theDebugTag} >  MEIN CORPUS ------------------------------------------------------")
    for i in myCorpus:
        print(i)


###########################################################################
# SCHRITT 2 > Embeddings > Typ=Corpus > embeddings generieren
###########################################################################

if 'MAIN' in myC.theLoglevels: print(f"\n{theDebugTag} >  EMBEDDINGS FÜR CORPUS BESTIMMEN -----------------------------------------------------")

for i in myCorpus:
    theNow=myTimestamp.generateDT()
    if 'MAIN' in myC.theLoglevels: print(f"\n{theDebugTag} >  EMBEDDING BESTIMMEN für {i['metadata']}")
    myEmbeddingID, myEmbeddingVektoren = myDB.readEmbedding(tabelle='embeddings_corpus',embedding_typ='Corpus', embedding_meta_anbieter=i["metadata_anbieter"],embedding_meta_produkt=i["metadata_produkt"], embedding_text=i["text"])
    if myEmbeddingID is None:
        if 'MAIN' in myC.theLoglevels: print(f"{theDebugTag} >  muss embedding durch Pinecone generieren ...")
        myEmbeddingVektoren = myPC.getEncodingCorpus(i)
        i["values"] = myEmbeddingVektoren
        if 'MAIN' in myC.theLoglevels: print(f"{theDebugTag} >  muss embedding in DB speichern...")
        myDB.writeEmbedding(tabelle='embeddings_corpus',embedding_id=i["id"], embedding_typ='Corpus', embedding_meta_anbieter=i["metadata_anbieter"],
                            embedding_meta_produkt=i["metadata_produkt"], embedding_text=i["text"], embedding_vektoren=f"{i['values']}",
                            meta_datum_erstellt=theNow, meta_datum_geaendert=theNow)
        iCopy = i.copy()
        iCompact = myPC.compact(iCopy)
        myDataForVectorDB = []
        myDataForVectorDB.append(iCompact)
        if 'MAIN' in myC.theLoglevels: print(f"{theDebugTag} >  muss embedding in Pinecone Index speichern ...")
        myPC.upsert(myDataForVectorDB, myC.thePineconeNSCorpus)
    else:
        i["id"] = myEmbeddingID
        i["values"] = myEmbeddingVektoren
        if 'MAIN' in myC.theLoglevels: print(f"{theDebugTag} >  Embedding vorhanden [ID={i['id']}]")


###########################################################################
# Embeddings > Recherche
###########################################################################

if 'MAIN' in myC.theLoglevels: print(f"\n{theDebugTag} >  RECHERCHE -----------------------------------------------------")

if 'MAIN' in myC.theLoglevels: print(f"\n{theDebugTag} >  RECHERCHE > EMBEDDINGS FÜR FRAGEN BESTIMMEN -----------------------------------------------------")
myFrage = "Tell me about the tech company known as Apple."
myFrageEmbedding = myPC.getEncodingFragen(myFrage)

if 'MAIN' in myC.theLoglevels: print(f"\n{theDebugTag} >  RECHERCHE > FRAGEN ABSETZEN -----------------------------------------------------")
myAntwortenRaw = myPC.recherche(myFrageEmbedding[0].values,myC.thePineconeNSCorpus)
myAntworten = myPC.antworten(myAntwortenRaw)
if 'MAIN' in myC.theLoglevels:
    print(f"{theDebugTag} >  RECHERCHE > ANTWORTEN ERHALTEN ------------------------------------")
    print(json.dumps(myAntworten,indent=4))

###########################################################################
# Beenden
###########################################################################

myDB.beenden()
print('Und Tschüss....')