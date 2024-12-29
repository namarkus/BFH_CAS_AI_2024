# ...

# bibliotheken laden
import config as myC
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
import time
import json

# klasse definieren
class pc:

    ###############################################################################
    def __init__(self):
        self.theDebugTag = "PC.PY"
        self.myPinecone = None
        self.myIndex = None


    ###############################################################################
    def start(self):
        if 'CLASS' in myC.theLoglevels: print(f"{self.theDebugTag} >  Pinecone gestartet")


    ###############################################################################
    def config(self,theIndex):
        self.myPinecone = Pinecone(api_key=myC.thePineconeKey)
        if not self.myPinecone.has_index(theIndex):
            self.myPinecone.create_index(
                name=theIndex,
                dimension=1024,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud='aws',
                    region='us-east-1'
                )
            )
        while not self.myPinecone.describe_index(theIndex).status['ready']:
            time.sleep(1)
        self.myIndex = self.myPinecone.Index(theIndex)
        if 'CLASS' in myC.theLoglevels: print(f"{self.theDebugTag} >  Pinecone konfiguriert")


    ###############################################################################
    def deleteIndex(self,theIndex):
        self.myPinecone = Pinecone(api_key=myC.thePineconeKey)
        if self.myPinecone.has_index(theIndex):
            self.myPinecone.delete_index(theIndex)
            if 'TEST' in myC.theLoglevels: print(f"{self.theDebugTag} >  Pinecone Index '{theIndex}' geleert")
        else:
            if 'TEST' in myC.theLoglevels: print(f"{self.theDebugTag} >  Pinecone Index '{theIndex}' nicht geleert da nicht vorhanden")

    ###############################################################################
    def getEncodingCorpus(self,theData):
        if 'TEST' in myC.theLoglevels: print(f"{self.theDebugTag} > Daten erhalten:")
        if 'TEST' in myC.theLoglevels: print(json.dumps(theData, indent=4))
        myEmbedding = self.myPinecone.inference.embed(
            model=myC.thePineconeModell,
            inputs=theData["text"],
            parameters={"input_type": "passage", "truncate": "END"}
        )
        myEmbedding = myEmbedding.data
        myEmbedding = myEmbedding.pop()
        if 'TEST' in myC.theLoglevels: print(f"{self.theDebugTag} > Embedding mit {len(myEmbedding['values'])} Vektoren generiert:")
        if 'TEST' in myC.theLoglevels: print(myEmbedding["values"][:20],"...")
        return myEmbedding["values"]


    ###############################################################################
    def getEncodingFragen(self,myFrage):
        myQueryEmbedding = self.myPinecone.inference.embed(
            model=myC.thePineconeModell,
            inputs=[myFrage],
            parameters={
                "input_type": "query"
            }
        )
        return myQueryEmbedding


    ###############################################################################
    def compact(self, theData):
        theData.pop("text")
        theData.pop("metadata_anbieter")
        theData.pop("metadata_produkt")
        return theData


    ###############################################################################
    def upsert(self, theData,theNS):
        self.myIndex.upsert(
            vectors=theData,
            namespace=theNS
        )
        if 'TEST' in myC.theLoglevels: print(f"{self.theDebugTag} > Daten in Index gespeichert [NS={theNS}]")


    ###############################################################################
    def recherche(self, theVectors,theNS):
        # Search the index for the three most similar vectors
        myResults = self.myIndex.query(
            namespace=myC.thePineconeNSCorpus,
            vector=theVectors,
            top_k=3,
            include_values=True,
            include_metadata=True
            # filter = {'group': GRP2}
        )
        return myResults

    ###############################################################################
    def antworten(self, myAntwortenRaw):
        myAntworten = []
        for x in myAntwortenRaw.matches:
            myAntworten.append({
                "id": x.id,
                "metadata": x.metadata,
                "score": x.score
            })
        return myAntworten