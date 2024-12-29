# Das Objekt myDatabase gehört zur Gruppe der System-Objekte.
# Das Objekt myDatabase ermöglicht den Zugriff auf die SQLite Datenbank und bietet
# diverse Methoden wie z.B. das Leeren der gesamten Datenbank oder einzelner Tabellen.

# bibliotheken laden
import os
import sqlite3
import config as myC

# klasse definieren
class db:

    ###############################################################################
    def __init__(self):
        self.theDebugTag = "DB.PY"
        self.myDBConnection = None
        self.myDBCursor = None


    ###############################################################################
    def start(self):
        if 'CLASS' in myC.theLoglevels: print(f"{self.theDebugTag} > DB gestartet")
        return(None)


    ###############################################################################
    def config(self,theDBPath):
        self.myDBPath=theDBPath
        try:
            self.myDBConnection = sqlite3.connect(self.myDBPath)
            self.myDBCursor = self.myDBConnection.cursor()
            if 'CLASS' in myC.theLoglevels: print(f"{self.theDebugTag} > DB konfiguriert")
        except os.error as e:
            if 'CLASS' in myC.theLoglevels: print('DB.PY > DB konnte nicht geöffnet werden')
            exit("ERROR @ DB.PY > DB start")


    ###############################################################################
    def beenden(self):
        try:
            self.myDBConnection.close()
            if 'CLASS' in myC.theLoglevels: print(f"{self.theDebugTag} > DB beendet")
        except os.error as e:
            if 'CLASS' in myC.theLoglevels: print('DB.PY > konnte nicht beendet werden')
            exit("ERROR @ DB.PY > DB beenden")


    ###############################################################################
    def leeren(self, theTabellen='alle'):
        # alle tabellen
        if theTabellen == 'alle':
            self.myDBCursor.execute("DELETE FROM embeddings_corpus WHERE 1 = 1")
            self.myDBConnection.commit()
            if 'CLASS' in myC.theLoglevels: print(f"{self.theDebugTag} > DB alle Tabellen in {self.myDBPath} geleerte")
        # embeddings tabelle
        elif theTabellen == 'embeddings_corpus':
            self.myDBCursor.execute("DELETE FROM embeddings_corpus WHERE 1 = 1")
            self.myDBConnection.commit()
            if 'CLASS' in myC.theLoglevels: print(f"{self.theDebugTag} > DB Tabelle embeddings_corpus geleerte")


    ###############################################################################
    def writeEmbedding(self, tabelle, embedding_id, embedding_typ, embedding_meta_anbieter, embedding_meta_produkt, embedding_text, embedding_vektoren,meta_datum_erstellt,meta_datum_geaendert):
        q = f"""INSERT INTO {tabelle} (embedding_id,embedding_typ,embedding_meta_anbieter,embedding_meta_produkt,embedding_text,embedding_vektoren,meta_datum_erstellt,meta_datum_geaendert)
                VALUES ("{embedding_id}","{embedding_typ}","{embedding_meta_anbieter}","{embedding_meta_produkt}","{embedding_text}","{embedding_vektoren}","{meta_datum_erstellt}","{meta_datum_geaendert}")"""
        if 'QUERIES' in myC.theLoglevels: print(f"{self.theDebugTag} > DB QUERY > {q}")
        self.myDBCursor.execute(q)
        self.myDBConnection.commit()
        if 'CLASS' in myC.theLoglevels: print(f"{self.theDebugTag} > DB > embedding gespeichert ")
        return(None)


    ###############################################################################
    def readEmbedding(self, tabelle, embedding_typ, embedding_meta_anbieter, embedding_meta_produkt, embedding_text):
        theID=None
        theWerte=None
        q = f'SELECT embedding_id, embedding_vektoren FROM {tabelle} WHERE embedding_typ ="{embedding_typ}" AND embedding_meta_anbieter ="{embedding_meta_anbieter}" AND embedding_meta_produkt ="{embedding_meta_produkt}" AND embedding_text ="{embedding_text}"'
        if 'QUERIES' in myC.theLoglevels: print(f"{self.theDebugTag} > DB QUERY > {q}")
        self.myDBCursor.execute(q)
        for r in self.myDBCursor.fetchall():
            theID = r[0]
            theWerte = r[1]
        return(theID,theWerte)
