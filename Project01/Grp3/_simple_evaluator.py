#!/usr/bin/env python
# -*- coding: utf-8 -*-
# _____[ _simple_evaluator.py ]____________________________________________________
"""
Einfache Evaluation der Qualität der trainierten Embeddings

Es werden einige sehr einfach Fragen gestellt (die möglichst wortwörtlich in einem der Dokumente vorkommen) und
geprüft, ob das richtige Dokument den Kontext liefert.
"""

if __name__ == '__main__':
    print("Dieses Modul ist nicht für den direkten Aufruf gedacht. Bitte nutze main vbc_chat ooder vbc_learn.")
    exit()

# _____[ Imports ]______________________________________________________________
from _logging import app_logger
from _apis import Evaluator
from _configs import VbcAction, VbcConfig

class SimpleEvaluator(Evaluator):
    def __init__(self, config: VbcConfig, embeddings_client, embedding_store):
        super().__init__(config, embeddings_client, embedding_store)

        self.simple_queries = [
            {"id": 1, "query": "Wieviel bezahlt die SWICA für medizinisch notwendige Brillengläser und fassungen sowie Kontaktlinsen?", "documents": {"Swica_ZB_SUPPLEMENTA.pdf"}},
            {"id": 2, "query": "Welchen Betrag übernimmt Helsana für präventivmedizinische Massnahmen", "documents": {"Helsana_ZVB_completa.pdf", "Helsana_ZVB_completa_plus.pdf", "Helsana_ZVB_sana.pdf"}},
            {"id": 3, "query": "Welchen Tarif vergütet die Visana für zahnärztliche Behandlungen höchstens?", "documents": {"Visana_ZB_Zahn.pdf"}},
            {"id": 4, "query": "In welchem Umfang übernimmt die Assura Kosten für ambulant durchgeführter Therapien?", "documents": {"Assura_BVB_Medna.pdf", "Assura_BVB_Complementa_Extra.pdf"}},
            {"id": 5, "query": "Wie kann man in den Genuss einer einmaligen jährlichen Bonusvergütung bei der CSS kommen?", "documents": {"CSS_ZB_spital_myflex.pdf", "CSS_ZB_alternativ_myflex.pdf", "CSS_ZB_ambulant_myflex.pdf"}}
        ]

    def is_simple(self):
        return True

    def is_advanced(self):
        return False

    def evaluate(self):
        top_k = 5

        queries_embedding = self.embeddings_client.get_embeddings([query['query'] for query in self.simple_queries])

        precision_scores = []
        recall_scores = []

        for embedding, simple_query in zip(queries_embedding, self.simple_queries):
            retrieved_docs = self.embedding_store.find_most_similar_docs(embedding, top_k=top_k)
            relevant_docs = simple_query['documents']

            app_logger().debug(f"Modell hat folgende Dokumente gewählt: {retrieved_docs}, die relevanten Dokumente sind: {relevant_docs}")

            # Count the number of relevant documents in the top-k
            relevant_retrieved = sum(1 for doc in retrieved_docs if doc in relevant_docs)
            precision_score = relevant_retrieved / top_k
            precision_scores.append(precision_score)
            recall_score = relevant_retrieved / len(relevant_docs)
            recall_scores.append(recall_score)

        precision_at_k = sum(precision_scores) / len(precision_scores)
        recall_at_k = sum(recall_scores) / len(recall_scores)

        app_logger().info(f"Modell evaluiert: Precision@k={precision_at_k:.3f} / Recall@k={recall_at_k:.3f}")

        return {'precision@k': precision_at_k, 'recall@k': recall_at_k}

    async def aevaluate(self):
        return
