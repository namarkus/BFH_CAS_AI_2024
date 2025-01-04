#!/usr/bin/env python
# -*- coding: utf-8 -*-
# _____[ _advanced_evaluator.py ]____________________________________________________
"""
Erweiterte Evaluation der Qualität der trainierten Embeddings

Hier werden mittels Framework llama-index verschiedene Metriken zur Modellevaluation ermittelt.
Der zugrunde liegende Mechanismus ist: Für jeden Text-Chunk (hier nur ein sampl von 10 Chunks, wegen Laufzeit)
wird ein LLM beauftragt, zwei Fragen dazu zu stellen. Danach wird geprüft, ob die gleichen Chunks als relevante Contexte
ermittelt werden.
"""

if __name__ == '__main__':
    print("Dieses Modul ist nicht für den direkten Aufruf gedacht. Bitte nutze main vbc_chat ooder vbc_learn.")
    exit()

# _____[ Imports ]______________________________________________________________
from _logging import app_logger
from _apis import Evaluator
from _configs import VbcConfig
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.evaluation import RetrieverEvaluator, generate_question_context_pairs
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding, OpenAIEmbeddingMode, OpenAIEmbeddingModelType

import random
import sys
import pandas as pd

class AdvancedEvaluator(Evaluator):
    def __init__(self, config: VbcConfig, embeddings_client, embedding_store):
        super().__init__(config, embeddings_client, embedding_store)

        embed_model = OpenAIEmbedding(mode=OpenAIEmbeddingMode.TEXT_SEARCH_MODE,
                                      model=OpenAIEmbeddingModelType.TEXT_EMBED_3_SMALL)
        self.query_model = OpenAI(model="gpt-4o", max_tokens=1000, temperature=0.1)

        # llama-index vector store aus chroma bilden
        vector_store = ChromaVectorStore(chroma_collection=embedding_store.collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        self.vector_index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store,
            storage_context=storage_context,
            embed_model=embed_model
        )

        self.retriever = self.vector_index.as_retriever(similarity_top_k=5, verbose=True)

        self.gen_qa_prompt = """\
            Context information is below.
            
            ---------------------
            {context_str}
            ---------------------
            
            Given the context information and no prior knowledge.
            generate only questions based on the below query.
            The language of the context information ist german.
            
            You are a Teacher/ Professor. Your task is to setup {num_questions_per_chunk} questions for an upcoming
            quiz/examination. The questions should be diverse in nature across the document.
            The language of the question has to be german. Restrict the questions to the context information provided."
            """

        self.qa_dataset = self.build_qa_dataset()

    def build_qa_dataset(self, sample_size=10):

        app_logger().debug(f"Erstellen eines QA Dataset auf {sample_size} zufällig ausgewählten Chunks")

        # get all nodes from store
        nodes = self.vector_index.storage_context.vector_store._get(limit=sys.maxsize, where={}).nodes
        smp_nodes = random.sample(nodes, sample_size)

        llm = OpenAI(model="gpt-4o")

        qa_dataset = generate_question_context_pairs(
            smp_nodes,
            llm=llm,
            qa_generate_prompt_tmpl=self.gen_qa_prompt,
            num_questions_per_chunk=2
        )
        # todo Da, dies eine aufwändig Funktion ist, sollte das Ergebnis irgendwo persistent gespeichert werden
        return qa_dataset

    def is_simple(self):
        return False

    def is_advanced(self):
        return True

    def evaluate(self):
        return

    async def aevaluate(self):

        metrics = ["hit_rate", "mrr", "precision", "recall", "ap", "ndcg"]

        retriever_evaluator = RetrieverEvaluator.from_metric_names(
            metrics, retriever=self.retriever
        )

        eval_results = await retriever_evaluator.aevaluate_dataset(self.qa_dataset)

        metric_dicts = []
        for eval_result in eval_results:
            metric_dict = eval_result.metric_vals_dict
            metric_dicts.append(metric_dict)

        full_df = pd.DataFrame(metric_dicts)
        metrics_df = full_df.mean(numeric_only=True).to_frame().T

        return metrics_df
