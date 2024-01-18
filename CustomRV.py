from llama_index import QueryBundle
from llama_index.retrievers import (
    BaseRetriever,
    VectorIndexRetriever,
)
from typing import List
from llama_index.schema import NodeWithScore
from llama_index import load_index_from_storage
import psycopg2
from sqlalchemy import make_url
from llama_index.vector_stores import PGVectorStore
from llama_index import StorageContext, VectorStoreIndex
import streamlit as st
import psycopg2
from llama_index import VectorStoreIndex, Document
import requests
from bs4 import BeautifulSoup
from llama_index import VectorStoreIndex, ServiceContext, Document,StorageContext
from llama_index.retrievers import VectorIndexRetriever
from llama_index.query_engine import RetrieverQueryEngine
from llama_index.postprocessor import SimilarityPostprocessor
from llama_index.memory import ChatMemoryBuffer
from llama_index.llms import OpenAI
import asyncio
import streamlit as st
from sqlalchemy import make_url
from llama_index.vector_stores import PGVectorStore
from llama_index import load_index_from_storage
from llama_index.indices.vector_store.retrievers import VectorIndexRetriever
from llama_index.query_engine.retriever_query_engine import (
    RetrieverQueryEngine,
)
from llama_index import get_response_synthesizer
from llama_index.prompts import PromptTemplate
from llama_index.indices.postprocessor import SimilarityPostprocessor
from llama_index.indices.postprocessor import LLMRerank
from llama_index.indices.postprocessor import SentenceEmbeddingOptimizer
from concurrent.futures import ThreadPoolExecutor


class CustomRetriever(BaseRetriever):
    """_summary_
        Custom retriever for search with title (method_1 old) and search with content (method_2)
    Args:
        BaseRetriever (_type_): Base class for retrievers
        
    Returns:
        CustomRetriever
    """
    def __init__( self, vector_retriever_1: VectorIndexRetriever,vector_retriever_2: VectorIndexRetriever) -> None:
        self.vector_retriever_1 = vector_retriever_1
        self.vector_retriever_2 = vector_retriever_2
        super().__init__()
        
    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Retrieve node with high score at title and content"""
        with ThreadPoolExecutor(max_workers=4) as executor:
            # Submit the retrieval tasks
            title_future = executor.submit(self.vector_retriever_1.retrieve, query_bundle)
            content_future = executor.submit(self.vector_retriever_2.retrieve, query_bundle)

            # Wait for the results
            title_nodes = title_future.result()
            content_nodes = content_future.result()

        title_ids = {n.node.node_id for n in title_nodes}
        content_ids = {n.node.node_id for n in content_nodes}

        combine_dict = {_.node.node_id: _ for _ in title_nodes}
        combine_dict.update({_.node.node_id: _ for _ in content_nodes})

        # Union title and content
        retrieve_ids = title_ids.union(content_ids)
        retrieve_nodes = [combine_dict[_] for _ in retrieve_ids]

        return retrieve_nodes