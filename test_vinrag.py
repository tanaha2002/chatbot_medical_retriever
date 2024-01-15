import nest_asyncio
nest_asyncio.apply()
import VinRAG
import importlib
#using importlib to reload module
importlib.reload(VinRAG)
from VinRAG import VinmecRetriever
import streamlit as st

DB_VECTOR = "vinmec_embedding_2"
DB_ROOT = "api"
url_pg_vector = st.secrets['url_pg_vector'] +"/{db}" #store information embedding

model = "gpt-3.5-turbo-1106"
api_key = st.secrets['api_key']
vin_retriever = VinmecRetriever(DB_VECTOR, DB_ROOT, url_pg_vector,model,api_key)

question = "trieu chung dau bung thuong xuyen"
answer = vin_retriever.behavior_controller(question, vin_retriever.hybrid_retriever_engine)
for text in answer:
    print(text, end="", flush=True)