import nest_asyncio
nest_asyncio.apply()
import VinRAG
import importlib
#using importlib to reload module
importlib.reload(VinRAG)
from VinRAG import VinmecRetriever
import streamlit as st

DB_VECTOR = "storage_index"
DB_ROOT = "api"
url_pg_vector = st.secrets['url_pg_vector'] +"/{db}" #store information embedding

model = "gpt-3.5-turbo-1106"
api_key = st.secrets['api_key']
vin_retriever = VinmecRetriever(DB_VECTOR, DB_ROOT, url_pg_vector,model,api_key)

<<<<<<< Updated upstream
question = "Bị mắc nghẹn vật to ở cổ thì nên làm thế nào?"
answer = vin_retriever.behavior_controller(question, vin_retriever.hybrid_retriever_engine)
for text in answer:
    print(text, end="", flush=True)
=======
question = "Bị nghẹn vật to ở cổ thì phải làm sao?"
# answer = vin_retriever.behavior_controller(question, vin_retriever.hybrid_retriever_engine)
# for text in answer:
    # print(text, end="", flush=True)
    
title,title_str = vin_retriever.get_title(question)
>>>>>>> Stashed changes
