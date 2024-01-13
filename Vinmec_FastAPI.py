from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import StreamingResponse
from VinRAG import VinmecRetriever
import asyncio
import json
import streamlit as st


DB_VECTOR = "vinmec_embedding_2"
DB_ROOT = "api"
url_pg_vector = st.secrets['url_pg_vector'] +"/{db}" #store information embedding
model = "gpt-3.5-turbo-1106"
api_key = st.secrets['api_key']


vinmec_engine = VinmecRetriever(DB_VECTOR, DB_ROOT, url_pg_vector,model,api_key)
app = FastAPI()


def generate_stream(question, retriever_func):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    def generate():
        # response_stream = retriever_func(question)
        response_stream = vinmec_engine.behavior_controller(question, retriever_func)
        if response_stream:
            for response in response_stream:
                yield response
        else:
            yield "Unsupported retriever type"
    return StreamingResponse(generate(), media_type="event-stream")


def retrieve_retriever(question, retriever_type):
    if retriever_type == "chat_engine_2":
        return generate_stream(question, vinmec_engine.create_retriever_stupid_2)
    elif retriever_type == "chat_engine":
        return generate_stream(question, vinmec_engine.create_retriever_stupid)
    elif retriever_type == "hybrid_engine":
        return generate_stream(question, vinmec_engine.hybrid_retriever_engine)
    else:
        return generate_stream(question, None)

@app.get("/{retriever_type}")
def stream_chat(retriever_type: str, question: str):
    return retrieve_retriever(question, retriever_type)

# @app.get("/chat_engine_2")
# def stream(question: str):
#     # Create a new event loop
#     loop = asyncio.new_event_loop()
#     # Set the event loop as the current event loop
#     asyncio.set_event_loop(loop)
#     def generate():
#         response_stream = vinmec_engine.create_retriever_stupid_2(question)
#         for response in response_stream:
#             yield response
#     return StreamingResponse(generate(), media_type="event-stream")


