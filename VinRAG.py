import os
import numpy as np
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
<<<<<<< Updated upstream
from llama_index.indices.postprocessor import SimilarityPostprocessor
from llama_index.indices.postprocessor import LLMRerank
from llama_index.indices.postprocessor import SentenceEmbeddingOptimizer
=======
from CustomRV import CustomRetriever

>>>>>>> Stashed changes
class VinmecRetriever:
    def __init__(self, db_vector, db_root, url_pg_vector,model,api_key, table_storage_index = "vinmec_storage_index"):
        os.environ["OPENAI_API_KEY"] = api_key
        self.DB_VECTOR = db_vector
        self.DB_ROOT = db_root
        self.url_pg_vector = url_pg_vector
        self.conn = self.connect_db(self.DB_VECTOR)
        self.llm = self.initialize_llm(model)
        self.service_context = ServiceContext.from_defaults(llm=self.llm,chunk_size=1024)
        self.connection_string = st.secrets['connection_string']
        self.table_storage_index = table_storage_index
        self.index = self.get_index_all()
        self.index_1,self.list_title = self.init_index1_and_title()
        self.chat_engine_2 = self.init_engine_2()
        self.hybrid_engine = self.retriever_query_engine()
        self.hybrid_engine = self.prompt_format(self.hybrid_engine)
        self.custom_rv = self.init_customRV()
    def connect_db(self, db_name):
        try:
            url_ = self.url_pg_vector.format(db=db_name)
            conn = psycopg2.connect(url_)
            print("connect successfully")
            return conn
        except Exception as e:
            print("connect failed")


    def initialize_llm(self,model):
        try:
            return OpenAI(model_name=model, temperature=0.55)
        except Exception as e:
            print(e)
            return None

    
    
    def create_retriever_stupid(self, question,k=3):
        
        # memory = ChatMemoryBuffer.from_defaults()
        index = self.index
        query_engine = index.as_chat_engine(
            chat_mode="best",
            streaming=True,
            system_prompt=(
                "Bạn là chatbot của Vinmec hỗ trợ người dùng trả lời các câu hỏi dựa trên thông tin chính xác. "
                "Vui lòng trình bày chi tiết để người dùng hiểu rõ hơn bằng tiếng việt. "
                "Vui lòng trích dẫn ý kiến của các bác sĩ nếu có trong nguồn dữ liệu. "
                "Lưu ý: Dù trong bất kỳ hoàn cảnh nào, hãy luôn luôn trả lời bằng tiếng Việt.\n\n"
                "Nếu không có dữ liệu để trả lời câu hỏi, vui lòng trả lời `Tôi hiện chưa được cập nhật thông tin này.`\n\n"
                
            ),
            context_prompt=(
                "Lưu ý: Dù trong bất kỳ hoàn cảnh nào, hãy luôn luôn trả lời bằng tiếng Việt.\n\n"
                "{context_str}"
                "\nInstruction: sử dụng ngữ cảnh bên trên để trả lời câu hỏi. Nếu ngữ cảnh không liên quan. Hãy trả lời `Tôi hiện chưa được cập nhật thông tin này.`\n\n"
    ),
        )
        response_stream = query_engine.stream_chat(question)
        yield "Tài liệu liên quan: \n"
        for node in response_stream.source_nodes:
            yield node.metadata['url']  + "\n"
        for text in response_stream.response_gen:
            yield text
    
    def get_all_docs(self, num_docs = None):
        if num_docs is None:
            query = """SELECT origin_information, link
                    FROM information_embedding"""
            cur = self.conn.cursor()
            cur.execute(query)
            all_docs = cur.fetchall()
            cur.close()
            return all_docs
        else:
            query = """SELECT origin_information, link
                    FROM information_embedding
                    ORDER BY RANDOM()
                    LIMIT %s"""
            cur = self.conn.cursor()
            cur.execute(query, (num_docs,))
            all_docs = cur.fetchall()
            cur.close()
            return all_docs
    
    def index_all(self):
        all_docs = self.get_all_docs()
        documents = self.llama_docs(all_docs)
        return VectorStoreIndex.from_documents(documents, service_context=self.service_context)
        
    
    def get_index_all(self,hybrid_search=True,text_search_config="english"):
        """_summary_
        Get index (method 2) from database
        Returns:
           index Object: index
           
        Notes:
            If you get error while hybrid search, that mean you don't have tsvector for searching
            
            ALTER TABLE your_table ADD COLUMN text_search_tsv tsvector;
            UPDATE your_table SET text_search_tsv = to_tsvector('english', content);
            CREATE INDEX text_search_idx ON your_table USING gin(text_search_tsv);
        """
        try:
            conn = psycopg2.connect(self.connection_string)
            conn.autocommit = True
            url = make_url(self.connection_string)
            vector_store = PGVectorStore.from_params(
                database=url.database,
                host=url.host,
                password=url.password,
                port=url.port,
                user=url.username,
                hybrid_search=hybrid_search,
                text_search_config=text_search_config,
                table_name=self.table_storage_index,
                embed_dim=1536, #openai embedding dim
                
            )
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            index = VectorStoreIndex.from_vector_store(storage_context.vector_store)
            return index
        except Exception as e:
            print(e)
            return None
    
    def init_engine_2(self):
        """_summary_
        Init chat engine on all documents for method 2
        Returns:
            _type_: Chat Engine
        """
        return self.index.as_chat_engine(
            chat_mode="condense_plus_context",
            streaming=True,
            system_prompt=(
                "Bạn là chatbot của Vinmec hỗ trợ người dùng trả lời các câu hỏi dựa trên thông tin chính xác. "
                "Vui lòng trình bày chi tiết để người dùng hiểu rõ hơn bằng tiếng việt. "
                "Vui lòng trích dẫn ý kiến của các bác sĩ nếu có trong nguồn dữ liệu. "
                "Lưu ý: Dù trong bất kỳ hoàn cảnh nào, hãy luôn luôn trả lời bằng tiếng Việt.\n\n"
                "Nếu không có dữ liệu để trả lời câu hỏi, vui lòng trả lời `Tôi hiện chưa được cập nhật thông tin này.`\n\n"
                
            ),
            context_prompt=(
                "Lưu ý: Dù trong bất kỳ hoàn cảnh nào, hãy luôn luôn trả lời bằng tiếng Việt.\n\n"
                "{context_str}"
                "\nInstruction: sử dụng ngữ cảnh bên trên để trả lời câu hỏi. Nếu ngữ cảnh không liên quan. Hãy trả lời `Tôi hiện chưa được cập nhật thông tin này.`\n\n"
    ),
        )
    
    def create_retriever_stupid_22(self,question):
        """_summary_

        Ask question on chat engine 2

        Returns:
            _type_: _string_
        """
        response_stream = self.chat_engine_2.stream_chat(question)
        response = ''
        response_source = 'Nguồn tài liệu liên quan: \n'
        for token in response_stream.response_gen:
            print(token, end="", flush=True)
            response += token
        for node in response_stream.source_nodes:
            print("\n", node.metadata['url'])
        return response,response_source

    def create_retriever_stupid_2(self,question):
        """_summary_

        Ask question on chat engine 2

        Returns:
            _type_: _string_
        """
        response_stream = self.chat_engine_2.stream_chat(question)
        yield "Tài liệu liên quan: \n"
        for node in response_stream.source_nodes:
            yield node.metadata['url']  + "\n"
        for text in response_stream.response_gen:
            yield text
        
        
    
    #another method implement for retriever instead of chat engine
    def retriever_query_engine(self, k=3,ivfflat_probes=2,hnsw_ef_search=40):
        """_summary_
        Using hybrid search in pgvector for searching retriever from database
        
        Returns:
            RetrieverQueryEngine: _object_
        """
        retriever_ = VectorIndexRetriever(
            index=self.index,
            vector_store_query_mode="default",
            similarity_top_k=k,
            # vector_storage_kwargs={"ivfflat_probes":ivfflat_probes,"hnsw_ef_search": hnsw_ef_search},
        )
        #adding some postprocessor
        similar_cutoff = SimilarityPostprocessor(similarity_cutoff=0.6)
        # sentence_optimizer = SentenceEmbeddingOptimizer(percentile_cutoff=0.3)
        # re_ranker = LLMRerank(choice_batch_size=3,top_n = 2,service_context=self.service_context)
        query_engine_ = RetrieverQueryEngine(
            retriever=retriever_,
            node_postprocessors=[similar_cutoff],
            response_synthesizer=get_response_synthesizer(response_mode="tree_summarize",streaming=True),
        )
        return query_engine_
    
    
    def hybrid_retriever_engine(self, question,):
        """_summary_
        Using hybrid search in pgvector for searching retriever from database
        
        Returns:
            full_answer: _string_
            full_source: _string_
        """
        response =  self.hybrid_engine.query(question)
        yield "Tài liệu liên quan: \n"
        for node in response.source_nodes:
            yield node.metadata['url']  + "\n"
        for text in response.response_gen:
            yield text
        
    
    def prompt_format(self,engine):
        template_en = (
            "You are a Vinmec chatbot that answer question about health.\n"
            "You using the following information to answer the question.\n"
            "We have the following text.\n"
            "---------------------\n"
            "{context_str}"
            "\n---------------------\n"
            "Assess the relevance of the query to the text.\nIf relevant:\n" 
            "Use the information from the text, regardless of the priority of the information.\n"
            "If not relevant:\n"
            "Respond with `Tôi hiện chưa được cập nhật thông tin này.`\n"
            "All text NEED TO BE IN VIETNAMESE INCLUDING YOUR ANSWER.\n"
            "Always sugestion user go to the Vinmec hospital for more information.\n"
            "Long answer for detail information if needed."
            "Please answer like a human.\n"
            "Question: {query_str}\n"
            "Answer: "
        )
        qa_template = PromptTemplate(template_en)
        engine.update_prompts(
            {"response_synthesizer:summary_template": qa_template}
        )
        return engine
        
    def behavior_controller(self, question,rag_type):
        """_summary_
        Behavior controller for chatbot

        Returns:
            _type_: _string_
        """
        behavior_prompt = """
        You are a helpful assistant that determines the type of query and response.  
        Always respond in Vietnamese.
        If the query is a greeting: Respond with a friendly greeting to the user.
        If the query is a thank you: Respond with a friendly thank you to the user.  
        If the query is about a health issue: Respond with `SEARCH + short the main task of query need to answer`.
        If the query is asking who you are: Respond you are chatbot assistant of Vinmec hospital.
        If the query contain subquery: Respond `SEARCH + combine all to one query`.
        If the query contain greeting and subquery: Respond `SEARCH + combine all to one query`.
        Query: {query}
        """
        query_ = PromptTemplate(behavior_prompt)
        response = self.llm.predict(query_, query= question)
        
        behavior = response.split("\n")[-1]
        print(behavior)
        if "SEARCH" in behavior:
            return rag_type(behavior.replace("SEARCH ",""))
        else:
            return behavior

    def init_index1_and_title(self):
        connection_string = st.secrets['connection_string']
        conn = psycopg2.connect(connection_string)
        conn.autocommit = True
        url = make_url(connection_string)
        vector_store = PGVectorStore.from_params(
            database=url.database,
            host=url.host,
            password=url.password,
            port=url.port,
            user=url.username,
            table_name="vinmec_retriever_method_1",
            embed_dim=1536, #openai embedding dim
            
        )

        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex.from_vector_store(vector_store=storage_context.vector_store)
        
        #get title list
        query = "SELECT text,metadata_::json->>'url' from data_vinmec_retriever_method_1"

        with self.conn.cursor() as cur:
            list_title = []
            cur.execute(query)
            for row in cur:
                list_title.append((row[1],row[0]))
                
        return index,list_title

    def init_customRV(self):
        vector_retriever_1 = VectorIndexRetriever(self.index_1,similarity_top_k=3)
        vector_retriever_2 = VectorIndexRetriever(self.index,similarity_top_k=3)
        return CustomRetriever(vector_retriever_1,vector_retriever_2)

    
    def get_customRV(self,question):
        ans = self.custom_rv.retrieve(question)
        list_link = []
        for a in ans:
            # print(a.metadata['url'])
            list_link.append(a.metadata['url'])
        
        title = []
        for i in self.list_title:
            if i[0] in list_link:
                title.append(i)
        
        title_str = "".join([f"{i + 1}. {value[1]}\n" for i, value in enumerate(title)])  
        return title,title_str
    
    def decide_index_retriever(self,question,title_str):
        query_gen_str = """
        You are a helpful assistant that determines the index relative to query.  
        Maybe some index contain a little the information about the query, make sure you read it carefully.
        If you think it is useful for you, you can select it. No need to explain.
        Always respond index number.
        Example:
        Query: What is the symptom of covid?
        Information:
        1. Covid is a disease caused by SARS-CoV-2 virus.
        2. The most common symptoms of COVID-19 are fever, dry cough, and tiredness.
        3. covid is done in 2023
        4. Some people become infected.
        Selected index: 1,2,4
        If there noone index relative then return `None`.
        Query: {query}\n
        Information: \n{infor}\n
        Selected index:
        """
        gen = query_gen_str.format(query=question,infor=title_str)
        print(gen)
        query_gen_prompt = PromptTemplate(gen)
        llm = OpenAI(model="gpt-3.5-turbo-1106")
        # response = llm.predict(query_gen_prompt, query= query,infor = infor)
        response = llm.predict(query_gen_prompt)
        
        return response
    
    def get_index(self,answer,title):
        if "Selected index" in answer:
            answer = answer.split("Selected index:")[1].strip()
        if answer == "None":
            return None
        index = [int(i) for i in answer.split(",")]
        link_selected = [title[i-1][0] for i in index]
        return link_selected
    
    def create_custom_rv(self,question):
        title,title_str = self.get_customRV(question)
        answer = self.decide_index_retriever(question,title_str)
        link_selected = self.get_index(answer,title)
        if link_selected is None:
            return "Tôi hiện chưa được cập nhật thông tin này."
        else:
            return link_selected