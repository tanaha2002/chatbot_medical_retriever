import os
import numpy as np
import re
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
from concurrent.futures import ThreadPoolExecutor

from llama_index.indices.postprocessor import SimilarityPostprocessor
from llama_index.indices.postprocessor import SentenceEmbeddingOptimizer
import json
from CustomRV import CustomRetriever
from llama_index.schema import NodeWithScore,TextNode,NodeRelationship,ObjectType,RelatedNodeInfo
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
        # self.node_data = self.get_whole_node()
        # self.node_data = None
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
                # hybrid_search=hybrid_search,
                # text_search_config=text_search_config,
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
        # similar_cutoff = SimilarityPostprocessor(similarity_cutoff=0.6)
        # sentence_optimizer = SentenceEmbeddingOptimizer(percentile_cutoff=0.3)
        # re_ranker = LLMRerank(choice_batch_size=3,top_n = 2,service_context=self.service_context)
        query_engine_ = RetrieverQueryEngine(
            retriever=retriever_,
            # node_postprocessors=[similar_cutoff],
            response_synthesizer=get_response_synthesizer(response_mode="tree_summarize",streaming=True),
        )
        return query_engine_
    
    
    def hybrid_retriever_engine(self, question):
        """_summary_
        Using hybrid search in pgvector for searching retriever from database
        
        Returns:
            full_answer: _string_
            full_source: _string_
        """
        response =  self.hybrid_engine.query(question)
        # print(f'-------{response}-------')
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
        llm = OpenAI(model="gpt-3.5-turbo-1106",temperature=0.55)
        query_ = PromptTemplate(behavior_prompt)
        response = llm.predict(query_, query= question)
        behavior = response.split("\n")[-1]
        if "SEARCH" in behavior:
            return rag_type(behavior.replace("SEARCH ",""))
        else:
            return behavior
    
    
    def search_link(self, link):
        query = """SELECT text
                    FROM data_vinmec_retriever_method_1
                    where metadata_::text ilike %s
                    """
        conn = psycopg2.connect(self.connection_string)
        cur = self.conn.cursor()
        cur.execute(query, (link,))
        content = cur.fetchall()
        cur.close()
        return content
    

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

    
    
    
    def decide_index_retriever(self,question,title_str):
        
        query_gen_str = """
        Bạn là người trợ giúp hữu ích trong việc giúp xác định các mục trong danh sách dưới đây tương ứng với cùng loại bệnh như trong truy vấn của tôi.
        Hãy chắc chắn rằng bạn đọc tất cả chúng một cách cẩn thận.
        Vui lòng xem xét danh sách các mô tả bệnh tật và chỉ ra các mục mô tả cùng loại bệnh như trong câu hỏi của tôi.
        Chỉ mục được chọn bao gồm các mục mô tả các triệu chứng hoặc vấn đề liên quan đến cùng loại bệnh như được đề cập trong truy vấn. Xác định tất cả các chỉ số có liên quan.
        Không cần phải giải thích.
        Luôn trả lời số chỉ mục.
        Ví dụ:
        Hỏi: Chăm sóc và điều trị bong gân cổ?
        Thông tin:
        1. Bị bong gân cổ phải làm sao?
        2. Trẻ mới biết đi từ 2-4 tuổi có bị trầm cảm không?
        3. Hướng dẫn chăm sóc trẻ bị ho
        4. Trẻ rất nhạy cảm
        5. Làm thế nào để giảm căng cơ cổ khi ngủ?
        6. Điều trị và chăm sóc trẻ nhỏ bị viêm phổi
        Chỉ số đã chọn: 1, 5
        Nếu không có chỉ mục tương đối thì trả về `None`.
        Truy vấn: {query}\n
        Thông tin: \n{infor}\n
        Chỉ số đã chọn:
        """

        gen = query_gen_str.format(query=question,infor=title_str)
        # print(gen)
        query_gen_prompt = PromptTemplate(gen)
        llm = OpenAI(model="gpt-3.5-turbo-1106")
        # response = llm.predict(query_gen_prompt, query= query,infor = infor)
        response = llm.predict(query_gen_prompt)
        
        return response
    
    def google_search(self, query):
        headers = {
            'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        }
        list_url = []
        title = []
        # Tạo một yêu cầu GET đến trang tìm kiếm của Google
        url = f"https://www.google.com/search?q={query.replace(' ', '%') + '%site:vinmec.com'}"
        # print(url)
        response = requests.get(url, headers=headers)
        # print(response.status_code)

        # Kiểm tra xem yêu cầu có thành công hay không (status code 200 là thành công)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            taget = soup.find_all('div', class_='MjjYud')
            for _ in taget:
                link_ = _.find('a').get('href')
                list_url.append(link_)
        
        for item in list_url[0:3]:
            content = ''
            if 'vinmec.com' in item:
                soup_ = BeautifulSoup(requests.get(item).text, 'html.parser')
                content = soup_.find('h1').text.strip()
                element = soup_.find_all('h2')
                h2 = '; '.join([_.text.strip() for _ in element])
                content += '; ' + h2
                title.append([item, content])
        title_str = "".join([f"{i + 1}. {value[1]}\n" for i, value in enumerate(title)])

        return title, title_str
    
    
    def get_index(self,answer,title):
        if "Selected index" in answer:
            answer = answer.split("Selected index:")[1].strip()
        if answer == "None":
            return None
        index = [int(i) for i in answer.split(",")]
        link_selected = [title[i-1][0] for i in index]
        return link_selected
    
    def get_customRV(self,question):
        self.title_nodes,self.content_nodes = self.custom_rv.retrieve(question)
        temporary_url_title_nodes = [node.metadata['url'] for node in self.title_nodes]
        temporary_url_content_nodes = []
        for node in self.content_nodes:
            temporary_url_content_nodes.append(node.metadata['url'])
        title = []
        for i in self.list_title:
            if i[0] in temporary_url_content_nodes or i[0] in temporary_url_title_nodes:
                title.append(i)
        title_str = "".join([f"{i + 1}. {value[1]}\n" for i, value in enumerate(title)])  
        # return 
        return title,title_str
    
    def create_custom_rv(self,question):
        # title,title_str = self.get_customRV(question)
        # answer = self.decide_index_retriever(question,title_str)
        # link_selected = self.get_index(answer,title)
        title, title_str = self.google_search(question)
        link_selected = [item[0].replace('/vi', '') for item in title]
        if link_selected is None:
            return None
        else:
            return link_selected
        
        
    #----------------PROCESS NODE FOR CUSTOM RETRIEVER----------------#
    
    def struct_create_node(self,node_list_):
        node_list = []
        for i in range(len(node_list_)):
                data_dict = json.loads(node_list_[i][1]['_node_content'])
                node=TextNode(
                    id_=data_dict['id_'],
                    embedding=data_dict['embedding'],
                    text=node_list_[i][0],
                    metadata=data_dict['metadata'],
                    excluded_embed_metadata_keys=data_dict['excluded_embed_metadata_keys'],
                    excluded_llm_metadata_keys=data_dict['excluded_llm_metadata_keys'],
                    hash=data_dict['hash'],
                )
                node.relationships[NodeRelationship.SOURCE] = RelatedNodeInfo(
                    node_id=data_dict['relationships'][NodeRelationship.SOURCE]['node_id'],
                    node_type=ObjectType(data_dict['relationships'][NodeRelationship.SOURCE]['node_type']),
                    metadata=data_dict['relationships'][NodeRelationship.SOURCE]['metadata'],
                    hash=data_dict['relationships'][NodeRelationship.SOURCE]['hash'],
                    class_name=data_dict['relationships'][NodeRelationship.SOURCE]['class_name'],
                )
                node.relationships[NodeRelationship.PREVIOUS] = RelatedNodeInfo(
                    node_id=data_dict['relationships'][NodeRelationship.PREVIOUS]['node_id'],
                    node_type=ObjectType(data_dict['relationships'][NodeRelationship.PREVIOUS]['node_type']),
                    metadata=data_dict['relationships'][NodeRelationship.PREVIOUS]['metadata'],
                    hash=data_dict['relationships'][NodeRelationship.PREVIOUS]['hash'],
                    class_name=data_dict['relationships'][NodeRelationship.PREVIOUS]['class_name'],
                )
                node.relationships[NodeRelationship.NEXT] = RelatedNodeInfo(
                    node_id=data_dict['relationships'][NodeRelationship.NEXT]['node_id'],
                    node_type=ObjectType(data_dict['relationships'][NodeRelationship.NEXT]['node_type']),
                    metadata=data_dict['relationships'][NodeRelationship.NEXT]['metadata'],
                    hash=data_dict['relationships'][NodeRelationship.NEXT]['hash'],
                    class_name=data_dict['relationships'][NodeRelationship.NEXT]['class_name'],
                )
                node_list.append(node)
        return node_list
    
    
    def get_final_list_nodes(self,link_selected):
        #check if node selected is in title or content
        # self.selected_node = []
        # self.need_to_search = []
        # print(link_selected)
        # for link in link_selected:
        #     if link in [node.metadata['url'] for node in self.content_nodes]:
        #         self.selected_node.append([node for node in self.content_nodes if node.metadata['url'] == link][0])
        #     else:
        #         self.need_to_search.append(link)
        # search_nodes = []
        
        with self.conn.cursor() as cur:
            for link in link_selected:
                query = "SELECT text,metadata_,node_id FROM data_vinmec_storage_index  WHERE  metadata_::json->>'url' = %s"
                cur.execute(query,(link,))
                search_nodes = cur.fetchall() # search_nodes chỉ lấy được các node của link cuối cùng, không lấy những link trước.
                
        # print(search_nodes)
        #convert to node
        # convert_node = self.struct_create_node(search_nodes)
        # final_list_nodes = self.selected_node + convert_node
        final_list_nodes = self.struct_create_node(search_nodes)
        # print(final_list_nodes)
        
        return final_list_nodes
    

    def pipeline_custom_retriever(self,question):
        link_selected = self.create_custom_rv(question)
        final_list_nodes = self.get_final_list_nodes(link_selected)
        storage_instance = StorageContext.from_defaults(docstore=self.index.docstore)
        instance_vectorindex = VectorStoreIndex(final_list_nodes,storage_context=storage_instance)
        instance_search = VectorIndexRetriever(instance_vectorindex,similarity_top_k=3)

        query_engine = RetrieverQueryEngine(
            retriever=instance_search,response_synthesizer=get_response_synthesizer(response_mode="tree_summarize",streaming=True)
            )

        query_engine = self.prompt_format(query_engine)
        response = query_engine.query(question)
        yield "Tài liệu liên quan: \n"
        for node in response.source_nodes:
            yield node.metadata['url']  + "\n"
        for text in response.response_gen:
            yield text
        
        
        
        
        
        #-----None of custom retriever above work well so i decide to use back hybrid and just remove duplicate and non-relevant url-----#
        
        
        
        
        
    
    
    