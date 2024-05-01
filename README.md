# Medical Retrieval-Augmented Generation (RAG) Chatbot

This repository contains the code and resources for a Retrieval-Augmented Generation (RAG) chatbot that assists users in asking questions related to medical health. The chatbot leverages a large language model (LLM) and a vector database of medical information to provide accurate and relevant responses.

## Project Overview

The project consists of the following main components:

1. **Data Acquisition and Processing**: Medical data was crawled from Vinmec, a healthcare provider. The raw text was processed and structured for efficient storage and retrieval.

2. **Vector Database**: A vector database was built using the processed medical data, enabling fast and accurate semantic searching. The vector database was stored in PostgreSQL (pg_vector) for persistence and scalability.

3. **RAG Pipeline**: A Retrieval-Augmented Generation (RAG) pipeline was implemented, combining the vector database and a large language model (ChatGPT 3.5) to generate contextually relevant responses to user queries.

4. **Advanced Techniques**: The RAG pipeline was enhanced with various advanced techniques, including re-ranking, chain of thought prompting, and hybrid search, to improve the quality and relevance of the generated responses.

5. **Inference and Deployment**: The complete RAG pipeline was optimized and deployed to a production environment, allowing users to interact with the medical chatbot seamlessly.

## Demo
