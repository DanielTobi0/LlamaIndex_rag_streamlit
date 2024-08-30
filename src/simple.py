from pathlib import Path
import os, time
from exceptions.operations_handler import system_logger, userops_logger, llmresponse_logger
import chromadb
from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    Settings,
    StorageContext,
    PromptTemplate,
    get_response_synthesizer,
    load_index_from_storage,
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.groq import Groq
from llama_index.core.llms import ChatMessage
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor



# LLM settings
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

def init_llm(model:str, temperature:float):
    """
    Initalize LLM and embedding models.
    Args:
        model(str): model name
        temperature(float): model experesiveness
    return:
        None
    """
    Settings.llm = Groq(model, request_timeout=60.0, api_key=GROQ_API_KEY, temperature=temperature)
    Settings.embed_model = HuggingFaceEmbedding(model_name='TaylorAI/bge-micro-v2')
    system_logger.info(f'Embedding and LLM model loaded')


def init_retriever():
    """Embed documents and return a query engine for retrieval."""
    db = chromadb.PersistentClient(path="./chroma_db")
    collection_name = 'quickstart'
    chroma_collection = db.get_or_create_collection(collection_name)

    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    documents = SimpleDirectoryReader("../data").load_data()
    Settings.chunk_size = 100
    Settings.chunk_overlap = 30
    system_logger.info(f'Document created with {len(documents)} chunk(s)')

    index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)
    # chat_engine = index.as_chat_engine(verbose=False)
    retriever = VectorIndexRetriever(index=index, similarity_top_k=5)

    template = (
        "Your goal is to provide insightful, accurate, and concise answers to questions in this domain.\n\n"
        "Here is some context related to the query:\n"
        "------------------------------------------\n"
        "{context}\n"
        "------------------------------------------\n"
        "Considering the above information, please respond to the following inquiry with detailed references to applicable laws, "
        "precedents, or principles where appropriate:\n\n"
        "Question: {query_str}\n\n"
    )

    # https://docs.llamaindex.ai/en/v0.10.17/api_reference/query/query_engines/retriever_query_engine.html
    query_engine = RetrieverQueryEngine(retriever=retriever) # to add template here
    system_logger.info('Query engine and index created\n\n')
    return query_engine


def generate_response(query_engine, query:str):
    """
    Generate a response to a query using the query engine.
    Args:
        query_engine: ...
        query(str): user prompt
        return response (str): user response
    """
    response = query_engine.query(query)
    llmresponse_logger.info(f'Query: {query} \nresponse: {response}\n')
    return response

models = [
    "llama-3.1-405b-reasoning",
    "llama-3.1-70b-versatile",
    "llama-3.1-8b-instant",
    "mixtral-8x7b-32768",
    "gemma2-9b-it"
    ]

def generate(query, model, temperature):
    init_llm(model=model, temperature=temperature)
    query_engine = init_retriever()
    llm_response = generate_response(query_engine, query)
    llmresponse_logger.info(f'Query: {query} \nresponse: {llm_response}\n')
    
    for word in str(llm_response).split():
        yield word + " "
        time.sleep(0.05)

def generate_no_stream(query, model, temperature):
    init_llm(model=model, temperature=temperature)
    query_engine = init_retriever()
    llm_response = generate_response(query_engine, query)
    llmresponse_logger.info(f'Query: {query} \nresponse: {llm_response}\n')
    return llm_response.response


if __name__ == '__main__':
    model = "gemma2-9b-it"
    query = 'Tell me about Emmanuel please'
    temperature = 0.1
    print(generate_no_stream(query=query, model=model, temperature=temperature))