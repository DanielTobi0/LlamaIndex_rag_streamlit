import os, time
from exceptions.operations_handler import system_logger, llmresponse_logger, userops_logger
import chromadb
from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    Settings,
    StorageContext,
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.groq import Groq
import argparse
from pathlib import Path
import shutil


# LLM settings
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


def init_llm(llm_model: str, llm_temperature: float):
    """
    Initialize LLM and embedding models.
    Args:
        llm_model(str): model name
        llm_temperature(float): model expressiveness
    return:
        None
    """
    Settings.llm = Groq(llm_model, request_timeout=60.0, api_key=GROQ_API_KEY, temperature=llm_temperature)
    Settings.embed_model = HuggingFaceEmbedding(model_name='TaylorAI/bge-micro-v2')
    system_logger.info(f'Embedding and LLM model loaded')


def init_retriever():
    """Embed documents and return a query engine for retrieval."""
    chromadb_path = './chroma_db'
    db = chromadb.PersistentClient(path=chromadb_path)
    collection_name = 'quickstart'
    chroma_collection = db.get_or_create_collection(collection_name)

    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    documents = SimpleDirectoryReader("../data").load_data()
    Settings.chunk_size = 100
    Settings.chunk_overlap = 30
    system_logger.info(f'Document created with {len(documents)} chunk(s)')

    
    SYSTEM_PROMPT = (
        "You are an AI assistant designed to help recruiters by answering questions based on a provided resume.\n\n"
        "The resume contains personal, educational, and professional details about a candidate. Use the information from the resume to answer questions accurately and concisely.\n"
        "If the question is outside the scope of the resume or requires subjective judgement, clarify this to the recruiter.\n"
        "Always ensure that your responses are relevant to the details provided in the resume.\n"
        "Below is some context related to the resume:\n"
        "--------------------------------------------\n"
        "{context}\n"
        "--------------------------------------------\n"
        "Considering the above information, please respond to the following query from the recruiter.\n\n"
        "Question: {query_str}\n\n"

    )
    index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)
    chat_engine = index.as_chat_engine(
        chat_mode='context',
        system_prompt=SYSTEM_PROMPT,
        similarity_top_k=3,
        verbose=False

    )
    # https://docs.llamaindex.ai/en/v0.10.17/api_reference/query/query_engines/retriever_query_engine.html
    system_logger.info('Query engine and index created\n\n')
    return chat_engine


def generate_response(query_engine, user_query: str):
    """
    Generate a response to a query using the query engine.
    Args:
        query_engine: ...
        user_query(str): user prompt
        return response (str): user response
    """
    response = query_engine.chat(user_query)
    llmresponse_logger.info(f'Query: {user_query} \nresponse: {response}\n')
    return response.response


models = [
    "llama-3.1-70b-versatile",
    "llama-3.1-8b-instant",
    "mixtral-8x7b-32768",
    "gemma2-9b-it"
]


def generate(query, model, temperature):
    init_llm(llm_model=model, llm_temperature=temperature)
    query_engine = init_retriever()
    llm_response = generate_response(query_engine, query)
    llmresponse_logger.info(f'Query: {query} \nresponse: {llm_response}\n')

    for word in str(llm_response).split():
        yield word + " "
        time.sleep(0.05)


def process_query(query, model, temperature):
    init_llm(llm_model=model, llm_temperature=temperature)
    query_engine = init_retriever()
    llm_response = generate_response(query_engine, query)
    llmresponse_logger.info(f"Query: {query} \nresponse: {str(llm_response)}\n")
    return llm_response


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process a query using a specified model.")
    parser.add_argument("--model", type=str, required=True, help="The model to use.")
    parser.add_argument("--query", type=str, required=True, help="The query to process")
    parser.add_argument("--temperature", type=float, default=0.1, help="The model temperature for output experessiveness")
    
    args = parser.parse_args()
    print(process_query(query=args.query, model=args.model, temperature=args.temperature))

# python script_name.py --model "gemma2-9b-it" --query "Tell me about Daniel Tobi" --temperature 0.1