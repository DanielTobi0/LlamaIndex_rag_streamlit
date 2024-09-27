import os, time, asyncio, yaml, chromadb, argparse
from dotenv import load_dotenv
from exceptions.operations_handler import system_logger, llmresponse_logger, userops_logger, evalresponse_logger
from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    Settings,
    StorageContext,
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.groq import Groq
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.evaluation import FaithfulnessEvaluator, RelevancyEvaluator, BatchEvalRunner
from llama_index.core.llama_dataset.generator import RagDatasetGenerator


with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)


def init_llm(llm_model: str, llm_temperature: float):
    """
    Initialize LLM and embedding models.
    Args:
        llm_model(str): model name
        llm_temperature(float): model expressiveness
    return:
        None
    """
    load_dotenv()
    GROQ_API_KEY = os.getenv("Groq_API")

    Settings.llm = Groq(llm_model, request_timeout=config['request_timeout'], api_key=GROQ_API_KEY,
                        temperature=llm_temperature)
    Settings.embed_model = HuggingFaceEmbedding(config['embed_model'])


def init_retriever():
    """Embed documents and return a query engine for retrieval."""
    chromadb_path = config['chromadb_path']
    db = chromadb.PersistentClient(path=chromadb_path)
    collection_name = config['collection_name']
    chroma_collection = db.get_or_create_collection(collection_name)

    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    documents = SimpleDirectoryReader(config["documents_path"]).load_data()
    Settings.chunk_size = config['Settings_chunk_size']
    Settings.chunk_overlap = config['Settings_chunk_overlap']

    
    SYSTEM_PROMPT = (
        "You are an AI assistant designed to help recruiters by answering questions based on a provided resume.\n\n"
        "The resume contains personal, educational, and professional details about a candidate. Use the information from the resume to answer questions accurately and concisely.\n"
        "If the question is outside the scope of the resume or requires subjective judgement, clarify this to the recruiter.\n"
        "Always ensure that your responses are relevant to the details provided in the resume.\n"
        "You are to sound in a first person view, like a recuriter chatting with the interviewee\n"
        "Below is some context related to the resume:\n"
        "--------------------------------------------\n"
        "{context}\n"
        "--------------------------------------------\n"
        "Considering the above information, please respond to the following query from the recruiter.\n\n"
        "Question: {query_str}\n\n"
    )
    SYSTEM_PROMPT = """
    You are an AI assistant designed to support recruiters by providing answers based on the candidate's resume.

    The resume contains personal, educational, and professional details. Use this information to answer questions clearly and concisely. If a question extends beyond the resume's scope or requires subjective judgement, inform the recruiter accordingly.
    
    Ensure all responses are directly relevant to the details within the resume, and always maintain a conversational tone, as if you're a recruiter engaging with the candidate.
    
    Below is the context from the resume:
    {context}
    Based on the provided details, please respond to the following query from the recruiter:
    
    Question: {query_str}
    """
    index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)
    memory = ChatMemoryBuffer.from_defaults(token_limit=config['token_limit'])
    chat_engine = index.as_chat_engine(
        chat_mode=config['chat_mode'],
        system_prompt=SYSTEM_PROMPT,
        similarity_top_k=config['similarity_top_k'],
        verbose=False,
        memory=memory
    )
    return chat_engine, index


def generate_response(query_engine, user_query: str):
    """
    Generate a response to a query using the query engine.
    Args:
        query_engine: ...
        user_query(str): user prompt
        return response (str): user response
    """
    response = query_engine.chat(user_query)
    return response.response


def evaluating_llm_response(vector_index, query):
    """
    Evaluating LLM response, based on faithfulness and relevancy.
    """
    llm = Groq(model=config['llm_model'], request_timeout=config['request_timeout'], api_key=GROQ_API_KEY,
               temperature=0.0)
    faithfulness_evaluator = FaithfulnessEvaluator(llm)
    query_engine = vector_index.as_query_engine()
    response = query_engine.query(query)
    faithfulness_result = faithfulness_evaluator.evaluate_response(response=response)

    relevancy_evaluator = RelevancyEvaluator(llm)
    response_relevancy = query_engine.query(query)
    result_relevancy = relevancy_evaluator.evaluate_response(query, response_relevancy)
    evalresponse_logger.info(
        f"-----Query: {query}-----Evaluation match: {str(faithfulness_result.passing)}-----\n-----Query: {query}-----Evaluating Query + Response Relevancy: {result_relevancy}\n\n")


def batch_evaluation():
    load_dotenv()
    GROQ_API_KEY = os.getenv("Groq_API")

    llm = Groq(model=config['llm_model'], request_timeout=60.0, api_key=GROQ_API_KEY, temperature=0.0)
    Settings.llm = llm
    embed_model = HuggingFaceEmbedding("TaylorAI/bge-micro-v2")
    Settings.embed_model = embed_model
    documents = SimpleDirectoryReader(config['documents_path']).load_data()
    index = VectorStoreIndex.from_documents(documents)

    dataset_generator = RagDatasetGenerator.from_documents(
        documents=documents,
        num_questions_per_chunk=1
    )
    rag_dataset = dataset_generator.generate_dataset_from_nodes()
    questions = [e.query for e in rag_dataset.examples]

    # model, faith and relevancy.
    faithfulness_evaluator = FaithfulnessEvaluator(llm)
    relevancy_evaluator = RelevancyEvaluator(llm)

    runner = BatchEvalRunner(
        {
            'faithfulness': faithfulness_evaluator,
            'relevancy': relevancy_evaluator
        },
        workers=8,
    )
    return runner, questions, index

async def run_with_delay(runner, vector_index, questions, delay_seconds):
    results = []
    for question in questions:
        result = await runner.aevaluate_queries(
            vector_index.as_query_engine(), queries=[question]
        )
        results.append(result)
        await asyncio.sleep(delay_seconds)
    return results

def get_eval_results(key, eval_results):
    correct = 0
    total = 0
    for result_set in eval_results:
        if key in result_set:  # Ensure the key exists in this result set
            results = result_set[key]
            total += len(results)
            for result in results:
                if result.passing:
                    correct += 1
    score = correct / total if total > 0 else 0
    print(f"{key} Score: {score}")
    return score

# runner, questions, index = batch_evaluation()
# eval_results = asyncio.run(run_with_delay(runner=runner, vector_index=index, questions=questions, delay_seconds=60))
# faithfulness_score = get_eval_results("faithfulness", eval_results)
# relevancy_score = get_eval_results("relevancy", eval_results)
# evalresponse_logger.info(f"Faithfullness score: {faithfulness_score}, Relevancy score: {relevancy_score}")


def process_query(query, model, temperature):
    init_llm(llm_model=model, llm_temperature=temperature)
    query_engine, index = init_retriever()
    # evaluating_llm_response(index, query)
    llm_response = generate_response(query_engine, query)
    # llmresponse_logger.info(f"Query: {query} \nresponse: {str(llm_response)}\n")
    return llm_response


if __name__ == '__main__':
    models = [
    "llama-3.1-70b-versatile",
    "llama-3.1-8b-instant",
    "mixtral-8x7b-32768",
    "gemma2-9b-it"
    ]
    parser = argparse.ArgumentParser(description="Process a query using a specified model.")
    parser.add_argument("--model", type=str, required=True, help="The model to use.")
    parser.add_argument("--query", type=str, required=True, help="The query to process")
    parser.add_argument("--temperature", type=float, default=0.1,
                        help="The model temperature for output expressiveness")

    args = parser.parse_args()
    process_query(query=args.query, model=args.model, temperature=args.temperature)
# python simple.py --model "gemma2-9b-it" --query "Tell me about Daniel Tobi" --temperature 0.1
