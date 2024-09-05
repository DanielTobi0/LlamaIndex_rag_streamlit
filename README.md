# Mid-Camp Project: RAG System for a Recuriter Use Case

## The objective of this project is to make job applications easier, where recuriters can collect wide range of resumes, chat the llm to perform wide range of tasks on them, example 1. Chat with a specific resume 2. Filter resume based on recuiters query.

The code can be run as a 
1. python script (simple.py), 
2. local fastapi (app.py)
3. streamlit (streamlit_app.py)
4. online streamlit deployment (url)


### Models
Groq models are called via API, the used models are 
"gemma2-9b-it",
"llama-3.1-70b-versatile",
"llama-3.1-8b-instant",
"mixtral-8x7b-32768"

## Embeddings
Chromadb was used at the vector database

## Frameword used
Llama index
FastAPI
Streamlit


## How to use
simple.py
Can be ran as a script, and pass the arguments
1. cd src
2. python script_name.py --model "gemma2-9b-it" --query "Tell me about Daniel Tobi" --temperature 0.1

app.py (FastAPI)
1. cd src
2. uvicorn app:app --host 127.0.0.1 --port 8000 --reload

streamlit_app.py
1. Make sure to be in the main folder (parent)
2. Parameter to pass  query, model and temperature
{
        "query": "Tell me about the applicant Daniel Tobi",
        "model": "gemma2-9b-it",
        "temperature": 0.1
}