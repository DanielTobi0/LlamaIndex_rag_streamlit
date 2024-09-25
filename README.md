# Mid-Camp Project: Retrival Agumented Generation System for Recuriters

## Description

This project is designed to ease the job application process by helping recruiters quickly go through many resumes, in a chat like manner. With the RAG system, recruiters can find resumes that match their queries and interact with the resumes in a conversational way. The system makes it easier for recruiters to get the information they need and makes the recruitment process smoother.

The system uses **FastAPI** as the backend for handling API requests, **Streamlit** for the frontend interface, and integrates several libraries for embeddings, vector storage and model calling.

## Tech Stack

- **Python 3.12**
- **FastAPI** - Backend API for handling user queries and returning AI-generated responses.
- **Streamlit** - Web-based frontend for interacting with the RAG system.
- **ChromaDB** - Vector database for document embeddings.
- **Groq API** - For accessing pre-trained text generation language models.
- **llama-index** - For document indexing and querying.
- **HuggingFace** - For embeddings via the HuggingFace embedding model.

## How to run locally

### Prerequisites

- Install **Python 3.12**.
- Install dependencies using `pip install -r requirements.txt`.
- Set up your **Groq API key** as an environment variable.

### Running the code

You can run this project in several ways

### 1. python script

cd src
python simple.py --model "model_name" --query "your query" --temperature float()

### 2. local fastapi

cd src
uvicorn app:app --host 127.0.0.1 --port 8000 --reload

To interact with the API

- a. set postman to Post method
- b. set url to http://127.0.0.1:8000/chat
- c. sample request body
  {
  "query": "query",
  "model": "model_name",
  "temperature": float()
  }

### 3. streamlit

- FastAPI must be running locally for Streamlit to function.
- Run the Streamlit app
  `streamlit run streamlit_app.py`
- Once launched, upload document(s), click to save, and generate embeddings before submitting your queries.
- Upload document(s), click to save to generate embeddings before sending in querys.

## Codebase Structure

- **simple.py**: Inializing the model, embedding document(s), retrieving documents, generating responses and evaluating the RAG system.
- **app.py**: Backend API using FastAPI that exposes endpoints for querying the model.
- **streamlit_app.py**: Streamlit frontend where users can upload resumes, select models, and interact with the AI system.

## Available models

The following models are accessed via the Groq API:

- gemma2-9b-it
- llama-3.1-70b-versatile
- llama-3.1-8b-instant
- mixtral-8x7b-32768

## Future work

Improve document indexing.
