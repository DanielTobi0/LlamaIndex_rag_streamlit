import streamlit as st
from simple import generate, process_query
from exceptions.operations_handler import system_logger, userops_logger, llmresponse_logger
import requests


st.title("Local RAG with Groq API")

# file upload functionality - yet to be integrated
st.sidebar.file_uploader(
    label='Upload documents',
    accept_multiple_files=True,
    key='file_uploader',
    )

# Model selection
model_select = st.sidebar.selectbox(
    'Select model',
    options=[
        "gemma2-9b-it",
        "llama-3.1-70b-versatile",
        "llama-3.1-8b-instant",
        "mixtral-8x7b-32768"
    ],
    index=0
)
st.session_state.model_name = model_select

# Temperature selection
temperature_select = st.sidebar.slider(
    'Temperature value',
    min_value=0.1, max_value=1.0, step=0.1, value=0.5
)
st.session_state.temperature = temperature_select


# initialize messages
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# handle user input
prompt = st.chat_input(placeholder='Message Groq')
if prompt:
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    userops_logger.info(f'User query: {prompt}\n\n')

    backend_url = "http://127.0.0.1:8000/chat"
    with st.spinner("Waiting for response..."):
        response = requests.post(backend_url, json={"query": prompt, "model": st.session_state.model_name, "temperature": st.session_state.temperature})

    if response.status_code == 200:
        bot_response = response.json().get("response", "")
        st.session_state.messages.append({"role": "assistant", "content": bot_response})
        llmresponse_logger.info(f'Groq response: {bot_response}')
        st.markdown(bot_response)

    else:
        st.error(f"Failed to get a response from the backend. Status code: {response.status_code}")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])