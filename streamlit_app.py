import streamlit as st
import os, time, requests
from src.exceptions.operations_handler import userops_logger, llmresponse_logger

UPLOAD_DIRECTORY = "../data"
if not os.path.exists(UPLOAD_DIRECTORY):
    os.makedirs(UPLOAD_DIRECTORY)


def save_uploaded_file(uploaded_file):
    if uploaded_file is not None:
        file_path = os.path.join(UPLOAD_DIRECTORY, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        return True
    return False


st.title("Local RAG with Groq API")

# File upload functionality
with st.sidebar:
    uploaded_files = st.sidebar.file_uploader(
        label='Upload documents',
        accept_multiple_files=True,
        key='file_uploader',
        type=['csv', 'pdf']
    )
    if uploaded_files:
        for uploaded_file in uploaded_files:
            file_details = {
                "Filename": uploaded_file.name,
                "Filetype": uploaded_file.type,
                "Filesize": uploaded_file.size
            }
        if st.sidebar.button("Save All Files"):
            for uploaded_file in uploaded_files:
                if save_uploaded_file(uploaded_file):
                    success = st.sidebar.success(f"File {uploaded_file.name} saved successfully!")
                    time.sleep(1.5)
                    success.empty()
                else:
                    st.sidebar.error(f"Failed to save the file {uploaded_file.name}.")

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

    BACKEND_URL = os.getenv("BACKEND_URL")
    with st.spinner("Waiting for response..."):
        response = requests.post(BACKEND_URL, json={"query": prompt, "model": st.session_state.model_name,
                                                    "temperature": st.session_state.temperature})

    if response.status_code == 200:
        bot_response = response.json().get("response", "")
        st.session_state.messages.append({"role": "assistant", "content": bot_response})
        llmresponse_logger.info(f'Groq response: {bot_response}')
        st.markdown(bot_response)
    else:
        st.error(f"Failed to get a response from the backend. Status code: {response.status_code}")