import streamlit as st
import os, time, requests, shutil, yaml
from exceptions.operations_handler import userops_logger, llmresponse_logger, system_logger

with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)


def delete_data_folder(folder_path, max_attempts=10, delay=2):
    """
    a forceful attempt to delete chroma db of the previous document(s)
    Args:
        folder_path(str): chroma db directory
        max_attempts(int): number of times to try the deletion
        delay(int): time interval to wait.
    """
    for attempt in range(max_attempts):
        if os.path.exists(folder_path):
            try:
                shutil.rmtree(folder_path)
                print(f'Deleted folder on attempt {attempt + 1}')
                return True
            except Exception as e:
                print(f'Attempting to force close file handle:')
                time.sleep(delay)
        else:
            print(f'Folder does not exist: {folder_path}')
            return True

    print(f'Failed to delete folder after {max_attempts} attempts')
    return False


if 'delete_chroma' not in st.session_state:
    if os.path.exists(config["chromadb_path_streamlit"]):
        delete_data_folder(config["chromadb_path_streamlit"])
        time.sleep(3)
    st.session_state.delete_chroma = True


def clear_data_folder(folder_path):
    """
    clear the document(s) in data folder
    Args:
        folder_path(str): chroma db directory
    """
    if os.path.exists(folder_path):
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                system_logger.error(f'Failed to delete {file_path}. Reason: {e}')


# clear data in data folder
if 'data_cleared' not in st.session_state:
    if not os.path.exists(config["DATA_UPLOAD_DIR_STREAMLIT"]):
        os.makedirs(config["DATA_UPLOAD_DIR_STREAMLIT"])
    else:
        clear_data_folder(config["DATA_UPLOAD_DIR_STREAMLIT"])
    st.session_state.data_cleared = True


def save_uploaded_file(upload_file):
    if upload_file is not None:
        file_path = os.path.join(config["DATA_UPLOAD_DIR_STREAMLIT"], upload_file.name)
        with open(file_path, "wb") as f:
            f.write(upload_file.getbuffer())
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
                "File size": uploaded_file.size
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
