import streamlit as st
from simple import generate
from exceptions.operations_handler import system_logger, userops_logger, llmresponse_logger


st.title("Local RAG with Groq API")

# select model
model_select = st.sidebar.selectbox(
    'Select model',
    options=[
        "llama-3.1-405b-reasoning",
        "llama-3.1-70b-versatile",
        "llama-3.1-8b-instant",
        "mixtral-8x7b-32768",
        "gemma2-9b-it"
    ],
    placeholder="gemma2-9b-it"
)
st.session_state.model_name = model_select

# select temperature
temperature_select = st.sidebar.slider(
    'Temperature value',
    0.1, 1.0
)
st.session_state.temperature = temperature_select


# initalize messages
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

    response = f"{prompt}"
    with st.chat_message("assistant"):
        response = st.write_stream(generate(response, model=str(st.session_state.model_name), temperature=float(st.session_state.temperature)))
    st.session_state.messages.append({"role": "assistant", "content": response})
    llmresponse_logger.info(f'Groq response: {response}')