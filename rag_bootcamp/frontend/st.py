import os

import streamlit as st
import requests

from model import (
    LLMRequest,
    LLMResponse,
)

fastapi_url = os.environ.get('FASTAPI_URL', "http://localhost:8000")

st.title("RAG Bootcamp")


def query(prompt: str):
    req = LLMRequest(
        query=prompt,
    )

    print(f"sending: {req.model_dump_json()}")
    resp = requests.post(url=f"{fastapi_url}/query", 
                         headers={"content-type": "application/json"},
                         data=req.model_dump_json())


    print(f"received: {LLMResponse.model_validate(resp.json())}")
    if resp.status_code != 200:
        return "asdf"

    llmResp = LLMResponse.model_validate(resp.json())

    return llmResp.response


# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


if prompt := st.chat_input("What is up?"):
    # Display user message in chat message container

    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response = st.write(query(prompt))


        