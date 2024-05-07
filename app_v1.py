import os
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.cohere import CohereEmbeddings
from langchain.llms import Cohere
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import RetrievalQA
from langchain.vectorstores import Qdrant
import qdrant_client

import qdrant_client
import streamlit as st

from dotenv import load_dotenv


import pymongo
from pymongo import MongoClient


from utils import insert_data, get_latest_data, get_full_data
from services import get_response


load_dotenv(dotenv_path=os.path.join('.env'))


st.set_option('deprecation.showPyplotGlobalUse', False)
os.environ["COHERE_API_KEY"] = os.getenv("COHERE_API_KEY")

st.set_page_config(page_title="PSG College of Technology Chatbot")

st.title("PSG College of Technology Chatbot")


user_question = st.chat_input("What is up?")

full_data = get_full_data(-2, 0)

for message in full_data:
    # print(message)
    with st.chat_message("user"):
        st.markdown(message["query"])
    with st.chat_message("assistant"):
        st.markdown(message["response"])



# Check if the user has entered a question
if user_question:
    # chat_history["query"].append(user_question)
    # Get response for the user's query
    # response = qa.run(user_question)
    response, context = get_response(user_question)

    insert_data(-2, 0, user_question, response)

    # chat_history["answer"].append(response)

    # Display the response
    # print(chat_history)

    with st.chat_message("user"):
        st.write(f"{user_question}")

    with st.chat_message("assistant"):
        st.write(f"{response}")


