import os
import streamlit as st
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.llms import Ollama

load_dotenv()

## Prompt Template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Please respond to the user's questions."),
        ("user", "Question: {question}")
    ]
)


## Streamlit framework
st.title("Langchain Demo with Ollama")
input_text = st.text_input("Search a topic you want")

## Google Gemini Model
# It automatically reads GOOGLE_API_KEY from the environment
model = Ollama(model="llma2")  ## You have to install the model in the local sysytem "ollama run gemma"

parser = StrOutputParser()

# Construct the chain
chain = prompt | model | parser

# Execute when user types text
if input_text:
    st.write(chain.invoke({'question': input_text}))