import os
import streamlit as st
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

# 1. Load environment variables from your .env file
load_dotenv()

## Prompt Template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Please respond to the user's questions."),
        ("user", "Question: {question}")
    ]
)

## Streamlit framework
st.title("Langchain Demo with Gemini")
input_text = st.text_input("Search a topic you want")

## Google Gemini Model
# It automatically reads GOOGLE_API_KEY from the environment
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

parser = StrOutputParser()

# Construct the chain
chain = prompt | model | parser

# Execute when user types text
if input_text:
    st.write(chain.invoke({'question': input_text}))