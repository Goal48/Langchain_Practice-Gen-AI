import requests
import streamlit as st

def get_gemini_response(input_text):
    # Fixed: Changed response.post to requests.post
    response = requests.post(
        "http://localhost:8501/essay/invoke",
        json={'input': {'topic': input_text}}
    )
    # Fixed: Simplified response parsing for LangServe output
    return response.json()['output']

def get_llama_response(input_text):
    # Fixed: Changed response.post to requests.post
    response = requests.post(
        "http://localhost:8501/poem/invoke",
        json={'input': {'topic': input_text}}
    )
    return response.json()['output']

# Streamlit framework UI
st.title("Langchain Demo with Llama and Gemini API")

input_text1 = st.text_input("Write an essay on:")
input_text2 = st.text_input("Write a poem on:")

if input_text1:
    st.write(get_gemini_response(input_text1))

if input_text2:
    st.write(get_llama_response(input_text2))