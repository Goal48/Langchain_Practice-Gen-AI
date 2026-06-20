import os 
from fastapi import FastAPI
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama  # Using the modern Ollama integration
from dotenv import load_dotenv
from langserve import add_routes
import uvicorn

load_dotenv()

app = FastAPI(
    title="Langchain Server",
    version='1.0',
    description="Simple API server"
)

# 1. Initialize Google Gemini (Pass explicit model name)
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

# 2. Initialize Local Llama (Fixed the naming crash to ChatOllama)
llm = ChatOllama(model='llama3.2:1b') 

# 3. Fixed prompt creators to use from_template
prompt1 = ChatPromptTemplate.from_template("write me essay about {topic} with 100 words")
prompt2 = ChatPromptTemplate.from_template("write me poem about {topic} with 100 words")

# Route for the raw Gemini model (Renamed path to /gemini for clarity)
add_routes(
    app,
    model,
    path="/gemini"
)

# Route for the Gemini Essay Chain
add_routes(
    app,
    prompt1 | model,
    path="/essay"
)

# Route for the local Llama Poem Chain
add_routes(
    app,
    prompt2 | llm,
    path="/poem"
)

if __name__ == "__main__" :
    uvicorn.run(app, host="localhost",port=8501)