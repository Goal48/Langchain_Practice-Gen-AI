from langchain_google_genai import GoogleGenerativeAI
from langchain_ollama import OllamaLLM
from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
import os
load_dotenv()


# llm = GoogleGenerativeAI(model="gemini-1.5-flash")
# model = OllamaLLM(model="llama3.2:1b")  
llm1 = HuggingFaceEndpoint(
    repo_id="meta-llama/Meta-Llama-3-8B-Instruct", 
    task="text-generation"
)
model1 = ChatHuggingFace(llm = llm1)

result = model1.invoke("what is the capital of west bengal?")
print(result.content)