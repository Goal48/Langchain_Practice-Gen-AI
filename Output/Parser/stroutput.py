import os
from langchain_community.llms import Ollama
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

llm = OllamaLLM(model = "llama3.2:1b")

template1 = PromptTemplate(
    template = "write a detailed report on the {topic}",
    input_variables = ['topic']
)

template2 = PromptTemplate(
    template = "write 5 line summary on the {text}",
    input_variables = ['text']
)

prompt1 = template1.invoke({"topic": "NIT Durgapur"})
result = llm.invoke(prompt1)

prompt2 = template2.invoke({"text" : result})
result1 = llm.invoke(prompt2)
print(result1)
