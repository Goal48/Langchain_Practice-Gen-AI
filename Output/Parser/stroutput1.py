import os
from langchain_community.llms import Ollama
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
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

parser = StrOutputParser()

chain = template1 | llm | parser | template2 | llm | parser

print(chain.invoke({"topic" : "NIT-Durapur"}))