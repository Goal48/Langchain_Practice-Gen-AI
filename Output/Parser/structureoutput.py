import os
from langchain_community.llms import Ollama
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_classic.output_parsers import StructuredOutputParser
from langchain_classic.output_parsers import ResponseSchema
from dotenv import load_dotenv

load_dotenv()

llm = OllamaLLM(model = "llama3.2:1b") 

schema = [
    ResponseSchema(name='fact_1',description="fact 1 about the topic"),
    ResponseSchema(name='fact_2',description="fact 2 about the topic"),
    ResponseSchema(name='fact_3',description="fact 3 about the topic")
]

parser = StructuredOutputParser.from_response_schemas(schema)

template = PromptTemplate(
    template = "give 3 fact about {topic} \n {format_instruction}",
    input_variables = ['topic'],
    partial_variables={'format_instruction' : parser.get_format_instructions()}
)

chain = template | llm | parser

result = chain.invoke({'topic' : 'NIT-Durgapur'})
print(result)