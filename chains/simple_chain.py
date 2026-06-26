from langchain_huggingface import ChatHuggingFace , HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
load_dotenv()

model = HuggingFaceEndpoint(
    repo_id= "meta-llama/Meta-Llama-3-8B-Instruct",
    task= "text-generation"
)

llm = ChatHuggingFace(llm = model)

parser = StrOutputParser()

prompt = PromptTemplate(
    template="write 5 important sentence about {topic}",
    input_variables=['topic']
)

chain = prompt | llm | parser
result = chain.invoke({'topic' : 'fifa-world cup 2026'})
print(result)

## visulization
# chain.get_graph().print_ascii() 