from langchain_google_genai import GoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence
from dotenv import load_dotenv

load_dotenv()

prompts1 = PromptTemplate(
    template="create a joke on {topic}",
    input_variables=['topic']
)

model = OllamaLLM(model="llama3.2:1b")
parser = StrOutputParser()

prompts2 = PromptTemplate(
    template="Explain the joke {text}",
    input_variables=['text']
)

chain = RunnableSequence(prompts1 , model , parser , prompts2 , model , parser)

print(chain.invoke({"topic" : 'AI'}))