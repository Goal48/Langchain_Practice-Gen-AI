from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence , RunnableParallel
from dotenv import load_dotenv

load_dotenv()

Prompt1 = PromptTemplate(
    template="write a description that will be post for tweet {topic}",
    input_variables=['topic']
)

Prompt2 = PromptTemplate(
    template="write a description that will be post for linkdin {topic}",
    input_variables=['topic']
)

model = OllamaLLM(model="llama3.2:1b")
parser = StrOutputParser()

paralles_chain = RunnableParallel({
    'tweet' : RunnableSequence(Prompt1,model,parser),
    'linkdin' : RunnableSequence(Prompt2,model,parser)
})

print(paralles_chain.invoke({"topic" : "AI"}))