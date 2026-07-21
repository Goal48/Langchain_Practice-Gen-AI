from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnableLambda , RunnableSequence , RunnablePassthrough
from langchain_ollama import OllamaLLM


load_dotenv()


model = OllamaLLM(model='llama3.2:1b')
parser = StrOutputParser()


# 1. Define prompts
generate_joke = PromptTemplate(
    template="generate a joke about {topic}",
    input_variables=['topic']
)

# define a function that count the words
def word_count(text) : 
    return len(text.split())

generate_chain = RunnableSequence(generate_joke , model , parser)
parallel_chain = RunnableParallel({
    'joke' : RunnablePassthrough(),
    'word count' : RunnableLambda(word_count)
})

result = RunnableSequence(generate_chain,parallel_chain)
print(result.invoke({'topic' : 'AI'}))