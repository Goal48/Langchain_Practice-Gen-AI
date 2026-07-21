from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_ollama import OllamaLLM

load_dotenv()

model = OllamaLLM(model='llama3.2:1b')
parser = StrOutputParser()

# 1. Define prompts
generate_joke = PromptTemplate(
    template="generate a joke about {topic}",
    input_variables=['topic']
)

explain_joke = PromptTemplate(
    template="explain the joke: {text}",
    input_variables=['text']
)

# 2. Build individual chains
gen_joke_chain = generate_joke | model | parser
explain_joke_chain = explain_joke | model | parser

# 3. Chain them sequentially using RunnableParallel to return both outputs
full_chain = (
    gen_joke_chain 
    | RunnableParallel({
        'joke': RunnablePassthrough(),
        'explain_joke': {'text': RunnablePassthrough()} | explain_joke_chain
    })
)

# 4. Invoke
result = full_chain.invoke({'topic': 'AI'})
print(result)