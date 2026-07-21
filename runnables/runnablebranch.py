from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnableLambda , RunnableSequence , RunnablePassthrough , RunnableBranch
from langchain_ollama import OllamaLLM


load_dotenv()


model = OllamaLLM(model='llama3.2:1b')
parser = StrOutputParser()


# 1. Define prompts
generate_joke = PromptTemplate(
    template="generate a description about {topic}",
    input_variables=['topic']
)

summary_joke = PromptTemplate(
    template='summary the topic {text}',
    input_variables=['text']
)
report_gen_chain = RunnableSequence(generate_joke , model , parser)

branch_chain = RunnableBranch(
    (lambda x : len(x.split()) > 300 , RunnableSequence(summary_joke,model ,parser)),
    RunnablePassthrough()
)
final_chain = RunnableSequence(report_gen_chain,branch_chain)
print(final_chain.invoke({'topic' : 'AI'}))