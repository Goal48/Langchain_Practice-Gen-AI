from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_google_genai import GoogleGenerativeAI
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel
from dotenv import load_dotenv

load_dotenv()

# --- 1. Prompts ---
prompt1 = PromptTemplate(
    template="make a short notes of the {topic}",
    input_variables=['topic']
)
prompt2 = PromptTemplate(
    template="write 5 question and answer of the {topic}",
    input_variables=['topic']
)
# Ensure both the template string and input_variables use 'quiz'
prompt3 = PromptTemplate(
    template="merge the short note and the quiz:\n\nNotes:\n{notes}\n\nQuiz:\n{quiz}",
    input_variables=['notes', 'quiz']
)

# --- 2. Models & Parser ---
llm = HuggingFaceEndpoint(repo_id="meta-llama/Meta-Llama-3-8B-Instruct", task="text-generation")
model1 = ChatHuggingFace(llm=llm)
model2 = OllamaLLM(model="llama3.2:1b")
model3 = GoogleGenerativeAI(model='gemini-2.5-flash')
parser = StrOutputParser()

# --- 3. Chains ---
chain1 = prompt1 | model1 | parser
chain2 = prompt2 | model2 | parser

# --- 4. Parallel Setup ---
# The keys here must MATCH prompt3's input variables exactly
combined_inputs = RunnableParallel({
    "notes": chain1,
    "quiz": chain2  # Cleaned up the 'e' here so it matches prompt3
})

# Complete pipeline layout
chain3 = combined_inputs | prompt3 | model3 | parser

# --- 5. Execution ---
result = chain3.invoke({'topic': ''})
print(result)