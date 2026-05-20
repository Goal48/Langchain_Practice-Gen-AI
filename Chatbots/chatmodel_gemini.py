from langchain_google_genai import GoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

llm = GoogleGenerativeAI(model="gemini-1.5-flash")
result = llm.invoke("what is the capital of west bengal?")
print(result)