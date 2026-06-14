from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint , HuggingFacePipeline
from dotenv import load_dotenv
import os

load_dotenv()

# Change this line:
token = os.getenv("HUGGINGFACEHUB_ACCESSS_TOKEN") 

# And pass it directly to the endpoint to be safe:
llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Meta-Llama-3-8B-Instruct",  # Supported modern model
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

response = model.invoke("What is the capital of India?")

print(response.content)