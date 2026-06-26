import os
from typing import TypedDict,Annotated,Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.llms import Ollama
from dotenv import load_dotenv
load_dotenv()

class Review(TypedDict):
    sentiment: Annotated[Optional[str],"Return the sentiment"]
    rating: Annotated[int,"Return the rating"]
    summary: Annotated[str,"Return the summary"]

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash"
)
 ## llms = Ollama(model = "llama3.2:1b") ## does not work in "with_structure_output"

# 3. Bind the structured output
structured_llm = model.with_structured_output(Review)

# 4. Invoke it
response = structured_llm.invoke("The product arrived on time and works beautifully! 5 stars.")
print(response)

