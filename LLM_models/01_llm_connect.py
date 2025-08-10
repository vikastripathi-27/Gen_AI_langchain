"""
This script connects to the Google Gemini LLM using the LangChain library.
It sends a message to the model and prints the response.
It is not ideal for chat conversations but demonstrates basic functionality.
"""

import os
from langchain_google_genai import GoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

llm = GoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.2, 
    google_api_key=os.getenv("GEMINI_API_KEY")
)

result = llm.invoke("what is power bi?")

print(result)