"""
This script connects to the Google Gemini API using the LangChain library to create a chat model.
It sends a message to the model and prints the response.
It is designed for chat conversations, allowing for a more structured interaction.
"""

import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import SystemMessage, HumanMessage
from dotenv import load_dotenv

load_dotenv()
#picks up GOOGLE_API_KEY from .env 
# name to be set as GOOGLE_API_KEY as that is what the ChatGoogleGenerativeAI expects 
# or else if using a different name, you will to pass that variable explicitly

llm = ChatGoogleGenerativeAI (
    model="gemini-2.5-flash",
    temperature=0.2
    #creativity parameter (randomness)
    # for factual questions, set temperature near 0; for creative tasks, set temperature higher
)

messages = [
    SystemMessage(content="You are a data analyst."),
    HumanMessage(content="what is analytics vs engineering?")
]
# Send the request to Gemini
response = llm.invoke(messages)

# Print output
print(response.content)