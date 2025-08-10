"""
the script contains the use of prompt from an existing prompt template
"""

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import load_prompt
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.2,
    max_new_tokens=1000
)

# UI title
st.title("Player profile")

# taking input from user
name = st.text_input("Enter the details of player you want to generate a report for:")

# Load the prompt from a JSON file
# using an existing prompt template
usr_prompt = load_prompt('player_name_prompt.json')

# storing the built prompt including the user inputs
input = usr_prompt.invoke({
    #storing in form of dictionary incase of multiple inputs
    #name is the variable which contain the user input
    #provide same key and value
    'name':name
    }
)

# button name and on clicking what action to perform
if st.button("Generate"):
    #passing the prompt to the model to generate the player profile
    result = model.invoke(input)
    # print output on UI
    st.write(result.content)
