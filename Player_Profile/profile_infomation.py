from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
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
player_name = st.text_input("Enter the details of player you want to generate a report for:")

# prompt template for generating player profile
usr_prompt = PromptTemplate(
    template = """
        Generate a detailed player profile for {player_name}.
        Include information such as career statistics, achievements, and notable performances.
            """,
    input_variables=["player_name"]
)

# storing the built prompt including the user inputs
input = usr_prompt.invoke({
    'player_name':player_name
    }
)

# button name and on clicking what action to perform
if st.button("Generate"):
    #passing the prompt to the model to generate the player profile
    result = model.invoke(input)
    # print output on UI
    st.write(result.content)
