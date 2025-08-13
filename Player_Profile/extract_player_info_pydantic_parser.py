from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from typing import Literal
from langchain_core.prompts import PromptTemplate

load_dotenv()

llm_model = ChatGoogleGenerativeAI(
    model='gemini-2.5-flash',
    temperature=0
)

# validating data types
class Player_profile(BaseModel):
    name: str = Field(description="Name of the player")
    age: int = Field(description="Age of the player")
    gender: Literal['Male', 'Female'] = Field(desciption='gender of the player')
    nationality: str = Field(description="nationality of the player")
    # only to provide input from the values provided, if not then provide default value
    profession: Literal['Football', 'Formula 1'] = Field(default="None", description="Profession of the player")
    net_worth: int = Field(default=0, description="lifetime networth of the player")

# output data to be parsed in pydantic
parser = PydanticOutputParser(pydantic_object=Player_profile)

prompt = PromptTemplate(
    template="Provide me information about {player_name}\n {format}",
    input_variables=['player_name'],
    partial_variables={'format':parser.get_format_instructions()}
)

chain = prompt | llm_model | parser

result = chain.invoke({"player_name": "ms dhoni"})

print(result)
print("\n")

#extacting name from output
print("Name: ", result.name)