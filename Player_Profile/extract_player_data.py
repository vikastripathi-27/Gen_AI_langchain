from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import Optional, Literal

load_dotenv()

model = ChatGoogleGenerativeAI(
    model='gemini-2.5-flash',
    temperature=0
)

#summary of the player passed to extract insights
player_summary = """
MS Dhoni is a legendary Indian cricketer and captain, renowned for his calm leadership and finishing skills.
He captained India to historic wins in the 2007 T20 World Cup, 2011 ODI World Cup, and 2013 Champions Trophy, becoming the only captain to win all three ICC titles.
A skilled wicketkeeper-batsman, Dhoni was famous for his sharp glove work and match-winning innings under pressure.
He retired from international cricket in 2020 but continues to captain Chennai Super Kings in the IPL.
His career is celebrated for consistency, tactical brilliance, and inspiring leadership.
"""

#insights to be extracted in below keys
class Player_info(BaseModel):
    name: str = Field(description="Name of the player")
    nationality: str  = Field(description="Nationality of the player") 
    #age greater than 0 and less than 100 --validation
    age: int = Field(gt=0, lt=100, description="Age of the player in years") 
    profession: str = Field(description="Profession of the player")
    #default value of runs scored to be set to 0
    runs_scored: int = Field(default=0, description="total runs scored by the player")
    gender: Literal['male', 'female'] = Field(default = 'Not specified', description="gender of the player")
    formats: list[str] = Field(description="formats the player played in")
    #optional field for phone number, if not provided, it will not be shown, remaining field are required as optional is not specified for them
    phone_no: Optional[str] = Field(description="phone number of the player")

# model to provide structured output
structured_data = model.with_structured_output(Player_info)

result = structured_data.invoke(player_summary)

print("AI output: \n",result)

# Convert the result to JSON schema
result_json = result.model_json_schema()

# print each key
print("Name: ", result.name)
print("Nationality: ", result.nationality)
print("Age: ", result.age)
print("Profession: ", result.profession)
print("Runs scored: ", result.runs_scored)
print("Gender: ", result.gender)
print("Formats: ", result.formats)
print("Formats: ", result.formats[1])
print("Phone number ", result.phone_no)

# print the json schema
print("json data: ", result_json)