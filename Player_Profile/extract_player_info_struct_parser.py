from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema

load_dotenv()

llm_model = ChatGoogleGenerativeAI(
    model='gemini-2.5-flash',
    temprature=0
)

# schema in which I want the output
output_schema = [
    ResponseSchema(name="name", description="Name of the player"),
    ResponseSchema(name="age", description="Age of the player"),
    ResponseSchema(name="gender", description="Gender of the player"),
    ResponseSchema(name="profession", description="Profession of the player"),
    ResponseSchema(name="net worth", description="Net worth of the player"),
    ResponseSchema(name="nationality", description="Nationality of the player")
]

# parser to extract data in mentioned schema
parser = StructuredOutputParser.from_response_schemas(output_schema)

# player name is variable we are passing
# instructions is for formatting in json
prompt = PromptTemplate(
    template="Provide me information about {player_name}\n {instructions}",
    input_variables=['player_name'],
    #parser provides the format using previously created object
    partial_variables={'instructions':parser.get_format_instructions()}
)

chains = prompt | llm_model | parser

#if there is no input to be passed then pass empty dictinonary {} in invoke function as chains require some argumnent
output = chains.invoke({'player_name':'Cristiano Ronaldo'})

# we get final output in json format
print(output)

#json is dict in python, so extracting values
print("Name: ", output['name'])