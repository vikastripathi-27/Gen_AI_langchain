from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

load_dotenv()

llm_model = ChatGoogleGenerativeAI(
    model='gemini-2.5-flash',
    temprature=0
)

parser = JsonOutputParser()

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
output = chains.invoke({'player_name':'Lewis Hamilton'})

# we get final output in json format
print(output)