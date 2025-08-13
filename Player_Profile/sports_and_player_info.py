from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel

load_dotenv()

llm_model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0
)

prompt_1 = PromptTemplate(
    template="Provide me information about {sport_name}.",
    input_variables=["sport_name"]
)

prompt_2 = PromptTemplate(
    template="Provide me information about the top 5 players in {sport_name}.",
    input_variables=["sport_name"]
)

prompt_3 = PromptTemplate(
    template="Combine the information provided from {info_1} and {info_2}",
    input_variables=['info_1', 'info_2']
)

parser = StrOutputParser()

# running 2 chains parallely, always passing a dictionary
run_parallel = RunnableParallel({
    'info_1': prompt_1 | llm_model | parser,
    'info_2': prompt_2 | llm_model | parser
})

# running a sequential chain
merge = prompt_3 | llm_model | parser

# combining parallel and sequential chain
chains = run_parallel | merge

result = chains.invoke({'sport_name':'cricket'})

print(result)

# prints the flow of the chains
chains.get_graph().print_ascii()