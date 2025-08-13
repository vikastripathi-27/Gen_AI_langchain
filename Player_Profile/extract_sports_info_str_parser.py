from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", 
    temperature=0.2
)

prompt1 = PromptTemplate(
    template="Provide me information about {sports_name}",
    input_variables=["sports_name"]
)

prompt2 = PromptTemplate(
    template="Summarize {llm_output1} in 3 lines.",
    input_variables=["llm_output1"]
)

parser = StrOutputParser()

#first prompt1 is provided with its prompt
#second llm is called for providing information about kabaddi
#third the output is extracted in string using parser --same as extracting content from llm output
#fourth the output is passed as input to prompt2
#fifth the llm is called again to summarize the output from prompt2
#sixth the parser is called to extract content from llm output
chains = prompt1 | model | parser | prompt2 | model | parser

# Invoke the chain by p[assing the input to first prompt
result = chains.invoke({'sports_name': 'kabaddi'})

print("Summarized data: ", result)

print("\n\n")

#for invoking normal parser to get data
#this will eliminate the user of .content and useful in chains as parser is a function itself anc can be invoked unlike content

prompt = prompt1.invoke({'sports_name':'kabaddi'})
detail = model.invoke(prompt)
detailed_output = parser.invoke(detail)

print("Detailed data: ", detailed_output)