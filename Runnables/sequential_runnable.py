from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence

load_dotenv()

llm_model = ChatGoogleGenerativeAI(
    model='gemini-2.5-flash',
    temperature=0
)

prompt = PromptTemplate(
    template='Provide information about {technology}.',
    input_variables=['technology']
)

parser = StrOutputParser()

# can use pipe as well for sequential chains
chain = RunnableSequence(prompt, llm_model, parser)

result = chain.invoke({'technology':'Machine Learning'})

print(result)

chain.get_graph().print_ascii()