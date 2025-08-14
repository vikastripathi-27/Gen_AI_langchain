from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel, RunnableSequence

load_dotenv()

llm_model = ChatGoogleGenerativeAI(
    model='gemini-2.5-flash',
    temperature=0
)

prompt = PromptTemplate(
    template='Summarize information about {technology} in 5 lines.',
    input_variables=['technology']
)

prompt2 = PromptTemplate(
    template='Suggest me best books on {technology}. Only list down the names of books',
    input_variables=['technology']
)

parser = StrOutputParser()

chain = RunnableParallel({
    'report': RunnableSequence(prompt, llm_model, parser),
    'books': RunnableSequence(prompt2, llm_model, parser)
})

result = chain.invoke({'technology':'Machine Learning'})

print(result['report'])
print(result['books'])
