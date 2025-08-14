from langchain_community.document_loaders import TextLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from dotenv import load_dotenv

load_dotenv()

llm_model = ChatGoogleGenerativeAI(
    model='gemini-2.5-flash',
    temperature=0
)

def doc_load(self):
    doc_loader = TextLoader(file_path='RAG/files/f1_2021.txt', encoding='utf-8')
    # loading the document
    result = doc_loader.load()
    # contains page content and metadata, extract only page content
    return result[0].page_content

# converting the function into runnable lambda
doc = RunnableLambda(doc_load)

prompt = PromptTemplate(
    template="Summarize the below text \n {report}",
    input_variables=['report']
)

parser = StrOutputParser()

chain = doc | prompt | llm_model | parser

output = chain.invoke({})

print(output)

'''
first loading the document
second passing that to llm as prompt
third fetching the output
'''
