from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

load_dotenv()

llm_model = ChatGoogleGenerativeAI(
    model='gemini-2.5-flash',
    temperature=0
)

embedding_model = GoogleGenerativeAIEmbeddings(model='models/embedding-001')

#extracting data from the passed url
url = 'https://medium.com/@social_65128/the-comprehensive-guide-to-understanding-generative-ai-c06bbf259786'
web_load = WebBaseLoader(web_path=url)

web_result = web_load.load()

# splitting the entire data
split = RecursiveCharacterTextSplitter(
    chunk_size=3000,
    chunk_overlap=0
)

text_split_result = split.split_documents(web_result)

# converting the text into vectors 
vector_store = FAISS.from_documents(documents=text_split_result, embedding=embedding_model)

retriever = vector_store.as_retriever (
    search_type='mmr',
    search_kwargs={'k':5, 'lambda_mult':0.4}
)

user_query = 'what is the history of AI?'

# retriving top 5 chunks based on the question asked
result_mmr = retriever.invoke(user_query)

context = []
for result in result_mmr:
    context.append(result.page_content)

# passing the text retrived based on the question as a 
# passing the question the llm needs answer based on the passed context
prompt = PromptTemplate(
    template='''
    You are helpful assistant. Based on the below context \n
    {context} \n
    Answer the question - {question}
    If the answer is not available in the context, plase say information is Not available. You should answer the question asked by the user based on the context only.
    ''',
    input_variables=['context', 'question']
)

question = user_query

parser = StrOutputParser()

chain = prompt | llm_model | parser

final_result = chain.invoke({'context':context, 'question':question})
print("Context is ", context)
print("*"*20)
print("question is ", question)
print("*"*20)
print(final_result)