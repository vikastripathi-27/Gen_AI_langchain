from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

load_dotenv()

llm_model = ChatGoogleGenerativeAI(
    model='gemini-2.5-flash',
    temperature=0
)

embedding_model = GoogleGenerativeAIEmbeddings(
    model='models/embedding-001'
)

pdf_loader = PyPDFLoader(
    file_path='RAG/files/Digital Campus - Black Book.pdf'
)

doc = pdf_loader.load()

vector_store = FAISS.from_documents(
    documents=doc,
    embedding=embedding_model
)

# if not search type provided then similarity retriever
# k is the maximum retrives it will do, it can do only 1 retrival if there is not much relevant info anywhere else in the doc
retriever = vector_store.as_retriever(
    search_type='mmr',
    search_kwargs={'k':5, 'lambda_mult': 0.4}
)

compressor = LLMChainExtractor.from_llm(llm_model)


# context compression retriever sends all the documents which are retrieved are sent to LLM again to extract only relevant part
compression_retriever = ContextualCompressionRetriever(
    base_retriever=retriever,
    base_compressor=compressor
)

# base retriver will retrieve the data based on the search type
# base compressor will send the retriver document to llm to extract only relevant par

user_query = "what is the author and developer of this project?"
result = compression_retriever.invoke(user_query)

i = 1
for result in result:
    print("result = ", i)
    print("Output: \n", result.page_content)
    print("Metadata = ", result.metadata)
    i = i + 1