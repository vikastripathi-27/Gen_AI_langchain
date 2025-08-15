from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.retrievers.multi_query import MultiQueryRetriever

load_dotenv()

llm_model = ChatGoogleGenerativeAI(
    model='gemini-2.5-flash',
    temperature=0
)

pdf_load = PyPDFLoader(file_path='RAG/files/Digital Campus - Black Book.pdf')

doc = pdf_load.load()

embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

vector_store = FAISS.from_documents(documents=doc,embedding=embedding_model)

# Maximum marginal relevance retriever
# retrievr output will vary from each other so we get diversity
mmr_retriever = vector_store.as_retriever(
    search_type='mmr',
    search_krwgs={'k':5, 'lambda_mult': 0.5}
    # lambda_mult provides the diversity in result
    # lesser the number more diverse the retrieval
    # 1 will make it same as similarity search
)

similarity_retriever = vector_store.as_retriever(
    search_type="similarity", 
    search_kwargs={"k": 5}
)

# multiple queries to llm based on user query and then using the queries provided by llm, retrieving the best matches
multi_query = vector_store.as_retriever(
    search_kwargs={"k": 5}
)

m_query_retriever = MultiQueryRetriever.from_llm(
    retriever=multi_query,
    llm=llm_model
)

query = "what is the tech stack used in this project?"

result_mmr = mmr_retriever.invoke(query)
print("MMR retriever \n")

index = 1
for result in result_mmr:
    print(index, "\n", result.page_content)
    index=index+1

print("--"*20)

result_similarity = similarity_retriever.invoke(query)
print("Similarity retriever \n")

for result in result_similarity:
    print(index, "\n", result.page_content)
    index=index+1

print("--"*20)

result_multi_query = m_query_retriever.invoke(query)
print("Multiquery retriever \n")

for result in result_multi_query:
    print(index, "\n", result.page_content)
    index=index+1