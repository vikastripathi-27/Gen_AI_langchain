from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
load_dotenv()

# each page is loaded into one list index
pdf_loader = PyPDFLoader(file_path='RAG/files/cricket.pdf')

result_doc = pdf_loader.load()

# creating an object to invoke embedding model
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

vector_store = FAISS.from_documents(documents=result_doc, embedding=embeddings)

# storing data locally
vector_store.save_local("RAG/Vectors/Database/")

result = vector_store.docstore._dict.items()

# View all documents
for doc_id, doc in result:
    print(f"ID: {doc_id}")
    print("Content: ", doc.page_content)
    print("Metadata: ", doc.metadata)
    print("-" * 40)

# search documents
output1 = vector_store.similarity_search(
    query='Who was the host nation?',
    k=1
)

print(output1[0].page_content)