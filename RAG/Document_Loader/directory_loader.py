from langchain_community.document_loaders import TextLoader, PyPDFLoader, DirectoryLoader

# creating different directory loader for different types of files
# it is used to load all file together
doc_load = DirectoryLoader(
    path='RAG/files/',
    glob='*.txt',
    loader_cls=TextLoader
)

doc_load2 = DirectoryLoader(
    path='RAG/files/',
    glob='*.pdf',
    loader_cls=PyPDFLoader
)

text_result = doc_load.load()

pdf_result = doc_load2.load()

print("Text file:\n",text_result[0].page_content)

print("\n PDF file:\n",pdf_result[0].page_content)