from langchain_community.document_loaders import PyPDFLoader

# each page is loaded into one list index
pdf_loader = PyPDFLoader(file_path='RAG/files/Cover letter - Vikas Tripathi.pdf')

result = pdf_loader.load()

print(result[0].page_content)