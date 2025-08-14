from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter

# loading data from pdf
doc_loader = PyPDFLoader(file_path='RAG/files/Cover letter - Vikas Tripathi.pdf')

text = doc_loader.load()

# splititng in chunks of 100 
# chunk overlap is number of characters repeating in each chunk so while splitting the words which are split get covered in other chunk if the overlap is within the mentioned limit, so getting context of the word is easy
# seperator provides when the chunk should be split, if you want to split chunks as sentence you can give .
split = CharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=0,
    separator=''
)

# splitting the passed document
# passing the doc loader variable
result = split.split_documents(text)

# result has document splits in list 
print("first chunk - ", result[0].page_content)

print("\n all chunks \n")
# all chunks
for i in result:
    print(i.page_content)