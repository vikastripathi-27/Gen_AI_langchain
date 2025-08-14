from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnableLambda

# convrting each step into Runnable lambda

def doc_load(self):
    doc_load = PyPDFLoader(file_path='RAG/files/Cover letter - Vikas Tripathi.pdf')
    return doc_load.load()

'''
recursively split the document into chunks size so that words are not cut in the middle
1 - split by paragrah
2 - split by sentences
3 - split by word
4 - split by characters
it keeps on comibing to have the best text split possible
'''
def splitting(text):
    split = RecursiveCharacterTextSplitter(
        chunk_size=50,
        chunk_overlap=0
    )
    return split.split_documents(text)

chain_1 = RunnableLambda(doc_load)

chain_2 = RunnableLambda(splitting)

# output of 1st doc loader is passed as input to splitting() --text contains the parameter
chain = chain_1 | chain_2

result = chain.invoke({})

print("count of chunks - ", len(result))

print("chunks")

for i in result:
    print(i.page_content)