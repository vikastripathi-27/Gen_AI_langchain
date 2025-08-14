from langchain_community.document_loaders import CSVLoader

# stores each record in each index of list
doc_load = CSVLoader(file_path='RAG/files/people_data.csv')

'''
lazy load to be  used when lots of files are present and high in volume
it loads each page/record one by one
load() first loads all of document in memeory and then carries out next operation
'''
result = doc_load.lazy_load()

# prining all the records from the csv file as each record is a list item indexed 0 to n
no = 1
for record in result:
    print("record number ",no,":\n", record.page_content)
    no = no + 1
