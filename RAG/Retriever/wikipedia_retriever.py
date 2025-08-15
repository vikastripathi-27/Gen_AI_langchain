from langchain_community.retrievers import WikipediaRetriever

# Initialize the retriever (optional: set language and top_k)
retriever = WikipediaRetriever(top_k_results=1, lang="en")

user_input = 'Football'

wiki_doc = retriever.invoke(user_input)

print(wiki_doc[0].page_content)