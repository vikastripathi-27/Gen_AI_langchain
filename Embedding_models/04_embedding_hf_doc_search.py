from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

embedding = HuggingFaceEmbeddings(
    model="sentence-transformers/all-MiniLM-L6-v2"
)

#providing data from which to search for relevant information
stadium_data_doc = [
    "Eden Gardens is a cricket stadium in Kolkata, India.",
    "Wankhede Stadium is a cricket stadium in Mumbai, India.",
    "M. A. Chidambaram Stadium is a cricket stadium in Chennai, India.",
    "M. Chinnaswamy Stadium is a cricket stadium in Bangalore, India.",
]

#providing a user query to search
user_query = "provide me some information about chennai super kings"

#convert the stadium data into embeddings
stadium_result = embedding.embed_documents(stadium_data_doc)

#convert the user query into embeddings
user_query_result = embedding.embed_query(user_query)

#provide paramers to cosine similarity in 2d list format 
#returns a 2d array with similarity scores after comparing user query with stadium data
#since we are comparing 1 query with multiple documents, use [0] to get only 1 result as we get a 2d array
matching = cosine_similarity([user_query_result], stadium_result)[0]

#enumerate provides index for matching scores so it will be index, matching
index_score = list(enumerate(matching))

#lambda is a small anonymous function so here we pass index_score and fetch its second index x[1]
#sort the index_score i.e. second index x[1] based on descending order[-1]
index, matching_value = sorted(index_score, key=lambda x:x[1])[-1]

print("user query - ", user_query)

# Print the document with the highest match 
print(stadium_data_doc[index])  

 # Print the similarity score of the matched document
print("Similarity Score:", matching_value) 