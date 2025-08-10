from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

embedding = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001"
)

#providing data from which to search for relevant information
stadium_data_doc = [
    "Old Trafford is home stadium of Manchester United Football Club.",
    "Anfield is the home stadium of Liverpool Football Club.",
    "Santaigo Bernabeu is the home stadium of Real Madrid Football Club.",
    "Camp Nou is the home stadium of FC Barcelona."
]

#providing a user query to search
user_query = "provide me some information about liverpool"

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
print(stadium_data_doc[index])  # Print the document with the highest match score
print("Similarity Score:", matching_value)  # Print the similarity score of the matched document