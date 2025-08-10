from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001"
)

#directly providing a query to generate vectors
result = embeddings.embed_query("Old Trafford is home stadium of Manchester United Football Club.")

# Print the embeddings as a string for better readability
print(str(result)) 