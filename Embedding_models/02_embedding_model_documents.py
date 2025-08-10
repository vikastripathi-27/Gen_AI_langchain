from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001"
)

#creating embeddings for a list of documents
doc =[
    "Old Trafford is home stadium of Manchester United Football Club.",
    "Anfield is the home stadium of Liverpool Football Club.",
    "Santaigo Bernabeu is the home stadium of Real Madrid Football Club.",
    "Camp Nou is the home stadium of FC Barcelona."
]

#passing the list to generate vectors
result = embeddings.embed_documents(doc)

print(str(result))  # Print the embeddings as a string for better readability