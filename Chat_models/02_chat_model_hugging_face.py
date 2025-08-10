from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv

load_dotenv()

llm_hf = HuggingFaceEndpoint(
    model="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
)

llm = ChatHuggingFace(llm=llm_hf)

result = llm.invoke("what is manchester united? ")

print(result.content)
        