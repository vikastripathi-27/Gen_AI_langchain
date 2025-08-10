from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

# for storing history
chat_details = []

llm = ChatGoogleGenerativeAI(
        model='gemini-2.5-flash',
        temperature=0.0,
        max_new_tokens=1000
    )

# loop to continue the chat unless exit command is given
while True:
    user_input = input("Vikas: ")
    if user_input.lower() == "exit":
        break
    else:
        #storing user input for history
        chat_details.append(user_input)
        #passing entire history to the model so it remebers older conversation and can answer if a question is asked on it
        ai_chat =  llm.invoke(chat_details)
        #storing AI response for history
        chat_details.append(ai_chat.content)
        print("AI: ", ai_chat.content)

print("\n")
print("Chat History:")
print(chat_details)