from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from dotenv import load_dotenv

load_dotenv()

llm = ChatGoogleGenerativeAI(
        model='gemini-2.5-flash',
        temperature=0.0,
        max_new_tokens=100
    )

#passingg the chat history as placeholder each time a prompt is invoked
history = ChatPromptTemplate([
    ('system', "You are a helpful sports analyst. You provide detailed and accurate information about various sports, including rules, player statistics, and game strategies. Provide summary in less than 100 words"),
    MessagesPlaceholder(variable_name='chat_details'),
    ('human', "{user_query}")
])

chat_details = []

#read chat history from chat_history.txt file if it contains any previous conversations
with open('chat_history.txt') as f:
    chat_details.extend(f.readlines())

print("Welcome to the Sports Chatbot! Which are you looking for?")
      
while True:
    user_query = input("Vikas: ")
    #append user query to chat history
    with open('chat_history.txt', 'a') as f:
        f.write(user_query)
    if user_query.lower() == "exit":
        break
    else:
        # load chat history
        with open('chat_history.txt') as f:
            chat_details.extend(f.readlines())

        #invoking prompt including chat history
        prompt = history.invoke({
            'chat_details':chat_details,
            'user_query': user_query
        })

        ai_chat =  llm.invoke(prompt)
        print("AI: ", ai_chat.content)
        #append AI response to chat history
        with open('chat_history.txt', 'a') as f:
            f.write(ai_chat.content)

print("\n")
print("Chat History:")
#variable that holds chat history from chat_history.txt file
print(chat_details)