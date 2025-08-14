'''
First we will extract a report on a particular topic
Then if the report length > 50 words we will ask LLM to summarize it in 50 words
Then we will ask to suggest 5 books on this topic
'''

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel, RunnableSequence, RunnablePassthrough, RunnableBranch

load_dotenv()

llm_model = ChatGoogleGenerativeAI(
    model='gemini-2.5-flash',
    temperature=0
)

prompt = PromptTemplate(
    template='Generate information about {technology} in 5 lines.',
    input_variables=['technology']
)

prompt2 = PromptTemplate(
    template='Suggest me best books below text \n {technology} \n Only list down the names of books',
    input_variables=['technology']
)

prompt3 = PromptTemplate(
    template='Summrize below text in 50 words \n {text}',
    input_variables=['text']
)

parser = StrOutputParser()

# extracting a report
report = prompt | llm_model | parser

# if the count of report > 50 then asking llm to summarize it in 50 words
# else pass the report as it is using Passthrough if len is less than or = to 50
chain_length = RunnableBranch(
    (lambda x: len(x.split())>50, prompt3 | llm_model | parser),
    RunnablePassthrough()
)

# summarzied report
chain_summarize = report | chain_length

# passing through the summarized report or normal report as it is
# fetching books on it
books = RunnableParallel({
    'report': RunnablePassthrough(),
    'books': RunnableSequence(prompt2, llm_model, parser)
})

chain = chain_summarize | books

result = chain.invoke({'technology':'Generative AI'})

print(result['report'])
print(result['books'])