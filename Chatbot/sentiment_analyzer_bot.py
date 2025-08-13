from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain.schema.runnable import RunnableBranch, RunnableLambda
from typing import Literal
from pydantic import BaseModel, Field

load_dotenv()

llm_model = ChatGoogleGenerativeAI(
    model='gemini-2.5-flash',
    temperature=0
)

class PersonSentiment(BaseModel):
    person_sentiment: Literal['Positive','Negative'] = Field(description="Sentiment of the text")

str_parser = StrOutputParser()

pydt_parser = PydanticOutputParser(pydantic_object=PersonSentiment)

prompt = PromptTemplate(
    template='Provide me the sentiment of below conversation \n {text} \n {format}',
    input_variables=['text'],
    partial_variables={'format':pydt_parser.get_format_instructions()}
)

# the branch chain to have correct input then only condition can be applied which is why, I have used pydantic parser
sentiment_chain = prompt | llm_model | pydt_parser

# if sentiment is negative then below prompt is sent to llm
prompt_negative = PromptTemplate(
    template='''
    The person is having negative sentiments. 
    Below is the person statement \n {text} \n 
    Analyze it and provide methods to have positive sentiments
    ''',
    input_variables=['text']
)

# if sentiment is positive then below prompt is sent to llm
prompt_positive = PromptTemplate(
    template='''
    The person is having positive sentiments. 
    Below is the person statement \n {text} \n 
    Analyze it and provide methods to continue having positive sentiments
    ''',
    input_variables=['text']
)

# passing a tuple
message_chain = RunnableBranch(
    #tuple (condition, chain)
    # based on sentiment of person, one of the chains is executed
    # person_sentiment stores the sentiment from the first prompt via pydantic parser
    (lambda x:x.person_sentiment == 'Positive', prompt_positive | llm_model | str_parser),
    (lambda x:x.person_sentiment == 'Negative', prompt_negative | llm_model | str_parser),
    # last to be default chain
    RunnableLambda(lambda x: "could not find sentiment")
    # since there is no default chain right now for this case then manually creating a runnable
) 

# first sentiment of the statement is extrcated and then based on the condition one of the branch chain is invoked
final_chain = sentiment_chain | message_chain

result = final_chain.invoke({'text': 'I feel like I am over the moon'})

print(result)

final_chain.get_graph().print_ascii()