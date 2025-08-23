from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool
from langchain.agents import create_react_agent, AgentExecutor
import requests
import os
from langchain import hub
# hub contains all the prompts from community

load_dotenv()
cricbuzz_api_key = os.getenv("CRICBUZZ_API_TOKEN")

search_tool = DuckDuckGoSearchRun()

@tool
def cricbuzz_data():
    #mandatory to give function description
    '''
    This function fetches data related to international cricket series
    '''
    
    url = "https://cricbuzz-cricket.p.rapidapi.com/series/v1/international"
    headers = {
    	"x-rapidapi-key": cricbuzz_api_key,
    	"x-rapidapi-host": "cricbuzz-cricket.p.rapidapi.com"
    }
    response = requests.get(url, headers=headers)
    data = response.json()

    cricket_data = []

    # Access each month and its series based on json data
    for month_data in data['seriesMapProto']:
        cricket_data.append(f"Month: {month_data['date']}")
        # Access each series in that month
        for series in month_data['series']:
            series = f"Series name: {series['name']}"
            cricket_data.append(series)

    return cricket_data
# can directly return json as well but the format of this json is causing an issue

llm_model = ChatGoogleGenerativeAI(
    model='gemini-2.5-flash',
    temperature=0.2
)

# in built prompts for react agent
# react agent means reason + acting
# reasoning is done by llm, acting is done by tool
# can write your own prompt as well but to get better results use the in built prompts
# prompt -> https://smith.langchain.com/hub/hwchase17/react
prompt = hub.pull("hwchase17/react")

# creating an agent
agent = create_react_agent(
    llm=llm_model,
    tools=[search_tool, cricbuzz_data],
    prompt=prompt
)

# exxecuting an agent
agent_executor = AgentExecutor(
    agent=agent,
    tools=[search_tool, cricbuzz_data],
    verbose=True #thinking
)

# first task is finding information on cricket - using duck duck go search too
# second task is finding list of international matches in 2026 - using cricbuzz api data
result = agent_executor.invoke({'input': 'Provide me information about cricket and also provide me list of cricket series in the year 2026'})

print(result)

print("*"*100)

print(result['output'])