from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools import DuckDuckGoSearchRun
from dotenv import load_dotenv
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub
from langchain_core.tools import tool
import requests
from config import API_KEY

load_dotenv()

# Initialize DuckDuckGo search tool
search_tool = DuckDuckGoSearchRun()

# Initialize Gemini LLM
llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro")

# Weather tool with error handling
@tool
def get_weather_data(city: str) -> str:
    """
    Fetches the current weather data for a given city using Weatherstack API.
    Always returns a human-readable string.
    """
    url = f"http://api.weatherstack.com/current?access_key={API_KEY}&query={city}"
    response = requests.get(url).json()

    if response.get("success") is False:
        return f"Weather API error: {response['error']['info']}"

    try:
        location = response["location"]["name"]
        temperature = response["current"]["temperature"]
        description = response["current"]["weather_descriptions"][0]
        return f"In {location}, it is currently {temperature}°C with {description}."
    except Exception:
        return "Could not parse weather data."

# Pull the ReAct prompt from LangChain Hub
prompt = hub.pull("hwchase17/react-chat")

# Create the ReAct agent
agent = create_react_agent(
    llm=llm,
    tools=[search_tool, get_weather_data],
    prompt=prompt
)

# Wrap with AgentExecutor
agent_executor = AgentExecutor(
    agent=agent,
    tools=[search_tool, get_weather_data],
    verbose=True
)

# Run agent
if __name__ == "__main__":
    response = agent_executor.invoke({
        "input": "Find the capital of Karnataka, then find its current weather condition",
        "chat_history": []
    })

    print("\nFinal Answer:", response["output"])
