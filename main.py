from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools import DuckDuckGoSearchRun
from dotenv import load_dotenv
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub
from langchain_core.tools import tool
import requests

load_dotenv()

# Initialize DuckDuckGo search tool
search_tool = DuckDuckGoSearchRun()

# Initialize Gemini LLM
llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro")


# Pull the ReAct prompt from LangChain Hub (chat version for Gemini)
prompt = hub.pull("hwchase17/react-chat")

# Create the ReAct agent
agent = create_react_agent(
    llm=llm,
    tools=[search_tool],
    prompt=prompt
)

# Wrap with AgentExecutor
agent_executor = AgentExecutor(
    agent=agent,
    tools=[search_tool],
    verbose=True
)

# Invoke with chat_history
response = agent_executor.invoke({
    "input": "Suggets me best places to visit in Bengaluru",
    "chat_history": []
})

print("\nFinal Answer:", response["output"])
