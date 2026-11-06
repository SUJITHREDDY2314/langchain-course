from urllib import response

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama

from tavily import TavilyClient

load_dotenv()

tavily_client = TavilyClient()

@tool
def search(query: str) -> str:
    """
    Tool that searches over internet
    Args:
        query (str): The search query
    Returns:
        str: The search results
    """
    print(f"Searching for: {query}")
    return tavily_client.search(query=query)


def main():

    llm = ChatOllama(
        temperature=0,
        model="qwen3.5:4b"
    )
    tools = [search]
    agent = create_agent(model=llm, tools = tools)
    result = agent.invoke(
        {
            "messages": [
                HumanMessage(content="What is the weather in irving, texas?")
            ]  
        }
    )
    print(result)
    response = result["messages"][-1].content
    print(response)



if __name__ == "__main__":
    main()
