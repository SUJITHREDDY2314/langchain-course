from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama

load_dotenv()

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
    return "The weather in irving is cold."


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
                HumanMessage(content="What is the weather in irving?")
            ]  
        }
    )
    print(result)



if __name__ == "__main__":
    main()
