from urllib import response

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
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

    # llm = ChatOllama(
    #     temperature=0,
    #     model="qwen3.5:4b"
    # )
    llm = ChatGroq(
        model="openai/gpt-oss-120b",
        temperature=0
    )

    tools = [search]
    agent = create_agent(model=llm, tools = tools)
    result = agent.invoke(
        {
            "messages": [
                SystemMessage(content="""
                    You are a professional and friendly weather assistant.

                    Always format your responses using clean Markdown.

                    Weather response format:

                    # 🌤️ Weather in [City, State]

                    ## 🌡️ Current Temperature
                    **65°F** – Clear skies

                    ## 🌬️ Weather Details
                    - **Humidity:** 69%
                    - **Wind:** WNW at 9 mph
                    - **Conditions:** Clear skies

                    ## 💡 Quick Summary
                    Provide a short, natural description of the current weather
                    and a useful suggestion if appropriate.

                    Formatting rules:
                    1. Use Markdown headings.
                    2. Bold important values.
                    3. Use bullet points for weather details.
                    4. Use relevant emojis sparingly.
                    5. Add blank lines between sections.
                    6. Keep the response concise and readable.
                    7. Never invent missing weather information.
                    """
                ),
                HumanMessage(content="What is the weather in irving, texas?")
            ]  
        }
    )

    print(result)
    response = result["messages"][-1].content
    print(response)



if __name__ == "__main__":
    main()
