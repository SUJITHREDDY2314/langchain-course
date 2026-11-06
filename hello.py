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

    response = tavily_client.search(
        query=query,
        max_results=3,
        search_depth="basic"
    )

    results = []

    for item in response["results"]:
        results.append(
            f"""
            Title: {item.get('title')}
            URL: {item.get('url')}
            Details: {item.get('content')}
            """
        )

    return "\n".join(results)


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
                    You are a job search assistant.

                    Your available tool is search.

                    Rules:
                    - Use only the search tool.
                    - Never call open or any other unavailable tool.
                    - Search LinkedIn for relevant job postings.
                    - Find 3 relevant Angular developer jobs requiring RxJS
                      in Richardson, Texas.
                    - Include job title, company, location, requirements
                      and job URL when available.
                    - Never invent missing information.
                    - Keep the response concise and use Markdown.
                """),
                HumanMessage(content="search for 3 job postings for an angular developer using rxjs in richardson texas on linkedin and their details?")
            ]  
        }
    )

    print(result)
    response = result["messages"][-1].content
    print(response)



if __name__ == "__main__":
    main()
