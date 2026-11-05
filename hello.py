from dotenv import load_dotenv

from langchain_core.prompts import PromptTemplate

from langchain_ollama import ChatOllama

load_dotenv()


def main():
    print("Hello from langchain-course!")

    information = """Elon Musk is a South African-born entrepreneur, engineer, and business leader known for his contributions to technology, transportation, and space exploration. He was born on June 28, 1971, in Pretoria, South Africa. From an early age, Musk showed an interest in computers, science, and technology. At the age of 12, he created and sold a video game called Blastar.

    Musk moved to Canada and later to the United States to pursue higher education and business opportunities. In 1995, he co-founded Zip2, a software company that was eventually sold for nearly $300 million. He later co-founded X.com, an online financial services company that became part of PayPal.

    In 2002, Musk founded SpaceX with the goal of making space travel more affordable and eventually enabling human settlement on Mars. He became a major investor and CEO of Tesla, helping accelerate the development of electric vehicles and sustainable energy. He also founded or became involved in companies such as Neuralink, The Boring Company, and xAI.

    In 2022, Musk acquired Twitter and later renamed it X. Known for his bold vision, risk-taking, and controversial public statements, Musk remains one of the most influential and debated figures in modern technology and business.
    """

    summary_template = """
    given the information {information} about a person I want you to create:
    1. A short summary
    2. two interesting facts about them
    """

    summary_prompt_template = PromptTemplate(
        input_variables=["information"], template=summary_template
    )

    llm = ChatOllama(temperature=0, model="gemma3:270m")

    chain = summary_prompt_template | llm
    response = chain.invoke(input={"information": information})
    print("trying..")
    print(response.content)


if __name__ == "__main__":
    main()
