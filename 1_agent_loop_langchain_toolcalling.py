from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain.tools import tool
from langchain_core.messages import HumanMessage,SystemMessage,ToolMessage

from langchain.chat_models import init_chat_model
from langsmith import traceable

load_dotenv()

MAX_ITERATIONS = 10
MODEL = "qwen3:1.7b"
price_catalogue = {
    "laptop": 999.99,
    "smartphone": 699.99,
    "headphones": 199.99,
    "monitor": 299.99,
    "keyboard": 49.99,
    "mouse": 29.99,
    "tablet": 399.99
}

discount_rates = {
    "bronze": 5,
    "silver": 10,
    "gold": 15,
    "platinum": 20
}

@tool
def get_product_price(product: str) -> float:
    """
    Look up the price of a product in the catalog
    """

    print(f"Looking up price for product: {product}")
    return price_catalogue.get(product,0)

@tool
def apply_discount(price: float, discount_tier: str) -> float:
    """
    Apply a discount to the price based on the discount tier and return the final price
    """
    discount_rate = discount_rates.get(discount_tier, 0)
    final_price = price * (1 - discount_rate / 100)
    print(f"Applying {discount_rate}% discount to price: {price}. Final price: {final_price}")
    return final_price


# Agent loop

@traceable("LangChain Agent Loop")
def run_agent(question: str):
    pass

if __name__ == "__main__":
    print("Hello LangChain agent(.bind_tools)!")

    result = run_agent(
        "What is the price of a laptop with a gold discount tier?"
    )