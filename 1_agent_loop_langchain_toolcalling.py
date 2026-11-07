from dotenv import load_dotenv
from langchain.tools import tool
from langchain_core.messages import HumanMessage,SystemMessage,ToolMessage

from langchain.chat_models import init_chat_model
from langsmith import traceable

load_dotenv()

MAX_ITERATIONS = 10
MODEL = "qwen/qwen3.8-27b"
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

@traceable(name="LangChain Agent Loop", run_type="chain")
def run_agent(question: str):
    tools = [get_product_price, apply_discount]
    tools_dict = {tool.name: tool for tool in tools}

    llm = init_chat_model(
        f"groq:{MODEL}",
        temperature=0,
    )
    llm_with_tools = llm.bind_tools(tools)

    messages = [
        SystemMessage(
            content=(
                "You are a helpful shopping assistant. "
                "You have access to a product catalog tool "
                "and a discount tool.\n\n"
                "STRICT RULES — you must follow these exactly:\n"
                "1. NEVER guess or assume any product price. "
                "You MUST call get_product_price first to get the real price.\n"
                "2. Only call apply_discount AFTER you have received "
                "a price from get_product_price. Pass the exact price "
                "returned by get_product_price — do NOT pass a made-up number.\n"
                "3. NEVER calculate discounts yourself using math. "
                "Always use the apply_discount tool.\n"
                "4. If the user does not specify a discount tier, "
                "ask them which tier to use — do NOT assume one."
            )
        ),
        HumanMessage(content=question),
    ]

    for iteration in range(MAX_ITERATIONS):
        print(f"\n--- Iteration {iteration + 1} ---")

        ai_message = llm_with_tools.invoke(messages)
        tool_calls = ai_message.tool_calls

        # if no tool calls, the agent has finished and we can return the answer
        if not tool_calls:
            print("Agent has finished.Final answer:\n",ai_message.content)
            return ai_message.content

        # process only the first tool call for simplicity
        tool_call = tool_calls[0]
        tool_name = tool_call.get("name")
        tool_args = tool_call.get("args", {})
        tool_call_id = tool_call.get("id")

        print(f"Tool call detected: {tool_name} with args: {tool_args}")

        function_to_call = tools_dict.get(tool_name)
        if function_to_call is None:
            raise ValueError(f"Tool {tool_name} not found.")

        observation = function_to_call.invoke(tool_args)

        print(f"Observation from tool {tool_name}: {observation}")

        # Append the tool's observation to the messages for the next iteration
        messages.append(ai_message)
        messages.append(
            ToolMessage(
                content=str(observation),
                tool_call_id=tool_call_id
            )
        )
    print("Max iterations reached. Returning the last answer.")
    return None



if __name__ == "__main__":
    print("Hello LangChain agent(.bind_tools)!")

    result = run_agent(
        "What is the price of a laptop with a gold discount tier?"
    )