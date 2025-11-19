from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from dotenv import load_dotenv

load_dotenv()

def main():
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    tools = []

    agent_executor = create_agent(model, tools)

    print("Welcome! I'm your AI assistant. Type 'quit' to exit.")

    while True:
        user_input = input("\nYou: ").strip()
        if user_input == "quit":
            break

        print("\nAssistant: ", end="")
        for chunk in agent_executor.stream(
            {"messages": [HumanMessage(content=user_input)]}
        ):
            if "model" in chunk and "messages" in chunk["model"]:
                for msg in chunk["model"]["messages"]:
                    print(msg.content, end="")
        print()


if __name__ == "__main__":
    main()
