import asyncio
from analyst_agent.agent import root_agent


async def main():
    print("AI Data Analyst Agent Ready!")
    print("Type 'exit' to quit.\n")

    while True:
        user_input = input("Ask a question: ")

        if user_input.lower() == "exit":
            break

        response = await root_agent.run(user_input)

        print("\nResult:")
        print(response)
        print("-" * 50)


if __name__ == "__main__":
    asyncio.run(main())