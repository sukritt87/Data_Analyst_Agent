from app.agent import DataAnalystAgent


def main():
    agent = DataAnalystAgent("data/sample_data.csv")

    print("AI Data Analyst Agent Ready!")
    print("Type 'exit' to quit.\n")

    while True:
        user_input = input("Ask a question: ")

        if user_input.lower() == "exit":
            break

        result = agent.handle_query(user_input)

        print("\nResult:")
        print(result)
        print("-" * 50)


if __name__ == "__main__":
    main()