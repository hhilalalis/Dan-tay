import argparse
# pyrefly: ignore [missing-import]
from dotenv import load_dotenv

from agentic_search import answer_question

# Load environment variables (like OPENAI_API_KEY)
load_dotenv()

def run_chat(max_cases: int = 5):
    print("Starting Interactive Danıştay Case Law Q&A Chat.")
    print("Type 'exit' or 'quit' to stop.")
    print("-" * 50)
    
    while True:
        try:
            user_input = input("\nYou: ")
            if user_input.strip().lower() in ['exit', 'quit']:
                break
                
            if not user_input.strip():
                continue
                
            print("Searching for relevant cases and generating answer...")
            response = answer_question(user_input, max_cases=max_cases)
            print(f"\nAssistant: {response}")
            
        except KeyboardInterrupt:
            break
            
    print("\nGoodbye!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Danıştay Case Law Agentic Search Tool")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Chat Command
    chat_parser = subparsers.add_parser("chat", help="Start an interactive Q&A chat using agentic search")
    chat_parser.add_argument("--max_cases", type=int, default=5, help="Maximum number of cases to retrieve per question")
    
    args = parser.parse_args()
    
    if args.command == "chat":
        run_chat(args.max_cases)
    else:
        parser.print_help()
