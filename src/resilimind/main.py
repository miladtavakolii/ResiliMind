import sys
from typing import Dict, Any
from resilimind.core.workflow import build_workflow


def run_cli() -> None:
    """
    Launches an interactive command-line chat session using the LangGraph workflow.
    """
    print("=" * 60)
    print("      🌱 ResiliMind: Resilience Assessment & Counseling 🌱")
    print("=" * 60)
    print("Type 'exit' or 'quit' to terminate the session.\n")

    # Build and compile the LangGraph app
    app = build_workflow()

    while True:
        try:
            user_input: str = input("\n👤 You: ").strip()
            if not user_input:
                continue

            if user_input.lower() in ["exit", "quit"]:
                print("\n👋 Goodbye! Wishing you strength and resilience.")
                break

            # Initial input state
            initial_state: Dict[str, Any] = {
                "user_message": user_input,
                "active_nodes": [],
                "subgraph_context": "",
                "assessments": [],
                "requires_disambiguation": False,
                "final_response": ""
            }

            # Run the state machine
            print("\n⏳ Processing response...")
            final_state: Dict[str, Any] = app.invoke(initial_state)

            # Display agent response
            response: str = final_state.get("final_response", "No response generated.")
            print(f"\n🤖 ResiliMind: {response}")

        except KeyboardInterrupt:
            print("\n\nSession interrupted. Exiting...")
            sys.exit(0)
        except Exception as e:
            print(f"\n❌ An error occurred: {str(e)}")


if __name__ == "__main__":
    run_cli()
