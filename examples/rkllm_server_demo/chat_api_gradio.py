from gradio_client import Client


def chat_with_rkllm(user_message, history=None):
    """Interact with the RKLLM Gradio server.

    Uses the new Gradio Chatbot message format:
        [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
    """
    if history is None:
        history = []

    client = Client("http://x.x.x.x:8080")

    # Step 1: Submit user input — server appends it to history
    _, history = client.predict(
        user_message=user_message,
        history=history,
        api_name="/get_user_input",
    )

    # Step 2: Get RKLLM response — server fills in the assistant reply
    result_history = client.predict(
        history=history,
        api_name="/get_RKLLM_output",
    )
    return result_history


if __name__ == "__main__":
    # History uses new Gradio format: list of {"role": ..., "content": ...} dicts
    result_history = []

    print("=" * 60)
    print("Enter your question to chat with the RKLLM model...")
    print("=" * 60)

    while True:
        try:
            user_message = input("\nPlease enter your question: ").strip()
            if not user_message:
                continue
            if user_message.lower() == "exit":
                print("Goodbye!")
                break

            result_history = chat_with_rkllm(user_message, result_history)

            # Print the last Q&A pair
            if result_history:
                # Extract text from content (handles both string and Gradio's [{"text":..., "type":"text"}] format)
                def _extract_text(msg):
                    if isinstance(msg, dict):
                        c = msg.get("content", "")
                        if isinstance(c, list):
                            return " ".join(p.get("text", "") for p in c if p.get("type") == "text")
                        return c
                    return str(msg)

                # result_history: [..., user_msg, assistant_msg]
                # Last is assistant, second-to-last is user
                if len(result_history) >= 2:
                    print(f"Q: {_extract_text(result_history[-2])}")
                    print(f"A: {_extract_text(result_history[-1])}")

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
            