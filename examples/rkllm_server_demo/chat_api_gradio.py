from gradio_client import Client

# This function interacts with the RKLLM model by calling the Gradio Client API.
def chat_with_rkllm(user_message, history=[]):
    # Instantiate the Gradio Client. Users need to modify according to their specific deployment URL.
    client = Client("http://172.16.10.79:8080")

    # Call the Gradio Client API for interaction. The internal APIs mainly include:
    # get_user_input: The model retrieves user input and adds it to the history record 'history'.
    # get_RKLLM_output: RKLLM generates a response using the historical record 'history' that contains the input.
    _, history = client.predict(user_message=user_message, history=history, api_name="/get_user_input")
    result_history = client.predict(history=history, api_name="/get_RKLLM_output")
    return result_history

if __name__ == '__main__':
    result_history = []

    print("============================")
    print("Enter your question in the terminal to have a conversation with the RKLLM model...")
    print("============================")
    # Enter a loop to continuously receive user input and have a conversation with the RKLLM model...
    while True:
        try:
            user_message = input("Please enter your question:")
            if user_message == "exit":
                print("============================")
                print("The RKLLM Server is stopping......")
                print("============================")
                break
            else:
                # Call the `chat_with_rkllm` function to get the model's response.
                result_history = chat_with_rkllm(user_message, result_history)

                # Print the history of chatting
                print("Q:", result_history[-1][0])
                print("A:", result_history[-1][1])
        except KeyboardInterrupt:
            print("\n")
            print("============================")
            print("The RKLLM Server is stopping......")
            print("============================")
            break
            