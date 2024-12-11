import sys
import requests
import json
  
# Set the address of the Server.
server_url = 'http://172.16.10.79:8080/rkllm_chat'
# Set whether to enable streaming mode.
is_streaming = True

# Create a session object.
session = requests.Session()
session.keep_alive = False  # Close the connection pool to maintain a long connection.
adapter = requests.adapters.HTTPAdapter(max_retries=5)
session.mount('https://', adapter)
session.mount('http://', adapter)

if __name__ == '__main__':
    print("============================")
    print("Input your question in the terminal to start a conversation with the RKLLM model...")
    print("============================")
    # Enter a loop to continuously get user input and converse with the RKLLM model.
    while True:
        try:
            user_message = input("\n*Please enter your question:")
            if user_message == "exit":
                print("============================")
                print("The RKLLM Server is stopping......")
                print("============================")
                break
            else:
                # Set the request headers; in this case, the headers have no actual effect and are only used to simulate the OpenAI interface design.
                headers = {
                    'Content-Type': 'application/json',
                    'Authorization': 'not_required'
                }

                # Prepare the data to be sent
                # model: The model defined by the user when setting up RKLLM-Server; this has no effect here
                # messages: The user's input question, which RKLLM-Server will use as input and return the model's reply; multiple questions can be added to messages
                # stream: Whether to enable streaming conversation, similar to the OpenAI interface
                data = {
                    "model": 'your_model_deploy_with_RKLLM_Server',
                    "messages": [{"role": "user", "content": user_message}],
                    "stream": is_streaming
                }

                # Send a POST request
                responses = session.post(server_url, json=data, headers=headers, stream=is_streaming, verify=False)

                if not is_streaming:
                    # Parse the response
                    if responses.status_code == 200:
                        print("Q:", data["messages"][-1]["content"])
                        print("A:", json.loads(responses.text)["choices"][-1]["message"]["content"])
                    else:
                        print("Error:", responses.text)
                else:
                    if responses.status_code == 200:
                        print("Q:", data["messages"][-1]["content"])
                        print("A:", end="")
                        for line in responses.iter_lines():
                            if line:
                                line = json.loads(line.decode('utf-8'))
                                if line["choices"][-1]["finish_reason"] != "stop":
                                    print(line["choices"][-1]["delta"]["content"], end="")
                                    sys.stdout.flush()
                    else:
                        print('Error:', responses.text)
                        
        except KeyboardInterrupt:
            # Capture Ctrl-C signal to close the session
            session.close()

            print("\n")
            print("============================")
            print("The RKLLM Server is stopping......")
            print("============================")
            break
