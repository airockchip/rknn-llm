import sys
import requests
import json
import re

# Set the address of the Server.
server_url = 'http://172.x.x.x:8080/rkllm_chat'

# Create a session object.
session = requests.Session()
session.keep_alive = False  # Close the connection pool to maintain a long connection.
adapter = requests.adapters.HTTPAdapter(max_retries=5)
session.mount('https://', adapter)
session.mount('http://', adapter)

def main_demo2(is_streaming=True):
    
    ## Define the function you need to call and its description
    TOOLS = [
        {
            "type": "function",
            "function": {
                "name": "get_current_temperature",
                "description": "Get current temperature at a location.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": 'The location to get the temperature for, in the format "City, State, Country".',
                        },
                        "unit": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"],
                            "description": 'The unit to return the temperature in. Defaults to "celsius".',
                        },
                    },
                    "required": ["location"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_temperature_date",
                "description": "Get temperature at a location and date.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": 'The location to get the temperature for, in the format "City, State, Country".',
                        },
                        "date": {
                            "type": "string",
                            "description": 'The date to get the temperature for, in the format "Year-Month-Day".',
                        },
                        "unit": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"],
                            "description": 'The unit to return the temperature in. Defaults to "celsius".',
                        },
                    },
                    "required": ["location", "date"],
                },
            },
        },
    ]
    
    def get_current_temperature(location: str, unit: str = "celsius"):
        """Get current temperature at a location.

        Args:
            location: The location to get the temperature for, in the format "City, State, Country".
            unit: The unit to return the temperature in. Defaults to "celsius". (choices: ["celsius", "fahrenheit"])

        Returns:
            the temperature, the location, and the unit in a dict
        """
        return {
            "temperature": 26.1,
            "location": location,
            "unit": unit,
        }


    def get_temperature_date(location: str, date: str, unit: str = "celsius"):
        """Get temperature at a location and date.

        Args:
            location: The location to get the temperature for, in the format "City, State, Country".
            date: The date to get the temperature for, in the format "Year-Month-Day".
            unit: The unit to return the temperature in. Defaults to "celsius". (choices: ["celsius", "fahrenheit"])

        Returns:
            the temperature, the location, the date and the unit in a dict
        """
        return {
            "temperature": 25.9,
            "location": location,
            "date": date,
            "unit": unit,
        }

    def get_function_by_name(name):
        if name == "get_current_temperature":
            return get_current_temperature
        if name == "get_temperature_date":
            return get_temperature_date
    
    
    print("============================")
    print("This is a demo about RKLLM function-call...")
    print("============================")
    
    messages = [
    {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.\n\nCurrent Date: 2024-09-30"},
    {"role": "user",  "content": "What's the temperature in San Francisco now? How about tomorrow?"},
    ]

    headers = {
        'Content-Type': 'application/json',
        'Authorization': 'not_required'
    }

    # Prepare the data to be sent
    # model: The model defined by the user when setting up RKLLM-Server; this has no effect here
    # messages: The user's input question, which RKLLM-Server will use as input and return the model's reply; multiple questions can be added to messages
    # stream: Whether to enable streaming generate, should be False
    data = {
        "model": 'your_model_deploy_with_RKLLM_Server',
        "messages": messages,
        "stream": False,
        "enable_thinking": False,
        "tools": TOOLS
    }

    # Send a POST request
    responses = session.post(server_url, json=data, headers=headers, stream=False, verify=False)

    # Parse the response
    if responses.status_code == 200:
        print("Q:", data["messages"][-1]["content"], '\n')
        
        server_answer = json.loads(responses.text)["choices"][-1]["message"]["content"]
        matches = re.findall(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", ''.join(server_answer), re.DOTALL)
        print("server_answer:", server_answer, '\n')
        
        result = [json.loads(match) for match in matches]
        for function_call in result:
            messages.append({'role': 'assistant', 'content': '', 'function_call':function_call}) 

        tool_calls = [{'function': result[i]} for i in range(len(result))]
        function_call = []
        for tool_call in tool_calls:
            if fn_call := tool_call.get("function"):
                fn_name: str = fn_call["name"]
                fn_args: dict = fn_call["arguments"]
                fn_res: str = json.dumps(get_function_by_name(fn_name)(**fn_args))
                messages.append({'role': 'tool', 'name': fn_name, 'content':fn_res})

        print("messages:", messages, '\n')
        
    else:
        print("Error:", responses.text)
        exit()
       
    data = {
        "model": 'your_model_deploy_with_RKLLM_Server',
        "messages": messages,
        "stream": is_streaming,
        "enable_thinking": False,
        "tools": TOOLS
    }

    # Send a POST request
    responses = session.post(server_url, json=data, headers=headers, stream=is_streaming, verify=False)
    
    if not is_streaming:
        # Parse the response
        if responses.status_code == 200:
            print("A:", json.loads(responses.text)["choices"][-1]["message"]["content"])
        else:
            print("Error:", responses.text)
    else:
        if responses.status_code == 200:
            print("A:", end="")
            for line in responses.iter_lines():
                if line:
                    line = json.loads(line.decode('utf-8'))
                    if line["choices"][-1]["finish_reason"] != "stop":
                        print(line["choices"][-1]["delta"]["content"], end="")
                        sys.stdout.flush()
        else:
            print('Error:', responses.text)
            
    print('\n')

def main_demo1(is_streaming=True):
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
                    "stream": is_streaming,
                    "enable_thinking": False,
                    "tools": None
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
        
if __name__ == '__main__':
    
    ## Demo1: RKLLM conversation
    main_demo1(True)
    
    # ## Demo2: RKLLM function-call
    # main_demo2(True)
    

