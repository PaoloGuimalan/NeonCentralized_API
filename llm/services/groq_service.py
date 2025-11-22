from groq import Groq
import os
import json
from ..utils.function_calls import trigger_function

# Initialize Groq client globally once


class GroqService:

    client = None
    model = None

    def __init__(self, api_key, model):
        if self.client is None:
            self.client = Groq(api_key=api_key)
            self.model = model

    def stream_groq_chat_completion(
        self, history, system_prompt, user_message, tool_response=None
    ):
        """
        Makes a streaming chat completion request to Groq with optional tool response context.

        Args:
            system_prompt (str): The system-level context or prompt.
            user_message (str): The user input message.
            tool_response (dict or None): Optional JSON result from an external tool to augment context.
            model (str): Groq model identifier.

        Yields:
            str: Incremental tokens returned from Groq streaming chat completion.
        """
        llm_encapsulated_tool_info = [
            {
                "name": tool["name"],
                "description": tool["description"],
                "parameters": tool["parameters_schema"],
            }
            for tool in tool_response
        ]

        messages = history
        messages += [
            {
                "role": "system",
                "content": f"{system_prompt}, these are the tools available for you to use, {json.dumps(llm_encapsulated_tool_info, indent=2)}. Always check this tools when asked for you capability, no matter what the history in the conversation says.",
            },
            {"role": "user", "content": user_message},
        ]

        # if tool_response:
        #     messages.append(
        #         {
        #             "role": "assistant",
        #             "content": json.dumps(llm_encapsulated_tool_info),
        #         }
        #     )

        # Call Groq chat completions with streaming enabled
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            stream=True,
            functions=llm_encapsulated_tool_info,
            function_call="auto",
            # tools=[{"type": "function", "function": func} for func in tool_response],
            # tool_choice="auto",
        )

        # print(messages)

        # Yield incremental tokens as they arrive
        for chunk in completion:
            delta = chunk.choices[0].delta

            # print(chunk)

            if delta.function_call:
                name = delta.function_call.name
                arguments = delta.function_call.arguments

                current_tool = [tool for tool in tool_response if tool["name"] == name][
                    0
                ]

                response = trigger_function(
                    "api_call",
                    current_tool["http_method"],
                    current_tool["api_endpoint"],
                    json.loads(arguments),
                    current_tool["param_type"],
                    current_tool["headers_schema"],
                )

                messages.append(
                    {
                        "role": "system",
                        "content": f"this is the request response of {name}: {json.dumps(response)}, understand the message and give a human readable feedback about users' request. Inform them what is the result of the requested process after this attempt",
                    }
                )

                # Ask the model to continue after function execution
                followup_response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    stream=True,
                )

                for followup in followup_response:
                    delta_followup = followup.choices[0].delta

                    yield delta_followup.content
            else:
                yield delta.content

    def summarize_messages(self, messages):
        summary_response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": f"Summarize these messages and format them for llm to understand for history referencing, but it is important to make the summary as SHORT as possible, but not too short, always. Make it details and concise and never forget to include important informations from the conversation. Messages: {json.dumps(messages)}",
                }
            ],
        )

        return summary_response.choices[0].message.content
