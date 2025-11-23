from openai import OpenAI  # assuming you use OpenAI's official Python client
from ..utils.function_calls import trigger_function
import json


class OpenAIService:
    client = None
    model = None

    def __init__(self, api_key, model):
        if self.client is None:
            self.client = OpenAI(api_key=api_key)
            self.model = model

    def stream_chat_completion(
        self, history, system_prompt, user_message, tool_response=None
    ):
        llm_encapsulated_tool_info = []
        if tool_response:
            llm_encapsulated_tool_info = [
                {
                    "type": "function",
                    "name": tool["name"],
                    "description": tool["description"],
                    "parameters": tool["parameters_schema"],
                }
                for tool in tool_response
            ]

        messages = history + [
            {
                "role": "system",
                "content": f"{system_prompt}, these are the tools available for you to use, {json.dumps(llm_encapsulated_tool_info, indent=2)}. Always check these tools when asked for your capability no matter what the history in the conversation says.",
            },
            {"role": "user", "content": user_message},
        ]

        completion = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            stream=True,
            functions=llm_encapsulated_tool_info,
            function_call="auto",
        )

        accumulated_args = ""
        current_function_name = None

        for chunk in completion:
            delta = chunk.choices[0].delta

            # If there's a function call delta part
            if hasattr(delta, "function_call") and delta.function_call:
                # If function name is present, start a new accumulation
                if delta.function_call.name:
                    current_function_name = delta.function_call.name
                    # accumulated_args = ""  # reset for new call

                # Accumulate argument chunks safely
                if delta.function_call.arguments:
                    accumulated_args += delta.function_call.arguments

            else:
                # Normal content chunk, yield directly
                yield delta.content

        # When the last chunk has arrived (detected by absence of new arguments or some end condition),
        # parse and trigger the function call
        if current_function_name and accumulated_args:
            # parse accumulated args if any, else empty dict
            args = json.loads(accumulated_args)

            current_tool = [
                tool for tool in tool_response if tool["name"] == current_function_name
            ][0]

            response = trigger_function(
                "api_call",
                current_tool["http_method"],
                current_tool["api_endpoint"],
                args,
                current_tool["param_type"],
                current_tool["headers_schema"],
            )

            messages.append(
                {
                    "role": "system",
                    "content": f"this is the request response of {current_function_name}: "
                    f"{json.dumps(response)}, understand the message and give a human readable feedback about the user's request. "
                    "Inform them what is the result of the requested process after this attempt",
                }
            )

            followup_response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                stream=True,
            )

            # Reset after completion
            current_function_name = None
            accumulated_args = ""

            for followup in followup_response:
                delta_followup = followup.choices[0].delta
                yield delta_followup.content

    def summarize_messages(self, messages):
        # Compose prompt directing chat model to summarize
        prompt = (
            "Summarize these messages into a short and detailed summary "
            "suitable for history referencing, including important info: "
        )
        full_input = prompt + "\n\nMessages:\n" + json.dumps(messages)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": full_input},
            ],
            max_tokens=300,
            temperature=0.3,
        )
        return response.choices[0].message.content
