from groq import Groq
import os
import json
from ..utils.function_calls import trigger_function

# Initialize Groq client globally once
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
default_model = os.getenv("GROQ_MODEL")


def stream_groq_chat_completion(
    history, system_prompt, user_message, tool_response=None, model=default_model
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
    messages = history
    messages += [
        {
            "role": "system",
            "content": f"{system_prompt}, these are the tools available for you to use, {json.dumps(tool_response)}",
        },
        {"role": "user", "content": user_message},
    ]

    if tool_response:
        messages.append(
            {
                "role": "assistant",
                "content": json.dumps(tool_response),
            }
        )

    # Call Groq chat completions with streaming enabled
    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        stream=True,
        functions=[
            {**tool, "parameters": tool["parameters_schema"]} for tool in tool_response
        ],
        function_call="auto",
        # tools=[{"type": "function", "function": func} for func in tool_response],
        # tool_choice="auto",
    )

    # Yield incremental tokens as they arrive
    for chunk in completion:
        delta = chunk.choices[0].delta

        if delta.function_call:
            name = delta.function_call.name
            arguments = delta.function_call.arguments

            current_tool = [tool for tool in tool_response if tool["name"] == name][0]

            response = trigger_function(
                "api_call",
                current_tool["http_method"],
                current_tool["api_endpoint"],
                json.loads(arguments),
                current_tool["headers_schema"],
            )

            messages.append(
                {
                    "role": "system",
                    "content": f"this is the request response of {name}: {json.dumps(response)}, understand the message and give a human readable feedback about users' request. Inform them what is the result of the requested process after this attempt",
                }
            )

            # Ask the model to continue after function execution
            followup_response = client.chat.completions.create(
                model=model,
                messages=messages,
                stream=True,
            )

            for followup in followup_response:
                delta_followup = followup.choices[0].delta

                yield delta_followup.content
        else:
            yield delta.content


def summarize_messages(messages, model=default_model):
    summary_response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": f"Summarize these messages and format them for llm to understand for history referencing. Messages: {json.dumps(messages)}",
            }
        ],
    )

    return summary_response.choices[0].message.content
