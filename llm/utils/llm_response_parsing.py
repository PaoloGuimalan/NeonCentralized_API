import re
import json

def handle_llm_response(llm_response):
    pattern = r'<function=(\w+)>(.*?)</function>'
    match = re.search(pattern, llm_response)
    if match:
        func_name = match.group(1)  # dynamically get function name
        params_json = match.group(2)
        params = json.loads(params_json)

        # Dispatch dynamically based on func_name
        response_for_user = dispatch_function_call(func_name, params)
        return response_for_user
    else:
        # No function call present, return normal LLM text
        return llm_response

def dispatch_function_call(func_name, params):
    # Implement dynamic lookup or calls here:
    # For example, mapping func_name to actual functions or handlers
    # or routing to modules/plugins
    
    # Example generic placeholder response:
    return f"Performing '{func_name}' with parameters {params}. Please provide required inputs if needed."
