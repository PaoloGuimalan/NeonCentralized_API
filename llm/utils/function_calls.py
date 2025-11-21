import requests


def api_call(method, url, parameters, param_type, headers=None):
    method = method.upper()
    try:
        if method == "GET":
            if param_type == "query":
                # For GET, send parameters as query string
                response = requests.get(url, params=parameters, headers=headers)
            elif param_type == "route":
                url = url.format(**parameters)
                # For GET, send parameters as route string
                response = requests.get(url, headers=headers)
            else:
                # For GET, send parameters as query string
                response = requests.get(url, params=parameters, headers=headers)
        elif method == "POST":
            # For POST, send parameters as JSON body
            response = requests.post(url, json=parameters, headers=headers)
        else:
            raise ValueError(f"Unsupported HTTP method: {method}")

        # Raise HTTPError for bad responses (4xx, 5xx)
        response.raise_for_status()

        # Try to parse JSON response
        try:
            return response.json()
        except ValueError:
            # Return raw text if not JSON
            return response.text

    except requests.RequestException as e:
        # Handle errors like connection issues
        return {"error": str(e)}


def trigger_function(func, method, url, parameters, param_type, headers):

    if func == "api_call":
        return api_call(
            method=method,
            url=url,
            parameters=parameters,
            param_type=param_type,
            headers=headers,
        )
