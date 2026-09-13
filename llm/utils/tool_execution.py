"""Turning a Tool row into something a model may call, and then calling it.

TWO REPRESENTATIONS, AND WHY THEY ARE NOT THE SAME OBJECT
---------------------------------------------------------
A Tool is described to the model and executed by us, and those need different
fields. The model needs a name, a description and a parameter schema. Execution
needs an endpoint, a method, a param style and - the reason this split exists -
a CREDENTIAL.

Previously both came from `ToolSerializer(fields="__all__")`, so `authentication`
was serialised alongside the rest and `json.dumps`'d into the system prompt. A
secret in the prompt is a secret in the model provider's logs, in any prompt
tracing, and in anything the model can be talked into repeating back. So the
LLM-facing spec is built from an explicit allow-list of three fields, and the
execution side reads the Tool row directly and never hands it to anybody.

BOUNDED ROUNDS
--------------
A model that calls a tool, reads the result and calls another is doing the
thing tools are for. A model that does it forever is a bill. MAX_TOOL_ROUNDS
bounds it; hitting the bound is logged and the model is asked to answer with
what it has rather than being cut off mid-sentence.
"""

import ipaddress
import json
import logging
import socket
from urllib.parse import urlparse

import requests

logger = logging.getLogger(__name__)

# How many times a single turn may go model -> tool -> model. Three covers
# "look something up, then look up something that depended on it"; past that a
# model is usually looping rather than reasoning.
MAX_TOOL_ROUNDS = 3

TOOL_TIMEOUT_SECONDS = 20

# A tool endpoint is a URL an ORG ADMIN typed, called by OUR server with
# arguments a MODEL chose. That is server-side request forgery with two
# untrusted inputs, so the destination is checked rather than trusted.
ALLOWED_SCHEMES = {"http", "https"}


class ToolExecutionError(Exception):
    """The tool could not be called at all.

    Distinct from a tool that ran and returned an error, which is a result the
    model should see and explain rather than a failure of the turn.
    """


def build_tool_specs(tools):
    """The model-facing description of each tool.

    An explicit allow-list, not a serializer: adding a field to Tool must never
    silently start shipping it to a model provider.
    """
    specs = []
    for tool in tools:
        specs.append(
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description or "",
                    # A provider rejects a function with no schema, and a tool
                    # that genuinely takes no parameters is legitimate.
                    "parameters": tool.parameters_schema
                    or {"type": "object", "properties": {}},
                },
            }
        )
    return specs


def _resolves_to_public_address(hostname):
    """Whether every address this hostname resolves to is publicly routable.

    Checked at call time rather than at Tool-save time on purpose: a hostname
    that resolved publicly when the tool was created can be repointed at
    169.254.169.254 afterwards, and a save-time check would never run again.
    """
    try:
        infos = socket.getaddrinfo(hostname, None)
    except socket.gaierror as ex:
        raise ToolExecutionError("Could not resolve tool host: " + str(ex))

    for info in infos:
        address = ipaddress.ip_address(info[4][0])
        if (
            address.is_private
            or address.is_loopback
            or address.is_link_local
            or address.is_reserved
            or address.is_multicast
        ):
            return False
    return bool(infos)


def _validate_endpoint(url):
    parsed = urlparse(url or "")
    if parsed.scheme not in ALLOWED_SCHEMES:
        raise ToolExecutionError(
            "Tool endpoint must be http or https, got: " + str(parsed.scheme)
        )
    if not parsed.hostname:
        raise ToolExecutionError("Tool endpoint has no host")
    if not _resolves_to_public_address(parsed.hostname):
        raise ToolExecutionError(
            "Tool endpoint resolves to a private or loopback address"
        )
    return parsed


def _build_headers(tool):
    """Headers for one call: the declared ones, plus the credential.

    Tool.authentication is accepted in either shape people actually store: a
    JSON object of header name -> value, or a bare string, which is treated as
    an Authorization value. Guessing between them beats a silent "why is my
    tool returning 401" with no indication of which was expected.
    """
    headers = {}
    if isinstance(tool.headers_schema, dict):
        headers.update({str(k): str(v) for k, v in tool.headers_schema.items()})

    if tool.requires_auth and tool.authentication:
        credential = tool.authentication.strip()
        try:
            parsed = json.loads(credential)
        except (ValueError, TypeError):
            parsed = None

        if isinstance(parsed, dict):
            headers.update({str(k): str(v) for k, v in parsed.items()})
        else:
            headers["Authorization"] = credential

    return headers


def execute_tool(tool, arguments):
    """Call one tool and return whatever it said.

    Never raises for a tool that answered badly - a 500 from someone's API is
    a fact the model should relay, not a reason to fail the whole turn. Only a
    tool we refused to call at all raises.
    """
    _validate_endpoint(tool.api_endpoint)
    method = (tool.http_method or "POST").upper()
    headers = _build_headers(tool)
    url = tool.api_endpoint

    if not isinstance(arguments, dict):
        arguments = {}

    try:
        if method == "GET":
            if tool.param_type == "route":
                try:
                    url = url.format(**arguments)
                except KeyError as ex:
                    raise ToolExecutionError(
                        "Tool route is missing parameter: " + str(ex)
                    ) from ex
                # Re-checked: formatting normally only changes the path, but a
                # template with a host placeholder could change the host.
                _validate_endpoint(url)
                response = requests.get(
                    url, headers=headers, timeout=TOOL_TIMEOUT_SECONDS
                )
            else:
                response = requests.get(
                    url,
                    params=arguments,
                    headers=headers,
                    timeout=TOOL_TIMEOUT_SECONDS,
                )
        elif method == "POST":
            if tool.param_type == "query":
                response = requests.post(
                    url,
                    params=arguments,
                    headers=headers,
                    timeout=TOOL_TIMEOUT_SECONDS,
                )
            else:
                response = requests.post(
                    url, json=arguments, headers=headers, timeout=TOOL_TIMEOUT_SECONDS
                )
        else:
            raise ToolExecutionError("Unsupported HTTP method: " + str(method))
    except requests.RequestException as ex:
        logger.warning("tool call failed", extra={"tool": tool.name, "error": str(ex)})
        return {"error": "The " + tool.name + " service could not be reached."}

    try:
        payload = response.json()
    except ValueError:
        payload = response.text

    if response.status_code >= 400:
        # Returned, not raised: the model is being asked to tell a person what
        # happened, and "the API said 404" is the answer.
        return {"status_code": response.status_code, "error": payload}

    return payload


class ToolCallAccumulator:
    """Reassembles tool_calls from a stream of deltas.

    A provider sends a call's NAME once, in the first delta that mentions it,
    and its ARGUMENTS as fragments across many. Parsing any single fragment as
    JSON fails - which is exactly what the previous Groq implementation did,
    calling json.loads on whatever arrived first.

    Keyed by the delta's `index` because a model may open several calls in one
    turn and their fragments interleave.
    """

    def __init__(self):
        self._calls = {}

    def add(self, delta_tool_calls):
        if not delta_tool_calls:
            return
        for call in delta_tool_calls:
            index = getattr(call, "index", 0) or 0
            entry = self._calls.setdefault(
                index, {"id": None, "name": None, "arguments": ""}
            )
            if getattr(call, "id", None):
                entry["id"] = call.id
            function = getattr(call, "function", None)
            if function is None:
                continue
            if getattr(function, "name", None):
                entry["name"] = function.name
            if getattr(function, "arguments", None):
                entry["arguments"] += function.arguments

    def finalize(self):
        """Completed calls, in the order the model opened them.

        A call whose arguments never parsed is dropped with a log rather than
        guessed at: inventing arguments for somebody's API is worse than not
        calling it.
        """
        finished = []
        for index in sorted(self._calls):
            entry = self._calls[index]
            if not entry["name"]:
                continue
            raw = entry["arguments"].strip() or "{}"
            try:
                arguments = json.loads(raw)
            except ValueError:
                logger.warning(
                    "discarding tool call with unparseable arguments",
                    extra={"tool": entry["name"], "arguments": raw[:200]},
                )
                continue
            finished.append(
                {
                    "id": entry["id"] or ("call_" + str(index)),
                    "name": entry["name"],
                    "arguments": arguments,
                }
            )
        return finished

    def __bool__(self):
        return any(entry["name"] for entry in self._calls.values())
