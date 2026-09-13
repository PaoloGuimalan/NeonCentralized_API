"""The streaming chat loop shared by every OpenAI-compatible provider.

WHY THIS IS ONE IMPLEMENTATION AND NOT TWO
------------------------------------------
GroqService and OpenAIService each carried their own copy of "stream tokens,
notice a tool call, run it, ask the model to continue". The copies drifted, and
both were wrong in different ways:

  * Groq called json.loads() on `delta.function_call.arguments` - a single
    STREAMING FRAGMENT. A provider sends a call's name once and its arguments
    in pieces, so that parse throws on anything but the shortest argument list.
  * Both used the deprecated `functions=` / `function_call=` parameters rather
    than `tools=` / `tool_calls`.
  * Both supported exactly one round: a model that needed to look something up
    and then act on the result had no way to.

The providers differ only in how the client is constructed, so that is the only
thing the subclasses own.

CLIENTS ARE PER INSTANCE, NOT PER CLASS
---------------------------------------
The previous implementations declared `client` and `model` as CLASS attributes
guarded by `if self.client is None`. The first organization to send a message
cached its API key and model on the class, and every organization afterwards
silently used them - billing one tenant for another's traffic and answering
with a model nobody chose. They are instance attributes here, and there is no
caching layer to get this wrong again.
"""

import json
import logging

from ..utils.tool_execution import (
    MAX_TOOL_ROUNDS,
    ToolCallAccumulator,
    ToolExecutionError,
    build_tool_specs,
    execute_tool,
)

logger = logging.getLogger(__name__)


class ChatCompletionService:
    """Streaming chat with a bounded tool-calling loop."""

    def __init__(self, api_key, model):
        if not api_key:
            raise ValueError("An API key is required to reach the model provider.")
        # Instance state. See the module docstring for what happened when this
        # was class state.
        self.client = self.build_client(api_key)
        self.model = model

    def build_client(self, api_key):
        raise NotImplementedError

    # ------------------------------------------------------------ messages --

    def build_messages(self, history, system_prompt, user_message, tools):
        """The opening message list.

        The tool CATALOG is not described in the prompt. The provider is told
        about tools through the `tools=` parameter, which is where a model
        actually looks; repeating them as prose was how `Tool.authentication`
        ended up in the prompt in the first place, and it competes with the
        schema rather than reinforcing it.
        """
        instructions = system_prompt or ""
        if tools:
            instructions += (
                "\n\nYou have tools available. Use them whenever they would "
                "answer the request more accurately than your own knowledge, "
                "and describe what you did in plain language afterwards."
            )

        return (
            list(history or [])
            + [{"role": "system", "content": instructions}]
            + [{"role": "user", "content": user_message}]
        )

    # ---------------------------------------------------------------- stream --

    def stream_chat_completion(
        self, history, system_prompt, user_message, tools=None
    ):
        """Yield reply tokens, running any tools the model asks for.

        `tools` is a sequence of llm.models.Tool INSTANCES, not serialized
        data - execute_tool() needs the credential, and the credential must
        never be in anything handed to a provider.
        """
        tools = list(tools or [])
        by_name = {tool.name: tool for tool in tools}
        tool_specs = build_tool_specs(tools) if tools else None

        messages = self.build_messages(history, system_prompt, user_message, tools)

        for round_index in range(MAX_TOOL_ROUNDS):
            # Tools are offered only while rounds remain. On the final pass
            # they are withheld, which is what turns "the model wants to call
            # something again" into a written answer rather than a truncated
            # turn.
            offer_tools = tool_specs and round_index < MAX_TOOL_ROUNDS - 1

            accumulator = ToolCallAccumulator()
            spoke = False

            kwargs = {"model": self.model, "messages": messages, "stream": True}
            if offer_tools:
                kwargs["tools"] = tool_specs
                kwargs["tool_choice"] = "auto"

            for chunk in self.client.chat.completions.create(**kwargs):
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta

                if getattr(delta, "tool_calls", None):
                    accumulator.add(delta.tool_calls)
                    continue

                content = getattr(delta, "content", None)
                if content:
                    spoke = True
                    yield content

            calls = accumulator.finalize() if accumulator else []
            if not calls:
                # Nothing more to do. If the model produced neither text nor a
                # usable call, say so rather than ending the stream silently -
                # an empty reply reads as the product being broken.
                if not spoke and round_index == 0:
                    yield (
                        "Sorry, I was not able to produce a reply to that. "
                        "Please try rephrasing."
                    )
                return

            messages.append(self._assistant_turn(calls))
            for call in calls:
                messages.append(self._tool_result(call, by_name))

        logger.info(
            "tool round limit reached", extra={"rounds": MAX_TOOL_ROUNDS}
        )

    # ------------------------------------------------------------- tool turns --

    @staticmethod
    def _assistant_turn(calls):
        """The assistant message recording what the model asked to call.

        Required before any tool result: a `role="tool"` message whose
        `tool_call_id` matches nothing in the preceding turn is rejected by
        the provider.
        """
        return {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": call["id"],
                    "type": "function",
                    "function": {
                        "name": call["name"],
                        "arguments": json.dumps(call["arguments"]),
                    },
                }
                for call in calls
            ],
        }

    @staticmethod
    def _tool_result(call, by_name):
        """One tool's result, as the model will read it.

        A tool that failed still produces a result message. The model is being
        asked to tell a person what happened, and "that lookup failed" is an
        answer; omitting the message would leave a dangling tool_call_id and
        fail the whole request.
        """
        tool = by_name.get(call["name"])
        if tool is None:
            payload = {"error": "No such tool: " + str(call["name"])}
        else:
            try:
                payload = execute_tool(tool, call["arguments"])
            except ToolExecutionError as ex:
                logger.warning(
                    "refused tool call",
                    extra={"tool": call["name"], "error": str(ex)},
                )
                payload = {"error": str(ex)}

        return {
            "role": "tool",
            "tool_call_id": call["id"],
            "content": json.dumps(payload, default=str),
        }

    # ------------------------------------------------------------ summaries --

    def summarize_messages(self, messages):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Summarize the following conversation for use as "
                        "history context by another model. Be concise but "
                        "keep every fact, decision and name that a later "
                        "turn might need."
                    ),
                },
                {"role": "user", "content": json.dumps(messages, default=str)},
            ],
        )
        return response.choices[0].message.content
