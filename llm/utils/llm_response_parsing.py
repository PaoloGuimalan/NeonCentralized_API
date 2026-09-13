"""Cleaning up function-call markup a model leaked into its prose.

WHAT THIS USED TO DO, AND WHY IT WAS WRONG
------------------------------------------
This module used to match `<function=name>{...}</function>` and hand it to
`dispatch_function_call`, which executed nothing and returned the literal
string:

    Performing 'name' with parameters {...}. Please provide required inputs...

That string was then saved as the assistant's message and shown to the user, so
any model that emitted inline call markup produced visible nonsense instead of
an answer - while `llm/utils/function_calls.py`, which could actually make the
call, was never reached from this path at all.

Tools are now executed by the provider's own `tools` / `tool_calls` protocol in
llm/services/base.py, before a token is ever yielded. So a model emitting this
markup as TEXT means it ignored the protocol and described a call instead of
making one. There is nothing to dispatch: the arguments were never validated
against a schema and the model has already moved on.

Stripping it is the honest handling. The surrounding prose is usually a
complete answer, and what remains is at worst short - never fabricated.
"""

import logging
import re

logger = logging.getLogger(__name__)

# Both shapes seen in the wild from Llama-family models: the XML-ish form and
# a bare <function_call> block.
FUNCTION_MARKUP = re.compile(
    r"<function(?:_call)?=?(\w+)?>.*?</function(?:_call)?>",
    re.DOTALL,
)


def handle_llm_response(llm_response):
    """Return the model's prose with any leaked call markup removed."""
    if not llm_response:
        return ""

    cleaned, replaced = FUNCTION_MARKUP.subn("", llm_response)
    if replaced:
        logger.info(
            "stripped inline function markup from a reply",
            extra={"occurrences": replaced},
        )

    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()

    if not cleaned:
        # The reply was markup and nothing else. Saying so is better than
        # saving an empty message that renders as a blank bubble.
        return (
            "Sorry, I was not able to complete that action. "
            "Please try rephrasing your request."
        )

    return cleaned
