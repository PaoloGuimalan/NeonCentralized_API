"""Which credential a tool actually sends.

A tool has two places to put one - the declared `headers_schema` and the
`authentication` convenience field - and for a while the second silently
overwrote the first. That cost a real afternoon: a tool whose form showed a
correct `Authorization: Bearer clt_...` kept returning 401, because a stale
value left in `authentication` from an earlier attempt was applied afterwards.
Nothing visible in the form was wrong.

So these assert precedence, not just presence.
"""

from django.test import TestCase

from llm.models import Tool
from llm.utils.tool_execution import _build_headers
from neon.testing import make_account, make_organization

BOT_TOKEN = "Bearer clt_aaaaaaaaaaaa_" + "b" * 64
STALE_JWT = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.stale.value"


class ToolHeaderTests(TestCase):
    def setUp(self):
        self.org = make_organization(make_account("alice"), "Acme", "acme")

    def _tool(self, **kwargs):
        return Tool(
            organization=self.org,
            name="search_entities",
            api_endpoint="https://example.test/v1/entities/search",
            http_method="GET",
            param_type="query",
            **kwargs,
        )

    def test_a_declared_authorization_is_not_overwritten(self):
        """The bug, exactly. Both fields set, and the declared one must win."""
        tool = self._tool(
            headers_schema={"Authorization": BOT_TOKEN},
            requires_auth=True,
            authentication=STALE_JWT,
        )

        self.assertEqual(_build_headers(tool)["Authorization"], BOT_TOKEN)

    def test_the_match_ignores_header_case(self):
        """HTTP header names are case-insensitive, so a lowercase declared
        header has to block the credential too - otherwise both go on the
        request and the HTTP library decides."""
        tool = self._tool(
            headers_schema={"authorization": BOT_TOKEN},
            requires_auth=True,
            authentication=STALE_JWT,
        )

        headers = _build_headers(tool)
        self.assertEqual(headers["authorization"], BOT_TOKEN)
        self.assertNotIn(STALE_JWT, headers.values())

    def test_the_credential_still_fills_a_gap(self):
        """It is a convenience for the common case, and that has to keep
        working - most tools set no headers at all."""
        tool = self._tool(headers_schema={}, requires_auth=True, authentication=BOT_TOKEN)

        self.assertEqual(_build_headers(tool)["Authorization"], BOT_TOKEN)

    def test_it_fills_gaps_beside_unrelated_declared_headers(self):
        tool = self._tool(
            headers_schema={"X-Tenant": "acme"},
            requires_auth=True,
            authentication=BOT_TOKEN,
        )

        headers = _build_headers(tool)
        self.assertEqual(headers["X-Tenant"], "acme")
        self.assertEqual(headers["Authorization"], BOT_TOKEN)

    def test_a_json_credential_merges_per_header(self):
        """`authentication` also accepts an object. Precedence is per HEADER,
        not all-or-nothing: the declared Authorization stands and the key the
        object also carries is still added."""
        tool = self._tool(
            headers_schema={"Authorization": BOT_TOKEN},
            requires_auth=True,
            authentication='{"Authorization": "Bearer stale", "X-Api-Key": "k-1"}',
        )

        headers = _build_headers(tool)
        self.assertEqual(headers["Authorization"], BOT_TOKEN)
        self.assertEqual(headers["X-Api-Key"], "k-1")

    def test_requires_auth_off_ignores_the_credential(self):
        tool = self._tool(
            headers_schema={"X-Tenant": "acme"},
            requires_auth=False,
            authentication=BOT_TOKEN,
        )

        self.assertNotIn("Authorization", _build_headers(tool))

    def test_a_tool_with_neither_sends_no_headers(self):
        tool = self._tool(headers_schema={}, requires_auth=False, authentication="")

        self.assertEqual(_build_headers(tool), {})

    def test_a_non_dict_headers_schema_is_ignored_rather_than_fatal(self):
        """The column is JSON and nothing stops a list being stored in it."""
        tool = self._tool(
            headers_schema=["not", "a", "mapping"],
            requires_auth=True,
            authentication=BOT_TOKEN,
        )

        self.assertEqual(_build_headers(tool)["Authorization"], BOT_TOKEN)
